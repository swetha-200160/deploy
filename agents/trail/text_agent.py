"""TextAgent — analysis for text columns."""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from core.groq_rate_limiter import groq_invoke

from config import Config, get_config
from core.state import AgentResult, ColumnRoute
from agents.chart_agent import ChartAgent
from agents.pandas_analysis_agent import run_analysis_from_df

log = logging.getLogger(__name__)

_BATCH_SIZE = 10


def _build_summary(col: str, table: str, a: dict) -> str:
    bs = a.get("basic_stats", {})
    sent = a.get("sentiment", {})
    words = a.get("top_words", {})
    comp = a.get("complaints", {})
    tq = a.get("text_quality", {})
    top_kw = list(words.keys())[:8]
    return (
        f"Column '{col}' | Table '{table}' | {bs.get('total_texts', 0)} text records\n"
        f"Avg length: {bs.get('avg_length')} chars, avg words: {bs.get('avg_word_count')}\n"
        f"Sentiment: {sent.get('positive_pct')}% positive, "
        f"{sent.get('negative_pct')}% negative, "
        f"{sent.get('neutral_pct')}% neutral\n"
        f"Complaints: {comp.get('complaint_count')} ({comp.get('complaint_pct')}%)\n"
        f"Top keywords: {', '.join(top_kw)}\n"
        f"Unique texts: {tq.get('unique_count')}, duplicates: {tq.get('duplicate_text_count')}"
    )


def _parse_batch_insights(content: str, cols: list[str], summaries: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for col in cols:
        pattern = rf'\[INSIGHTS:\s*{re.escape(col)}\]\s*(.*?)(?=\[INSIGHTS:|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            if text:
                result[col] = text
                continue
        log.warning("TextAgent: no LLM insight parsed for column '%s', using stats fallback", col)
        result[col] = summaries.get(col, "")
    return result


class TextAgent:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        self._chart_agent = ChartAgent(self.config)

    def _batch_insights(self, summaries: dict[str, str]) -> dict[str, str]:
        cols = list(summaries.keys())
        prompt_parts = [
            "Analyse the following text financial data columns. For each column, write 2-3 sentences "
            "of clear, actionable insights focusing on sentiment, complaints, and key themes.\n\n"
            "Use EXACTLY this format for every column (no extra text between entries):\n"
            "[INSIGHTS: column_name]\nYour 2-3 sentence insight here.\n\n"
        ]
        for col in cols:
            prompt_parts.append(f"--- Data for column '{col}' ---\n{summaries[col]}\n\n")

        try:
            batch_llm = self._llm.with_config({"max_tokens": min(4096, max(self.config.llm_max_tokens, len(cols) * 120))})
            msgs = [
                SystemMessage("You are a financial text analyst. Return insights for each column in the exact format requested. Do not skip any column."),
                HumanMessage("".join(prompt_parts)),
            ]
            response = groq_invoke(batch_llm, msgs, agent_name="text")
            content = response.content if isinstance(response.content, str) else ""
            return _parse_batch_insights(content, cols, summaries)
        except Exception as exc:
            log.error("TextAgent batch LLM error: %s", exc)
            return {col: summaries[col] for col in cols}

    def _run_table(self, df: pd.DataFrame, table: str, routes: list[ColumnRoute]) -> list[AgentResult]:
        # Step 1: compute all analyses (pure pandas, no LLM, no DB)
        analyses: dict[str, dict] = {}
        for route in routes:
            col = route["column"]
            try:
                analyses[col] = run_analysis_from_df(df, col, "text", self.config)
            except Exception as exc:
                log.error("TextAgent compute error on %s.%s: %s", table, col, exc)
                analyses[col] = {}

        # Step 2: build summaries
        summaries = {col: _build_summary(col, table, a) for col, a in analyses.items()}

        # Step 3: batched LLM calls
        insights_map: dict[str, str] = {}
        col_list = [r["column"] for r in routes]
        for i in range(0, len(col_list), _BATCH_SIZE):
            batch_cols = col_list[i: i + _BATCH_SIZE]
            batch_summaries = {c: summaries[c] for c in batch_cols if c in summaries}
            insights_map.update(self._batch_insights(batch_summaries))
            log.info("TextAgent %s: batch %d-%d insights done", table, i + 1, i + len(batch_cols))

        # Step 4: assemble results
        results: list[AgentResult] = []
        for route in routes:
            col = route["column"]
            analysis = analyses.get(col, {})
            try:
                words = analysis.get("top_words", {})
                kw_labels = list(words.keys())[:15]
                kw_values = [float(words[k]) for k in kw_labels]
                sent = analysis.get("sentiment", {})
                sentiment_labels = ["Positive", "Negative", "Neutral"]
                sentiment_values = [
                    sent.get("positive_count", 0),
                    sent.get("negative_count", 0),
                    sent.get("neutral_count", 0),
                ]
                chart_data = {
                    "keyword_labels": kw_labels,
                    "keyword_values": kw_values,
                    "sentiment_labels": sentiment_labels,
                    "sentiment_values": sentiment_values,
                }
                chart_type = self._chart_agent.pick_text_chart(
                    kw_labels, kw_values, sentiment_labels, sentiment_values
                )
            except Exception as exc:
                log.error("ChartAgent error on %s.%s: %s", table, col, exc)
                chart_type, chart_data = "bar", {}

            results.append(AgentResult(
                agent_name="text",
                table=table,
                column=col,
                analysis=analysis,
                insights_text=insights_map.get(col, summaries.get(col, "")),
                chart_data=chart_data,
                chart_type=chart_type,
                error=None,
            ))
        return results

    def run(self, raw_data: dict, routes: list[ColumnRoute]) -> list[AgentResult]:
        by_table: dict[str, list[ColumnRoute]] = defaultdict(list)
        for route in routes:
            table = route["table"]
            if raw_data.get(table) is not None and not raw_data[table].empty:
                by_table[table].append(route)

        results: list[AgentResult] = []
        for table, table_routes in by_table.items():
            results.extend(self._run_table(raw_data[table], table, table_routes))
        return results
