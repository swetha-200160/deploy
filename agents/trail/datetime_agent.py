"""DatetimeAgent — analysis for datetime columns."""
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
    v = a.get("validation", {})
    md = a.get("missing_dates", {})
    tf = a.get("time_features", {})
    tr = a.get("trend", {})
    sea = a.get("seasonality", {})
    anom = a.get("anomalies", {})
    fc = a.get("forecast", {})
    return (
        f"Column '{col}' | Table '{table}'\n"
        f"Date range: {v.get('min_date')} → {v.get('max_date')} ({v.get('date_range_days')} days)\n"
        f"Valid: {v.get('valid_count')}, Invalid: {v.get('invalid_count')}\n"
        f"Completeness: {md.get('completeness_score')}%, missing dates: {md.get('missing_count')}, "
        f"largest gap: {md.get('largest_gap_days')} days\n"
        f"Busiest: day={tf.get('busiest_day')}, month={tf.get('busiest_month')}, "
        f"quarter={tf.get('busiest_quarter')}, weekend%={tf.get('weekend_pct')}\n"
        f"Trend: {tr.get('trend_direction', 'N/A')}, slope={tr.get('slope', 'N/A')}\n"
        f"Seasonality strength: {sea.get('seasonality_strength', 'N/A')}, "
        f"peak month: {sea.get('peak_month', 'N/A')}, trough: {sea.get('trough_month', 'N/A')}\n"
        f"Anomalies: {len(anom.get('spike_dates', []))} spikes, {len(anom.get('dip_dates', []))} dips\n"
        f"Forecast: {fc.get('direction', 'N/A')}, slope={fc.get('trend_slope', 'N/A')}"
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
        log.warning("DatetimeAgent: no LLM insight parsed for column '%s', using stats fallback", col)
        result[col] = summaries.get(col, "")
    return result


class DatetimeAgent:
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
            "Analyse the following datetime financial data columns. For each column, write 2-3 sentences "
            "of clear, actionable insights focusing on trends, seasonality, and anomalies.\n\n"
            "Use EXACTLY this format for every column (no extra text between entries):\n"
            "[INSIGHTS: column_name]\nYour 2-3 sentence insight here.\n\n"
        ]
        for col in cols:
            prompt_parts.append(f"--- Data for column '{col}' ---\n{summaries[col]}\n\n")

        try:
            batch_llm = self._llm.with_config({"max_tokens": min(4096, max(self.config.llm_max_tokens, len(cols) * 120))})
            msgs = [
                SystemMessage("You are a financial data analyst. Return insights for each column in the exact format requested. Do not skip any column."),
                HumanMessage("".join(prompt_parts)),
            ]
            response = groq_invoke(batch_llm, msgs, agent_name="datetime")
            content = response.content if isinstance(response.content, str) else ""
            return _parse_batch_insights(content, cols, summaries)
        except Exception as exc:
            log.error("DatetimeAgent batch LLM error: %s", exc)
            return {col: summaries[col] for col in cols}

    def _run_table(self, df: pd.DataFrame, table: str, routes: list[ColumnRoute]) -> list[AgentResult]:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        value_col = num_cols[0] if num_cols else None

        # Step 1: compute all analyses (pure pandas, no LLM, no DB)
        analyses: dict[str, dict] = {}
        for route in routes:
            col = route["column"]
            try:
                analyses[col] = run_analysis_from_df(df, col, "datetime", self.config)
            except Exception as exc:
                log.error("DatetimeAgent compute error on %s.%s: %s", table, col, exc)
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
            log.info("DatetimeAgent %s: batch %d-%d insights done", table, i + 1, i + len(batch_cols))

        # Step 4: assemble results
        results: list[AgentResult] = []
        for route in routes:
            col = route["column"]
            analysis = analyses.get(col, {})
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                valid_mask = dt_series.notna()
                date_sample = dt_series[valid_mask].dt.strftime("%Y-%m-%d").tolist()
                values = df.loc[valid_mask, value_col].tolist() if value_col else []
                trend = analysis.get("trend", {})
                chart_type, chart_data = self._chart_agent.pick_datetime_chart(
                    col, date_sample, values, trend, value_col
                )
            except Exception as exc:
                log.error("ChartAgent error on %s.%s: %s", table, col, exc)
                chart_type, chart_data = "line", {}

            results.append(AgentResult(
                agent_name="datetime",
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
