from __future__ import annotations

import json
import logging

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import Config, get_config

log = logging.getLogger(__name__)

# ── Categorical chart tools (pie · bar · stacked_bar · countplot) ─────────────
_CATEGORICAL_TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "make_countplot",
        "description": (
            "Frequency bar chart — each bar = count of one category value. "
            "Best when unique categories > 6 and you want to show raw count distribution."
        ),
        "parameters": {"type": "object", "properties": {
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Category labels (x-axis)"},
            "values": {"type": "array", "items": {"type": "integer"}, "description": "Row count for each label"},
        }, "required": ["labels", "values"]},
    }},
    {"type": "function", "function": {
        "name": "make_bar_chart",
        "description": (
            "Standard bar chart — compares a numeric value across categories. "
            "Use when there is no stacking needed and categories > 6."
        ),
        "parameters": {"type": "object", "properties": {
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Category labels"},
            "values": {"type": "array", "items": {"type": "number"}, "description": "Numeric value per label"},
        }, "required": ["labels", "values"]},
    }},
    {"type": "function", "function": {
        "name": "make_pie_chart",
        "description": (
            "Pie/donut chart — shows proportional share of each category. "
            "Best when unique categories <= 6 (part-to-whole relationship)."
        ),
        "parameters": {"type": "object", "properties": {
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Slice labels"},
            "values": {"type": "array", "items": {"type": "number"}, "description": "Slice sizes (counts or proportions)"},
        }, "required": ["labels", "values"]},
    }},
    {"type": "function", "function": {
        "name": "make_stacked_bar_chart",
        "description": (
            "Stacked bar chart — one bar per category, stacked by a numeric metric series. "
            "Best when the table has numeric columns and you want a category vs. metric breakdown."
        ),
        "parameters": {"type": "object", "properties": {
            "x_axis": {"type": "array", "items": {"type": "string"}, "description": "Category labels (x-axis)"},
            "series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "values": {"type": "array", "items": {"type": "number"}},
                    },
                },
                "description": "One or more series, each with a name and per-category values",
            },
        }, "required": ["x_axis", "series"]},
    }},
]

# ── Numerical chart tools (histogram · heatmap) ───────────────────────────────
_NUMERICAL_TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "make_histogram",
        "description": (
            "Histogram showing frequency distribution of a single numeric column. "
            "Best when only one numeric column is available."
        ),
        "parameters": {"type": "object", "properties": {
            "bins":      {"type": "array", "items": {"type": "integer"}, "description": "Frequency count per bin"},
            "bin_edges": {"type": "array", "items": {"type": "number"},  "description": "Bin boundary values"},
            "column":    {"type": "string", "description": "Column name"},
        }, "required": ["bins", "bin_edges", "column"]},
    }},
    {"type": "function", "function": {
        "name": "make_heatmap",
        "description": (
            "Correlation heatmap across multiple numeric columns. "
            "Best when 2 or more numeric columns are available."
        ),
        "parameters": {"type": "object", "properties": {
            "correlation": {"type": "object", "description": "Pairwise correlation dict {col: {col: value}}"},
            "columns":     {"type": "array", "items": {"type": "string"}, "description": "Column names"},
        }, "required": ["correlation", "columns"]},
    }},
]

# ── Datetime chart tool (line) ────────────────────────────────────────────────
_DATETIME_TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "make_line_chart",
        "description": "Time series line chart with optional rolling average. Always use for datetime/temporal columns.",
        "parameters": {"type": "object", "properties": {
            "dates":        {"type": "array", "items": {"type": "string"}, "description": "ISO date strings (x-axis)"},
            "values":       {"type": "array", "items": {"type": "number"}, "description": "Numeric values per date"},
            "rolling_mean": {"type": "array", "items": {"type": "number"}, "description": "Optional rolling average series"},
            "value_column": {"type": "string", "description": "Name of the value column"},
            "column":       {"type": "string", "description": "Name of the datetime column"},
        }, "required": ["dates", "values"]},
    }},
]

# ── Text chart tools (bar · pie) ──────────────────────────────────────────────
_TEXT_TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "make_bar_chart",
        "description": "Bar chart showing top keyword TF-IDF importance scores. Best when the primary insight is keyword distribution.",
        "parameters": {"type": "object", "properties": {
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Keyword labels"},
            "values": {"type": "array", "items": {"type": "number"}, "description": "TF-IDF scores"},
        }, "required": ["labels", "values"]},
    }},
    {"type": "function", "function": {
        "name": "make_pie_chart",
        "description": "Pie chart showing Positive/Negative/Neutral sentiment split. Best when sentiment distribution is the primary insight.",
        "parameters": {"type": "object", "properties": {
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Sentiment category labels"},
            "values": {"type": "array", "items": {"type": "number"}, "description": "Count per sentiment category"},
        }, "required": ["labels", "values"]},
    }},
]

_TOOL_TO_CHART_TYPE: dict[str, str] = {
    "make_countplot":        "countplot",
    "make_bar_chart":        "bar",
    "make_pie_chart":        "pie",
    "make_stacked_bar_chart":"stacked_bar",
    "make_histogram":        "histogram",
    "make_heatmap":          "heatmap",
    "make_line_chart":       "line",
}


class ChartAgent:
    """Single agent that exposes all 7 chart types as LLM tools.

    Each data pipeline (categorical, numerical, datetime, text) has its own
    pick_* method with a focused tool subset.  Callers receive (chart_type, chart_data)
    ready to pass to the matching make_* render function in chart_generator.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        _base = dict(
            google_api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
            temperature=0,
            max_output_tokens=512,
            thinking_budget=0,
        )
        self._llm_categorical = ChatGoogleGenerativeAI(**_base).bind_tools(_CATEGORICAL_TOOL_DEFS)
        self._llm_numerical   = ChatGoogleGenerativeAI(**_base).bind_tools(_NUMERICAL_TOOL_DEFS)
        self._llm_datetime    = ChatGoogleGenerativeAI(**_base).bind_tools(_DATETIME_TOOL_DEFS)
        self._llm_text        = ChatGoogleGenerativeAI(**_base).bind_tools(_TEXT_TOOL_DEFS)

    # ── Categorical ───────────────────────────────────────────────────────────

    def pick_chart(
        self,
        column_name: str,
        series: pd.Series,
        df: pd.DataFrame,
    ) -> tuple[str, dict]:
        """Return (chart_type, chart_data) for a categorical column.

        Tools available: make_countplot · make_bar_chart · make_pie_chart · make_stacked_bar_chart
        """
        full_series = (
            df[column_name].dropna() if column_name in df.columns else series.dropna()
        )
        counts = full_series.value_counts()
        labels = [str(k) for k in counts.index.tolist()]
        values = counts.values.tolist()
        unique = int(full_series.nunique())
        num_cols = df.select_dtypes(include="number").columns.tolist()

        seg_data: dict = {}
        if num_cols and unique <= 20:
            try:
                num_col = num_cols[0]
                seg = (
                    df.dropna(subset=[column_name, num_col])
                    .groupby(column_name)[num_col]
                    .mean()
                    .round(4)
                )
                seg_data = {
                    "x_axis": [str(x) for x in seg.index.tolist()],
                    "series": [{"name": num_col, "values": seg.values.tolist()}],
                }
            except Exception:
                pass

        system = (
            "You are a data visualisation expert. Given a categorical column's statistics, "
            "call EXACTLY ONE chart tool that best suits the data.\n"
            "  make_countplot        → unique > 6, show raw count distribution\n"
            "  make_pie_chart        → unique ≤ 6, show percentage/proportion\n"
            "  make_stacked_bar_chart → categories + numeric metric available\n"
            "  make_bar_chart        → general comparison when others don't apply\n"
            "You MUST call a tool — do not reply with plain text."
        )
        context = [
            f"Column: '{column_name}' | Unique values: {unique}",
            f"Numeric columns in table: {num_cols[:5] or 'none'}",
            f"Top value counts (label → count): {json.dumps(dict(zip(labels[:10], values[:10])))}",
        ]
        if seg_data:
            context.append(
                f"Numeric segmentation (mean of '{num_cols[0]}' per category): "
                f"{json.dumps(seg_data, default=str)[:400]}"
            )

        messages = [SystemMessage(content=system), HumanMessage(content="\n".join(context))]
        for _ in range(3):
            response = self._llm_categorical.invoke(messages)
            if response.tool_calls:
                tc = response.tool_calls[0]
                if tc["name"] in _TOOL_TO_CHART_TYPE:
                    return _TOOL_TO_CHART_TYPE[tc["name"]], tc["args"]
            messages.append(response)

        log.warning("ChartAgent: no tool call for categorical '%s', using fallback.", column_name)
        if unique <= 6 and labels:
            return "pie", {"labels": labels, "values": values}
        if seg_data:
            return "stacked_bar", seg_data
        return "countplot", {"labels": labels[:20], "values": values[:20]}

    # ── Numerical ─────────────────────────────────────────────────────────────

    def pick_numerical_chart(
        self,
        col: str,
        num_cols: list[str],
        histogram_data: dict,
        heatmap_data: dict,
    ) -> tuple[str, dict]:
        """Return (chart_type, chart_data) for a numerical column.

        Tools available: make_histogram · make_heatmap
        """
        system = (
            "You are a data visualisation expert. Choose the best chart for this numerical column.\n"
            "  make_histogram → single numeric column, show value distribution shape\n"
            "  make_heatmap   → 2+ numeric columns, show pairwise correlation matrix\n"
            "You MUST call a tool."
        )
        context = "\n".join([
            f"Column: '{col}' | Total numeric columns in table: {len(num_cols)} {num_cols[:6]}",
            f"Histogram data ready: {len(histogram_data.get('bins', []))} bins",
            f"Heatmap data ready: {len(heatmap_data.get('columns', []))} columns in correlation matrix",
        ])
        messages = [SystemMessage(content=system), HumanMessage(content=context)]
        for _ in range(3):
            response = self._llm_numerical.invoke(messages)
            if response.tool_calls:
                tool_name = response.tool_calls[0]["name"]
                if tool_name == "make_histogram":
                    return "histogram", histogram_data
                if tool_name == "make_heatmap":
                    return "heatmap", heatmap_data
            messages.append(response)

        log.warning("ChartAgent: no tool call for numerical '%s', using fallback.", col)
        return ("heatmap", heatmap_data) if len(num_cols) > 1 else ("histogram", histogram_data)

    # ── Datetime ──────────────────────────────────────────────────────────────

    def pick_datetime_chart(
        self,
        col: str,
        date_sample: list,
        values: list,
        trend_data: dict,
        value_col: str | None,
    ) -> tuple[str, dict]:
        """Return ('line', chart_data) for a datetime column.

        Tools available: make_line_chart
        """
        chart_data = {
            "dates":        trend_data.get("dates", date_sample),
            "values":       trend_data.get("values", values),
            "rolling_mean": trend_data.get("rolling_mean", []),
            "column":       col,
            "value_column": value_col or "count",
        }

        # Skip LLM entirely when there are no valid date points — the tool
        # would not be called anyway and we'd waste 3 API round-trips.
        if not chart_data.get("dates"):
            log.debug("ChartAgent: datetime '%s' has no date points, using line fallback.", col)
            return "line", chart_data

        system = (
            "You are a data visualisation expert. "
            "For datetime/temporal data call make_line_chart with the time series. "
            "You MUST call the tool."
        )
        context = (
            f"Datetime column: '{col}' | {len(date_sample)} date points | "
            f"Value column: '{value_col or 'count'}' | "
            f"Trend: {trend_data.get('trend_direction', 'unknown')}"
        )
        messages = [SystemMessage(content=system), HumanMessage(content=context)]
        for _ in range(3):
            response = self._llm_datetime.invoke(messages)
            if response.tool_calls and response.tool_calls[0]["name"] == "make_line_chart":
                return "line", chart_data
            messages.append(response)

        log.warning("ChartAgent: no tool call for datetime '%s', using fallback.", col)
        return "line", chart_data

    # ── Text ──────────────────────────────────────────────────────────────────

    def pick_text_chart(
        self,
        kw_labels: list,
        kw_values: list,
        sentiment_labels: list,
        sentiment_values: list,
    ) -> str:
        """Return the primary chart_type for a text column ('bar' or 'pie').

        Tools available: make_bar_chart (keywords) · make_pie_chart (sentiment)
        Note: chart_data in AgentResult stays as the compound dict so that
        _build_text_charts() can still produce both keyword and sentiment charts.
        """
        system = (
            "You are a data visualisation expert. Choose the primary chart for this text analysis.\n"
            "  make_bar_chart → show top keyword TF-IDF importance scores\n"
            "  make_pie_chart → show Positive/Negative/Neutral sentiment split\n"
            "You MUST call a tool."
        )
        context = "\n".join([
            f"Top keywords: {kw_labels[:5]} | scores: {[round(v, 4) for v in kw_values[:5]]}",
            f"Sentiment: {dict(zip(sentiment_labels, sentiment_values))}",
        ])
        messages = [SystemMessage(content=system), HumanMessage(content=context)]
        for _ in range(3):
            response = self._llm_text.invoke(messages)
            if response.tool_calls and response.tool_calls[0]["name"] in _TOOL_TO_CHART_TYPE:
                return _TOOL_TO_CHART_TYPE[response.tool_calls[0]["name"]]
            messages.append(response)

        log.warning("ChartAgent: no tool call for text, using fallback.")
        return "bar"
