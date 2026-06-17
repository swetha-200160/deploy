"""NumericalAgent — row-wise entity-level analysis.

Flow per table:
  1. Pre-compute IQR bounds/means for all numerical columns  (zero LLM tokens)
  2. For each row, build a compact value summary with outlier flags (!)
  3. Batch rows → ONE LLM call per batch → row-level audit insights

Tables ≤ _MAX_ROWS_FOR_FULL rows  → analyze every row.
Larger tables                      → only rows with ≥1 IQR outlier (capped at 200).

Token budget (10 rows/batch, 15 cols):
  Input : ~150 tok/row × 10 = ~1,500   Output: ~70 tok/row × 10 = ~700
  Total : ~2,200/batch — fits comfortably in llama-3.1-8b-instant TPM.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from core.groq_rate_limiter import groq_invoke

from config import Config, get_config
from core.state import AgentResult, ColumnRoute
from agents.chart_agent import ChartAgent
from agents.pandas_analysis_agent import run_analysis_from_df

log = logging.getLogger(__name__)

# Row-wise batching — ~160 tokens per row (input+output), TPM=6000 → 10 rows/call safe
_MAX_COLS_PER_CALL = 9        # kept for _table_insights (column-wise fallback)
_MAX_ROWS_PER_CALL = 10       # rows per LLM batch for row-wise analysis
_MAX_ROWS_FOR_FULL = 100      # analyze ALL rows if table size ≤ this; else outliers only

_SYSTEM = """You are a senior financial audit analyst specializing in anomaly detection and fraud risk assessment.

You will receive individual transaction records from financial tables.
Values marked with ! are IQR statistical outliers compared to the column's normal range.

For each record write a 2-3 sentence insight that:
1. States the RISK LEVEL: [HIGH RISK] / [MEDIUM RISK] / [LOW RISK]
2. Names the specific anomalous values (e.g. "amount=172,600 is 25x above the column mean of 6,800")
3. Explains the BUSINESS IMPLICATION — what does this mean in financial audit terms?
4. Calls out cross-column patterns (e.g. high amount + rejected status + zero cleared = total rejection of large cheque)
5. Ends with a recommended audit ACTION (investigate, escalate, flag for review, etc.)

Do NOT say "statistical outlier" — explain what it means for the business.
Do NOT repeat the column names mechanically — tell the story of the transaction.
"""


# ── 7 analysis tools — pure pandas, zero LLM calls ───────────────────────────

def _detect_missing(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    total = len(s)
    nc = int(s.isna().sum())
    pct = round(nc / total * 100, 2) if total else 0.0
    label = ("Complete" if pct == 0 else
             "MCAR-low" if pct < 5 else
             "MAR-moderate" if pct < 20 else "MNAR-high")
    return f"{nc}/{total} ({pct}%) [{label}]"


def _detect_outliers(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return "no data"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_n = int(((s < lo) | (s > hi)).sum())
    std = float(s.std())
    z_n = int((((s - s.mean()) / std).abs() > 3).sum()) if std > 0 else 0
    return (
        f"IQR={iqr_n}({round(iqr_n/n*100,1)}%) "
        f"bounds=[{round(lo,2)},{round(hi,2)}] "
        f"Zscore={z_n}({round(z_n/n*100,1)}%)"
    )


def _analyse_distribution(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 3:
        return "insufficient data"
    skew = round(float(s.skew()), 3)
    kurt = round(float(s.kurt()), 3)
    shape = ("normal" if abs(skew) < 0.5 else
             "right-skewed" if skew > 1 else
             "left-skewed" if skew < -1 else "moderate-skew")
    return (
        f"{shape} skew={skew} kurt={kurt} "
        f"mean={round(float(s.mean()),2)} "
        f"median={round(float(s.median()),2)} "
        f"std={round(float(s.std()),2)}"
    )


def _analyse_correlation(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce")
    # Skip if this column has zero variance — correlation is undefined
    if s.dropna().std() == 0:
        return "skipped (zero variance)"
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corrs = {}
    for c in num_cols:
        if c == col:
            continue
        other = pd.to_numeric(df[c], errors="coerce")
        # Skip zero-variance columns to avoid divide-by-zero warnings
        if other.dropna().std() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r = s.corr(other)
        if not pd.isna(r):
            corrs[c] = round(float(r), 3)
    top3 = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    return ",".join(f"{c}:{v}" for c, v in top3) if top3 else "none"


def _scale_normalize(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return "no data"
    skew = float(s.skew())
    rec = "zscore" if abs(skew) > 1 else "minmax"
    return (
        f"range=[{round(float(s.min()),2)},{round(float(s.max()),2)}] "
        f"mean={round(float(s.mean()),2)} std={round(float(s.std()),2)} "
        f"rec={rec}(skew={round(skew,2)})"
    )


def _engineer_features(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return "no data"
    feats = []
    if float(s.min()) > 0:
        feats.append("log_transform")
    if float(s.min()) >= 0:
        feats.append("sqrt_transform")
    q = [round(float(s.quantile(v)), 2) for v in [0.25, 0.5, 0.75]]
    feats.append(f"quartile_bins{q}")
    rn = round(float((s % 100 == 0).sum() / n * 100), 1)
    feats.append(f"round_number_flag({rn}%)")
    neg = round(float((s < 0).sum() / n * 100), 1)
    if neg > 0:
        feats.append(f"negative_flag({neg}%)")
    return " | ".join(feats)


def _get_suspicious_rows(df: pd.DataFrame, col: str) -> str:
    """Compact summary of actual outlier rows — value + co-occurring anomalies.

    Gives LLM row-level context without sending raw data.
    Example: "amount=95000|co_outliers:[vendor_amount=88000,tax=0]"
    """
    s = pd.to_numeric(df[col], errors="coerce")
    clean = s.dropna()
    if len(clean) == 0:
        return "no data"

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_df = df[(s < lo) | (s > hi)]

    if outlier_df.empty:
        return "no outlier rows"

    num_cols = df.select_dtypes(include="number").columns.tolist()
    summaries = []
    for _, row in outlier_df.head(5).iterrows():
        val = round(float(row[col]), 2) if pd.notna(row[col]) else "NULL"
        co = []
        for c in num_cols:
            if c == col or pd.isna(row[c]):
                continue
            s2 = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s2) < 4:
                continue
            q1c, q3c = s2.quantile(0.25), s2.quantile(0.75)
            iqrc = q3c - q1c
            v = float(row[c])
            if v < q1c - 1.5 * iqrc or v > q3c + 1.5 * iqrc:
                co.append(f"{c}={round(v,2)}")
        co_str = ",".join(co[:3]) if co else "none"
        summaries.append(f"{col}={val}|co:[{co_str}]")

    return f"{len(outlier_df)} outlier rows. top5: " + "; ".join(summaries)


# ── Row-wise helpers ──────────────────────────────────────────────────────────

def _get_col_bounds(df: pd.DataFrame) -> dict[str, dict]:
    """Pre-compute outlier bounds + mean/std for every numerical column.

    Strategy:
    - Binary/constant (≤2 unique values): no outlier detection, show value as-is.
    - Normal IQR > 0: use IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    - Zero-IQR but >2 unique values (near-constant, e.g. price_var_pct mostly 0,
      grn_count mostly 1): fall back to mean ± 3*std. If std=0 too, treat as binary.
    - Lower bound is clamped to 0 for columns that cannot be negative
      (amount, qty, count, price, sum columns) — prevents nonsense like
      "normal range includes -700" in the LLM prompt.
    """
    bounds: dict[str, dict] = {}
    for col in df.select_dtypes(include="number").columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue

        unique_vals = s.nunique()
        mean = float(s.mean())
        std = float(s.std()) if float(s.std()) > 0 else 0.0
        actual_min = float(s.min())

        if unique_vals <= 2:
            # Binary or constant — no outlier flagging
            bounds[col] = {"lo": None, "hi": None, "mean": mean, "std": std, "binary": True}
            continue

        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1

        if iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
        elif std > 0:
            # Zero-IQR but variance exists — fall back to mean ± 3σ
            lo = mean - 3 * std
            hi = mean + 3 * std
        else:
            # Truly constant — treat as binary
            bounds[col] = {"lo": None, "hi": None, "mean": mean, "std": 0.0, "binary": True}
            continue

        # Dynamic clamping: if the actual data minimum is ≥ 0, this column
        # cannot be negative in reality — clamp the computed lower bound to 0.
        # This handles any column name (amount, userid, status, po_line_c, etc.)
        # without hardcoding. Columns with real negatives (price_var_pct, etc.)
        # keep their negative lower bound as-is.
        if actual_min >= 0:
            lo = max(lo, 0.0)

        bounds[col] = {
            "lo":     round(lo, 4),
            "hi":     round(hi, 4),
            "mean":   round(mean, 4),
            "std":    round(std, 4),
            "binary": False,
        }
    return bounds


def _row_summary(row: pd.Series, df: pd.DataFrame, bounds: dict) -> str:
    """Compact single-row summary: ALL categorical context + numerical values with ! flags.

    Binary/constant columns show value without ! flag (IQR outlier detection not applicable).
    Truly anomalous numerical columns are marked with ! so the LLM focuses on them.
    """
    parts: list[str] = []
    # ALL non-numerical columns — full entity context (cheque number, vendor, account, etc.)
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    for col in cat_cols:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}={str(row[col])[:30]}")
    # Numerical columns
    for col, b in bounds.items():
        if col not in row.index or pd.isna(row[col]):
            parts.append(f"{col}=NULL")
            continue
        val = float(row[col])
        if b.get("binary"):
            # Binary column — show value, no outlier flag (0/1 are both valid)
            parts.append(f"{col}={round(val, 2)}")
        else:
            flag = "!" if (val < b["lo"] or val > b["hi"]) else ""
            parts.append(f"{col}={round(val, 2)}{flag}")
    return " | ".join(parts)


# ── Tool registry ─────────────────────────────────────────────────────────────

_TOOLS: list[tuple[str, any]] = [
    ("missing",         _detect_missing),
    ("outliers",        _detect_outliers),
    ("distribution",    _analyse_distribution),
    ("correlation",     _analyse_correlation),
    ("scaling",         _scale_normalize),
    ("features",        _engineer_features),
    ("suspicious_rows", _get_suspicious_rows),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_schema_line(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce")
    clean = s.dropna()
    n = len(clean)
    null_pct = round(s.isna().mean() * 100, 1)
    if n > 0:
        return (
            f"min={clean.min():.4g} max={clean.max():.4g} "
            f"mean={clean.mean():.4g} std={clean.std():.4g} "
            f"rows={n} null={null_pct}%"
        )
    return "all null"


def _parse_insights(content: str, cols: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for col in cols:
        pattern = rf'\[INSIGHTS:\s*{re.escape(col)}\]\s*(.*?)(?=\[INSIGHTS:|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            if text:
                result[col] = text
                continue
        log.warning("NumericalAgent: no insight parsed for '%s'", col)
        result[col] = ""
    return result


# ── Agent ─────────────────────────────────────────────────────────────────────

class NumericalAgent:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model="llama-3.1-8b-instant",  # 500K TPD — saves 70b budget for insight/story agents
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        self._chart_agent = ChartAgent(self.config)

    def _table_insights(
        self, df: pd.DataFrame, table: str, cols: list[str]
    ) -> dict[str, str]:
        """One LLM call for ALL columns in the table.

        Step 1 — run all 7 tools locally (pandas, zero API calls)
        Step 2 — build one compact prompt with all results
        Step 3 — single LLM call → parse all insights
        """
        # Step 1 — pre-compute all tools, all columns
        computed: dict[str, dict[str, str]] = {}
        for col in cols:
            computed[col] = {}
            for tool_name, tool_fn in _TOOLS:
                try:
                    computed[col][tool_name] = tool_fn(df, col)
                except Exception as exc:
                    computed[col][tool_name] = f"err:{exc}"

        # Step 2 — compact prompt (CSV-style, token efficient per research)
        lines = [
            f"Table: '{table}' | {df.shape[0]} rows × {df.shape[1]} cols",
            "Pre-computed analysis for each numerical column:",
            "",
        ]
        for col in cols:
            lines.append(f"[{col}]")
            lines.append(f"  schema         : {_build_schema_line(df, col)}")
            for tool_name, result in computed[col].items():
                lines.append(f"  {tool_name:<15}: {result}")
            lines.append("")

        lines += [
            "Write 2-3 sentence audit insight per column grounded in the numbers above.",
            "Pay special attention to suspicious_rows for row-level anomalies.",
            "Use EXACTLY this format:",
            "[INSIGHTS: column_name]",
            "Your insight here.",
            "",
        ]

        # Dynamic max_tokens: ~60 tokens per column for output
        max_tokens = max(self.config.llm_max_tokens, len(cols) * 70)

        # Step 3 — single LLM call (rate-limited)
        try:
            llm = self._llm.with_config({"max_tokens": max_tokens})
            msgs = [SystemMessage(content=_SYSTEM), HumanMessage(content="\n".join(lines))]
            response = groq_invoke(llm, msgs, agent_name="numerical")
            content = response.content if hasattr(response, "content") else str(response)
            log.info("NumericalAgent %s: %d cols → 1 LLM call", table, len(cols))
            return _parse_insights(content, cols)
        except Exception as exc:
            log.error("NumericalAgent LLM error on %s: %s", table, exc)
            return {col: "" for col in cols}

    def _table_row_insights(
        self,
        df: pd.DataFrame,
        table: str,
        row_indices: list[int],
        bounds: dict,
    ) -> dict[str, str]:
        """One LLM call for a batch of rows — returns {row_key: insight_text}."""
        # Column baseline context — only non-binary columns (binary ones have no outlier range)
        col_lines = []
        for c, b in bounds.items():
            if b.get("binary"):
                col_lines.append(f"  {c}: binary/indicator (0 or 1), mean={round(b['mean'], 2)}")
            else:
                col_lines.append(
                    f"  {c}: normal_range=[{round(b['lo'], 2)}, {round(b['hi'], 2)}]"
                    f"  mean={round(b['mean'], 2)}  std={round(b['std'], 2)}"
                )

        lines = [
            f"Table: '{table}' | {df.shape[0]} total rows × {df.shape[1]} cols",
            "Column baselines (values marked ! are outside normal IQR range):",
        ] + col_lines + [""]

        row_keys: list[str] = []
        for idx in row_indices:
            key = f"row_{idx}"
            row_keys.append(key)
            lines.append(f"[{key}]")
            lines.append(f"  {_row_summary(df.iloc[idx], df, bounds)}")

        lines += [
            "",
            "For each record above, write a 2-3 sentence financial audit insight.",
            "Follow the system instructions: include risk level, name exact values, explain business impact, suggest action.",
            "Use EXACTLY this format (one block per row, no extra text):",
            "[INSIGHTS: row_N]",
            "Your insight here.",
            "",
        ]
        max_tokens = max(self.config.llm_max_tokens, len(row_indices) * 100)
        try:
            llm = self._llm.with_config({"max_tokens": max_tokens})
            msgs = [SystemMessage(content=_SYSTEM), HumanMessage(content="\n".join(lines))]
            response = groq_invoke(llm, msgs, agent_name="numerical")
            content = response.content if hasattr(response, "content") else str(response)
            log.info("NumericalAgent %s: %d rows → 1 LLM call", table, len(row_indices))
            return _parse_insights(content, row_keys)
        except Exception as exc:
            log.error("NumericalAgent row LLM error on %s: %s", table, exc)
            return {k: "" for k in row_keys}

    def _run_table(
        self, df: pd.DataFrame, table: str, routes: list[ColumnRoute]
    ) -> list[AgentResult]:
        # Step 1 — pre-compute bounds for every numerical column (zero LLM)
        bounds = _get_col_bounds(df)

        # Step 2 — decide which rows to analyze
        # Only non-binary columns participate in outlier detection for row selection
        outlier_bounds = {c: b for c, b in bounds.items() if not b.get("binary") and b["lo"] is not None}

        total_rows = len(df)
        if total_rows <= _MAX_ROWS_FOR_FULL:
            row_indices = list(range(total_rows))
        else:
            # Large table: only rows that contain at least one IQR outlier in a non-binary column
            anomalous: list[int] = []
            for idx in range(total_rows):
                row = df.iloc[idx]
                for col, b in outlier_bounds.items():
                    if col in row.index and pd.notna(row[col]):
                        val = float(row[col])
                        if val < b["lo"] or val > b["hi"]:
                            anomalous.append(idx)
                            break
            row_indices = anomalous[:200]   # cap to keep token budget sane

        # Step 3 — LLM insight calls, batched by _MAX_ROWS_PER_CALL
        insights_map: dict[str, str] = {}
        for i in range(0, len(row_indices), _MAX_ROWS_PER_CALL):
            batch = row_indices[i: i + _MAX_ROWS_PER_CALL]
            insights_map.update(self._table_row_insights(df, table, batch, bounds))

        # Step 4 — assemble one AgentResult per row
        results: list[AgentResult] = []
        for idx in row_indices:
            row = df.iloc[idx]
            row_key = f"row_{idx}"
            row_vals: dict[str, float] = {}
            anomalous_cols: list[str] = []
            for col, b in bounds.items():
                if col in row.index and pd.notna(row[col]):
                    val = float(row[col])
                    row_vals[col] = round(val, 4)
                    # Only flag as anomalous if non-binary and outside IQR range
                    if not b.get("binary") and b["lo"] is not None:
                        if val < b["lo"] or val > b["hi"]:
                            anomalous_cols.append(col)

            analysis = {
                "row_index": idx,
                "values": row_vals,
                "anomalous_columns": anomalous_cols,
                "col_means": {c: round(b["mean"], 4) for c, b in bounds.items()},
            }
            chart_data = {
                "row_index": idx,
                "values": row_vals,
                "means": {c: round(b["mean"], 2) for c, b in bounds.items()},
                "anomalies": anomalous_cols,
            }
            results.append(AgentResult(
                agent_name="numerical",
                table=table,
                column=row_key,
                analysis=analysis,
                insights_text=insights_map.get(row_key, ""),
                chart_data=chart_data,
                chart_type="bar",
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
