"""EnterpriseRowAgent — per-row data quality and fraud risk analysis.

Architecture mirrors UniversalRowAgent:
  1. Pre-compute: build quality and fraud baselines for the full table (zero LLM)
  2. Detect:      for each row, collect data quality / fraud signals
  3. LLM batch:   send flagged rows + baselines → audit insights

Signals detected per row:
  NULL     : missing value in a column that is normally populated
  DUP      : exact duplicate of another row
  TYPE     : value does not match the column's expected data type
  FRAUD    : amount is near a reporting threshold, extremely large, or a suspicious round number
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from core.groq_rate_limiter import groq_invoke
from config import Config, get_config
from core.state import AgentResult, ColumnRoute

log = logging.getLogger(__name__)

_STRUCTURING_THRESHOLDS = [10_000, 9_000, 5_000, 4_500, 3_000]

_SYSTEM_PROMPT = """\
You are a senior data quality auditor and fraud analyst.

For each flagged record write a 2-3 sentence audit insight that:
1. States RISK LEVEL: [HIGH RISK] / [MEDIUM RISK] / [LOW RISK]
2. Names the specific data quality issues or fraud signals with exact values
3. Explains the BUSINESS IMPLICATION in audit terms
4. Recommends a specific audit ACTION

Fields marked with ! are flagged — meanings:
  ! on NULL  : missing value in a column that should be populated
  ! on DUP   : this row is an exact duplicate of another row
  ! on TYPE  : value does not match the column's expected data type
  ! on FRAUD : amount is near a reporting threshold, extreme, or suspicious round number

Do NOT say 'data quality issue' generically — name the specific problem.
Do NOT list column names mechanically — tell the audit story of the record.\
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_row_insights(content: str, row_keys: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    missing: list[str] = []
    for key in row_keys:
        # tolerant of: **[INSIGHTS: row_N]**, *[INSIGHT: row_N]*, bold/italic markdown wrappers
        pattern = rf'[*_]*\[INSIGHTS?:\s*{re.escape(key)}\][*_]*\s*(.*?)(?=[*_]*\[INSIGHTS?:|$)'
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        result[key] = m.group(1).strip() if m and m.group(1).strip() else ""
        if not result[key]:
            missing.append(key)
    if missing:
        log.warning("EnterpriseRowAgent: no insight parsed for %s", missing)
    return result


def _is_numeric_str(v: str) -> bool:
    try:
        float(str(v).replace(",", "").replace("$", "").strip())
        return True
    except ValueError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _precompute(df: pd.DataFrame) -> tuple[set, dict, dict, dict]:
    """Build quality and fraud baselines for the full table (zero LLM).

    Returns:
        dup_indices  — set of row indices that are exact duplicates
        null_rates   — {col: null_rate} for columns with any nulls
        type_profile — {col: "numeric"|"text"|"datetime"} expected type per column
        fraud_bounds — {col: {p995, thresholds}} for amount-like columns
    """
    # Duplicate detection — keep='first' so the second+ occurrence is flagged
    dup_indices: set = set(df.index[df.duplicated(keep="first")].tolist())

    # Null profile
    null_rates: dict = {}
    for col in df.columns:
        rate = float(df[col].isna().mean())
        if rate > 0:
            null_rates[col] = round(rate, 3)

    # Type profile
    type_profile: dict = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            type_profile[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            type_profile[col] = "datetime"
        else:
            numeric_frac = pd.to_numeric(s.astype(str), errors="coerce").notna().mean()
            type_profile[col] = "numeric" if numeric_frac > 0.8 else "text"

    # Fraud baselines for amount-like numerical columns
    fraud_bounds: dict = {}
    amount_kw = ("amount", "price", "value", "revenue", "total", "sum", "cost", "fee", "payment")
    for col in df.select_dtypes(include="number").columns:
        if any(k in col.lower() for k in amount_kw):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) >= 10:
                fraud_bounds[col] = {
                    "p995": round(float(s.quantile(0.995)), 2),
                    "thresholds": _STRUCTURING_THRESHOLDS,
                }

    return dup_indices, null_rates, type_profile, fraud_bounds


# ═══════════════════════════════════════════════════════════════════════════════
# ROW SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _row_signals(
    row: pd.Series,
    idx: int,
    dup_indices: set,
    null_rates: dict,
    type_profile: dict,
    fraud_bounds: dict,
) -> list[str]:
    signals: list[str] = []

    # NULL signals — only flag columns that are normally populated (null_rate < 50%)
    for col in row.index:
        if col in null_rates and null_rates[col] < 0.5 and pd.isna(row[col]):
            signals.append(f"null:{col}(col_null_rate={null_rates[col]*100:.0f}%)")

    # DUPLICATE signal
    if idx in dup_indices:
        signals.append("dup:EXACT_DUPLICATE")

    # TYPE signals
    for col, expected in type_profile.items():
        if col not in row.index or pd.isna(row[col]):
            continue
        if expected == "numeric" and not pd.api.types.is_numeric_dtype(type(row[col])):
            if not _is_numeric_str(str(row[col])):
                signals.append(f"type:{col}='{str(row[col])[:20]}'(expected_numeric)")

    # FRAUD signals
    for col, bounds in fraud_bounds.items():
        if col not in row.index or pd.isna(row[col]):
            continue
        try:
            val = float(row[col])
        except (ValueError, TypeError):
            continue
        if val > bounds["p995"]:
            signals.append(f"fraud:{col}={round(val,2)}(extreme>p99.5={bounds['p995']})")
        else:
            for t in bounds["thresholds"]:
                if t * 0.9 < val < t:
                    signals.append(f"fraud:{col}={round(val,2)}(structuring<{t})")
                    break
        if val > 0 and val % 1000 == 0:
            signals.append(f"fraud:{col}={round(val,2)}(round_1000)")

    return signals


def _row_summary(
    row: pd.Series,
    idx: int,
    df: pd.DataFrame,
    dup_indices: set,
    null_rates: dict,
    fraud_bounds: dict,
) -> str:
    parts: list[str] = []

    if idx in dup_indices:
        parts.append("STATUS=DUPLICATE!")

    for col in df.columns:
        if col not in row.index:
            continue
        raw = row[col]
        flag = ""

        if pd.isna(raw):
            if col in null_rates and null_rates[col] < 0.5:
                flag = "!"
            parts.append(f"{col}=NULL{flag}")
            continue

        if col in fraud_bounds:
            try:
                val = float(raw)
                b = fraud_bounds[col]
                if val > b["p995"]:
                    flag = "!"
                elif any(t * 0.9 < val < t for t in b["thresholds"]):
                    flag = "!"
                elif val > 0 and val % 1000 == 0:
                    flag = "!"
            except (ValueError, TypeError):
                pass

        parts.append(f"{col}={str(raw)[:25]}{flag}")

    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA-ONLY FLOW: ONE LLM CALL → TEMPLATES, THEN PURE PYTHON PER ROW
# ═══════════════════════════════════════════════════════════════════════════════

def _ent_extract_json(raw: str) -> dict:
    """Parse JSON from LLM output, tolerating markdown code fences."""
    if isinstance(raw, list):
        raw = " ".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw)
    for attempt in (raw, re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw),
                    re.search(r"\{[\s\S]+\}", raw)):
        text = attempt.group(1) if hasattr(attempt, "group") else (attempt or "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _ent_signal_prefix(sig: str) -> str:
    """Extract the signal prefix from an enterprise signal string (null/dup/type/fraud)."""
    return sig.split(":")[0] if ":" in sig else "unknown"


def _ent_signal_detail(sig: str) -> str:
    """Convert a raw enterprise signal string into a readable {detail} filler."""
    if ":" not in sig:
        return sig
    prefix, rest = sig.split(":", 1)
    if prefix == "dup":
        return "this row is an exact duplicate of another record in the table"
    return rest


def _ent_fetch_templates(
    llm,
    table: str,
    signals_cache: dict[int, list[str]],
) -> dict[str, str]:
    """ONE LLM call per table — signal types only, no row data, no baselines.

    Template keys are the signal prefixes actually present in flagged rows
    (null / dup / type / fraud). The system message defines what each means.
    Baseline stats are excluded to keep the prompt within token limits.
    """
    present_prefixes = sorted({
        _ent_signal_prefix(s)
        for sigs in signals_cache.values()
        for s in sigs
        if _ent_signal_prefix(s) != "unknown"
    })
    if not present_prefixes:
        return {}

    prompt_lines = [
        f"Financial audit table: '{table}'.",
        "",
        f"Signal prefixes detected in flagged rows: {present_prefixes}",
        "(Refer to the system message for what each prefix means.)",
        "",
        "Write one 2-3 sentence audit insight TEMPLATE for each signal prefix.",
        "Rules:",
        "  • Use {detail} as the ONLY placeholder — Python fills it with the actual "
        "column name, value, and context at runtime.",
        "  • Start with RISK LEVEL: [HIGH RISK] / [MEDIUM RISK] / [LOW RISK]",
        "  • Explain the data quality or fraud implication.",
        "  • End with a concrete recommended audit action.",
        "  • Keep it generic — no specific column names or values.",
        "",
        f"Return a JSON object with exactly these keys: {present_prefixes}",
    ]

    try:
        msgs = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content="\n".join(prompt_lines))]
        resp = groq_invoke(llm, msgs, agent_name="enterprise")
        raw  = resp.content if hasattr(resp, "content") else str(resp)
        parsed = _ent_extract_json(raw)
        return {k: v if isinstance(v, str) else " ".join(v.values()) if isinstance(v, dict) else str(v)
                for k, v in parsed.items()}
    except Exception as exc:
        log.error("EnterpriseRowAgent template fetch failed for '%s': %s", table, exc)
        return {}


def _ent_render_insight(signals: list[str], templates: dict[str, str]) -> str:
    """Pure Python — render a row insight from pre-fetched templates."""
    if not signals:
        return ""
    lead_key = _ent_signal_prefix(signals[0])
    template = templates.get(lead_key, "")
    detail   = _ent_signal_detail(signals[0])
    insight  = template.replace("{detail}", detail) if template else f"[FLAGGED] {detail}"
    if len(signals) > 1:
        others  = "; ".join(_ent_signal_detail(s) for s in signals[1:])
        insight += f" Additional flags: {others}."
    return insight


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class EnterpriseRowAgent:
    """Row-by-row data quality and fraud risk analysis agent.

    Mirrors UniversalRowAgent architecture:
      1. Pre-compute quality/fraud baselines (zero LLM)
      2. Flag rows with signals
      3. Batch flagged rows to Gemini for audit insights
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=2048,
            max_retries=0,
        )

    def _run_table(self, df: pd.DataFrame, table: str) -> list[AgentResult]:

        # Extract Excel-compatible DB row numbers and drop the metadata column.
        db_row_nums: dict[int, int] = {}
        if "__db_row_num__" in df.columns:
            for i in range(len(df)):
                db_row_nums[i] = int(df["__db_row_num__"].iloc[i])
            df = df.drop(columns=["__db_row_num__"]).reset_index(drop=True)

        # STEP 1 — pre-compute baselines
        dup_indices, null_rates, type_profile, fraud_bounds = _precompute(df)
        log.info(
            "EnterpriseRowAgent %s: dup=%d null_cols=%d fraud_cols=%d",
            table, len(dup_indices), len(null_rates), len(fraud_bounds),
        )

        # STEP 2 — find flagged rows and cache signals
        total = len(df)
        signals_cache: dict[int, list[str]] = {}
        for idx in range(total):
            sigs = _row_signals(df.iloc[idx], idx, dup_indices, null_rates, type_profile, fraud_bounds)
            if sigs:
                signals_cache[idx] = sigs
        row_indices = list(signals_cache.keys())[:200]

        if not row_indices:
            log.info("EnterpriseRowAgent %s: no flagged rows found", table)
            return []

        # STEP 3 — build schema context (baselines only, no row data)
        baseline_lines: list[str] = []
        if null_rates:
            null_str = ", ".join(f"{c}={v*100:.0f}%" for c, v in null_rates.items())
            baseline_lines.append(f"  Null rates: {null_str}")
        if dup_indices:
            baseline_lines.append(
                f"  Duplicate rows: {len(dup_indices)}/{total} ({len(dup_indices)/total*100:.1f}%)"
            )
        for col, b in fraud_bounds.items():
            baseline_lines.append(
                f"  [FRAUD] {col}: extreme_threshold={b['p995']}, "
                f"structuring_thresholds={b['thresholds']}"
            )

        # STEP 4 — ONE LLM call: schema + baselines → insight templates (no row data sent)
        templates = _ent_fetch_templates(self._llm, table, signals_cache)

        # STEP 4b — pure Python: render one insight per flagged row (zero more LLM calls)
        insights_map: dict[str, str] = {
            f"row_{idx}": _ent_render_insight(signals_cache[idx], templates)
            for idx in row_indices
        }

        # STEP 5 — assemble AgentResult per row
        results: list[AgentResult] = []
        for idx in row_indices:
            row     = df.iloc[idx]
            row_key = f"row_{idx}"
            signals = signals_cache[idx]

            # ALL column values for this row — every column locked together
            all_values: dict[str, str] = {}
            for col in df.columns:
                raw_val = row[col] if col in row.index else None
                all_values[col] = "NULL" if (raw_val is None or (not isinstance(raw_val, str) and pd.isna(raw_val))) else str(raw_val)

            results.append(AgentResult(
                agent_name="enterprise_row",
                table=table,
                column=row_key,
                analysis={
                    "row_index":      db_row_nums.get(idx, idx + 2),  # Excel row number
                    "is_duplicate":   idx in dup_indices,
                    "null_columns":   [
                        c for c in row.index
                        if pd.isna(row[c]) and c in null_rates and null_rates[c] < 0.5
                    ],
                    "signals":        signals,
                    "signal_count":   len(signals),
                    "all_values":     all_values,
                },
                insights_text=insights_map.get(row_key, ""),
                chart_data={
                    "row_index":    db_row_nums.get(idx, idx + 2),  # Excel row number
                    "signals":      signals,
                    "is_duplicate": idx in dup_indices,
                },
                chart_type="bar",
                error=None,
            ))

        log.info("EnterpriseRowAgent %s: %d row results", table, len(results))
        return results

    def run(self, raw_data: dict) -> list[AgentResult]:
        results: list[AgentResult] = []
        for table, df in raw_data.items():
            if df is None or df.empty:
                continue
            log.info("EnterpriseRowAgent: table '%s' (%d rows)", table, len(df))
            results.extend(self._run_table(df, table))
        log.info("EnterpriseRowAgent complete: %d total row results", len(results))
        return results
