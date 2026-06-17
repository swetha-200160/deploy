from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from config import Config, get_config
from core.state import AgentResult

log = logging.getLogger(__name__)


# ── Tool implementations ──────────────────────────────────────────────────────

def _data_quality_score(dataframe_dict: dict) -> dict:
    """Overall data quality score across completeness, uniqueness, consistency."""
    try:
        df = pd.DataFrame(dataframe_dict)
        total_cells = df.size
        null_cells = int(df.isnull().sum().sum())
        completeness = round((1 - null_cells / max(total_cells, 1)) * 100, 2)

        dup_rows = int(df.duplicated().sum())
        uniqueness = round((1 - dup_rows / max(len(df), 1)) * 100, 2)

        # Consistency: columns that have mixed types
        inconsistent_cols = []
        for col in df.columns:
            sample = df[col].dropna().astype(str).tolist()[:50]
            has_numeric = any(_is_numeric(v) for v in sample)
            has_text = any(not _is_numeric(v) for v in sample)
            if has_numeric and has_text:
                inconsistent_cols.append(col)
        consistency = round((1 - len(inconsistent_cols) / max(df.shape[1], 1)) * 100, 2)

        overall = round((completeness + uniqueness + consistency) / 3, 2)
        return {
            "overall_quality_score": overall,
            "completeness_score": completeness,
            "uniqueness_score": uniqueness,
            "consistency_score": consistency,
            "null_cell_count": null_cells,
            "duplicate_row_count": dup_rows,
            "inconsistent_columns": inconsistent_cols,
            "quality_grade": "A" if overall >= 90 else ("B" if overall >= 75 else ("C" if overall >= 60 else "D")),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _is_numeric(v: str) -> bool:
    try:
        float(v.replace(",", "").replace("$", "").strip())
        return True
    except ValueError:
        return False


def _detect_data_drift(current_data: list, baseline_data: list, column_name: str) -> dict:
    """Detect distribution drift between current and baseline data."""
    try:
        cur = pd.Series([x for x in current_data if x is not None], dtype=float).dropna()
        base = pd.Series([x for x in baseline_data if x is not None], dtype=float).dropna()
        if cur.empty or base.empty:
            return {"error": "insufficient_data"}

        mean_shift = round(float(cur.mean() - base.mean()), 4)
        cur_std = cur.std() if len(cur) > 1 else 0.0
        base_std = base.std() if len(base) > 1 else 0.0
        std_shift = round(float(cur_std - base_std), 4)
        mean_shift_pct = round(mean_shift / max(abs(float(base.mean())), 1e-9) * 100, 2)

        # Population Stability Index (PSI) approximation
        n_bins = 10
        base_hist, edges = np.histogram(base, bins=n_bins)
        cur_hist, _ = np.histogram(cur, bins=edges)
        base_pct = base_hist / max(base_hist.sum(), 1) + 1e-9
        cur_pct = cur_hist / max(cur_hist.sum(), 1) + 1e-9
        psi = float(np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct)))

        drift_level = "high" if psi > 0.2 else ("medium" if psi > 0.1 else "low")
        return {
            "column": column_name,
            "mean_shift": mean_shift,
            "mean_shift_pct": mean_shift_pct,
            "std_shift": std_shift,
            "psi_score": round(psi, 4),
            "drift_level": drift_level,
            "drift_detected": psi > 0.1,
            "current_mean": round(float(cur.mean()), 4),
            "baseline_mean": round(float(base.mean()), 4),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _detect_fraud_advanced(dataframe_dict: dict, amount_col: str) -> dict:
    """Advanced fraud detection: velocity, round-tripping, structuring patterns."""
    try:
        df = pd.DataFrame(dataframe_dict)
        if amount_col not in df.columns:
            return {"error": f"column {amount_col!r} not found"}
        amounts = pd.to_numeric(df[amount_col], errors="coerce").dropna()

        # Structuring detection (just below reporting thresholds)
        thresholds = [9000, 4500, 3000]
        structuring_flags = {}
        for t in thresholds:
            count = int(((amounts > t * 0.9) & (amounts < t)).sum())
            if count > 0:
                structuring_flags[f"below_{t}"] = count

        # Round number concentration
        round_100 = int((amounts % 100 == 0).sum())
        round_1000 = int((amounts % 1000 == 0).sum())

        # Extreme outliers (>99.5th percentile)
        extreme_threshold = float(amounts.quantile(0.995))
        extreme_txns = int((amounts > extreme_threshold).sum())

        # Negative amounts (refunds/reversals)
        negative = int((amounts < 0).sum())

        fraud_score = (
            min(len(structuring_flags) * 2, 4)
            + (2 if round_100 / max(len(amounts), 1) > 0.3 else 0)
            + (2 if extreme_txns > len(amounts) * 0.01 else 0)
            + (1 if negative > 0 else 0)
        )
        return {
            "amount_column": amount_col,
            "fraud_risk_score": min(fraud_score, 10),
            "fraud_risk_level": "high" if fraud_score >= 6 else ("medium" if fraud_score >= 3 else "low"),
            "structuring_signals": structuring_flags,
            "round_number_100_count": round_100,
            "round_number_1000_count": round_1000,
            "extreme_transaction_count": extreme_txns,
            "negative_amount_count": negative,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _root_cause_analysis(dataframe_dict: dict, target_col: str) -> dict:
    """Identify which columns most correlate with anomalies in the target column."""
    try:
        df = pd.DataFrame(dataframe_dict).apply(pd.to_numeric, errors="coerce")
        if target_col not in df.columns:
            return {"error": f"column {target_col!r} not found"}
        target = df[target_col].dropna()
        if target.empty:
            return {"error": "target_column_empty"}

        # IQR-based anomaly mask on target
        q1, q3 = target.quantile(0.25), target.quantile(0.75)
        iqr = q3 - q1
        anomaly_mask = (target < q1 - 1.5 * iqr) | (target > q3 + 1.5 * iqr)
        n_anomalies = int(anomaly_mask.sum())

        correlations = {}
        for col in df.columns:
            if col == target_col:
                continue
            col_data = df[col].dropna()
            if len(col_data) < 5:
                continue
            try:
                corr = float(df[[target_col, col]].dropna().corr().iloc[0, 1])
                if not math.isnan(corr):
                    correlations[col] = round(corr, 4)
            except Exception:
                pass

        top_causes = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        anomaly_stats = {}
        for col, _ in top_causes:
            col_data = df[col]
            anomaly_mean = round(float(col_data[anomaly_mask].mean()), 4) if anomaly_mask.sum() > 0 else None
            normal_mean = round(float(col_data[~anomaly_mask].mean()), 4) if (~anomaly_mask).sum() > 0 else None
            if anomaly_mean is not None and normal_mean is not None:
                anomaly_stats[col] = {"anomaly_mean": anomaly_mean, "normal_mean": normal_mean,
                                      "diff_pct": round((anomaly_mean - normal_mean) / max(abs(normal_mean), 1e-9) * 100, 2)}

        return {
            "target_column": target_col,
            "anomaly_count": n_anomalies,
            "anomaly_pct": round(n_anomalies / max(len(target), 1) * 100, 2),
            "top_correlated_causes": dict(top_causes),
            "anomaly_vs_normal_stats": anomaly_stats,
            "root_cause_summary": f"Top driver: {top_causes[0][0]} (corr={top_causes[0][1]})" if top_causes else "No strong correlations found",
        }
    except Exception as exc:
        return {"error": str(exc)}


def _generate_recommendations(analysis_summary: dict) -> dict:
    """Generate actionable recommendations based on analysis results."""
    recs = []

    quality = analysis_summary.get("data_quality_score", {})
    if quality.get("completeness_score", 100) < 90:
        recs.append({"priority": "high", "action": "impute_missing_values",
                     "detail": f"Completeness is {quality.get('completeness_score')}% — apply median/mode imputation"})
    if quality.get("duplicate_row_count", 0) > 0:
        recs.append({"priority": "medium", "action": "remove_duplicates",
                     "detail": f"{quality.get('duplicate_row_count')} duplicate rows detected"})

    fraud = analysis_summary.get("fraud_signals", {})
    if fraud.get("fraud_risk_level") in ("high", "medium"):
        recs.append({"priority": "high", "action": "investigate_fraud",
                     "detail": f"Fraud risk score: {fraud.get('fraud_risk_score')} — review flagged transactions"})

    drift = analysis_summary.get("data_drift", {})
    if drift.get("drift_detected"):
        recs.append({"priority": "medium", "action": "retrain_models",
                     "detail": f"Data drift detected (PSI={drift.get('psi_score')}) — models may need retraining"})

    if not recs:
        recs.append({"priority": "low", "action": "routine_monitoring",
                     "detail": "Data quality is good — continue scheduled monitoring"})

    return {
        "recommendation_count": len(recs),
        "high_priority": [r for r in recs if r["priority"] == "high"],
        "medium_priority": [r for r in recs if r["priority"] == "medium"],
        "low_priority": [r for r in recs if r["priority"] == "low"],
    }


# ── Tool registry ─────────────────────────────────────────────────────────────

_TOOL_MAP = {
    "data_quality_score":    lambda args, cfg: _data_quality_score(args["dataframe_dict"]),
    "detect_data_drift":     lambda args, cfg: _detect_data_drift(args["current_data"], args["baseline_data"], args.get("column_name", "")),
    "detect_fraud_advanced": lambda args, cfg: _detect_fraud_advanced(args["dataframe_dict"], args.get("amount_col", "")),
    "root_cause_analysis":   lambda args, cfg: _root_cause_analysis(args["dataframe_dict"], args.get("target_col", "")),
    "generate_recommendations": lambda args, _: _generate_recommendations(args.get("analysis_summary", {})),
}

_TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "data_quality_score",
        "description": "Overall data quality score: completeness, uniqueness, consistency — returns A/B/C/D grade. Data is injected server-side.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "detect_data_drift",
        "description": "PSI-based drift detection comparing current vs baseline distribution",
        "parameters": {"type": "object", "properties": {
            "column_name": {"type": "string"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "detect_fraud_advanced",
        "description": "Advanced fraud detection: structuring, round numbers, extreme transactions, reversals",
        "parameters": {"type": "object", "properties": {
            "amount_col": {"type": "string", "description": "Name of the amount/value column to analyze"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "root_cause_analysis",
        "description": "Identify which columns correlate most with anomalies in a target column",
        "parameters": {"type": "object", "properties": {
            "target_col": {"type": "string", "description": "Column to find root causes for"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "generate_recommendations",
        "description": "Generate prioritized actionable recommendations from the full analysis summary. Analysis summary is injected server-side.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


# ── Agent class ───────────────────────────────────────────────────────────────

class EnterpriseAgent:
    """Section 6 — Advanced Enterprise Features agent.

    Runs data quality scoring, drift detection, fraud detection,
    root cause analysis, and recommendation generation.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
            temperature=self.config.llm_temperature,
            max_output_tokens=self.config.llm_max_tokens,
            thinking_budget=0,
        ).bind_tools(_TOOL_DEFS)

    def run(self, raw_data: dict) -> dict[str, Any]:
        """Run enterprise analysis across all tables and return a combined results dict."""
        results: dict[str, Any] = {}

        for table_name, df in raw_data.items():
            if df is None or df.empty:
                continue
            results[table_name] = self._analyze_table(table_name, df)

        return results

    def _analyze_table(self, table_name: str, df: pd.DataFrame) -> dict[str, Any]:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        amount_col = next(
            (c for c in num_cols if any(k in c.lower() for k in ("amount", "price", "value", "revenue", "total"))),
            num_cols[0] if num_cols else "",
        )
        target_col = amount_col

        row_cols = [c for c in df.columns if str(c).strip() and not df[c].replace("", None).isna().all()]

        schema_info = {col: str(df[col].dtype) for col in row_cols}

        system = (
            "You are an enterprise data analyst. Run ALL provided tools on this dataset. "
            "Call: data_quality_score, detect_fraud_advanced, root_cause_analysis, generate_recommendations. "
            "Data is injected automatically into each tool — do NOT include raw data in your response. "
            "Then summarize enterprise risk findings."
        )
        user = (
            f"Enterprise analysis for table: '{table_name}'\n"
            f"Shape: {len(df)} rows × {len(row_cols)} columns\n"
            f"Column schema (name → dtype): {json.dumps(schema_info)}\n"
            f"Numeric columns: {num_cols[:8]}\n"
            f"Amount/target column: '{amount_col}'"
        )

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        tool_results: dict[str, Any] = {}

        def _json_safe(v):
            if isinstance(v, pd.Timestamp):
                return v.isoformat()
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        # Sample to avoid Gemini token limits (1M token cap).
        sample_df = df.sample(n=min(500, len(df)), random_state=42) if len(df) > 500 else df
        full_df_dict = {c: [_json_safe(v) for v in sample_df[c].tolist()] for c in row_cols if str(c).strip()}

        for _ in range(8):
            response = self.llm.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                fn = tc["name"]
                # Copy args to avoid mutating the AIMessage already stored in messages,
                # which would inject full_df_dict into the next LLM call's context.
                args = dict(tc["args"])
                # Inject real data server-side (never goes to LLM)
                args["dataframe_dict"] = full_df_dict
                if fn == "detect_data_drift":
                    col_vals = sample_df[amount_col].dropna().tolist() if amount_col else []
                    mid = max(len(col_vals) // 2, 1)
                    args["baseline_data"] = col_vals[:mid]
                    args["current_data"] = col_vals[mid:]
                    args["column_name"] = amount_col
                if fn in ("detect_fraud_advanced", "root_cause_analysis"):
                    args["amount_col"] = amount_col
                    args["target_col"] = target_col
                if fn == "generate_recommendations":
                    args["analysis_summary"] = tool_results

                result = _TOOL_MAP[fn](args, self.config)
                tool_results[fn] = result
                messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc["id"]))

        insights = response.content if isinstance(response.content, str) and response.content else ""
        return {"table": table_name, "tool_results": tool_results, "insights": insights}
