"""UniversalRowAgent — single-pass, row-wise analysis across ALL column types.

Architecture — dynamic tool registry:
  • Tools are registered with @register_tool(col_type) decorator
  • Agent discovers tools at runtime by column type — no hardcoding
  • Add a new tool: just write the function + @register_tool("numerical")
  • Remove a tool: delete the function — agent auto-adjusts

Flow per table:
  1. Pre-compute: for each column, run all registered tools of its type (zero LLM)
  2. Detect:      for each row, collect anomaly signals across ALL column types
  3. LLM batch:   send flagged rows + pre-computed baselines → insights
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from core.groq_rate_limiter import groq_invoke
from config import Config, get_config
from core.state import AgentResult, ColumnRoute

log = logging.getLogger(__name__)

_RARE_CAT_THRESHOLD = 0.03
_HIGH_CARD_RATIO    = 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC KEYWORD REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Maps keyword → reason (shown in system prompt and signals)
_KEYWORD_REGISTRY: dict[str, str] = {}


def register_keyword(keyword: str, reason: str = "suspicious term"):
    """Register a suspicious text keyword.

    Usage:
        register_keyword("urgent",    "pressure tactic — common in fraud")
        register_keyword("offshore",  "high-risk jurisdiction indicator")
    """
    _KEYWORD_REGISTRY[keyword.lower()] = reason


def get_keywords() -> dict[str, str]:
    """Return all registered suspicious keywords."""
    return dict(_KEYWORD_REGISTRY)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC ANOMALY FLAG REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Maps col_type → human-readable description of what ! means for that type
_ANOMALY_DESCRIPTIONS: dict[str, str] = {}


def register_anomaly_description(col_type: str, description: str):
    """Register what the ! flag means for a column type.

    Usage:
        register_anomaly_description("numerical", "IQR or Z-score outlier")
        register_anomaly_description("datetime",  "off-hours or future date")
    """
    _ANOMALY_DESCRIPTIONS[col_type] = description


def _build_system_prompt() -> str:
    """Build the LLM system prompt dynamically from all registries.

    Re-evaluated at runtime so adding/removing tools or keywords
    automatically updates what the LLM is told about the analysis.
    """
    lines = [
        "You are a senior financial audit analyst specialising in anomaly detection and fraud risk.",
        "",
        "Column baselines were pre-computed using these registered analysis tools:",
    ]

    for col_type, tools in sorted(_TOOL_REGISTRY.items()):
        names = ", ".join(name for name, _ in tools)
        lines.append(f"  {col_type.upper():12s}: {names}")

    lines += [
        "",
        "Fields marked with ! are anomalous — meanings per column type:",
    ]
    for col_type, desc in sorted(_ANOMALY_DESCRIPTIONS.items()):
        lines.append(f"  ! on {col_type:12s}: {desc}")

    if _KEYWORD_REGISTRY:
        lines += [
            "",
            "Registered suspicious text keywords (trigger ! on text fields):",
        ]
        for kw, reason in sorted(_KEYWORD_REGISTRY.items()):
            lines.append(f"  '{kw}': {reason}")

    lines += [
        "",
        "For each record write a 2-3 sentence insight that:",
        "1. States RISK LEVEL: [HIGH RISK] / [MEDIUM RISK] / [LOW RISK]",
        "2. Names the specific anomalous values and what makes them unusual",
        "3. Explains the BUSINESS IMPLICATION in audit / fraud-detection terms",
        "4. Highlights CROSS-COLUMN patterns (e.g. large amount + off-hours + rare payee)",
        "5. Ends with a recommended audit ACTION",
        "",
        "Do NOT say 'statistical outlier' — explain the business meaning.",
        "Do NOT list column names mechanically — tell the story of the transaction.",
    ]
    return "\n".join(lines)


# ── Default keywords (add more with register_keyword() anywhere in codebase) ──
register_keyword("urgent",    "pressure tactic — common in fraud")
register_keyword("test",      "likely a test/dummy record")
register_keyword("temp",      "temporary entry — may not be auditable")
register_keyword("delete",    "deletion marker in description")
register_keyword("error",     "explicit error flag in text")
register_keyword("dummy",     "dummy/placeholder record")
register_keyword("fake",      "explicit fake indicator")
register_keyword("cancel",    "cancellation — check if reversal is authorised")
register_keyword("void",      "voided entry — requires authorisation trail")
register_keyword("n/a",       "missing or not-applicable description")
register_keyword("none",      "empty description placeholder")
register_keyword("unknown",   "unknown entity — high due-diligence risk")

# ── Default anomaly descriptions (shown in dynamic system prompt) ──
register_anomaly_description("numerical",   "IQR or Z-score outlier — value outside the column's normal range")
register_anomaly_description("categorical", "rare value (< 3% frequency in column) or missing/empty")
register_anomaly_description("datetime",    "off-hours (before 7 AM / after 9 PM), future date, or > 5 yr old")
register_anomaly_description("text",        "empty, too short (< 3 chars), or contains a suspicious keyword")


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Registry maps column_type → list of (tool_name, tool_fn)
_TOOL_REGISTRY: dict[str, list[tuple[str, Callable]]] = defaultdict(list)


def register_tool(col_type: str, name: str | None = None):
    """Decorator to register an analysis tool for a specific column type.

    Usage:
        @register_tool("numerical")
        def my_tool(df, col): ...

        @register_tool("categorical", name="freq_analysis")
        def my_freq_tool(df, col): ...
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__.lstrip("_")
        _TOOL_REGISTRY[col_type].append((tool_name, fn))
        return fn
    return decorator


def get_tools(col_type: str) -> list[tuple[str, Callable]]:
    """Return all registered tools for a column type."""
    return list(_TOOL_REGISTRY.get(col_type, []))


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@register_tool("numerical", name="missing")
def _num_missing(df: pd.DataFrame, col: str) -> str:
    """Null and missing value detection."""
    s = df[col]
    total = len(s)
    nc = int(s.isna().sum())
    pct = round(nc / total * 100, 2) if total else 0.0
    label = ("complete" if pct == 0 else
             "MCAR-low" if pct < 5 else
             "MAR-moderate" if pct < 20 else "MNAR-high")
    return f"{nc}/{total} ({pct}%) [{label}]"


@register_tool("numerical", name="outliers")
def _num_outliers(df: pd.DataFrame, col: str) -> str:
    """Outlier detection — IQR and Z-score."""
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
        f"IQR_outliers={iqr_n}({round(iqr_n/n*100,1)}%) "
        f"bounds=[{round(lo,2)},{round(hi,2)}] "
        f"Zscore_outliers={z_n}({round(z_n/n*100,1)}%)"
    )


@register_tool("numerical", name="distribution")
def _num_distribution(df: pd.DataFrame, col: str) -> str:
    """Distribution analysis — skew, kurtosis, shape."""
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
        f"mean={round(float(s.mean()),2)} median={round(float(s.median()),2)} "
        f"std={round(float(s.std()),2)}"
    )


@register_tool("numerical", name="correlation")
def _num_correlation(df: pd.DataFrame, col: str) -> str:
    """Correlation analysis — top 3 correlated numerical columns."""
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().std() == 0:
        return "skipped (zero variance)"
    corrs = {}
    for c in df.select_dtypes(include="number").columns:
        if c == col:
            continue
        other = pd.to_numeric(df[c], errors="coerce")
        if other.dropna().std() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r = s.corr(other)
        if not pd.isna(r):
            corrs[c] = round(float(r), 3)
    top3 = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    return ",".join(f"{c}:{v}" for c, v in top3) if top3 else "none"


@register_tool("numerical", name="scaling")
def _num_scaling(df: pd.DataFrame, col: str) -> str:
    """Scaling and normalisation — range and recommendation."""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return "no data"
    skew = float(s.skew())
    rec = "log+zscore" if abs(skew) > 1 else "minmax"
    return (
        f"range=[{round(float(s.min()),2)},{round(float(s.max()),2)}] "
        f"mean={round(float(s.mean()),2)} std={round(float(s.std()),2)} "
        f"rec={rec}(skew={round(skew,2)})"
    )


@register_tool("numerical", name="features")
def _num_features(df: pd.DataFrame, col: str) -> str:
    """Feature engineering hints — round numbers, negatives, transforms."""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return "no data"
    feats = []
    if float(s.min()) > 0:
        feats.append("log_transform_ok")
    q = [round(float(s.quantile(v)), 2) for v in [0.25, 0.5, 0.75]]
    feats.append(f"quartile_bins{q}")
    rn = round(float((s % 100 == 0).sum() / n * 100), 1)
    if rn > 10:
        feats.append(f"round_numbers({rn}%)")
    neg = round(float((s < 0).sum() / n * 100), 1)
    if neg > 0:
        feats.append(f"negatives({neg}%)")
    return " | ".join(feats) if feats else "standard"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORICAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@register_tool("categorical", name="missing")
def _cat_missing(df: pd.DataFrame, col: str) -> str:
    """Missing value and type validation."""
    s = df[col]
    total = len(s)
    null_n = int(s.isna().sum())
    empty_n = int((s.astype(str).str.strip() == "").sum()) - null_n
    mixed = len(s.dropna().apply(type).unique()) > 1
    pct = round((null_n + empty_n) / total * 100, 1) if total else 0.0
    return (
        f"nulls={null_n} empty={empty_n} missing_pct={pct}% "
        f"mixed_types={'yes' if mixed else 'no'}"
    )


@register_tool("categorical", name="frequency")
def _cat_frequency(df: pd.DataFrame, col: str) -> str:
    """Frequency and imbalance analysis."""
    s = df[col].astype(str).str.strip()
    s = s[s.str.lower() != "nan"]
    if len(s) == 0:
        return "no data"
    vc = s.value_counts(normalize=True)
    top3 = [(v, round(f * 100, 1)) for v, f in vc.head(3).items()]
    rare_n = int((vc < _RARE_CAT_THRESHOLD).sum())
    dominant = round(float(vc.iloc[0]) * 100, 1)
    imbalance = "high" if dominant > 80 else "moderate" if dominant > 50 else "balanced"
    top_str = " ".join(f"{v}({f}%)" for v, f in top3)
    return f"top3=[{top_str}] rare_values={rare_n} imbalance={imbalance}({dominant}%)"


@register_tool("categorical", name="encoding")
def _cat_encoding(df: pd.DataFrame, col: str) -> str:
    """Encoding recommendation — label / one-hot / hash / target."""
    s = df[col].dropna().astype(str)
    n_unique = s.nunique()
    n_total = len(s)
    if n_unique <= 2:
        return f"rec=binary_label n_unique={n_unique}"
    if n_unique <= 10:
        return f"rec=one_hot n_unique={n_unique}"
    if n_unique / max(n_total, 1) > _HIGH_CARD_RATIO:
        return f"rec=hash_encoding n_unique={n_unique} (high cardinality)"
    return f"rec=target_encoding n_unique={n_unique}"


@register_tool("categorical", name="cardinality")
def _cat_cardinality(df: pd.DataFrame, col: str) -> str:
    """Cardinality analysis."""
    s = df[col].dropna().astype(str)
    n_unique = s.nunique()
    n_total = len(df)
    ratio = round(n_unique / n_total, 3) if n_total else 0
    level = ("binary" if n_unique <= 2 else
             "low" if n_unique <= 10 else
             "medium" if n_unique <= 50 else
             "high" if ratio < _HIGH_CARD_RATIO else "very_high(id-like)")
    return f"unique={n_unique}/{n_total} ratio={ratio} level={level}"


# ═══════════════════════════════════════════════════════════════════════════════
# DATETIME TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@register_tool("datetime", name="validation")
def _dt_validation(df: pd.DataFrame, col: str) -> str:
    """Datetime validation — invalid, future, very old dates."""
    raw = df[col]
    total = len(raw)
    parsed = pd.to_datetime(raw, errors="coerce")
    invalid = int(parsed.isna().sum())
    now = pd.Timestamp.now()
    future = int((parsed > now).sum())
    old = int(((now - parsed).dt.days > 365 * 5).sum())
    return (
        f"valid={total - invalid}/{total} invalid={invalid} "
        f"future={future} older_5yr={old}"
    )


@register_tool("datetime", name="trend")
def _dt_trend(df: pd.DataFrame, col: str) -> str:
    """Trend and seasonality — transaction volume trend."""
    parsed = pd.to_datetime(df[col], errors="coerce").dropna()
    if len(parsed) < 4:
        return "insufficient data"
    daily = parsed.dt.date.value_counts().sort_index()
    if len(daily) < 2:
        return "single day"
    first_h = int(daily.iloc[:len(daily)//2].mean())
    second_h = int(daily.iloc[len(daily)//2:].mean())
    trend = ("increasing" if second_h > first_h * 1.1 else
             "decreasing" if second_h < first_h * 0.9 else "stable")
    return f"trend={trend} avg_per_day={round(daily.mean(),1)} peak_day={daily.idxmax()}"


@register_tool("datetime", name="decomposition")
def _dt_decomposition(df: pd.DataFrame, col: str) -> str:
    """Time decomposition — hour, weekday, month distribution."""
    parsed = pd.to_datetime(df[col], errors="coerce").dropna()
    if len(parsed) == 0:
        return "no data"
    n = len(parsed)
    off_hours = int(((parsed.dt.hour < 7) | (parsed.dt.hour > 21)).sum())
    weekends  = int((parsed.dt.dayofweek >= 5).sum())
    top_hour  = int(parsed.dt.hour.value_counts().idxmax())
    top_month = int(parsed.dt.month.value_counts().idxmax())
    return (
        f"off_hours={off_hours}({round(off_hours/n*100,1)}%) "
        f"weekends={weekends}({round(weekends/n*100,1)}%) "
        f"peak_hour={top_hour}h peak_month={top_month}"
    )


@register_tool("datetime", name="rolling_avg")
def _dt_rolling(df: pd.DataFrame, col: str) -> str:
    """Rolling averages — 7-day window transaction count."""
    parsed = pd.to_datetime(df[col], errors="coerce").dropna()
    if len(parsed) < 7:
        return "insufficient data for rolling average"
    daily = parsed.dt.date.value_counts().sort_index()
    if len(daily) < 7:
        return f"only {len(daily)} days of data"
    roll7 = daily.rolling(7).mean().dropna()
    return (
        f"7day_avg={round(float(roll7.mean()),2)} "
        f"7day_peak={round(float(roll7.max()),2)} "
        f"days_covered={len(daily)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@register_tool("text", name="stats")
def _txt_stats(df: pd.DataFrame, col: str) -> str:
    """Text cleaning and tokenisation stats."""
    s = df[col].dropna().astype(str).str.strip()
    s = s[s.str.lower() != "nan"]
    if len(s) == 0:
        return "no data"
    lengths = s.str.len()
    empty = int((s == "").sum())
    avg_tok = round(float(s.str.split().str.len().dropna().mean()), 1)
    duplicates = int(s.duplicated().sum())
    return (
        f"count={len(s)} empty={empty} avg_len={round(float(lengths.mean()),1)} "
        f"avg_tokens={avg_tok} duplicates={duplicates}"
    )


@register_tool("text", name="sentiment")
def _txt_sentiment(df: pd.DataFrame, col: str) -> str:
    """Sentiment analysis — keyword-based positive/negative/neutral."""
    POS = {"good", "approved", "cleared", "success", "valid", "complete"}
    NEG = {"rejected", "failed", "error", "invalid", "suspicious", "fraud",
           "cancel", "void", "urgent", "dispute", "overdue"}
    s = df[col].dropna().astype(str).str.lower()
    pos = int(s.apply(lambda x: any(w in x for w in POS)).sum())
    neg = int(s.apply(lambda x: any(w in x for w in NEG)).sum())
    neu = len(s) - pos - neg
    return (
        f"positive={pos} negative={neg} neutral={neu} "
        f"neg_rate={round(neg/max(len(s),1)*100,1)}%"
    )


@register_tool("text", name="keywords")
def _txt_keywords(df: pd.DataFrame, col: str) -> str:
    """Keyword extraction — top 5 most frequent meaningful words."""
    STOP = {"the", "a", "an", "is", "in", "of", "for", "to", "and", "or",
            "with", "this", "that", "was", "are", "has", "have", "been"}
    s = df[col].dropna().astype(str).str.lower()
    words: list[str] = []
    for text in s:
        words.extend([w.strip(".,!?") for w in text.split()
                      if w not in STOP and len(w) > 2])
    top5 = Counter(words).most_common(5)
    return ",".join(f"{w}({c})" for w, c in top5) if top5 else "none"


@register_tool("text", name="topics")
def _txt_topics(df: pd.DataFrame, col: str) -> str:
    """Topic modelling — rule-based cluster detection."""
    TOPICS = {
        "payment":    {"payment", "pay", "transfer", "remit", "amount"},
        "rejection":  {"reject", "refuse", "decline", "denied", "invalid"},
        "approval":   {"approve", "clear", "pass", "authorise", "authorize"},
        "fraud_risk": {"fraud", "suspicious", "urgent", "fake", "duplicate"},
        "correction": {"correct", "amend", "fix", "adjust", "reverse"},
    }
    s = df[col].dropna().astype(str).str.lower()
    counts = {topic: int(s.apply(lambda x: any(kw in x for kw in kws)).sum())
              for topic, kws in TOPICS.items()}
    found = {t: c for t, c in counts.items() if c > 0}
    if not found:
        return "no clear topics detected"
    return " ".join(f"{t}={c}" for t, c in sorted(found.items(), key=lambda x: -x[1]))


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTATION  — run all registered tools once per column (zero LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def _precompute(
    df: pd.DataFrame,
    column_routes: list[ColumnRoute],
) -> tuple[dict, dict, list, list, dict]:
    """Run all registered tools for every column dynamically.

    Returns:
        num_bounds  — IQR bounds for numerical columns (per-row flagging)
        cat_rares   — rare category sets (per-row flagging)
        dt_cols     — datetime column names (per-row flagging)
        txt_cols    — text column names (per-row flagging)
        baseline    — { col: { tool_name: result_str } } → LLM context
    """
    type_map: dict[str, str] = {r["column"]: r["detected_type"] for r in column_routes}

    num_bounds: dict[str, dict] = {}
    cat_rares:  dict[str, set]  = {}
    dt_cols:    list[str]       = []
    txt_cols:   list[str]       = []
    baseline:   dict[str, dict] = {}

    for col in df.columns:
        ctype = type_map.get(col)
        if ctype is None:
            # Auto-detect if not in routes
            if pd.api.types.is_numeric_dtype(df[col]):
                ctype = "numerical"
            else:
                continue

        baseline[col] = {}

        # Run all registered tools for this column type dynamically
        for tool_name, tool_fn in get_tools(ctype):
            try:
                baseline[col][tool_name] = tool_fn(df, col)
            except Exception as exc:
                baseline[col][tool_name] = f"err:{exc}"

        # Build per-row anomaly helpers
        if ctype == "numerical":
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) >= 4:
                unique_vals = s.nunique()
                mean = float(s.mean())
                std  = float(s.std()) if float(s.std()) > 0 else 0.0
                actual_min = float(s.min())
                if unique_vals <= 2:
                    num_bounds[col] = {"lo": None, "hi": None, "mean": mean,
                                       "std": std, "binary": True}
                else:
                    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    elif std > 0:
                        lo, hi = mean - 3 * std, mean + 3 * std
                    else:
                        num_bounds[col] = {"lo": None, "hi": None, "mean": mean,
                                           "std": 0.0, "binary": True}
                        continue
                    if actual_min >= 0:
                        lo = max(lo, 0.0)
                    num_bounds[col] = {"lo": round(lo,4), "hi": round(hi,4),
                                       "mean": round(mean,4), "std": round(std,4),
                                       "binary": False}

        elif ctype == "categorical":
            try:
                s = df[col].astype(str).str.strip()
                freq = s.value_counts(normalize=True)
                rare = set(freq[freq < _RARE_CAT_THRESHOLD].index)
                if rare:
                    cat_rares[col] = rare
            except Exception:
                pass

        elif ctype == "datetime":
            dt_cols.append(col)

        elif ctype == "text":
            txt_cols.append(col)

    return num_bounds, cat_rares, dt_cols, txt_cols, baseline


# ═══════════════════════════════════════════════════════════════════════════════
# ROW ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _row_signals(
    row: pd.Series,
    num_bounds: dict,
    cat_rares: dict,
    dt_cols: list[str],
    txt_cols: list[str],
) -> list[str]:
    signals: list[str] = []

    for col, b in num_bounds.items():
        if col not in row.index or pd.isna(row[col]) or b.get("binary"):
            continue
        val = float(row[col])
        if val < b["lo"] or val > b["hi"]:
            signals.append(f"num:{col}={round(val,2)}(mean={b['mean']})")

    for col, rare_set in cat_rares.items():
        if col not in row.index:
            continue
        val = str(row[col]).strip() if pd.notna(row[col]) else ""
        if not val or val.lower() in ("nan", "none", ""):
            signals.append(f"cat:{col}=MISSING")
        elif val in rare_set:
            signals.append(f"cat:{col}={val[:30]}(rare<3%)")

    for col in dt_cols:
        if col not in row.index or pd.isna(row[col]):
            continue
        try:
            ts = pd.to_datetime(row[col], errors="coerce")
            if pd.isna(ts):
                continue
            h = ts.hour
            now = pd.Timestamp.now()
            if h < 7 or h > 21:
                signals.append(f"dt:{col}={ts.strftime('%H:%M')}(off-hours)")
            if ts > now:
                signals.append(f"dt:{col}={ts.date()}(future)")
            if (now - ts).days > 365 * 5:
                signals.append(f"dt:{col}={ts.date()}(>5yr)")
        except Exception:
            pass

    keywords = get_keywords()
    for col in txt_cols:
        if col not in row.index:
            continue
        val = str(row[col]).strip() if pd.notna(row[col]) else ""
        if not val or val.lower() in ("nan", "none", ""):
            signals.append(f"txt:{col}=EMPTY")
        elif len(val) < 3:
            signals.append(f"txt:{col}={val!r}(too_short)")
        elif any(kw in val.lower() for kw in keywords):
            hits = [kw for kw in keywords if kw in val.lower()]
            signals.append(f"txt:{col}={val[:40]!r}(kw:{','.join(hits)})")

    return signals


def _row_summary(
    row: pd.Series,
    df: pd.DataFrame,
    num_bounds: dict,
    cat_rares: dict,
    dt_cols: list[str],
    txt_cols: list[str],
) -> str:
    parts: list[str] = []
    for col in df.columns:
        if col not in row.index:
            continue
        raw = row[col]
        flag = ""

        if col in num_bounds:
            if pd.isna(raw):
                parts.append(f"{col}=NULL")
                continue
            val = float(raw)
            b = num_bounds[col]
            if not b.get("binary") and (val < b["lo"] or val > b["hi"]):
                flag = "!"
            parts.append(f"{col}={round(val,2)}{flag}")

        elif col in cat_rares:
            val_str = str(raw).strip() if pd.notna(raw) else ""
            if not val_str or val_str.lower() in ("nan","none",""):
                parts.append(f"{col}=MISSING!")
            elif val_str in cat_rares[col]:
                parts.append(f"{col}={val_str[:25]}!")
            else:
                parts.append(f"{col}={val_str[:25]}")

        elif col in dt_cols:
            try:
                ts = pd.to_datetime(raw, errors="coerce")
                if pd.isna(ts):
                    parts.append(f"{col}=NULL")
                else:
                    flag = "!" if ts.hour < 7 or ts.hour > 21 else ""
                    parts.append(f"{col}={ts.strftime('%Y-%m-%d %H:%M')}{flag}")
            except Exception:
                parts.append(f"{col}={str(raw)[:20]}")

        elif col in txt_cols:
            val_str = str(raw).strip() if pd.notna(raw) else ""
            keywords = get_keywords()
            if not val_str or val_str.lower() in ("nan","none",""):
                flag = "!"
            elif len(val_str) < 3:
                flag = "!"
            elif any(kw in val_str.lower() for kw in keywords):
                flag = "!"
            parts.append(f"{col}={val_str[:30]}{flag}")

        else:
            parts.append(f"{col}={str(raw)[:25] if pd.notna(raw) else 'NULL'}")

    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════



def _baseline_lines(
    baseline: dict,
    num_bounds: dict,
    cat_rares: dict,
    dt_cols: list[str],
    txt_cols: list[str],
) -> list[str]:
    """Build column-baseline section for LLM prompt."""
    lines: list[str] = []
    for col, tool_results in baseline.items():
        if not tool_results:
            continue
        if col in num_bounds:
            b = num_bounds[col]
            hdr = (f"  [NUM] {col}: binary(mean={round(b['mean'],2)})" if b.get("binary")
                   else f"  [NUM] {col}: normal=[{round(b['lo'],2)},{round(b['hi'],2)}]"
                        f" mean={round(b['mean'],2)} std={round(b['std'],2)}")
        elif col in (set(cat_rares) | set()):
            hdr = f"  [CAT] {col}:"
        elif col in dt_cols:
            hdr = f"  [DT]  {col}:"
        elif col in txt_cols:
            hdr = f"  [TXT] {col}:"
        else:
            hdr = f"  [COL] {col}:"
        tool_str = " | ".join(f"{k}={v}" for k, v in tool_results.items())
        lines.append(f"{hdr} {tool_str}")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA-ONLY FLOW: ONE LLM CALL → TEMPLATES, THEN PURE PYTHON PER ROW
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json_safe(raw: str) -> dict:
    """Parse JSON from LLM output, tolerating markdown code fences."""
    if isinstance(raw, list):
        raw = " ".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw)
    for attempt in (raw, re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw),
                    re.search(r"(\{[\s\S]+\})", raw)):
        text = attempt.group(1) if hasattr(attempt, "group") else (attempt or "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _uni_signal_prefix(sig: str) -> str:
    """Extract the column-type prefix from a signal string (num / cat / dt / txt)."""
    return sig.split(":")[0] if ":" in sig else "unknown"


def _uni_signal_detail(sig: str) -> str:
    """Convert a raw signal string into a readable {detail} filler for templates."""
    if ":" not in sig:
        return sig
    _, rest = sig.split(":", 1)
    for marker, readable in (
        ("(rare<3%)",   " — rare value (< 3% of records)"),
        ("MISSING",     "is missing / empty"),
        ("(off-hours)", " — outside business hours (07:00–21:00)"),
        ("(future)",    " — future-dated timestamp"),
        ("(>5yr)",      " — older than 5 years"),
        ("EMPTY",       "is empty"),
        ("(too_short)", " — suspiciously short (< 3 chars)"),
    ):
        rest = rest.replace(marker, readable)
    return rest


def _uni_fetch_templates(
    llm,
    table: str,
    signals_cache: dict[int, list[str]],
) -> dict[str, str]:
    """ONE LLM call per table — signal type descriptions only, no row data, no baselines.

    Template keys are the signal prefixes actually present (num / cat / dt / txt).
    Descriptions are pulled from the existing _ANOMALY_DESCRIPTIONS and keyword
    registries so the prompt adapts automatically when registries change.
    Baseline stats are excluded to keep the prompt well within token limits.
    """
    prefix_to_coltype = {"num": "numerical", "cat": "categorical",
                         "dt": "datetime",   "txt": "text"}

    present_prefixes = sorted({
        _uni_signal_prefix(s)
        for sigs in signals_cache.values()
        for s in sigs
        if _uni_signal_prefix(s) in prefix_to_coltype
    })
    if not present_prefixes:
        return {}

    # Descriptions from the existing _ANOMALY_DESCRIPTIONS dynamic registry
    desc_lines = [
        f"  {pfx} ({prefix_to_coltype[pfx]}): "
        f"{_ANOMALY_DESCRIPTIONS.get(prefix_to_coltype[pfx], 'column anomaly')}"
        for pfx in present_prefixes
    ]

    # Keyword context from dynamic registry (only when txt signals present)
    kw_lines: list[str] = []
    if "txt" in present_prefixes:
        kws = get_keywords()
        if kws:
            kw_lines = [
                "  Suspicious keywords registered: "
                + ", ".join(f"'{k}' ({v})" for k, v in list(kws.items())[:8])
            ]

    prompt_lines = [
        f"Financial audit table: '{table}'.",
        "",
        "Signal type prefixes detected in flagged rows and their meanings:",
    ] + desc_lines + kw_lines + [
        "",
        "Write one 2-3 sentence financial audit insight TEMPLATE for each signal prefix.",
        "Rules:",
        "  • Use {detail} as the ONLY placeholder — Python fills it with the actual "
        "column name, value, and statistical context at runtime.",
        "  • Start with RISK LEVEL: [HIGH RISK] / [MEDIUM RISK] / [LOW RISK]",
        "  • Explain the business / audit implication of this signal type.",
        "  • End with a concrete recommended audit action.",
        "  • Keep it generic — no specific column names or values.",
        "",
        f"Return a JSON object with exactly these keys: {present_prefixes}",
    ]

    try:
        msgs = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content="\n".join(prompt_lines)),
        ]
        resp = groq_invoke(llm, msgs, agent_name="universal")
        raw  = resp.content if hasattr(resp, "content") else str(resp)
        parsed = _extract_json_safe(raw)
        return {k: v if isinstance(v, str) else " ".join(v.values()) if isinstance(v, dict) else str(v)
                for k, v in parsed.items()}
    except Exception as exc:
        log.error("UniversalRowAgent template fetch failed for '%s': %s", table, exc)
        return {}


def _uni_render_insight(signals: list[str], templates: dict[str, str]) -> str:
    """Pure Python — render a row insight from pre-fetched templates.

    Lead signal: first signal in the list (detection order from _row_signals).
    Secondary signals: appended as brief additional flags.
    """
    if not signals:
        return ""
    lead_key = _uni_signal_prefix(signals[0])
    template = templates.get(lead_key, "")
    detail   = _uni_signal_detail(signals[0])
    insight  = template.replace("{detail}", detail) if template else f"[FLAGGED] {detail}"
    if len(signals) > 1:
        others  = "; ".join(_uni_signal_detail(s) for s in signals[1:])
        insight += f" Additional flags: {others}."
    return insight


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class UniversalRowAgent:
    """Single agent replacing the 4 type-based agents.

    Tools are discovered dynamically from the registry at runtime.
    To add a new analysis capability, register a function with @register_tool.
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

    def _run_table(
        self,
        df: pd.DataFrame,
        table: str,
        table_routes: list[ColumnRoute],
    ) -> list[AgentResult]:

        # Extract Excel-compatible DB row numbers injected by db.load(), then drop the
        # metadata column so it does not affect type detection or anomaly analysis.
        # db_row_nums[i] = Excel row number for df.iloc[i] (header row = 1, data starts at 2).
        db_row_nums: dict[int, int] = {}
        if "__db_row_num__" in df.columns:
            for i in range(len(df)):
                db_row_nums[i] = int(df["__db_row_num__"].iloc[i])
            df = df.drop(columns=["__db_row_num__"]).reset_index(drop=True)

        # STEP 1 — run all registered tools per column (zero LLM)
        num_bounds, cat_rares, dt_cols, txt_cols, baseline = _precompute(df, table_routes)
        registered = {t: len(get_tools(t)) for t in ("numerical","categorical","datetime","text")}
        # log.info("UniversalRowAgent %s: tools registered=%s", table, registered)

        # STEP 2 — find flagged rows and cache signals (avoids recomputation in STEP 5)
        total = len(df)
        signals_cache: dict[int, list[str]] = {}
        for idx in range(total):
            sigs = _row_signals(df.iloc[idx], num_bounds, cat_rares, dt_cols, txt_cols)
            if sigs:
                signals_cache[idx] = sigs
        row_indices = list(signals_cache.keys())[:200]

        if not row_indices:
            log.info("UniversalRowAgent %s: no anomalous rows found", table)
            return []

        # STEP 3 — build schema context from pre-computed baselines (no row data)
        col_bl = _baseline_lines(baseline, num_bounds, cat_rares, dt_cols, txt_cols)

        # STEP 4 — ONE LLM call: schema + baselines → insight templates
        templates = _uni_fetch_templates(self._llm, table, signals_cache)

        # STEP 4b — pure Python: render one insight per flagged row (zero more LLM calls)
        insights_map: dict[str, str] = {
            f"row_{idx}": _uni_render_insight(signals_cache[idx], templates)
            for idx in row_indices
        }

        # STEP 5 — assemble AgentResult per row
        results: list[AgentResult] = []
        for idx in row_indices:
            row     = df.iloc[idx]
            row_key = f"row_{idx}"
            signals = signals_cache[idx]

            row_vals: dict[str, float] = {}
            anomalous_cols: list[str] = []
            for col, b in num_bounds.items():
                if col in row.index and pd.notna(row[col]):
                    val = float(row[col])
                    row_vals[col] = round(val, 4)
                    if not b.get("binary") and b["lo"] is not None:
                        if val < b["lo"] or val > b["hi"]:
                            anomalous_cols.append(col)

            # ALL column values for this row — every column in the table,
            # stored as strings so the LLM sees the complete transaction record
            # with all fields locked together on one row.
            all_values: dict[str, str] = {}
            for col in df.columns:
                raw_val = row[col] if col in row.index else None
                if pd.isna(raw_val) if raw_val is not None else True:
                    all_values[col] = "NULL"
                else:
                    all_values[col] = str(raw_val)

            results.append(AgentResult(
                agent_name="universal",
                table=table,
                column=row_key,
                analysis={
                    "row_index":         db_row_nums.get(idx, idx + 2),  # Excel row number
                    "values":            row_vals,
                    "anomalous_columns": anomalous_cols,
                    "all_signals":       signals,
                    "signal_count":      len(signals),
                    "col_means":         {c: round(b["mean"],4) for c, b in num_bounds.items()},
                    "all_values":        all_values,
                },
                insights_text=insights_map.get(row_key, ""),
                chart_data={
                    "row_index": db_row_nums.get(idx, idx + 2),  # Excel row number
                    "values":    row_vals,
                    "means":     {c: round(b["mean"],2) for c, b in num_bounds.items()},
                    "anomalies": anomalous_cols,
                    "signals":   signals,
                },
                chart_type="bar",
                error=None,
            ))
        return results

    def run(self, raw_data: dict, column_routes: list[ColumnRoute]) -> list[AgentResult]:
        by_table: dict[str, list[ColumnRoute]] = defaultdict(list)
        for route in column_routes:
            table = route["table"]
            if raw_data.get(table) is not None and not raw_data[table].empty:
                by_table[table].append(route)

        results: list[AgentResult] = []
        for table, table_routes in by_table.items():
            log.info("UniversalRowAgent: table '%s' (%d routes)", table, len(table_routes))
            results.extend(self._run_table(raw_data[table], table, table_routes))

        log.info("UniversalRowAgent complete: %d total row results", len(results))
        return results
