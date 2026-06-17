"""Pandas DataFrame analysis — pure Python/pandas/numpy computation, no LLM.

Stats are computed deterministically in-process. The LLM is only called once
per column (in the calling agent) for the insights sentence.
"""
from __future__ import annotations

import asyncio
import logging
import re

import numpy as np
import pandas as pd

from config import Config, get_config
from db import DB

log = logging.getLogger(__name__)


# ── DataFrame normalisation ───────────────────────────────────────────────────

def _to_scalar(x):
    if isinstance(x, (str, bytes)) or x is None:
        return x
    if hasattr(x, "__iter__"):
        try:
            items = list(x)
            return _to_scalar(items[0]) if items else None
        except Exception:
            return x
    return x


def _is_non_scalar_iterable(x) -> bool:
    return (
        x is not None
        and not isinstance(x, (str, bytes, bool, int, float, np.integer, np.floating))
        and hasattr(x, "__iter__")
    )


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if df[col].apply(_is_non_scalar_iterable).any():
            df[col] = df[col].apply(_to_scalar)
    return df


# ── Numerical ─────────────────────────────────────────────────────────────────

def _compute_numerical(df: pd.DataFrame, col: str, cfg: Config) -> dict:
    series = pd.to_numeric(df[col], errors="coerce")
    clean  = series.dropna()
    total  = len(series)
    n      = len(clean)

    mean   = float(clean.mean())          if n > 0 else None
    median = float(clean.median())        if n > 0 else None
    mode   = float(clean.mode().iloc[0])  if n > 0 else None
    std    = float(clean.std())           if n > 1 else None
    var    = float(clean.var())           if n > 1 else None
    skew   = float(clean.skew())          if n > 2 else None
    kurt   = float(clean.kurt())          if n > 3 else None
    min_v  = float(clean.min())           if n > 0 else None
    max_v  = float(clean.max())           if n > 0 else None
    p25    = float(clean.quantile(0.25))  if n > 0 else None
    p50    = float(clean.quantile(0.50))  if n > 0 else None
    p75    = float(clean.quantile(0.75))  if n > 0 else None
    p95    = float(clean.quantile(0.95))  if n > 0 else None

    null_count = int(series.isna().sum())
    null_pct   = round(null_count / total * 100, 2) if total > 0 else 0.0
    if null_pct == 0:        classification = "Complete"
    elif null_pct < 5:       classification = "MCAR"
    elif null_pct < 20:      classification = "MAR"
    else:                    classification = "MNAR"

    dup_count = int(series.duplicated().sum())
    dup_pct   = round(dup_count / total * 100, 2) if total > 0 else 0.0

    if n > 0:
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr     = q3 - q1
        iqr_mask = (clean < q1 - cfg.iqr_multiplier * iqr) | (clean > q3 + cfg.iqr_multiplier * iqr)
        iqr_count = int(iqr_mask.sum())
        iqr_pct   = round(iqr_count / n * 100, 2)
        outlier_idx = clean[iqr_mask].abs().nlargest(5).index
        outlier_rows = df.loc[outlier_idx].to_dict(orient="records")
    else:
        iqr_count = 0; iqr_pct = 0.0
        outlier_rows = []

    if n > 1 and std and std > 0:
        z       = (clean - clean.mean()) / clean.std()
        z_count = int((z.abs() > cfg.zscore_threshold).sum())
        z_pct   = round(z_count / n * 100, 2)
    else:
        z_count = 0; z_pct = 0.0

    if skew is not None:
        if abs(skew) < 0.5:   shape = "normal"
        elif skew > 1:        shape = "right_skewed"
        elif skew < -1:       shape = "left_skewed"
        else:                 shape = "moderately_skewed"
    else:
        shape = "unknown"

    if n > 0:
        hist_counts, bin_edges = np.histogram(clean, bins=cfg.distribution_bins)
        histogram_bins = [int(x) for x in hist_counts]
        bin_edges_list = [float(x) for x in bin_edges]
    else:
        histogram_bins = []; bin_edges_list = []

    total_sum = float(clean.sum())          if n > 0 else 0.0
    var_95    = float(clean.quantile(0.05)) if n > 0 else None
    cvar_95   = float(clean[clean <= clean.quantile(0.05)].mean()) if n > 0 else None
    sharpe    = float(clean.mean() / clean.std()) if n > 1 and std and std > 0 else None

    if n > 1:
        mid = n // 2
        fh, sh = float(clean.iloc[:mid].mean()), float(clean.iloc[mid:].mean())
        trend_dir = "up" if sh > fh * 1.01 else ("down" if sh < fh * 0.99 else "flat")
    else:
        trend_dir = "flat"

    if n > 0:
        rn_pct   = round(float((clean % 100 == 0).sum() / n * 100), 2)
        neg_cnt  = int((clean < 0).sum())
        p99      = float(clean.quantile(0.99))
        ext_high = int((clean > p99).sum())
        fraud_risk = ("high"   if rn_pct > 30 or ext_high > n * 0.01 else
                      "medium" if rn_pct > 15 or neg_cnt > 0 else "low")
    else:
        rn_pct = 0.0; neg_cnt = 0; ext_high = 0; fraud_risk = "low"

    num_cols = df.select_dtypes(include="number").columns.tolist()
    correlation: dict = {}
    if n > 1 and std and std > 0:
        for c in num_cols:
            if c == col:
                continue
            other = pd.to_numeric(df[c], errors="coerce")
            # Skip zero-variance columns — correlation is undefined, causes RuntimeWarning
            if other.dropna().std() == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = series.corr(other, method=cfg.correlation_method)
            if not np.isnan(r):
                correlation[c] = round(float(r), 4)
        correlation = dict(
            sorted(correlation.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        )

    return {
        "descriptive_stats": {
            "mean": mean, "median": median, "mode": mode, "std": std,
            "variance": var, "skewness": skew, "kurtosis": kurt,
            "min": min_v, "max": max_v, "p25": p25, "p50": p50,
            "p75": p75, "p95": p95, "count": n,
        },
        "missing":         {"null_count": null_count, "null_pct": null_pct, "classification": classification},
        "duplicates":      {"duplicate_count": dup_count, "duplicate_pct": dup_pct},
        "outliers_iqr":    {"count": iqr_count, "pct": iqr_pct, "top_rows": outlier_rows},
        "outliers_zscore": {"count": z_count, "pct": z_pct},
        "distribution":    {"shape": shape, "skewness": skew, "kurtosis": kurt,
                            "histogram_bins": histogram_bins, "bin_edges": bin_edges_list},
        "financial_kpis":  {"total_sum": total_sum, "trend_direction": trend_dir,
                            "var_95": var_95, "cvar_95": cvar_95, "sharpe_ratio": sharpe},
        "fraud_signals":   {"round_number_pct": rn_pct, "negative_value_count": neg_cnt,
                            "extreme_high_count": ext_high, "fraud_risk_level": fraud_risk},
        "correlation":     correlation,
    }


# ── Categorical ───────────────────────────────────────────────────────────────

def _compute_categorical(df: pd.DataFrame, col: str, cfg: Config) -> dict:
    series     = df[col]
    total      = len(series)
    null_count = int(series.isna().sum())
    null_pct   = round(null_count / total * 100, 2) if total > 0 else 0.0
    mode_val   = str(series.mode().iloc[0]) if series.notna().any() else ""

    vc           = series.value_counts()
    total_unique = int(series.nunique())
    top_20       = {str(k): int(v) for k, v in vc.head(20).items()}
    dominant     = str(vc.index[0]) if len(vc) > 0 else ""
    dominant_pct = round(float(vc.iloc[0] / total * 100), 2) if total > 0 else 0.0
    rare_count   = int((vc < total * cfg.rare_category_threshold).sum())

    cardinality_ratio = round(total_unique / total, 4) if total > 0 else 0.0
    cardinality_type  = "low_cardinality" if total_unique <= 10 else "high_cardinality"

    if len(vc) >= 2:
        most, least  = int(vc.iloc[0]), int(vc.iloc[-1])
        ratio        = round(most / least, 2) if least > 0 else float(most)
        dominant_cls = str(vc.index[0])
        minority_cls = str(vc.index[-1])
        is_imbalanced = ratio > cfg.imbalance_ratio_threshold
    else:
        ratio = 0.0; dominant_cls = dominant; minority_cls = dominant; is_imbalanced = False

    non_null    = series.dropna().astype(str)
    case_groups = max(0, int(non_null.str.lower().nunique() - non_null.nunique())) if len(non_null) > 0 else 0
    ws_count    = int(non_null.str.contains(r'^\s|\s$', regex=True).sum()) if len(non_null) > 0 else 0
    recommended = "standardize" if case_groups > 0 or ws_count > 0 else "ok"

    sorted_unique    = sorted(series.dropna().astype(str).unique())[:10]
    encoding_preview = {v: i for i, v in enumerate(sorted_unique)}

    segmentation: dict = {}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        try:
            grp = df.groupby(col)[num_cols[0]].mean()
            segmentation = {str(k): round(float(v), 4) for k, v in grp.items() if not np.isnan(v)}
        except Exception:
            pass

    return {
        "frequency":    {"top_20": top_20, "total_rows": total, "total_unique": total_unique,
                         "dominant_category": dominant, "dominant_pct": dominant_pct,
                         "rare_category_count": rare_count},
        "cardinality":  {"unique_count": total_unique, "cardinality_ratio": cardinality_ratio, "type": cardinality_type},
        "imbalance":    {"ratio": ratio, "dominant_class": dominant_cls, "minority_class": minority_cls, "is_imbalanced": is_imbalanced},
        "missing":      {"null_count": null_count, "null_pct": null_pct, "mode_value": mode_val},
        "data_quality": {"case_inconsistency_groups": case_groups, "whitespace_issue_count": ws_count, "recommended_action": recommended},
        "encoding_preview": encoding_preview,
        "segmentation": segmentation,
    }


# ── Datetime ──────────────────────────────────────────────────────────────────

def _compute_datetime(df: pd.DataFrame, col: str, cfg: Config) -> dict:
    _EMPTY = {
        "validation":    {"valid_count": 0, "invalid_count": len(df), "min_date": None, "max_date": None, "date_range_days": 0},
        "missing_dates": {"total_dates": 0, "expected_dates": 0, "missing_count": 0, "missing_pct": 0.0, "completeness_score": 0.0, "largest_gap_days": 0},
        "time_features": {}, "trend": {}, "seasonality": {}, "mom_yoy": {}, "anomalies": {}, "forecast": {},
    }

    dt    = pd.to_datetime(df[col], errors="coerce")
    valid = dt.dropna()
    if len(valid) == 0:
        return _EMPTY

    valid_count   = int(dt.notna().sum())
    invalid_count = int(dt.isna().sum())
    min_dt, max_dt = valid.min(), valid.max()
    date_range_days = (max_dt - min_dt).days

    all_cal   = pd.date_range(min_dt, max_dt, freq="D")
    expected  = len(all_cal)
    actual    = set(valid.dt.normalize())
    miss_cnt  = len(set(all_cal) - actual)
    miss_pct  = round(miss_cnt / expected * 100, 2) if expected > 0 else 0.0
    completeness = round(100 - miss_pct, 2)

    sorted_v = valid.sort_values()
    gaps     = sorted_v.diff().dt.days.dropna()
    max_gap  = int(gaps.max()) if len(gaps) > 0 else 0

    dow_dist     = {str(k): int(v) for k, v in valid.dt.day_name().value_counts().items()}
    month_dist   = {int(k): int(v) for k, v in valid.dt.month.value_counts().sort_index().items()}
    quarter_dist = {int(k): int(v) for k, v in valid.dt.quarter.value_counts().sort_index().items()}
    weekend_cnt  = int(valid.dt.dayofweek.isin([5, 6]).sum())
    weekend_pct  = round(weekend_cnt / len(valid) * 100, 2)
    busiest_day  = str(valid.dt.day_name().value_counts().index[0])
    busiest_mon  = int(valid.dt.month.value_counts().index[0])
    busiest_qtr  = int(valid.dt.quarter.value_counts().index[0])

    trend = {}; seasonality = {}; mom_yoy = {}; anomalies = {}; forecast = {}

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        num_s  = pd.to_numeric(df[num_cols[0]], errors="coerce")
        mask   = dt.notna() & num_s.notna()
        dt_c   = dt[mask]; num_c = num_s[mask]

        if len(num_c) > 1:
            x = np.arange(len(num_c))
            slope, intercept = np.polyfit(x, num_c.values, 1)
            slope = float(slope)
            t_dir = "up" if slope > 0.01 else ("down" if slope < -0.01 else "flat")
            rolling = num_c.rolling(cfg.rolling_window, min_periods=1).mean()
            trend = {"trend_direction": t_dir, "slope": round(slope, 6),
                     "rolling_mean": [round(float(v), 4) for v in rolling.tolist()[-20:]]}

            m_grp = num_c.groupby(dt_c.dt.month).mean()
            m_avg = {int(k): round(float(v), 4) for k, v in m_grp.items()}
            sea_str = round(float(m_grp.std() / m_grp.mean()), 4) if float(m_grp.mean()) != 0 else 0.0
            seasonality = {"monthly_averages": m_avg, "peak_month": int(m_grp.idxmax()),
                           "trough_month": int(m_grp.idxmin()), "seasonality_strength": abs(sea_str)}

            mean_n = float(num_c.mean()); std_n = float(num_c.std())
            anomalies = {
                "spike_dates": dt_c[num_c > mean_n + 3 * std_n].dt.strftime("%Y-%m-%d").tolist()[:10],
                "dip_dates":   dt_c[num_c < mean_n - 3 * std_n].dt.strftime("%Y-%m-%d").tolist()[:10],
            }

            last_x = len(num_c)
            f_vals = [round(float(slope * xi + intercept), 4) for xi in range(last_x, last_x + cfg.forecast_periods)]
            forecast = {"method": "linear_trend", "forecasted_values": f_vals,
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "trend_slope": round(slope, 6)}

            tmp = pd.DataFrame({"date": dt_c.values, "value": num_c.values})
            tmp["period"] = pd.to_datetime(tmp["date"]).dt.to_period("M")
            m_sum   = tmp.groupby("period")["value"].sum()
            periods = m_sum.index.tolist()
            mom_pct: dict = {}; yoy_pct: dict = {}
            for p in periods[-cfg.seasonality_period:]:
                pi = periods.index(p)
                if pi > 0 and m_sum[periods[pi - 1]] != 0:
                    mom_pct[str(p)] = round(float((m_sum[p] - m_sum[periods[pi - 1]]) / m_sum[periods[pi - 1]] * 100), 2)
                yoy_p = p - 12
                if yoy_p in m_sum and m_sum[yoy_p] != 0:
                    yoy_pct[str(p)] = round(float((m_sum[p] - m_sum[yoy_p]) / m_sum[yoy_p] * 100), 2)
            mom_yoy = {"mom_pct_change": mom_pct, "yoy_pct_change": yoy_pct}

    return {
        "validation":    {"valid_count": valid_count, "invalid_count": invalid_count,
                          "min_date": min_dt.strftime("%Y-%m-%d"), "max_date": max_dt.strftime("%Y-%m-%d"),
                          "date_range_days": date_range_days},
        "missing_dates": {"total_dates": valid_count, "expected_dates": expected,
                          "missing_count": miss_cnt, "missing_pct": miss_pct,
                          "completeness_score": completeness, "largest_gap_days": max_gap},
        "time_features": {"day_of_week_distribution": dow_dist, "month_distribution": month_dist,
                          "quarter_distribution": quarter_dist, "weekend_count": weekend_cnt,
                          "weekend_pct": weekend_pct, "busiest_day": busiest_day,
                          "busiest_month": busiest_mon, "busiest_quarter": busiest_qtr},
        "trend": trend, "seasonality": seasonality,
        "mom_yoy": mom_yoy, "anomalies": anomalies, "forecast": forecast,
    }


# ── Text ──────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','was','are','were','be','been','this','that','it','its','we',
    'you','he','she','they','not','no','have','has','had','do','does','did',
    'will','would','can','could','should','may','might','our','their','as',
    'if','so','than','then','there',
}
_POSITIVE_WORDS  = {"good","great","excellent","approved","cleared","success",
                    "profit","gain","increase","valid","complete"}
_NEGATIVE_WORDS  = {"bad","poor","rejected","failed","error","loss","decrease",
                    "complaint","issue","dispute","invalid","missing","delay","wrong"}
_COMPLAINT_WORDS = {"complaint","issue","problem","error","rejected","failed",
                    "dispute","wrong","missing","delay","incorrect","discrepancy"}


def _compute_text(df: pd.DataFrame, col: str, cfg: Config) -> dict:
    raw        = df[col]
    null_count = int(raw.isna().sum())
    series     = raw.dropna().astype(str)
    total      = len(series)

    if total == 0:
        return {
            "basic_stats":  {"total_texts": 0, "null_count": null_count, "avg_length": 0.0,
                             "min_length": 0, "max_length": 0, "avg_word_count": 0.0},
            "sentiment":    {"positive_count": 0, "negative_count": 0, "neutral_count": 0,
                             "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0},
            "top_words":    {},
            "complaints":   {"complaint_count": 0, "complaint_pct": 0.0},
            "text_quality": {"unique_count": 0, "duplicate_text_count": 0, "avg_unique_words_per_text": 0.0},
        }

    lengths     = series.str.len()
    word_counts = series.str.split().str.len().fillna(0)

    pos = neg = neu = complaint_count = 0
    word_freq: dict[str, int] = {}
    unique_word_counts: list[int] = []

    for text in series:
        tokens   = re.findall(r'[a-z]+', text.lower())
        word_set = set(tokens)

        has_pos = bool(word_set & _POSITIVE_WORDS)
        has_neg = bool(word_set & _NEGATIVE_WORDS)
        if has_pos and not has_neg:  pos += 1
        elif has_neg:                neg += 1
        else:                        neu += 1

        if word_set & _COMPLAINT_WORDS:
            complaint_count += 1

        filtered = [t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS]
        for t in filtered:
            word_freq[t] = word_freq.get(t, 0) + 1
        unique_word_counts.append(len(set(filtered)))

    top_words       = dict(sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)[:cfg.max_keywords])
    unique_count    = int(series.nunique())
    avg_unique_words = round(float(np.mean(unique_word_counts)), 1) if unique_word_counts else 0.0

    return {
        "basic_stats": {
            "total_texts": total, "null_count": null_count,
            "avg_length": round(float(lengths.mean()), 1),
            "min_length": int(lengths.min()), "max_length": int(lengths.max()),
            "avg_word_count": round(float(word_counts.mean()), 1),
        },
        "sentiment": {
            "positive_count": pos, "negative_count": neg, "neutral_count": neu,
            "positive_pct": round(pos / total * 100, 2),
            "negative_pct": round(neg / total * 100, 2),
            "neutral_pct":  round(neu / total * 100, 2),
        },
        "top_words": top_words,
        "complaints": {"complaint_count": complaint_count,
                       "complaint_pct": round(complaint_count / total * 100, 2)},
        "text_quality": {"unique_count": unique_count,
                         "duplicate_text_count": total - unique_count,
                         "avg_unique_words_per_text": avg_unique_words},
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────

_COMPUTE_MAP = {
    "numerical":   _compute_numerical,
    "categorical": _compute_categorical,
    "datetime":    _compute_datetime,
    "text":        _compute_text,
}


# ── Sync core ─────────────────────────────────────────────────────────────────

def _run_analysis_sync(
    tables: list[str],
    col: str,
    analysis_type: str,
    cfg: Config,
) -> dict:
    db = DB()
    try:
        dfs: list[pd.DataFrame] = []
        for table in tables:
            try:
                df = db.load(table)
                if df is not None and not df.empty:
                    dfs.append(_normalize_df(df))
                else:
                    log.warning("Table '%s' is empty or not found", table)
            except Exception as exc:
                log.error("Failed to load table '%s': %s", table, exc)
    finally:
        db.close()

    if not dfs:
        return {"error": f"No data loaded from tables: {tables}"}

    target_df = next((d for d in dfs if col in d.columns), dfs[0])

    compute_fn = _COMPUTE_MAP.get(analysis_type)
    if compute_fn is None:
        return {"error": f"Unknown analysis_type: {analysis_type}"}

    try:
        return compute_fn(target_df, col, cfg)
    except Exception as exc:
        log.error("pandas_analysis [%s/%s]: %s", analysis_type, col, exc)
        return {"error": str(exc)}


# ── Public sync API (in-memory DataFrame) ─────────────────────────────────────

def run_analysis_from_df(
    df: pd.DataFrame,
    col: str,
    analysis_type: str,
    config: Config | None = None,
) -> dict:
    """Compute column statistics from an already-loaded DataFrame (no DB access).

    Args:
        df:            DataFrame containing the column to analyse.
        col:           Column to analyse.
        analysis_type: 'numerical' | 'categorical' | 'datetime' | 'text'
        config:        App config (auto-loaded if None).
    """
    cfg = config or get_config()
    df = _normalize_df(df)
    compute_fn = _COMPUTE_MAP.get(analysis_type)
    if compute_fn is None:
        return {"error": f"Unknown analysis_type: {analysis_type}"}
    try:
        return compute_fn(df, col, cfg)
    except Exception as exc:
        log.error("pandas_analysis [%s/%s]: %s", analysis_type, col, exc)
        return {"error": str(exc)}


# ── Public async API ──────────────────────────────────────────────────────────

async def run_pandas_analysis(
    tables: str | list[str],
    col: str,
    analysis_type: str,
    config: Config | None = None,
) -> dict:
    """Compute column statistics from DB tables using pure pandas/numpy.

    Args:
        tables:        DB table name or list of table names.
        col:           Column to analyse.
        analysis_type: 'numerical' | 'categorical' | 'datetime' | 'text'
        config:        App config (auto-loaded if None).
    """
    cfg = config or get_config()
    if isinstance(tables, str):
        tables = [tables]
    return await asyncio.to_thread(_run_analysis_sync, tables, col, analysis_type, cfg)
