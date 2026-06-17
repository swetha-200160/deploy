"""
Pure data-accessor functions.
The LLM calls these via tool-use; they access raw_data from Python memory
and return compact JSON results. No logic about what is "interesting"
lives here — that is the LLM's job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Schema discovery ──────────────────────────────────────────────────────────

def list_tables(raw_data: dict[str, pd.DataFrame]) -> list[str]:
    return list(raw_data.keys())


def get_schema(raw_data: dict[str, pd.DataFrame], table: str) -> dict:
    if table not in raw_data:
        return {"error": f"Table '{table}' not found. Available: {list(raw_data.keys())}"}
    df = raw_data[table]
    columns = []
    for col in df.columns:
        columns.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique_count": int(df[col].nunique()),
        })
    return {"table": table, "row_count": len(df), "columns": columns}


def get_column_stats(raw_data: dict[str, pd.DataFrame], table: str, column: str) -> dict:
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    s = df[column].dropna()
    base = {
        "table": table, "column": column,
        "dtype": str(df[column].dtype),
        "count": int(s.count()),
        "null_pct": round(float(df[column].isna().mean() * 100), 2),
    }
    if pd.api.types.is_numeric_dtype(df[column]):
        base.update({
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "median": round(float(s.median()), 4),
        })
    else:
        base["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
    return base


def get_column_coverage(raw_data: dict[str, pd.DataFrame], table: str) -> list[dict]:
    """Null%, unique count, filled% for every column — used to spot dark/unused data."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    return [
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "filled_pct": round(float(df[col].notna().mean() * 100), 2),
            "unique_count": int(df[col].nunique()),
        }
        for col in df.columns
    ]


# ── Correlation ───────────────────────────────────────────────────────────────

def get_correlation(raw_data: dict[str, pd.DataFrame], table: str,
                    col_a: str, col_b: str) -> dict:
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if col_a not in df.columns or col_b not in df.columns:
        return {"error": f"Column not found. Available: {list(df.columns)}"}
    both = df[[col_a, col_b]].dropna()
    if len(both) < 5:
        return {"error": "Insufficient non-null rows"}
    c = float(both[col_a].corr(both[col_b]))
    return {"col_a": col_a, "col_b": col_b, "correlation": round(c, 4), "n": len(both)}


def get_correlation_matrix(raw_data: dict[str, pd.DataFrame], table: str) -> list[dict]:
    """All numeric column-pair correlations sorted by absolute value."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return [{"error": "Need at least 2 numeric columns"}]
    corr = numeric.corr()
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if not np.isnan(c):
                pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(float(c), 4)})
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:20]


def get_cross_table_correlation(raw_data: dict[str, pd.DataFrame],
                                 table_a: str, col_a: str,
                                 table_b: str, col_b: str) -> dict:
    for t, c in [(table_a, col_a), (table_b, col_b)]:
        if t not in raw_data:
            return {"error": f"Table '{t}' not found. Available: {list(raw_data.keys())}"}
        if c not in raw_data[t].columns:
            return {"error": f"Column '{c}' not found in '{t}'"}
    min_len = min(len(raw_data[table_a]), len(raw_data[table_b]))
    s_a = raw_data[table_a][col_a].reset_index(drop=True).iloc[:min_len]
    s_b = raw_data[table_b][col_b].reset_index(drop=True).iloc[:min_len]
    both = pd.concat([s_a, s_b], axis=1).dropna()
    if len(both) < 5:
        return {"error": "Insufficient aligned rows"}
    c = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
    return {"table_a": table_a, "col_a": col_a,
            "table_b": table_b, "col_b": col_b,
            "correlation": round(c, 4), "n": len(both)}


# ── Sorting / Ranking ─────────────────────────────────────────────────────────

def get_sorted_rows(raw_data: dict[str, pd.DataFrame], table: str,
                    sort_col: str, show_cols: list[str],
                    ascending: bool = True, top_n: int = 10) -> list[dict]:
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    all_cols = list(dict.fromkeys([sort_col] + show_cols))
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        return [{"error": f"Columns not found: {missing}. Available: {list(df.columns)}"}]
    result = df[all_cols].dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending).head(top_n)
    return [
        {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else str(v))
         for k, v in row.items()}
        for row in result.to_dict(orient="records")
    ]


# ── Segment breakdown ─────────────────────────────────────────────────────────

def get_segment_breakdown(raw_data: dict[str, pd.DataFrame], table: str,
                           group_col: str, value_col: str) -> list[dict]:
    """Sum, count, mean and % share per segment — LLM decides what is disproportionate."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    if group_col not in df.columns or value_col not in df.columns:
        return [{"error": f"Column not found. Available: {list(df.columns)}"}]
    grp = df.groupby(group_col)[value_col].agg(["sum", "count", "mean"]).reset_index()
    grp.columns = [group_col, "total_value", "count", "mean_value"]
    total_val = float(grp["total_value"].sum())
    total_cnt = float(grp["count"].sum())
    rows = []
    for _, r in grp.iterrows():
        val_pct = round(float(r["total_value"]) / total_val * 100, 2) if total_val else 0
        cnt_pct = round(float(r["count"]) / total_cnt * 100, 2) if total_cnt else 0
        rows.append({
            "segment": str(r[group_col]),
            "count": int(r["count"]),
            "count_pct": cnt_pct,
            "total_value": round(float(r["total_value"]), 4),
            "value_pct": val_pct,
            "mean_value": round(float(r["mean_value"]), 4),
        })
    rows.sort(key=lambda x: x["value_pct"], reverse=True)
    return rows[:15]


# ── Time / Trend ──────────────────────────────────────────────────────────────

def get_time_trend(raw_data: dict[str, pd.DataFrame], table: str,
                   date_col: str, value_col: str, window: int = 30) -> dict:
    """Recent window mean vs prior window mean — LLM judges significance."""
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": f"Column not found. Available: {list(df.columns)}"}
    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.dropna(subset=[date_col]).sort_values(date_col)
        if len(df2) < window * 2:
            return {"error": "Not enough rows for trend comparison"}
        recent = float(df2[value_col].tail(window).mean())
        earlier = float(df2[value_col].iloc[-window * 2: -window].mean())
        change_pct = (
            round((recent - earlier) / abs(earlier) * 100, 2)
            if abs(earlier) > 1e-9 else None
        )
        return {
            "recent_mean": round(recent, 4),
            "earlier_mean": round(earlier, 4),
            "change_pct": change_pct,
        }
    except Exception as e:
        return {"error": str(e)}


def get_entity_activity_change(raw_data: dict[str, pd.DataFrame], table: str,
                                id_col: str, date_col: str,
                                activity_col: str, top_n: int = 10) -> list[dict]:
    """Recent 60-day activity vs prior period per entity — LLM spots declining ones."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    if not all(c in df.columns for c in [id_col, date_col, activity_col]):
        return [{"error": f"Column not found. Available: {list(df.columns)}"}]
    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.dropna(subset=[date_col])
        cutoff = df2[date_col].max() - pd.Timedelta(days=60)
        recent = df2[df2[date_col] >= cutoff].groupby(id_col)[activity_col].sum()
        older  = df2[df2[date_col] <  cutoff].groupby(id_col)[activity_col].sum()
        both = pd.DataFrame({"recent": recent, "older": older}).dropna()
        both = both[both["older"] > 0]
        both["change_pct"] = (both["recent"] - both["older"]) / both["older"].abs() * 100
        result = both.sort_values("change_pct").head(top_n)
        return [
            {"entity": str(idx),
             "recent_activity": round(float(r["recent"]), 4),
             "prior_activity": round(float(r["older"]), 4),
             "change_pct": round(float(r["change_pct"]), 2)}
            for idx, r in result.iterrows()
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ── Seasonality / Cycles ──────────────────────────────────────────────────────

def get_autocorrelation(raw_data: dict[str, pd.DataFrame], table: str,
                         date_col: str, value_col: str,
                         max_lag: int = 52) -> list[dict]:
    """Autocorrelation at each lag — LLM identifies significant cyclical patterns."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    if date_col not in df.columns or value_col not in df.columns:
        return [{"error": f"Column not found. Available: {list(df.columns)}"}]
    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        s = df2.sort_values(date_col)[value_col].dropna().reset_index(drop=True)
        result = []
        for lag in range(1, min(max_lag + 1, len(s) // 2)):
            c = s.corr(s.shift(lag))
            if not np.isnan(c):
                result.append({"lag": lag, "autocorrelation": round(float(c), 4)})
        result.sort(key=lambda x: abs(x["autocorrelation"]), reverse=True)
        return result[:10]
    except Exception as e:
        return [{"error": str(e)}]


def get_fft_dominant_period(raw_data: dict[str, pd.DataFrame], table: str,
                             date_col: str, value_col: str) -> dict:
    """Dominant cycle period via FFT — LLM interprets whether the cycle is non-calendar."""
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": f"Column not found. Available: {list(df.columns)}"}
    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        s = df2.sort_values(date_col)[value_col].dropna().values
        if len(s) < 10:
            return {"error": "Insufficient data"}
        fft  = np.fft.fft(s - s.mean())
        freqs = np.fft.fftfreq(len(s))
        mags  = np.abs(fft)
        pos   = np.where(freqs > 0)[0]
        if len(pos) == 0:
            return {"error": "No positive frequencies found"}
        top    = pos[np.argmax(mags[pos])]
        period = round(1.0 / freqs[top]) if freqs[top] != 0 else 0
        strength = round(float(mags[top] / (mags.sum() + 1e-9)), 4)
        return {"dominant_period_units": int(period), "cycle_strength": strength}
    except Exception as e:
        return {"error": str(e)}


# ── Predictive lag ────────────────────────────────────────────────────────────

def get_lag_correlation(raw_data: dict[str, pd.DataFrame], table: str,
                         col_a: str, col_b: str,
                         max_lag: int = 30) -> list[dict]:
    """Correlation at each lag offset — LLM identifies the strongest predictive lag."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    if col_a not in df.columns or col_b not in df.columns:
        return [{"error": f"Column not found. Available: {list(df.columns)}"}]
    a = df[col_a].reset_index(drop=True)
    b = df[col_b].reset_index(drop=True)
    result = []
    for lag in range(-max_lag, max_lag + 1):
        try:
            a_s = a.shift(lag) if lag >= 0 else a
            b_s = b if lag >= 0 else b.shift(-lag)
            both = pd.concat([a_s, b_s], axis=1).dropna()
            if len(both) >= 5:
                c = both.iloc[:, 0].corr(both.iloc[:, 1])
                if not np.isnan(c):
                    result.append({"lag_days": lag, "correlation": round(float(c), 4)})
        except Exception:
            pass
    result.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return result[:10]


# ── Change points ─────────────────────────────────────────────────────────────

def get_change_points(raw_data: dict[str, pd.DataFrame], table: str,
                       col: str, window: int = 10) -> list[dict]:
    """Positions where a column's rolling mean shifts significantly."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    if col not in df.columns:
        return [{"error": f"Column '{col}' not found. Available: {list(df.columns)}"}]
    s = df[col].dropna().reset_index(drop=True)
    if len(s) < window * 3:
        return [{"error": "Not enough rows for change-point detection"}]
    changes = []
    for i in range(window, len(s) - window):
        before = float(s.iloc[i - window:i].mean())
        after  = float(s.iloc[i:i + window].mean())
        if abs(before) > 1e-9:
            chg = (after - before) / abs(before) * 100
            if abs(chg) > 20:
                changes.append({
                    "position": i,
                    "change_pct": round(chg, 2),
                    "before_mean": round(before, 4),
                    "after_mean": round(after, 4),
                })
    changes.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return changes[:5]


# ── Simpsons Paradox ──────────────────────────────────────────────────────────

def get_group_vs_aggregate_correlation(raw_data: dict[str, pd.DataFrame], table: str,
                                        group_col: str, x_col: str, y_col: str) -> dict:
    """Overall correlation vs per-group correlation — LLM decides if a paradox exists."""
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if not all(c in df.columns for c in [group_col, x_col, y_col]):
        return {"error": f"Column not found. Available: {list(df.columns)}"}
    try:
        overall = float(df[[x_col, y_col]].corr().iloc[0, 1])
        by_group = {}
        for g, gdf in df.groupby(group_col):
            if len(gdf) >= 5:
                c = gdf[[x_col, y_col]].corr().iloc[0, 1]
                if not np.isnan(c):
                    by_group[str(g)] = round(float(c), 4)
        return {"overall_correlation": round(overall, 4), "by_group": by_group}
    except Exception as e:
        return {"error": str(e)}


# ── Narrative shift ───────────────────────────────────────────────────────────

def get_driver_shift(raw_data: dict[str, pd.DataFrame], table: str,
                      target_col: str, driver_cols: list[str],
                      window: int = 30) -> list[dict]:
    """Early vs recent correlation between target and each driver — LLM spots the shift."""
    if table not in raw_data:
        return [{"error": f"Table '{table}' not found"}]
    df = raw_data[table]
    if target_col not in df.columns:
        return [{"error": f"Column '{target_col}' not found. Available: {list(df.columns)}"}]
    result = []
    for col in driver_cols:
        if col == target_col or col not in df.columns:
            continue
        try:
            early = float(df.head(window)[[target_col, col]].corr().iloc[0, 1])
            late  = float(df.tail(window)[[target_col, col]].corr().iloc[0, 1])
            if not (np.isnan(early) or np.isnan(late)):
                result.append({
                    "driver": col,
                    "early_correlation": round(early, 4),
                    "recent_correlation": round(late, 4),
                    "shift": round(abs(late - early), 4),
                })
        except Exception:
            pass
    result.sort(key=lambda x: x["shift"], reverse=True)
    return result[:8]


# ── Outlier / Anomaly ─────────────────────────────────────────────────────────

def get_outlier_stats(raw_data: dict[str, pd.DataFrame], table: str,
                       col: str, iqr_multiplier: float = 1.5) -> dict:
    """Outlier count, %, bounds and sample values — LLM assesses severity."""
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if col not in df.columns:
        return {"error": f"Column '{col}' not found. Available: {list(df.columns)}"}
    s = df[col].dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr   = q3 - q1
    mask  = (df[col] < q1 - iqr_multiplier * iqr) | (df[col] > q3 + iqr_multiplier * iqr)
    return {
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(float(mask.mean() * 100), 2),
        "lower_bound": round(q1 - iqr_multiplier * iqr, 4),
        "upper_bound": round(q3 + iqr_multiplier * iqr, 4),
        "sample_outlier_values": df.loc[mask, col].head(5).round(4).tolist(),
    }


def get_outlier_time_clustering(raw_data: dict[str, pd.DataFrame], table: str,
                                 col: str, date_col: str,
                                 iqr_multiplier: float = 1.5) -> dict:
    """Whether outliers cluster in time — LLM judges if it signals a real event."""
    if table not in raw_data:
        return {"error": f"Table '{table}' not found"}
    df = raw_data[table]
    if col not in df.columns or date_col not in df.columns:
        return {"error": f"Column not found. Available: {list(df.columns)}"}
    s = df[col].dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr   = q3 - q1
    mask  = (df[col] < q1 - iqr_multiplier * iqr) | (df[col] > q3 + iqr_multiplier * iqr)
    n_out = int(mask.sum())
    if n_out == 0:
        return {"outlier_count": 0, "temporal_clustering": False}
    try:
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        dates = df2.loc[mask, date_col].dropna().sort_values()
        if len(dates) > 1:
            gaps = dates.diff().dt.days.dropna()
            close_pct = float((gaps < 7).mean())
            return {
                "outlier_count": n_out,
                "outlier_pct": round(float(mask.mean() * 100), 2),
                "temporal_clustering": close_pct > 0.5,
                "pct_within_7_days_of_each_other": round(close_pct * 100, 2),
            }
        return {"outlier_count": n_out, "temporal_clustering": False}
    except Exception as e:
        return {"outlier_count": n_out, "error": str(e)}
