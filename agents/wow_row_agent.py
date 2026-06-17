"""WowRowAgent — dynamic per-row WoW pattern detection.

Architecture mirrors UniversalRowAgent's dynamic tool registry:
  • Signals are registered with @register_wow_signal(name, description)
  • Each signal class has 3 static methods:
      precompute(df, ti)           → dict    (runs once per table, zero LLM)
      detect(row, idx, baseline)  → list[str] (runs per row)
      summarize(baseline)          → list[str] (LLM context lines)
  • Adding a signal : write a class + @register_wow_signal decorator
  • Removing a signal: delete the class — agent auto-adjusts
  • No hardcoding of signal names anywhere in the agent

12 registered signals:
  paradox     Paradox Detector         — counter-intuitive metric movements
  champion    Hidden Champion          — low input, disproportionate output
  churn       Silent Churn Predictor   — pre-churn engagement decline signals
  seasonal    Invisible Seasonality    — non-calendar cyclical anomalies
  leading     Leading Indicator        — predictive metric at extreme level
  butterfly   Butterfly Effect         — small-variance column at extreme
  simpsons    Simpson's Paradox        — subgroup reverses overall trend
  redundancy  Redundancy and Noise     — recording gaps / zero-value KPIs
  crossdomain Cross-Domain Correlation — entity anomaly across tables
  cluster     Anomaly Cluster          — multi-dimensional outlier
  narrative   Narrative Shift          — dominant driver changed over time
  dark        Dark Data                — value in normally-empty column
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from core.groq_rate_limiter import groq_invoke
from config import Config, get_config
from core.state import AgentResult, ColumnRoute

log = logging.getLogger(__name__)



# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC SIGNAL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_WOW_SIGNAL_REGISTRY: dict[str, type] = {}
_WOW_DESCRIPTIONS:    dict[str, str]  = {}


def register_wow_signal(name: str, description: str = ""):
    """Decorator to register a WoW signal class.

    The decorated class must implement three static methods:
      precompute(df, ti)          → dict   (table-level baselines, zero LLM)
      detect(row, idx, baseline)  → list[str] (per-row signal strings)
      summarize(baseline)          → list[str] (lines shown in LLM context)

    Usage::

        @register_wow_signal("paradox", "Counter-intuitive metric movements")
        class _ParadoxSignal:
            @staticmethod
            def precompute(df, ti): ...
            @staticmethod
            def detect(row, idx, baseline): ...
            @staticmethod
            def summarize(baseline): ...
    """
    def decorator(cls: type) -> type:
        _WOW_SIGNAL_REGISTRY[name] = cls
        _WOW_DESCRIPTIONS[name]    = description
        return cls
    return decorator


def get_wow_signals() -> dict[str, type]:
    """Return all registered WoW signal classes."""
    return dict(_WOW_SIGNAL_REGISTRY)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED TYPE-INFO CONTEXT  (computed once per table, passed to every signal)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TypeInfo:
    """Pre-built type context shared across all signal precompute calls."""
    df:           pd.DataFrame
    table_name:   str
    all_raw_data: dict
    num_cols:     list[str]
    cat_cols:     list[str]
    dt_cols:      list[str]
    num_df:       pd.DataFrame
    corr_matrix:  pd.DataFrame


def _build_type_info(
    df: pd.DataFrame,
    column_routes: list[ColumnRoute],
    all_raw_data: dict | None = None,
    table_name:   str  | None = None,
) -> _TypeInfo:
    type_map = {r["column"]: r["detected_type"] for r in column_routes}
    num_cols = [
        c for c in df.columns
        if type_map.get(c) == "numerical" and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [c for c in df.columns if type_map.get(c) == "categorical"]
    dt_cols  = [c for c in df.columns if type_map.get(c) == "datetime"]
    num_df   = df[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame()
    if len(num_cols) >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_matrix = num_df.corr()
    else:
        corr_matrix = pd.DataFrame()
    return _TypeInfo(
        df=df, table_name=table_name or "",
        all_raw_data=all_raw_data or {},
        num_cols=num_cols, cat_cols=cat_cols, dt_cols=dt_cols,
        num_df=num_df, corr_matrix=corr_matrix,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _iqr_bounds(s: pd.Series) -> tuple[float | None, float | None, float, float]:
    s = s.dropna()
    if len(s) < 4:
        v = float(s.median()) if len(s) > 0 else 0.0
        return None, None, v, v
    q1, q3   = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr      = q3 - q1
    med, mean = float(s.median()), float(s.mean())
    if iqr > 0:
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr, med, mean
    std = float(s.std())
    if std > 0:
        return mean - 3 * std, mean + 3 * std, med, mean
    return None, None, med, mean


# ═══════════════════════════════════════════════════════════════════════════════
# 12 REGISTERED WOW SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

@register_wow_signal(
    "paradox",
    "Paradox Detector — counter-intuitive metric movements (cheapest product = highest margin)",
)
class _ParadoxSignal:
    _MIN_CORR = -0.5

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        pairs: list[dict] = []
        seen: set = set()
        for ca in ti.num_cols:
            for cb in ti.num_cols:
                if ca == cb or (cb, ca) in seen:
                    continue
                seen.add((ca, cb))
                try:
                    r = float(ti.corr_matrix.loc[ca, cb])
                except Exception:
                    continue
                if pd.isna(r) or r >= _ParadoxSignal._MIN_CORR:
                    continue
                sa, sb = ti.num_df[ca].dropna(), ti.num_df[cb].dropna()
                if len(sa) < 4 or len(sb) < 4:
                    continue
                pairs.append({
                    "col_a": ca, "col_b": cb, "corr": round(r, 3),
                    "a_med": round(float(sa.median()), 4),
                    "b_med": round(float(sb.median()), 4),
                })
        pairs.sort(key=lambda x: x["corr"])
        return {"pairs": pairs[:3]}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for pp in baseline.get("pairs", []):
            ca, cb = pp["col_a"], pp["col_b"]
            if ca not in row.index or cb not in row.index:
                continue
            if pd.isna(row[ca]) or pd.isna(row[cb]):
                continue
            try:
                va, vb = float(row[ca]), float(row[cb])
            except (ValueError, TypeError):
                continue
            both_high = va > pp["a_med"] and vb > pp["b_med"]
            both_low  = va < pp["a_med"] and vb < pp["b_med"]
            if both_high or both_low:
                direction = "both_high" if both_high else "both_low"
                signals.append(
                    f"paradox:{ca}={round(va,2)},{cb}={round(vb,2)}"
                    f"({direction},corr={pp['corr']})"
                )
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        return [
            f"  [PARADOX]  {pp['col_a']} vs {pp['col_b']}: corr={pp['corr']} — "
            f"normally when {pp['col_a']} rises, {pp['col_b']} falls"
            for pp in baseline.get("pairs", [])
        ]


@register_wow_signal(
    "champion",
    "Hidden Champion — 5% cohort drives 38% revenue; low input, disproportionate output",
)
class _ChampionSignal:
    _COST_KW    = ("cost", "spend", "expense", "price", "fee", "budget")
    _OUTCOME_KW = ("revenue", "profit", "margin", "return", "sales", "income", "gain")

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        input_cols  = [c for c in ti.num_cols if any(k in c.lower() for k in _ChampionSignal._COST_KW)]
        output_cols = [c for c in ti.num_cols if any(k in c.lower() for k in _ChampionSignal._OUTCOME_KW)]
        pairs: list[dict] = []
        for ic in input_cols[:2]:
            for oc in output_cols[:2]:
                si = ti.num_df[ic].dropna()
                so = ti.num_df[oc].dropna()
                if len(si) >= 4 and len(so) >= 4:
                    pairs.append({
                        "input_col":  ic, "output_col": oc,
                        "input_p25":  round(float(si.quantile(0.25)), 4),
                        "output_p75": round(float(so.quantile(0.75)), 4),
                        "input_med":  round(float(si.median()), 4),
                        "output_med": round(float(so.median()), 4),
                    })
        return {"pairs": pairs}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for cp in baseline.get("pairs", []):
            ic, oc = cp["input_col"], cp["output_col"]
            if ic not in row.index or oc not in row.index:
                continue
            if pd.isna(row[ic]) or pd.isna(row[oc]):
                continue
            try:
                vi, vo = float(row[ic]), float(row[oc])
            except (ValueError, TypeError):
                continue
            if vi <= cp["input_p25"] and vo >= cp["output_p75"]:
                signals.append(
                    f"champion:{ic}={round(vi,2)}<p25={round(cp['input_p25'],2)},"
                    f"{oc}={round(vo,2)}>p75={round(cp['output_p75'],2)}"
                )
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        return [
            f"  [CHAMPION] {cp['input_col']}(input,p25={cp['input_p25']}) → "
            f"{cp['output_col']}(output,p75={cp['output_p75']})"
            for cp in baseline.get("pairs", [])
        ]


@register_wow_signal(
    "churn",
    "Silent Churn Predictor — 60-90 day early warning; recent record but critically low engagement",
)
class _ChurnSignal:
    _ENGAGEMENT_KW = (
        "count", "visit", "order", "session", "login", "active",
        "purchase", "click", "interact", "frequency", "engage",
    )
    _CHURN_WINDOW_DAYS = 90

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        eng_cols: dict = {}
        for col in ti.num_cols:
            if any(k in col.lower() for k in _ChurnSignal._ENGAGEMENT_KW):
                s = ti.num_df[col].dropna()
                if len(s) >= 4:
                    eng_cols[col] = {
                        "p25":  round(float(s.quantile(0.25)), 4),
                        "mean": round(float(s.mean()), 4),
                    }
        latest_date = None
        for col in ti.dt_cols[:3]:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(parsed) > 0:
                    latest_date = parsed.max()
                    break
            except Exception:
                pass
        return {"eng_cols": eng_cols, "latest_date": latest_date, "dt_cols": ti.dt_cols}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        eng_cols    = baseline.get("eng_cols", {})
        latest_date = baseline.get("latest_date")
        if not eng_cols or latest_date is None:
            return []
        low_engagement: list[str] = []
        for col, t in eng_cols.items():
            if col not in row.index or pd.isna(row[col]):
                continue
            try:
                val = float(row[col])
            except (ValueError, TypeError):
                continue
            if val < t["p25"]:
                low_engagement.append(f"{col}={round(val,2)}<p25={t['p25']}")
        if not low_engagement:
            return []
        for dt_col in baseline.get("dt_cols", []):
            if dt_col not in row.index or pd.isna(row[dt_col]):
                continue
            try:
                ts = pd.to_datetime(row[dt_col], errors="coerce")
                if pd.isna(ts):
                    continue
                days_from_latest = int((latest_date - ts).days)
                if 0 <= days_from_latest <= _ChurnSignal._CHURN_WINDOW_DAYS:
                    return [
                        f"churn:{','.join(low_engagement[:3])}"
                        f"(recent={ts.date()},days_from_latest={days_from_latest})"
                    ]
            except Exception:
                pass
        return []

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        lines: list[str] = []
        for col, t in baseline.get("eng_cols", {}).items():
            lines.append(f"  [CHURN]    engagement {col}: low_threshold=p25={t['p25']}")
        return lines


@register_wow_signal(
    "seasonal",
    "Invisible Seasonality — non-calendar cycles; sales dip every 11th week, unknown cause",
)
class _SeasonalSignal:
    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        baselines: dict = {}
        for col in ti.dt_cols[:3]:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(parsed) < 14:
                    continue
                weekly = parsed.dt.isocalendar().week.value_counts()
                wk_mean, wk_std = float(weekly.mean()), float(weekly.std())
                if wk_std == 0:
                    continue
                low_weeks  = set(weekly[weekly < wk_mean - wk_std].index.tolist())
                high_weeks = set(weekly[weekly > wk_mean + wk_std].index.tolist())
                dow_counts = parsed.dt.dayofweek.value_counts()
                dow_mean, dow_std = float(dow_counts.mean()), float(dow_counts.std())
                low_dow  = set(dow_counts[dow_counts < dow_mean - dow_std].index.tolist()) if dow_std > 0 else set()
                high_dow = set(dow_counts[dow_counts > dow_mean + dow_std].index.tolist()) if dow_std > 0 else set()
                if low_weeks or high_weeks or low_dow or high_dow:
                    baselines[col] = {
                        "low_weeks": low_weeks, "high_weeks": high_weeks,
                        "low_dow":   low_dow,   "high_dow":   high_dow,
                        "wk_mean":   round(wk_mean, 1),
                    }
            except Exception:
                pass
        return baselines

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        _DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        signals: list[str] = []
        for dt_col, s in baseline.items():
            if dt_col not in row.index or pd.isna(row[dt_col]):
                continue
            try:
                ts = pd.to_datetime(row[dt_col], errors="coerce")
                if pd.isna(ts):
                    continue
                week = int(ts.isocalendar().week)
                dow  = int(ts.dayofweek)
                if week in s.get("low_weeks", set()):
                    signals.append(f"seasonal:{dt_col}={ts.date()}(low_activity_week={week})")
                elif week in s.get("high_weeks", set()):
                    signals.append(f"seasonal:{dt_col}={ts.date()}(high_activity_week={week})")
                elif dow in s.get("low_dow", set()):
                    signals.append(f"seasonal:{dt_col}={ts.date()}(low_activity_dow={_DOW[dow]})")
                elif dow in s.get("high_dow", set()):
                    signals.append(f"seasonal:{dt_col}={ts.date()}(high_activity_dow={_DOW[dow]})")
            except Exception:
                pass
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        lines: list[str] = []
        for col, s in baseline.items():
            lines.append(
                f"  [SEASONAL] {col}: low_weeks={sorted(s['low_weeks'])[:5]}, "
                f"high_weeks={sorted(s['high_weeks'])[:5]}, avg={s['wk_mean']}/wk"
            )
        return lines


@register_wow_signal(
    "leading",
    "Leading Indicator — Tuesday sessions predict Friday revenue; N-day time-lag correlation",
)
class _LeadingSignal:
    _MIN_CORR = 0.5
    _LAGS     = (7, 14)

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        if not ti.dt_cols or len(ti.num_cols) < 2:
            return {}
        try:
            dt_col = ti.dt_cols[0]
            parsed = pd.to_datetime(df[dt_col], errors="coerce")
            if parsed.isna().mean() > 0.5:
                return {}
            temp = ti.num_df.copy()
            temp["_dt"] = parsed
            temp = temp.dropna(subset=["_dt"]).sort_values("_dt").set_index("_dt")
            daily = temp[ti.num_cols].resample("D").mean().interpolate(limit=3)
            if len(daily) < 14:
                return {}
            pairs: list[dict] = []
            seen: set = set()
            for lag in _LeadingSignal._LAGS:
                for ca in ti.num_cols[:8]:
                    for cb in ti.num_cols[:8]:
                        if ca == cb or (ca, cb, lag) in seen:
                            continue
                        seen.add((ca, cb, lag))
                        try:
                            shifted = daily[ca].shift(lag)
                            valid   = daily[[cb]].join(shifted.rename("_lag")).dropna()
                            if len(valid) < 7:
                                continue
                            with np.errstate(invalid="ignore", divide="ignore"):
                                r = float(valid[cb].corr(valid["_lag"]))
                            if pd.isna(r) or abs(r) < _LeadingSignal._MIN_CORR:
                                continue
                            s_lead = ti.num_df[ca].dropna()
                            lo, hi, _, _ = _iqr_bounds(s_lead)
                            if lo is None:
                                continue
                            pairs.append({
                                "leading_col": ca, "outcome_col": cb,
                                "lag_days": lag, "corr": round(r, 3),
                                "lead_lo": round(lo, 4), "lead_hi": round(hi, 4),
                            })
                        except Exception:
                            pass
            pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
            return {"pairs": pairs[:4]}
        except Exception:
            return {}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for lp in baseline.get("pairs", []):
            col = lp["leading_col"]
            if col not in row.index or pd.isna(row[col]):
                continue
            try:
                val = float(row[col])
            except (ValueError, TypeError):
                continue
            if val < lp["lead_lo"] or val > lp["lead_hi"]:
                direction = "up" if val > lp["lead_hi"] else "down"
                signals.append(
                    f"leading:{col}={round(val,2)}"
                    f"(predicts_{lp['outcome_col']}_{lp['lag_days']}days,"
                    f"corr={lp['corr']},signal={direction})"
                )
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        return [
            f"  [LEADING]  {lp['leading_col']} predicts {lp['outcome_col']} "
            f"in {lp['lag_days']} days (corr={lp['corr']})"
            for lp in baseline.get("pairs", [])
        ]


@register_wow_signal(
    "butterfly",
    "Butterfly Effect — 2% packing drop causes 19% complaints; small metric, large downstream impact",
)
class _ButterflySignal:
    _MAX_CV    = 0.25   # tight column: coefficient of variation < 25%
    _MIN_CORR  = 0.5
    _IMPACT_KW = (
        "revenue", "sales", "complaint", "return", "churn", "loss",
        "profit", "incident", "error", "defect", "cost", "damage",
    )

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        impact_cols = [c for c in ti.num_cols if any(k in c.lower() for k in _ButterflySignal._IMPACT_KW)]
        pairs: list[dict] = []
        for col in ti.num_cols:
            s    = ti.num_df[col].dropna()
            if len(s) < 4:
                continue
            mean = float(s.mean())
            std  = float(s.std())
            if mean == 0 or std == 0:
                continue
            cv = std / abs(mean)
            if cv >= _ButterflySignal._MAX_CV:
                continue
            lo, hi, _, _ = _iqr_bounds(s)
            if lo is None:
                continue
            for ic in impact_cols:
                if ic == col:
                    continue
                try:
                    r = float(ti.corr_matrix.loc[col, ic])
                except Exception:
                    continue
                if pd.isna(r) or abs(r) < _ButterflySignal._MIN_CORR:
                    continue
                pairs.append({
                    "leverage_col": col, "impact_col": ic,
                    "corr": round(r, 3), "cv": round(cv, 3),
                    "normal_lo": round(lo, 4), "normal_hi": round(hi, 4),
                })
        pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
        return {"pairs": pairs[:3]}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for bp in baseline.get("pairs", []):
            col = bp["leverage_col"]
            if col not in row.index or pd.isna(row[col]):
                continue
            try:
                val = float(row[col])
            except (ValueError, TypeError):
                continue
            if val < bp["normal_lo"] or val > bp["normal_hi"]:
                signals.append(
                    f"butterfly:{col}={round(val,2)}"
                    f"(outside_tight_range=[{round(bp['normal_lo'],2)},{round(bp['normal_hi'],2)}],"
                    f"cv={bp['cv']},impacts_{bp['impact_col']},corr={bp['corr']})"
                )
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        return [
            f"  [BUTTERFLY] {bp['leverage_col']}(cv={bp['cv']}) → "
            f"{bp['impact_col']}(corr={bp['corr']}) — tight column, high leverage"
            for bp in baseline.get("pairs", [])
        ]


@register_wow_signal(
    "simpsons",
    "Simpson's Paradox — grouped vs subgroup trend reversal; misleading aggregates mask true signal",
)
class _SimpsonsSignal:
    _MIN_OVERALL_CORR  = 0.3
    _REVERSAL_DIFF     = 0.3

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        reversals: list[dict] = []
        for cat_col in ti.cat_cols[:4]:
            cats   = df[cat_col].astype(str).str.strip()
            groups = cats.groupby(cats).groups
            if len(groups) < 2 or len(groups) > 30:
                continue
            for ca in ti.num_cols[:5]:
                for cb in ti.num_cols[:5]:
                    if ca == cb:
                        continue
                    try:
                        overall_r = float(ti.corr_matrix.loc[ca, cb])
                    except Exception:
                        continue
                    if pd.isna(overall_r) or abs(overall_r) < _SimpsonsSignal._MIN_OVERALL_CORR:
                        continue
                    reversed_cats: list[str] = []
                    for cat_val, idx_group in groups.items():
                        sub = ti.num_df.iloc[idx_group][[ca, cb]].dropna()
                        if len(sub) < 5:
                            continue
                        with np.errstate(invalid="ignore", divide="ignore"):
                            sub_r = float(sub[ca].corr(sub[cb]))
                        if pd.isna(sub_r):
                            continue
                        if (overall_r > _SimpsonsSignal._REVERSAL_DIFF and sub_r < -_SimpsonsSignal._REVERSAL_DIFF) or \
                           (overall_r < -_SimpsonsSignal._REVERSAL_DIFF and sub_r > _SimpsonsSignal._REVERSAL_DIFF):
                            reversed_cats.append(cat_val)
                    if reversed_cats:
                        reversals.append({
                            "cat_col": cat_col, "col_a": ca, "col_b": cb,
                            "overall_corr":  round(overall_r, 3),
                            "reversed_cats": reversed_cats[:5],
                        })
        return {"reversals": reversals[:3]}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for sp in baseline.get("reversals", []):
            cat_col = sp["cat_col"]
            if cat_col not in row.index or pd.isna(row[cat_col]):
                continue
            if str(row[cat_col]).strip() in sp["reversed_cats"]:
                signals.append(
                    f"simpsons:{cat_col}={str(row[cat_col])[:20]}"
                    f"(subgroup_reverses_{sp['col_a']}_vs_{sp['col_b']},"
                    f"overall_corr={sp['overall_corr']})"
                )
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        return [
            f"  [SIMPSONS] {sp['col_a']} vs {sp['col_b']} by {sp['cat_col']}: "
            f"overall_corr={sp['overall_corr']}, reversed in {sp['reversed_cats'][:3]}"
            for sp in baseline.get("reversals", [])
        ]


@register_wow_signal(
    "redundancy",
    "Redundancy and Noise — zero-value KPI identification; stop measuring what does not matter",
)
class _RedundancySignal:
    _NORMAL_ZERO_RATE   = 0.20   # column is "normally populated" if < 20% zeros
    _TRIGGER_FRACTION   = 0.33   # flag row if >= 1/3 of normally-populated cols are zero

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        normal_cols: list[str] = []
        for col in ti.num_cols:
            s = ti.num_df[col].dropna()
            if len(s) < 4:
                continue
            if float((s == 0).mean()) < _RedundancySignal._NORMAL_ZERO_RATE:
                normal_cols.append(col)
        threshold = max(2, int(len(normal_cols) * _RedundancySignal._TRIGGER_FRACTION))
        return {"normal_cols": normal_cols, "threshold": threshold}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        normal_cols = baseline.get("normal_cols", [])
        threshold   = baseline.get("threshold", 3)
        zero_cols: list[str] = [
            col for col in normal_cols
            if col in row.index and (pd.isna(row[col]) or row[col] == 0)
        ]
        if len(zero_cols) >= threshold:
            return [
                f"redundancy:{','.join(zero_cols[:5])}"
                f"({len(zero_cols)}/{len(normal_cols)}_normally_populated_cols_are_zero)"
            ]
        return []

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        if not baseline.get("normal_cols"):
            return []
        return [
            f"  [REDUNDANCY] {len(baseline['normal_cols'])} normally-populated cols; "
            f"flag if >={baseline['threshold']} are zero"
        ]


@register_wow_signal(
    "crossdomain",
    "Cross-Domain Correlation — employee satisfaction predicts NPS with 45-day lag; unrelated datasets connect",
)
class _CrossDomainSignal:
    _ID_KW = ("id", "code", "number", "ref", "key", "no", "num")

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        if not ti.all_raw_data or len(ti.all_raw_data) < 2:
            return {}
        candidate_cols = [
            c for c in df.columns
            if any(k in c.lower() for k in _CrossDomainSignal._ID_KW)
            and df[c].nunique() > 5
        ]
        if not candidate_cols:
            return {}
        links: list[dict] = []
        for other_tbl, other_df in ti.all_raw_data.items():
            if other_tbl == ti.table_name or other_df is None or other_df.empty:
                continue
            for col in candidate_cols[:3]:
                if col not in other_df.columns:
                    continue
                this_vals  = set(df[col].dropna().astype(str).unique())
                other_vals = set(other_df[col].dropna().astype(str).unique())
                common = this_vals & other_vals
                if len(common) < 3:
                    continue
                links.append({
                    "join_col":      col,
                    "other_table":   other_tbl,
                    "common_count":  len(common),
                    "common_sample": list(common)[:10],
                })
        return {"links": links} if links else {}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for link in baseline.get("links", [])[:2]:
            join_col = link["join_col"]
            if join_col not in row.index or pd.isna(row[join_col]):
                continue
            entity_val = str(row[join_col]).strip()
            if entity_val in link.get("common_sample", []):
                signals.append(
                    f"crossdomain:{join_col}={entity_val[:20]}"
                    f"(entity_in_{link['other_table']},"
                    f"{link['common_count']}_shared_entities)"
                )
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        return [
            f"  [CROSSDOMAIN] {link['join_col']}: {link['common_count']} shared entities "
            f"with {link['other_table']}"
            for link in baseline.get("links", [])
        ]


@register_wow_signal(
    "cluster",
    "Anomaly Cluster — geographic and temporal clustering; pattern in outlier groups, signal vs noise",
)
class _ClusterSignal:
    _THRESHOLD = 3   # min outlier columns to trigger

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        bounds: dict = {}
        for col in ti.num_cols:
            s = ti.num_df[col].dropna()
            if len(s) < 4 or s.nunique() <= 2:
                continue
            lo, hi, _, mean = _iqr_bounds(s)
            if lo is None:
                continue
            if float(s.min()) >= 0:
                lo = max(lo, 0.0)
            bounds[col] = {
                "lo": round(lo, 4), "hi": round(hi, 4),
                "mean": round(mean, 4), "std": round(float(s.std()), 4),
            }
        return {"bounds": bounds}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        outlier_cols: list[str] = []
        for col, b in baseline.get("bounds", {}).items():
            if col not in row.index or pd.isna(row[col]):
                continue
            try:
                val = float(row[col])
            except (ValueError, TypeError):
                continue
            if val < b["lo"] or val > b["hi"]:
                outlier_cols.append(f"{col}={round(val,2)}")
        if len(outlier_cols) >= _ClusterSignal._THRESHOLD:
            return [
                f"cluster:{','.join(outlier_cols[:5])}"
                f"({len(outlier_cols)}_cols_outlier_simultaneously)"
            ]
        return []

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        n = len(baseline.get("bounds", {}))
        if n == 0:
            return []
        return [f"  [CLUSTER]   monitoring {n} numerical cols; flag if outlier in >={_ClusterSignal._THRESHOLD}"]


@register_wow_signal(
    "narrative",
    "Narrative Shift — price replaced by delivery speed as top predictor; driver change detection over time",
)
class _NarrativeSignal:
    _TARGET_KW = ("revenue", "sales", "profit", "amount", "total", "value")

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        if not ti.dt_cols or len(ti.num_cols) < 3:
            return {}
        try:
            dt_col = ti.dt_cols[0]
            parsed = pd.to_datetime(df[dt_col], errors="coerce")
            if parsed.isna().mean() > 0.5:
                return {}
            sorted_idx = parsed.dropna().sort_values().index
            n = len(sorted_idx)
            if n < 20:
                return {}
            first_idx  = sorted_idx[:n // 2]
            second_idx = sorted_idx[n // 2:]
            shift_date = parsed[second_idx[0]]
            target_col = next(
                (c for c in ti.num_cols if any(k in c.lower() for k in _NarrativeSignal._TARGET_KW)),
                ti.num_cols[0],
            )

            def _top_driver(idx_set):
                sub = ti.num_df.iloc[idx_set]
                best_col, best_r = None, 0.0
                for col in ti.num_cols:
                    if col == target_col:
                        continue
                    valid = sub[[target_col, col]].dropna()
                    if len(valid) < 5:
                        continue
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = abs(float(valid[target_col].corr(valid[col])))
                    if not pd.isna(r) and r > best_r:
                        best_r, best_col = r, col
                return best_col, round(best_r, 3)

            pre_driver,  pre_r  = _top_driver(first_idx)
            post_driver, post_r = _top_driver(second_idx)
            if pre_driver and post_driver and pre_driver != post_driver:
                return {
                    "shift_date":  shift_date,
                    "target_col":  target_col,
                    "pre_driver":  pre_driver,  "pre_corr":  pre_r,
                    "post_driver": post_driver, "post_corr": post_r,
                    "dt_col":      dt_col,
                }
        except Exception:
            pass
        return {}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        if not baseline:
            return []
        dt_col     = baseline.get("dt_col", "")
        shift_date = baseline.get("shift_date")
        if not dt_col or dt_col not in row.index or shift_date is None or pd.isna(row[dt_col]):
            return []
        try:
            ts = pd.to_datetime(row[dt_col], errors="coerce")
            if not pd.isna(ts) and ts >= shift_date:
                return [
                    f"narrative:row_date={ts.date()}"
                    f"(post_shift={shift_date.date()},"
                    f"new_driver={baseline['post_driver']}(r={baseline['post_corr']}),"
                    f"old_driver={baseline['pre_driver']}(r={baseline['pre_corr']}))"
                ]
        except Exception:
            pass
        return []

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        if not baseline:
            return []
        return [
            f"  [NARRATIVE] shift at {baseline['shift_date'].date()}: "
            f"{baseline['pre_driver']}(r={baseline['pre_corr']}) → "
            f"{baseline['post_driver']}(r={baseline['post_corr']}) "
            f"as driver of {baseline['target_col']}"
        ]


@register_wow_signal(
    "dark",
    "Dark Data — collected but unanalyzed fields; hidden signal in unused data",
)
class _DarkDataSignal:
    _NULL_RATE_THRESHOLD = 0.75

    @staticmethod
    def precompute(df: pd.DataFrame, ti: _TypeInfo) -> dict:
        dark_cols = [
            col for col in df.columns
            if float(df[col].isna().mean()) > _DarkDataSignal._NULL_RATE_THRESHOLD
        ]
        return {"dark_cols": dark_cols}

    @staticmethod
    def detect(row: pd.Series, idx: int, baseline: dict) -> list[str]:
        signals: list[str] = []
        for col in baseline.get("dark_cols", []):
            if col not in row.index:
                continue
            val = row[col]
            if pd.notna(val) and str(val).strip() not in ("", "nan", "None", "NULL"):
                signals.append(f"dark:{col}='{str(val)[:30]}'(col_normally_empty>75%)")
        return signals

    @staticmethod
    def summarize(baseline: dict) -> list[str]:
        cols = baseline.get("dark_cols", [])
        if not cols:
            return []
        return [f"  [DARK]      normally-empty columns (>75% null): {', '.join(cols[:5])}"]


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME HELPERS  (no hardcoding — everything driven by the registry)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    """Build the LLM system prompt dynamically from registered WoW signals."""
    lines = [
        "You are a senior business intelligence analyst specialising in counter-intuitive pattern detection.",
        "",
        "Registered WoW signal types (fields marked ! are flagged):",
    ]
    for name, desc in _WOW_DESCRIPTIONS.items():
        lines.append(f"  [{name.upper():12s}]: {desc}")
    lines += [
        "",
        "For each flagged record write a 2-3 sentence business insight that:",
        "1. States the WOW TYPE from the list above",
        "2. Describes what makes this record surprising with EXACT values",
        "3. Explains the BUSINESS OPPORTUNITY or RISK",
        "4. Recommends a specific ACTION with measurable outcome",
        "",
        "Do NOT say 'statistical anomaly' — explain the real business story.",
    ]
    return "\n".join(lines)


def _parse_row_insights(content: str, row_keys: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    missing: list[str] = []
    for key in row_keys:
        pattern = rf'[*_]*\[INSIGHTS?:\s*{re.escape(key)}\][*_]*\s*(.*?)(?=[*_]*\[INSIGHTS?:|$)'
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        result[key] = m.group(1).strip() if m and m.group(1).strip() else ""
        if not result[key]:
            missing.append(key)
    if missing:
        log.warning("WowRowAgent: no insight parsed for %s", missing)
    return result


def _run_all_precomputes(df: pd.DataFrame, ti: _TypeInfo) -> dict[str, dict]:
    """Run every registered signal's precompute and return {signal_name: baseline}."""
    baselines: dict[str, dict] = {}
    for name, cls in _WOW_SIGNAL_REGISTRY.items():
        try:
            baselines[name] = cls.precompute(df, ti)
        except Exception as exc:
            log.warning("WowRowAgent: precompute failed for signal '%s': %s", name, exc)
            baselines[name] = {}
    registered = {n: len(b) if isinstance(b, dict) else 0 for n, b in baselines.items()}
    log.info("WowRowAgent: signals registered=%s", registered)
    return baselines


def _run_all_signals(row: pd.Series, idx: int, baselines: dict[str, dict]) -> list[str]:
    """Run every registered signal's detect function and return combined signals."""
    signals: list[str] = []
    for name, cls in _WOW_SIGNAL_REGISTRY.items():
        baseline = baselines.get(name, {})
        if not baseline:
            continue
        try:
            signals.extend(cls.detect(row, idx, baseline))
        except Exception as exc:
            log.warning("WowRowAgent: detect failed for signal '%s': %s", name, exc)
    return signals


def _build_baseline_lines(baselines: dict[str, dict]) -> list[str]:
    """Collect LLM context lines from every registered signal's summarize function."""
    lines: list[str] = []
    for name, cls in _WOW_SIGNAL_REGISTRY.items():
        baseline = baselines.get(name, {})
        if not baseline:
            continue
        try:
            lines.extend(cls.summarize(baseline))
        except Exception:
            pass
    return lines


def _row_summary(row: pd.Series, df: pd.DataFrame, baselines: dict) -> str:
    """Format a row for the LLM prompt, flagging outlier numerical columns."""
    cluster_bounds = baselines.get("cluster", {}).get("bounds", {})
    dark_cols      = set(baselines.get("dark", {}).get("dark_cols", []))
    parts: list[str] = []
    for col in df.columns:
        if col not in row.index:
            continue
        raw  = row[col]
        flag = ""
        if pd.isna(raw):
            parts.append(f"{col}=NULL")
            continue
        if col in cluster_bounds:
            try:
                val = float(raw)
                b   = cluster_bounds[col]
                if val < b["lo"] or val > b["hi"]:
                    flag = "!"
            except (ValueError, TypeError):
                pass
        if col in dark_cols:
            flag = "!"
        parts.append(f"{col}={str(raw)[:25]}{flag}")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA-ONLY FLOW: ONE LLM CALL → TEMPLATES, THEN PURE PYTHON PER ROW
# ═══════════════════════════════════════════════════════════════════════════════

def _wow_extract_json(raw: str) -> dict:
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


def _wow_signal_name(sig: str) -> str:
    """Extract the WoW signal name from a signal string (text before the first ':')."""
    return sig.split(":")[0] if ":" in sig else "unknown"


def _wow_signal_detail(sig: str) -> str:
    """Convert a WoW signal string into a readable {detail} filler."""
    if ":" not in sig:
        return sig
    name, rest = sig.split(":", 1)
    return f"{name} pattern — {rest}"


def _wow_fetch_templates(
    llm,
    table: str,
    signals_cache: dict[int, list[str]],
) -> dict[str, str]:
    """ONE LLM call per table — signal type descriptions only, no row data, no baselines.

    Template keys are the WoW signal names actually present in flagged rows.
    Descriptions come from the _WOW_DESCRIPTIONS dynamic registry, populated
    automatically by each @register_wow_signal decorator.
    Baseline stats are excluded to keep the prompt within token limits.
    """
    present_names = sorted({
        _wow_signal_name(s)
        for sigs in signals_cache.values()
        for s in sigs
        if _wow_signal_name(s) != "unknown"
    })
    if not present_names:
        return {}

    # Descriptions from _WOW_DESCRIPTIONS — populated by @register_wow_signal
    desc_lines = [
        f"  {name}: {_WOW_DESCRIPTIONS.get(name, name)}"
        for name in present_names
    ]

    prompt_lines = [
        f"Business analytics table: '{table}'.",
        "",
        "WoW signal types detected in flagged rows (from registered signal classes):",
    ] + desc_lines + [
        "",
        "Write one 2-3 sentence business insight TEMPLATE for each WoW signal type.",
        "Rules:",
        "  • Use {detail} as the ONLY placeholder — Python fills it with the actual "
        "signal values and column context at runtime.",
        "  • Name the WoW pattern type, explain the business risk or opportunity,",
        "    and recommend a measurable action.",
        "  • Keep it generic — no specific column names or values.",
        "",
        f"Return a JSON object with exactly these keys: {present_names}",
    ]

    try:
        msgs = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content="\n".join(prompt_lines)),
        ]
        resp = groq_invoke(llm, msgs, agent_name="wow")
        raw  = resp.content if hasattr(resp, "content") else str(resp)
        parsed = _wow_extract_json(raw)
        return {k: v if isinstance(v, str) else " ".join(v.values()) if isinstance(v, dict) else str(v)
                for k, v in parsed.items()}
    except Exception as exc:
        log.error("WowRowAgent template fetch failed for '%s': %s", table, exc)
        return {}


def _wow_render_insight(signals: list[str], templates: dict[str, str]) -> str:
    """Pure Python — render a row insight from pre-fetched WoW templates."""
    if not signals:
        return ""
    lead_name = _wow_signal_name(signals[0])
    template  = templates.get(lead_name, "")
    detail    = _wow_signal_detail(signals[0])
    insight   = template.replace("{detail}", detail) if template else f"[WOW] {detail}"
    if len(signals) > 1:
        others  = "; ".join(_wow_signal_detail(s) for s in signals[1:])
        insight += f" Additional WoW signals: {others}."
    return insight


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class WowRowAgent:
    """Per-row business WoW pattern detection agent.

    All 12 signal types are registered via @register_wow_signal — the agent
    itself contains no hardcoded signal logic. Add a new signal by writing
    a class with precompute/detect/summarize and decorating it.
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
        all_raw_data: dict | None = None,
    ) -> list[AgentResult]:

        # Extract Excel-compatible DB row numbers and drop the metadata column.
        db_row_nums: dict[int, int] = {}
        if "__db_row_num__" in df.columns:
            for i in range(len(df)):
                db_row_nums[i] = int(df["__db_row_num__"].iloc[i])
            df = df.drop(columns=["__db_row_num__"]).reset_index(drop=True)

        # STEP 1 — build shared type context + run all registered precomputes
        ti        = _build_type_info(df, table_routes, all_raw_data, table)
        baselines = _run_all_precomputes(df, ti)

        if not any(baselines.values()):
            log.info("WowRowAgent %s: no baselines computed, skipping", table)
            return []

        # STEP 2 — find flagged rows and cache signals
        total = len(df)
        signals_cache: dict[int, list[str]] = {}
        for idx in range(total):
            sigs = _run_all_signals(df.iloc[idx], idx, baselines)
            if sigs:
                signals_cache[idx] = sigs
        row_indices = list(signals_cache.keys())[:200]

        if not row_indices:
            log.info("WowRowAgent %s: no rows with WoW signals found", table)
            return []

        # STEP 4 — ONE LLM call: signal types → insight templates (no row data)
        templates = _wow_fetch_templates(self._llm, table, signals_cache)

        # STEP 4b — pure Python: render one insight per flagged row (zero more LLM calls)
        insights_map: dict[str, str] = {
            f"row_{idx}": _wow_render_insight(signals_cache[idx], templates)
            for idx in row_indices
        }

        # STEP 5 — assemble AgentResult per row
        results: list[AgentResult] = []
        for idx in row_indices:
            row     = df.iloc[idx]
            row_key = f"row_{idx}"
            signals = signals_cache.get(idx, [])

            # ALL column values for this row — every column locked together
            all_values: dict[str, str] = {}
            for col in df.columns:
                raw_val = row[col] if col in row.index else None
                all_values[col] = "NULL" if (raw_val is None or (not isinstance(raw_val, str) and pd.isna(raw_val))) else str(raw_val)

            results.append(AgentResult(
                agent_name="wow_row",
                table=table,
                column=row_key,
                analysis={
                    "row_index":    db_row_nums.get(idx, idx + 2),  # Excel row number
                    "wow_signals":  signals,
                    "signal_count": len(signals),
                    "signal_types": list({s.split(":")[0] for s in signals}),
                    "all_values":   all_values,
                },
                insights_text=insights_map.get(row_key, ""),
                chart_data={"row_index": db_row_nums.get(idx, idx + 2), "signals": signals},
                chart_type="bar",
                error=None,
            ))

        log.info("WowRowAgent %s: %d row results", table, len(results))
        return results

    def run(self, raw_data: dict, column_routes: list[ColumnRoute]) -> list[AgentResult]:
        by_table: dict[str, list[ColumnRoute]] = defaultdict(list)
        for route in column_routes:
            table = route["table"]
            if raw_data.get(table) is not None and not raw_data[table].empty:
                by_table[table].append(route)

        results: list[AgentResult] = []
        for table, table_routes in by_table.items():
            log.info("WowRowAgent: table '%s' (%d routes)", table, len(table_routes))
            results.extend(
                self._run_table(raw_data[table], table, table_routes, all_raw_data=raw_data)
            )

        log.info("WowRowAgent complete: %d total row results", len(results))
        return results
