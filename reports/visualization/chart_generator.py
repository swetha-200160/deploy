from __future__ import annotations

import html as _html_lib
import json
import logging
import os
from typing import Any

import plotly.graph_objects as go

from config import Config, get_config
from core.state import AgentResult

log = logging.getLogger(__name__)

# Map pyecharts theme names → Plotly templates
_THEME_MAP = {
    "dark": "plotly_dark",
    "chalk": "plotly_dark",
    "purple_passion": "plotly_dark",
    "white": "simple_white",
    "westeros": "plotly_white",
    "essos": "plotly_white",
    "macarons": "plotly_white",
    "romantic": "plotly_white",
    "walden": "plotly_white",
    "wonderland": "plotly_white",
    "shine": "seaborn",
    "vintage": "ggplot2",
    "infographic": "plotly",
    "roma": "plotly",
}


def _px(val: Any) -> int:
    """Parse '900px' or 900 → int pixels."""
    if isinstance(val, int):
        return val
    return int(str(val).replace("px", "").strip())


def _layout(title: str, config: Config) -> dict:
    template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
    return dict(
        title=dict(text=title, font=dict(size=16, color="#1a237e")),
        template=template,
        width=_px(config.chart_width),
        height=_px(config.chart_height),
        margin=dict(l=60, r=40, t=70, b=90),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )


def _to_html(fig: go.Figure) -> str:
    """Render figure as a fully self-contained HTML — no CDN, works offline."""
    return fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True, "displayModeBar": True},
    )


def _fig_to_data(fig: go.Figure) -> dict:
    """Extract simplified data dict from a Plotly figure for JSON responses."""
    traces = []
    for trace in fig.data:
        t: dict = {"type": trace.type}
        for attr in ("x", "y", "labels", "values", "z", "name", "text"):
            val = getattr(trace, attr, None)
            if val is not None:
                try:
                    t[attr] = list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else val
                except Exception:
                    pass
        traces.append(t)
    title = ""
    if fig.layout.title and fig.layout.title.text:
        title = fig.layout.title.text
    return {"traces": traces, "title": title}


# ── Public chart builders ─────────────────────────────────────────────────────

def make_countplot(data: dict, title: str, config: Config) -> str:
    labels = [str(x) for x in data.get("labels", [])]
    values = [int(v) for v in data.get("values", [])]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        text=values, textposition="outside",
        marker_color="#4e79a7",
    ))
    fig.update_layout(**_layout(title, config), xaxis_tickangle=-30)
    return _to_html(fig)


def make_bar_chart(data: dict, title: str, config: Config) -> str:
    labels = [str(x) for x in data.get("labels", [])]
    values = data.get("values", [])
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        text=[str(v) for v in values], textposition="outside",
        marker_color="#4e79a7",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**_layout(title, config), xaxis_tickangle=-30)
    return _to_html(fig)


def make_pie_chart(data: dict, title: str, config: Config) -> str:
    labels = [str(x) for x in data.get("labels", [])]
    values = data.get("values", [])
    if not labels or not values:
        return "<div style='font-family:sans-serif;color:#888;padding:16px'>No data available</div>"
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.35,
        textinfo="label+percent+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**_layout(title, config))
    return _to_html(fig)


def make_stacked_bar_chart(data: dict, title: str, config: Config) -> str:
    x_axis = [str(x) for x in data.get("x_axis", [])]
    series_list = data.get("series", [])
    fig = go.Figure()
    for s in series_list:
        fig.add_trace(go.Bar(
            name=s.get("name", ""),
            x=x_axis,
            y=s.get("values", []),
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>",
        ))
    fig.update_layout(
        **_layout(title, config),
        barmode="stack",
        xaxis_tickangle=-30,
    )
    return _to_html(fig)


def make_heatmap(data: dict, title: str, config: Config) -> str:
    correlation = data.get("correlation", {})
    columns = data.get("columns", list(correlation.keys()))
    if not correlation or not columns:
        return "<div>No correlation data</div>"
    col_list = [str(c) for c in columns]
    z = []
    for col_a in col_list:
        row = []
        for col_b in col_list:
            val = correlation.get(col_a, {}).get(col_b, 0) if isinstance(correlation.get(col_a), dict) else 0
            row.append(round(float(val), 4))
        z.append(row)
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        x=col_list, y=col_list, z=z,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=text, texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_layout(title, config))
    return _to_html(fig)


def make_line_chart(data: dict, title: str, config: Config) -> str:
    dates = [str(d) for d in data.get("dates", [])]
    values = data.get("values", [])
    rolling = data.get("rolling_mean", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers",
        name=data.get("value_column", "Value"),
        line=dict(width=2, color="#4e79a7"),
        marker=dict(size=4),
        hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
    ))
    if rolling and len(rolling) == len(dates):
        fig.add_trace(go.Scatter(
            x=dates, y=rolling,
            mode="lines",
            name="Rolling Avg",
            line=dict(dash="dash", width=2, color="#e15759"),
            hovertemplate="<b>%{x}</b><br>Rolling Avg: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**_layout(title, config), xaxis_tickangle=-30)
    return _to_html(fig)


def make_histogram(data: dict, title: str, config: Config) -> str:
    bins = data.get("bins", [])
    edges = data.get("bin_edges", [])
    if not bins or not edges:
        return "<div>No histogram data</div>"
    bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    fig = go.Figure(go.Bar(
        x=bin_labels, y=bins,
        text=bins, textposition="outside",
        marker_color="#76b7b2",
        hovertemplate="<b>%{x}</b><br>Frequency: %{y}<extra></extra>",
    ))
    fig.update_layout(**_layout(title, config), xaxis_tickangle=-45, bargap=0.05)
    return _to_html(fig)


def make_boxplot(data: dict, title: str, config: Config) -> str:
    col = data.get("column", "Value")
    stats = data.get("stats", {})
    fig = go.Figure(go.Box(
        name=col,
        q1=[stats.get("p25", 0)],
        median=[stats.get("p50", 0)],
        q3=[stats.get("p75", 0)],
        lowerfence=[stats.get("min", 0)],
        upperfence=[stats.get("max", 0)],
        marker_color="#59a14f",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Min: " + str(stats.get("min", 0)) + "<br>"
            "Q1: " + str(stats.get("p25", 0)) + "<br>"
            "Median: " + str(stats.get("p50", 0)) + "<br>"
            "Q3: " + str(stats.get("p75", 0)) + "<br>"
            "Max: " + str(stats.get("max", 0)) +
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_layout(title, config))
    return _to_html(fig)


def make_scatter_chart(data: dict, title: str, config: Config) -> str:
    x_data = [round(float(v), 4) for v in data.get("x", []) if v is not None]
    y_data = [round(float(v), 4) for v in data.get("y", []) if v is not None]
    x_label = data.get("x_label", "X")
    y_label = data.get("y_label", "Y")
    fig = go.Figure(go.Scatter(
        x=x_data, y=y_data,
        mode="markers",
        marker=dict(size=6, opacity=0.7, color="#af7aa1"),
        hovertemplate=f"<b>{x_label}</b>: %{{x}}<br><b>{y_label}</b>: %{{y}}<extra></extra>",
    ))
    fig.update_layout(
        **_layout(title, config),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return _to_html(fig)


def make_violin_chart(data: dict, title: str, config: Config) -> str:
    """Mirrored histogram bars to approximate violin shape."""
    bins = data.get("bins", [])
    edges = data.get("bin_edges", [])
    if not bins or not edges:
        return "<div>No violin data</div>"
    bin_labels = [f"{edges[i]:.2f}" for i in range(len(edges) - 1)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Freq +", x=bin_labels, y=bins,
        marker_color="#4e79a7",
        hovertemplate="<b>%{x}</b><br>Freq: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Freq −", x=bin_labels, y=[-v for v in bins],
        marker_color="#e15759",
        hovertemplate="<b>%{x}</b><br>Freq: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_layout(title, config),
        barmode="relative",
        xaxis_tickangle=-45,
    )
    return _to_html(fig)


_DISPATCH = {
    "countplot": make_countplot,
    "bar": make_bar_chart,
    "pie": make_pie_chart,
    "stacked_bar": make_stacked_bar_chart,
    "heatmap": make_heatmap,
    "line": make_line_chart,
    "histogram": make_histogram,
    "boxplot": make_boxplot,
    "scatter": make_scatter_chart,
    "violin": make_violin_chart,
}


# ── Pipeline summary chart builders ──────────────────────────────────────────

def _build_universal_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    """Build charts from UniversalRowAgent results.

    Charts produced (all derived dynamically):
      1. bar  — flagged row count per table
      2. pie  — signal type distribution (prefix before ':' in all_signals)
      3. bar  — top anomalous columns by frequency
      4. bar  — rows grouped by signal count (0, 1, 2 … signals)
      5. bar  — actual flagged transaction amounts per row vs. column mean
      6. bar  — anomalous actual values vs. column means (grouped, side-by-side)
      7. bar  — cleared vs. rejected amounts per flagged row
    """
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return charts

    # 1. Flagged rows per table
    table_counts: dict[str, int] = {}
    for r in valid:
        t = r.get("table", "unknown")
        table_counts[t] = table_counts.get(t, 0) + 1
    if table_counts:
        sorted_tables = sorted(table_counts.items(), key=lambda x: -x[1])
        lbls, vals = zip(*sorted_tables)
        try:
            charts["bar__universal_flagged_by_table"] = make_bar_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"Audit Flags by Table — {list(lbls)[0] if len(lbls)==1 else f'{len(lbls)} Tables'} ({sum(vals)} total)",
                config,
            )
        except Exception as exc:
            log.error("universal table bar chart failed: %s", exc)

    # 2. Signal type distribution (dynamic: parse prefix before ':')
    type_counts: dict[str, int] = {}
    for r in valid:
        for sig in r.get("analysis", {}).get("all_signals", []):
            prefix = sig.split(":")[0] if ":" in sig else "other"
            type_counts[prefix] = type_counts.get(prefix, 0) + 1
    if type_counts:
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        lbls, vals = zip(*sorted_types)
        try:
            dominant = sorted_types[0][0] if sorted_types else "mixed"
            charts["pie__universal_signal_types"] = make_pie_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"Audit Signal Breakdown by Type (dominant: {dominant})",
                config,
            )
        except Exception as exc:
            log.error("universal signal pie chart failed: %s", exc)

    # 3. Top anomalous columns
    col_counts: dict[str, int] = {}
    for r in valid:
        for col in r.get("analysis", {}).get("anomalous_columns", []):
            col_counts[col] = col_counts.get(col, 0) + 1
    if col_counts:
        top_cols = sorted(col_counts.items(), key=lambda x: -x[1])[:15]
        lbls, vals = zip(*top_cols)
        try:
            charts["bar__universal_top_anomalous_columns"] = make_bar_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"High-Risk Columns by Anomaly Frequency (top {len(top_cols)})",
                config,
            )
        except Exception as exc:
            log.error("universal column bar chart failed: %s", exc)

    # 4. Signal count distribution (how many rows have 1, 2, 3 … signals)
    sc_dist: dict[int, int] = {}
    for r in valid:
        sc = r.get("analysis", {}).get("signal_count", 0)
        sc_dist[sc] = sc_dist.get(sc, 0) + 1
    if sc_dist:
        sorted_sc = sorted(sc_dist.items())
        lbls = [f"{k} signal{'s' if k != 1 else ''}" for k, _ in sorted_sc]
        vals = [v for _, v in sorted_sc]
        try:
            charts["bar__universal_signal_count_dist"] = make_bar_chart(
                {"labels": lbls, "values": vals},
                f"Transaction Risk Distribution by Signal Count ({len(valid)} flagged rows)",
                config,
            )
        except Exception as exc:
            log.error("universal signal count dist chart failed: %s", exc)

    # 5. Actual flagged transaction amounts per row (real values from inside the rows)
    amounts: list[float] = []
    row_labels: list[str] = []
    amount_mean: float | None = None
    for r in valid:
        analysis = r.get("analysis", {})
        vals = analysis.get("values", {})
        if "amount" in vals and vals["amount"] is not None:
            amounts.append(float(vals["amount"]))
            row_labels.append(f"row_{analysis.get('row_index', '?')}")
            if amount_mean is None:
                amount_mean = analysis.get("col_means", {}).get("amount")
    if amounts:
        try:
            template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
            threshold = (amount_mean or 0) * 2
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=row_labels, y=amounts,
                text=[f"{a:,.0f}" for a in amounts], textposition="outside",
                marker_color=["#e15759" if a > threshold else "#4e79a7" for a in amounts],
                name="Amount",
                hovertemplate="<b>%{x}</b><br>Amount: %{y:,.0f}<extra></extra>",
            ))
            if amount_mean is not None:
                fig.add_hline(
                    y=amount_mean, line_dash="dash", line_color="#e15759",
                    annotation_text=f"Mean: {amount_mean:,.2f}",
                    annotation_position="top right",
                )
            fig.update_layout(
                title=dict(text=f"Flagged Transaction Amounts — {len(amounts)} Rows (red = 2× above mean)", font=dict(size=16, color="#1a237e")),
                template=template,
                width=_px(config.chart_width), height=_px(config.chart_height),
                margin=dict(l=60, r=40, t=70, b=90),
                xaxis_tickangle=-45,
                yaxis_title="Amount",
                xaxis_title="Flagged Row",
            )
            charts["bar__universal_flagged_amounts"] = _to_html(fig)
        except Exception as exc:
            log.error("universal flagged amounts chart failed: %s", exc)

    # 6. Anomalous numeric values vs. column means (actual vs. expected)
    anom_rows: list[str] = []
    anom_actual: list[float] = []
    anom_means: list[float] = []
    anom_col_labels: list[str] = []
    for r in valid:
        analysis = r.get("analysis", {})
        vals = analysis.get("values", {})
        means = analysis.get("col_means", {})
        row_idx = analysis.get("row_index", "?")
        for col in analysis.get("anomalous_columns", []):
            if col in vals and col in means and vals[col] is not None:
                anom_rows.append(f"row_{row_idx}")
                anom_actual.append(float(vals[col]))
                anom_means.append(float(means[col]))
                anom_col_labels.append(col)
    if anom_actual:
        try:
            template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
            x_labels = [f"{r}:{c}" for r, c in zip(anom_rows, anom_col_labels)]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Actual Value", x=x_labels, y=anom_actual,
                marker_color="#e15759",
                hovertemplate="<b>%{x}</b><br>Actual: %{y:,.2f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="Column Mean", x=x_labels, y=anom_means,
                marker_color="#76b7b2",
                hovertemplate="<b>%{x}</b><br>Mean: %{y:,.2f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"Anomalous Values vs. Column Means ({len(anom_actual)} anomalies)", font=dict(size=16, color="#1a237e")),
                template=template,
                width=_px(config.chart_width), height=_px(config.chart_height),
                margin=dict(l=60, r=40, t=70, b=90),
                barmode="group",
                xaxis_tickangle=-45,
                yaxis_title="Value",
            )
            charts["bar__universal_anomalous_vs_mean"] = _to_html(fig)
        except Exception as exc:
            log.error("universal anomalous vs mean chart failed: %s", exc)

    # 7. Cleared vs. Rejected amount breakdown per flagged row
    cleared_amounts: list[float] = []
    rejected_amounts: list[float] = []
    cr_row_labels: list[str] = []
    for r in valid:
        analysis = r.get("analysis", {})
        vals = analysis.get("values", {})
        if "clearedamount" in vals and "rejectedamount" in vals:
            cleared_amounts.append(float(vals.get("clearedamount") or 0))
            rejected_amounts.append(float(vals.get("rejectedamount") or 0))
            cr_row_labels.append(f"row_{analysis.get('row_index', '?')}")
    if cr_row_labels and any(v > 0 for v in cleared_amounts + rejected_amounts):
        try:
            template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Cleared Amount", x=cr_row_labels, y=cleared_amounts,
                marker_color="#59a14f",
                hovertemplate="<b>%{x}</b><br>Cleared: %{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="Rejected Amount", x=cr_row_labels, y=rejected_amounts,
                marker_color="#e15759",
                hovertemplate="<b>%{x}</b><br>Rejected: %{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"Cleared vs. Rejected Amounts per Flagged Row ({len(cr_row_labels)} rows)", font=dict(size=16, color="#1a237e")),
                template=template,
                width=_px(config.chart_width), height=_px(config.chart_height),
                margin=dict(l=60, r=40, t=70, b=90),
                barmode="group",
                xaxis_tickangle=-45,
                yaxis_title="Amount",
                xaxis_title="Flagged Row",
            )
            charts["bar__universal_cleared_vs_rejected"] = _to_html(fig)
        except Exception as exc:
            log.error("universal cleared vs rejected chart failed: %s", exc)

    return charts


def _build_enterprise_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    """Build charts from EnterpriseRowAgent results.

    Charts produced (all derived dynamically):
      1. bar — signal type counts (NULL / DUP / TYPE / FRAUD — parsed from signals)
      2. bar — flagged row count per table
    """
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return charts

    # 1. Signal type distribution (dynamic: prefix before ':')
    type_counts: dict[str, int] = {}
    for r in valid:
        for sig in r.get("analysis", {}).get("signals", []):
            prefix = sig.split(":")[0] if ":" in sig else sig[:10]
            type_counts[prefix] = type_counts.get(prefix, 0) + 1
    if type_counts:
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        lbls, vals = zip(*sorted_types)
        try:
            dominant_ent = sorted_types[0][0] if sorted_types else "mixed"
            charts["bar__enterprise_signal_types"] = make_bar_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"Data Quality & Fraud Control Signals (dominant: {dominant_ent})",
                config,
            )
        except Exception as exc:
            log.error("enterprise signal bar chart failed: %s", exc)

    # 2. Flagged rows per table
    table_counts: dict[str, int] = {}
    for r in valid:
        t = r.get("table", "unknown")
        table_counts[t] = table_counts.get(t, 0) + 1
    if len(table_counts) > 1:
        sorted_tables = sorted(table_counts.items(), key=lambda x: -x[1])
        lbls, vals = zip(*sorted_tables)
        try:
            charts["bar__enterprise_flagged_by_table"] = make_bar_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"Data Quality & Fraud Flags by Table ({sum(vals)} total)",
                config,
            )
        except Exception as exc:
            log.error("enterprise table bar chart failed: %s", exc)

    return charts


def _build_wow_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    """Build charts from WowRowAgent results.

    Charts produced (all derived dynamically):
      1. bar — pattern type frequency (from analysis.signal_types)
      2. bar — flagged row count per table
    """
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return charts

    # 1. Pattern type frequency (dynamic: values come from signal_types list)
    type_counts: dict[str, int] = {}
    for r in valid:
        for sig_type in r.get("analysis", {}).get("signal_types", []):
            type_counts[sig_type] = type_counts.get(sig_type, 0) + 1
    if type_counts:
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        lbls, vals = zip(*sorted_types)
        try:
            dominant_wow = sorted_types[0][0] if sorted_types else "mixed"
            charts["bar__wow_pattern_types"] = make_bar_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"Advanced Pattern Anomaly Frequency (dominant: {dominant_wow})",
                config,
            )
        except Exception as exc:
            log.error("wow pattern bar chart failed: %s", exc)

    # 2. Flagged rows per table
    table_counts: dict[str, int] = {}
    for r in valid:
        t = r.get("table", "unknown")
        table_counts[t] = table_counts.get(t, 0) + 1
    if len(table_counts) > 1:
        sorted_tables = sorted(table_counts.items(), key=lambda x: -x[1])
        lbls, vals = zip(*sorted_tables)
        try:
            charts["bar__wow_flagged_by_table"] = make_bar_chart(
                {"labels": list(lbls), "values": list(vals)},
                f"Business Pattern Flags by Table ({sum(vals)} total)",
                config,
            )
        except Exception as exc:
            log.error("wow table bar chart failed: %s", exc)

    return charts


# ── DEPRECATED helpers (kept to avoid import errors) ─────────────────────────

def _get_cat_counts(result: AgentResult) -> tuple[list[str], list[int]]:
    freq = result.get("analysis", {}).get("analyze_frequency", {}).get("frequency", {})
    if freq:
        pairs = sorted(freq.items(), key=lambda kv: -kv[1])
        return [str(k) for k, _ in pairs], [int(v) for _, v in pairs]
    d = result.get("chart_data", {})
    ct = result.get("chart_type", "")
    if ct in ("bar", "countplot", "pie"):
        lbls = [str(x) for x in d.get("labels", [])]
        vals = [int(v) for v in d.get("values", [])]
        return lbls, vals
    return [], []


def _build_categorical_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error") and r.get("chart_data")]
    if not valid:
        return charts

    countable = [r for r in valid if _get_cat_counts(r)[1]]

    if countable:
        countplot_src = max(countable, key=lambda r: len(_get_cat_counts(r)[0]))
        lbls, vals = _get_cat_counts(countplot_src)
        top20 = sorted(zip(lbls, vals), key=lambda x: -x[1])[:20]
        if top20:
            pl, pv = zip(*top20)
            try:
                charts["countplot__categorical_summary"] = make_countplot(
                    {"labels": list(pl), "values": list(pv)},
                    f"Value Count Distribution — {countplot_src['column']}",
                    config,
                )
            except Exception as exc:
                log.error("Countplot summary failed: %s", exc)

    bar_lbls, bar_vals = [], []
    for r in valid:
        _, cnts = _get_cat_counts(r)
        bar_lbls.append(r["column"])
        bar_vals.append(cnts[0] if cnts else 0)
    if any(bar_vals):
        try:
            charts["bar__categorical_summary"] = make_bar_chart(
                {"labels": bar_lbls, "values": bar_vals},
                "Dominant Category Count per Column",
                config,
            )
        except Exception as exc:
            log.error("Categorical bar summary failed: %s", exc)

    if countable:
        pie_src = min(countable, key=lambda r: len(_get_cat_counts(r)[0]))
        lbls, vals = _get_cat_counts(pie_src)
        top8 = sorted(zip(lbls, vals), key=lambda x: -x[1])[:8]
        if top8:
            pl, pv = zip(*top8)
            try:
                charts["pie__categorical_summary"] = make_pie_chart(
                    {"labels": list(pl), "values": list(pv)},
                    f"Category Distribution — {pie_src['column']}",
                    config,
                )
            except Exception as exc:
                log.error("Pie summary failed: %s", exc)

    col_counts: dict[str, dict[str, int]] = {}
    all_top_cats: list[str] = []
    for r in valid:
        lbls, vals = _get_cat_counts(r)
        top5 = list(zip(lbls, vals))[:5] if vals else []
        col_counts[r["column"]] = dict(top5)
        for cat, _ in top5:
            if cat not in all_top_cats:
                all_top_cats.append(cat)
    col_names = [r["column"] for r in valid]
    series = [
        {"name": cat, "values": [col_counts[col].get(cat, 0) for col in col_names]}
        for cat in all_top_cats
    ]
    if series:
        try:
            charts["stacked_bar__categorical_summary"] = make_stacked_bar_chart(
                {"x_axis": col_names, "series": series},
                "Category Breakdown Across All Columns",
                config,
            )
        except Exception as exc:
            log.error("Stacked bar summary failed: %s", exc)

    return charts


def _build_numerical_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error") and r.get("chart_data")]
    if not valid:
        return charts

    hist_data: dict | None = None
    hist_col: str = ""
    for r in sorted(valid, key=lambda r: len(r["chart_data"].get("bins", [])), reverse=True):
        if r.get("chart_type") == "histogram" and r["chart_data"].get("bins"):
            hist_data = r["chart_data"]
            hist_col = r["column"]
            break
    if hist_data is None:
        for r in valid:
            dist = r.get("analysis", {}).get("analyze_distribution", {})
            if dist.get("bins") and dist.get("bin_edges"):
                hist_data = dist
                hist_col = r["column"]
                break
    if hist_data:
        try:
            charts["histogram__numerical_summary"] = make_histogram(
                hist_data, f"Distribution — {hist_col}", config,
            )
        except Exception as exc:
            log.error("Histogram summary failed: %s", exc)

    heatmap_src = next(
        (r for r in valid if r.get("chart_type") == "heatmap" and r["chart_data"].get("correlation")),
        None,
    )
    if heatmap_src:
        try:
            charts["heatmap__numerical_summary"] = make_heatmap(
                heatmap_src["chart_data"],
                "Correlation Heatmap — All Numerical Columns",
                config,
            )
        except Exception as exc:
            log.error("Heatmap summary failed: %s", exc)

    return charts


def _build_datetime_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error") and r.get("chart_data") and r["chart_data"].get("dates")]
    if not valid:
        return charts

    best = max(valid, key=lambda r: len(r["chart_data"].get("dates", [])))
    try:
        charts["line__datetime_summary"] = make_line_chart(
            best["chart_data"],
            f"Time Series — {best['column']}",
            config,
        )
    except Exception as exc:
        log.error("Datetime line chart failed: %s", exc)

    return charts


def _build_text_charts(results: list[AgentResult], config: Config) -> dict[str, str]:
    charts: dict[str, str] = {}
    valid = [r for r in results if not r.get("error") and r.get("chart_data")]
    if not valid:
        return charts

    combined_kw: dict[str, float] = {}
    combined_sentiment = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in valid:
        d = r["chart_data"]
        for kw, score in zip(d.get("keyword_labels", []), d.get("keyword_values", [])):
            combined_kw[kw] = combined_kw.get(kw, 0) + float(score)
        sv = d.get("sentiment_values", [0, 0, 0])
        combined_sentiment["Positive"] += sv[0] if len(sv) > 0 else 0
        combined_sentiment["Negative"] += sv[1] if len(sv) > 1 else 0
        combined_sentiment["Neutral"] += sv[2] if len(sv) > 2 else 0

    top_kw = sorted(combined_kw.items(), key=lambda x: -x[1])[:15]
    if top_kw:
        kl, kv = zip(*top_kw)
        try:
            charts["bar__text_keywords"] = make_bar_chart(
                {"labels": list(kl), "values": [round(v, 4) for v in kv]},
                "Top Keywords — All Text Columns",
                config,
            )
        except Exception as exc:
            log.error("Keywords bar chart failed: %s", exc)

    if any(combined_sentiment.values()):
        try:
            charts["bar__text_sentiment"] = make_bar_chart(
                {"labels": list(combined_sentiment.keys()), "values": list(combined_sentiment.values())},
                "Sentiment Distribution — All Text Columns",
                config,
            )
        except Exception as exc:
            log.error("Sentiment bar chart failed: %s", exc)

    return charts


def _build_raw_data_charts(raw_data: dict, config: Config) -> dict[str, str]:
    """Build exactly 5 charts from all raw DataFrames combined.

    Fully dynamic — zero hardcoded column names. Scans every table,
    picks the best candidate column(s) per chart type, then produces
    ONE chart per type across ALL tables:

      hist__raw_data    — Histogram  : highest-variance numeric col
      pie__raw_data     — Pie        : most balanced categorical col
      line__raw_data    — Line       : best datetime × numeric pair
      bar__raw_data     — Bar        : best categorical × numeric pair
      heatmap__raw_data — Heatmap    : table with the most numeric columns
    """
    import pandas as pd
    import warnings

    _MAX_CAT_CARD    = 30
    _MIN_CAT_CARD    = 2
    _DT_PARSE_THRESH = 0.8
    _PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    charts: dict[str, str] = {}
    template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
    base_layout = dict(
        template=template,
        width=_px(config.chart_width),
        height=_px(config.chart_height),
        margin=dict(l=60, r=40, t=70, b=90),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )

    # ── Candidate lists — scan all tables, collect best options per chart type ───
    hist_cands:    list[tuple] = []  # (std,    tname, num_col, df)
    pie_cands:     list[tuple] = []  # (n_uniq, tname, cat_col, df)
    line_cands:    list[tuple] = []  # (n_rows, tname, dt_col, num_col, df)
    bar_cands:     list[tuple] = []  # (std,    tname, cat_col, num_col, df)
    heatmap_cands: list[tuple] = []  # (n_num,  tname, num_cols, df)

    for tname, df in raw_data.items():
        if df is None or df.empty:
            continue

        # numeric cols: non-binary, non-ID, sorted by std desc
        num_cols: list[str] = []
        for c in df.select_dtypes(include="number").columns:
            s = df[c].dropna()
            if len(s) < 5 or s.nunique() <= 2 or s.nunique() == len(s):
                continue
            num_cols.append(c)
        num_cols.sort(
            key=lambda c: float(pd.to_numeric(df[c], errors="coerce").std() or 0),
            reverse=True,
        )

        # categorical cols: bounded cardinality, sorted by n_unique asc
        cat_cols: list[str] = []
        for c in df.select_dtypes(include=["object", "category"]).columns:
            n = df[c].nunique()
            if _MIN_CAT_CARD <= n <= _MAX_CAT_CARD:
                cat_cols.append(c)
        cat_cols.sort(key=lambda c: df[c].nunique())

        # datetime cols: dtype first, then sniff strings (no deprecated flags)
        dt_cols: list[str] = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        for c in df.select_dtypes(include="object").columns:
            if c in dt_cols:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[c].dropna().head(20), errors="coerce")
                if parsed.notna().mean() >= _DT_PARSE_THRESH:
                    dt_cols.append(c)
            except Exception:
                pass

        if num_cols:
            std = float(pd.to_numeric(df[num_cols[0]], errors="coerce").std() or 0)
            hist_cands.append((std, tname, num_cols[0], df))

        if cat_cols:
            pie_cands.append((df[cat_cols[0]].nunique(), tname, cat_cols[0], df))

        if dt_cols and num_cols:
            line_cands.append((len(df), tname, dt_cols[0], num_cols[0], df))

        if cat_cols and num_cols:
            std = float(pd.to_numeric(df[num_cols[0]], errors="coerce").std() or 0)
            bar_cands.append((std, tname, cat_cols[0], num_cols[0], df))

        if len(num_cols) >= 3:
            heatmap_cands.append((len(num_cols), tname, num_cols[:10], df))

    # ── 1. HISTOGRAM ─────────────────────────────────────────────────────────────
    if hist_cands:
        _, tname, nc, df = max(hist_cands, key=lambda x: x[0])
        try:
            vals = pd.to_numeric(df[nc], errors="coerce").dropna()
            mean_val = float(vals.mean())
            fig = go.Figure(go.Histogram(
                x=vals.tolist(), nbinsx=30, marker_color=_PALETTE[0],
                hovertemplate=f"{nc}: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))
            fig.add_vline(x=mean_val, line_dash="dash", line_color="#e15759",
                          annotation_text=f"Mean: {mean_val:,.2f}",
                          annotation_position="top right")
            fig.update_layout(
                title=dict(text=f"{nc} Distribution — {len(vals):,} rows ({tname})",
                           font=dict(size=16, color="#1a237e")),
                xaxis_title=nc, yaxis_title="Count", **base_layout,
            )
            charts["hist__raw_data"] = _to_html(fig)
        except Exception as exc:
            log.error("raw histogram failed: %s", exc)

    # ── 2. PIE ───────────────────────────────────────────────────────────────────
    if pie_cands:
        # closest to 5 unique values = cleanest pie
        _, tname, cc, df = min(pie_cands, key=lambda x: abs(x[0] - 5))
        try:
            counts = df[cc].astype(str).value_counts().head(10)
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(),
                hole=0.35, textinfo="label+percent+value",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"{cc} Distribution — {tname}",
                           font=dict(size=16, color="#1a237e")),
                **base_layout,
            )
            charts["pie__raw_data"] = _to_html(fig)
        except Exception as exc:
            log.error("raw pie failed: %s", exc)

    # ── 3. LINE ──────────────────────────────────────────────────────────────────
    if line_cands:
        _, tname, dc, nc, df = max(line_cands, key=lambda x: x[0])
        try:
            df_t = df[[dc, nc]].copy()
            df_t[nc] = pd.to_numeric(df_t[nc], errors="coerce")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_t[dc] = pd.to_datetime(df_t[dc], errors="coerce")
            df_t = df_t.dropna()
            if len(df_t) >= 3:
                df_t["_p"] = df_t[dc].dt.to_period("M").astype(str)
                monthly = df_t.groupby("_p")[nc].sum().sort_index()
                fig = go.Figure(go.Scatter(
                    x=monthly.index.tolist(), y=monthly.values.tolist(),
                    mode="lines+markers",
                    line=dict(width=2, color=_PALETTE[0]), marker=dict(size=6),
                    hovertemplate=f"<b>%{{x}}</b><br>{nc} total: %{{y:,.0f}}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(text=f"Monthly {nc} Trend — {tname}",
                               font=dict(size=16, color="#1a237e")),
                    xaxis_tickangle=-30, yaxis_title=f"Total {nc}",
                    xaxis_title="Period", **base_layout,
                )
                charts["line__raw_data"] = _to_html(fig)
        except Exception as exc:
            log.error("raw line chart failed: %s", exc)

    # ── 4. BAR ───────────────────────────────────────────────────────────────────
    if bar_cands:
        _, tname, cc, nc, df = max(bar_cands, key=lambda x: x[0])
        try:
            df_p = df[[cc, nc]].copy()
            df_p[nc] = pd.to_numeric(df_p[nc], errors="coerce")
            grp = df_p.groupby(cc)[nc].sum().sort_values(ascending=False).head(20)
            if len(grp) >= 2:
                fig = go.Figure(go.Bar(
                    x=grp.index.tolist(), y=grp.values.tolist(),
                    text=[f"{v:,.0f}" for v in grp.values], textposition="outside",
                    marker_color=_PALETTE[0],
                    hovertemplate=f"<b>%{{x}}</b><br>Total {nc}: %{{y:,.0f}}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(text=f"Total {nc} by {cc} — {tname}",
                               font=dict(size=16, color="#1a237e")),
                    xaxis_tickangle=-30, yaxis_title=f"Total {nc}", **base_layout,
                )
                charts["bar__raw_data"] = _to_html(fig)
        except Exception as exc:
            log.error("raw bar chart failed: %s", exc)

    # ── 5. HEATMAP ───────────────────────────────────────────────────────────────
    if heatmap_cands:
        _, tname, hm_cols, df = max(heatmap_cands, key=lambda x: x[0])
        try:
            corr = df[hm_cols].apply(pd.to_numeric, errors="coerce").corr()
            z    = [[round(float(corr.loc[a, b]), 4) for b in hm_cols] for a in hm_cols]
            text = [[f"{v:.2f}" for v in row] for row in z]
            fig = go.Figure(go.Heatmap(
                x=hm_cols, y=hm_cols, z=z,
                colorscale="RdBu", zmin=-1, zmax=1,
                text=text, texttemplate="%{text}",
                hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"Numeric Column Correlations — {tname}",
                           font=dict(size=16, color="#1a237e")),
                **base_layout,
            )
            charts["heatmap__raw_data"] = _to_html(fig)
        except Exception as exc:
            log.error("raw heatmap failed: %s", exc)

    return charts


def _build_table_specific_charts(raw_data: dict, config: Config) -> tuple[dict[str, str], dict[str, dict]]:
    """Build data-insight charts dynamically for every table in raw_data.

    Zero hardcoded column names.  For each table the function:
      1. Profiles every column by role (flag / amount / qty / price / group / status / datetime)
         using keyword matching on the column name.
      2. Builds the best available chart(s) for that table's actual columns:
         - Grouped-bar comparison  : when 2+ numeric-value cols share a grouping col
         - Flag-filtered amount bar: flag col(s) × amount col × grouping col
         - Categorical × numeric bar: group col × any numeric col
         - Status / severity pie  : low-cardinality categorical col (≤ 15 unique)
         - Exception-flags pie    : all flag cols aggregated into one pie
      3. Falls back gracefully — if a table has none of the above patterns, nothing is emitted.
    """
    import pandas as pd

    charts: dict[str, str] = {}
    chart_data_store: dict[str, dict] = {}
    template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
    base_layout = dict(
        template=template,
        width=_px(config.chart_width),
        height=_px(config.chart_height),
        margin=dict(l=60, r=40, t=70, b=90),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    _PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    # ── keyword sets for role detection ─────────────────────────────────────────
    _PRICE_KW    = {"price", "rate", "unitprice", "unit_price", "cost"}
    _AMOUNT_KW   = {"amount", "amt", "value", "total", "sum", "cleared", "rejected"}
    _QTY_KW      = {"qty", "quantity", "count", "grn_qty", "po_qty", "inv_qty"}
    _GROUP_KW    = {"vendor", "supplier", "material", "product", "branch",
                    "region", "category", "type", "name", "desc", "item",
                    "company", "entity", "plant", "buyer", "department"}
    _STATUS_KW   = {"status", "severity", "priority", "action", "result",
                    "outcome", "state", "condition", "cheque_status", "cleared",
                    "rejected", "processed"}
    _FLAG_KW     = {"exc_", "flag_", "missing", "exception", "error", "issue",
                    "warning", "anomal"}
    _DATE_KW     = {"date", "period", "month", "year", "week", "time", "created",
                    "processed"}

    def _role(col: str) -> set[str]:
        """Return a set of roles this column likely plays (may have multiple)."""
        lc = col.lower()
        roles: set[str] = set()
        if any(lc.startswith(k) or k in lc for k in _FLAG_KW):
            roles.add("flag")
        if any(k in lc for k in _PRICE_KW):
            roles.add("price")
        if any(k in lc for k in _AMOUNT_KW):
            roles.add("amount")
        if any(k in lc for k in _QTY_KW):
            roles.add("qty")
        if any(k in lc for k in _GROUP_KW):
            roles.add("group")
        if any(k in lc for k in _STATUS_KW):
            roles.add("status")
        if any(k in lc for k in _DATE_KW):
            roles.add("date")
        return roles

    def _flag_count(series: "pd.Series") -> int:
        if series.dtype == bool:
            return int(series.sum())
        if series.dtype.kind in ("i", "f"):
            return int((series != 0).sum())
        return int(series.astype(str).str.upper().isin({"Y", "YES", "TRUE", "1"}).sum())

    def _filter_flag(df: "pd.DataFrame", col: str) -> "pd.DataFrame":
        s = df[col]
        if s.dtype == bool:
            return df[s]
        if s.dtype.kind in ("i", "f"):
            return df[s.astype(bool)]
        return df[s.astype(str).str.upper().isin({"Y", "YES", "TRUE", "1"})]

    for tname, df in raw_data.items():
        if df is None or df.empty:
            continue

        # ── Classify every column ────────────────────────────────────────────────
        price_cols:  list[str] = []
        amount_cols: list[str] = []
        qty_cols:    list[str] = []
        group_cols:  list[str] = []
        status_cols: list[str] = []
        flag_cols:   list[str] = []
        date_cols:   list[str] = []
        num_cols:    list[str] = []   # all numeric, non-binary

        for col in df.columns:
            roles = _role(col)
            s = df[col].dropna()
            is_num = pd.api.types.is_numeric_dtype(df[col])
            is_obj = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])

            if "flag" in roles:
                flag_cols.append(col)
            if "price" in roles and is_num:
                price_cols.append(col)
            if "amount" in roles and is_num:
                amount_cols.append(col)
            if "qty" in roles and is_num:
                qty_cols.append(col)
            if "group" in roles and is_obj and 1 < s.nunique() <= 50:
                group_cols.append(col)
            if "status" in roles and is_obj and 1 < s.nunique() <= 15:
                status_cols.append(col)
            if "date" in roles and not is_num:
                date_cols.append(col)
            if is_num:
                n = s.nunique()
                if n > 2 and n < len(s):
                    num_cols.append(col)

        # Best grouping col: prefer lower cardinality (cleaner charts)
        grp = sorted(group_cols, key=lambda c: df[c].nunique())[:1]
        grp_col = grp[0] if grp else None

        safe_tname = tname.replace(" ", "_").replace("/", "_")

        # ── Chart A: Grouped-bar — compare numeric value columns per group ────────
        # Covers: price comparison, qty comparison, amount comparison
        value_groups = [
            ("price",  price_cols,  "Price"),
            ("qty",    qty_cols,    "Quantity"),
            ("amount", amount_cols, "Amount"),
        ]
        for vtype, vcols, ylabel in value_groups:
            if len(vcols) >= 2 and grp_col:
                try:
                    use_cols = vcols[:4]  # max 4 series
                    dfp = df[[grp_col] + use_cols].copy()
                    for c in use_cols:
                        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
                    dfp = dfp.dropna(subset=use_cols[:2])
                    grpd = (dfp.groupby(grp_col)[use_cols]
                               .mean()
                               .sort_values(use_cols[0], ascending=False)
                               .head(20))
                    if len(grpd) < 2:
                        continue
                    fig = go.Figure()
                    for i, col in enumerate(use_cols):
                        fig.add_trace(go.Bar(
                            name=col, x=grpd.index.tolist(), y=grpd[col].tolist(),
                            marker_color=_PALETTE[i % len(_PALETTE)],
                            text=[f"{v:,.2f}" for v in grpd[col]], textposition="outside",
                            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,.2f}}<extra></extra>",
                        ))
                    fig.update_layout(
                        title=dict(
                            text=f"{tname} — {ylabel} Comparison by {grp_col} (Top 20, avg)",
                            font=dict(size=16, color="#1a237e"),
                        ),
                        barmode="group", xaxis_tickangle=-35,
                        yaxis_title=ylabel, xaxis_title=grp_col, **base_layout,
                    )
                    key = f"chart__{safe_tname}__{vtype}_comparison"
                    chart_data_store[key] = _fig_to_data(fig)
                    charts[key] = _to_html(fig)
                except Exception as exc:
                    log.error("%s %s comparison chart failed: %s", tname, vtype, exc)

        # ── Chart B: Flag-filtered bar — PO/amount at risk where flag = True ─────
        if flag_cols and amount_cols and grp_col:
            for flag_col in flag_cols[:3]:  # at most 3 flag charts per table
                flagged_count = _flag_count(df[flag_col])
                if flagged_count == 0:
                    continue
                amt_col = amount_cols[0]
                try:
                    dff = _filter_flag(df, flag_col)
                    dff = dff[[grp_col, amt_col]].copy()
                    dff[amt_col] = pd.to_numeric(dff[amt_col], errors="coerce")
                    dff = dff.dropna()
                    grpd = dff.groupby(grp_col)[amt_col].sum().sort_values(ascending=False).head(20)
                    if len(grpd) < 1:
                        continue
                    fig = go.Figure(go.Bar(
                        x=grpd.index.tolist(), y=grpd.values.tolist(),
                        text=[f"{v:,.0f}" for v in grpd.values], textposition="outside",
                        marker_color="#e15759",
                        hovertemplate=f"<b>%{{x}}</b><br>{flag_col} — {amt_col}: %{{y:,.0f}}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=dict(
                            text=f"{tname} — {flag_col}: {amt_col} at Risk by {grp_col} (Top 20)",
                            font=dict(size=16, color="#1a237e"),
                        ),
                        xaxis_tickangle=-35, yaxis_title=amt_col,
                        xaxis_title=grp_col, **base_layout,
                    )
                    safe_flag = flag_col.replace(" ", "_")
                    key = f"chart__{safe_tname}__flag_{safe_flag}"
                    chart_data_store[key] = _fig_to_data(fig)
                    charts[key] = _to_html(fig)
                except Exception as exc:
                    log.error("%s flag bar chart (%s) failed: %s", tname, flag_col, exc)

        # ── Chart C: Single numeric × group bar (best numeric col by variance) ───
        if not (price_cols or qty_cols or amount_cols) and num_cols and grp_col:
            try:
                best_num = max(
                    num_cols,
                    key=lambda c: float(pd.to_numeric(df[c], errors="coerce").std() or 0),
                )
                dfb = df[[grp_col, best_num]].copy()
                dfb[best_num] = pd.to_numeric(dfb[best_num], errors="coerce")
                dfb = dfb.dropna()
                grpd = dfb.groupby(grp_col)[best_num].sum().sort_values(ascending=False).head(20)
                if len(grpd) >= 2:
                    fig = go.Figure(go.Bar(
                        x=grpd.index.tolist(), y=grpd.values.tolist(),
                        text=[f"{v:,.0f}" for v in grpd.values], textposition="outside",
                        marker_color=_PALETTE[0],
                        hovertemplate=f"<b>%{{x}}</b><br>{best_num}: %{{y:,.0f}}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=dict(
                            text=f"{tname} — Total {best_num} by {grp_col} (Top 20)",
                            font=dict(size=16, color="#1a237e"),
                        ),
                        xaxis_tickangle=-35, yaxis_title=best_num,
                        xaxis_title=grp_col, **base_layout,
                    )
                    key = f"chart__{safe_tname}__top_by_group"
                    chart_data_store[key] = _fig_to_data(fig)
                    charts[key] = _to_html(fig)
            except Exception as exc:
                log.error("%s top-by-group chart failed: %s", tname, exc)

        # ── Chart D: Status / severity pie ───────────────────────────────────────
        for st_col in status_cols[:2]:
            try:
                cnt = df[st_col].astype(str).value_counts().head(12)
                if len(cnt) < 2:
                    continue
                sev_colors = [
                    "#e15759" if any(k in str(v).lower() for k in ("high", "critical", "reject", "missing"))
                    else "#f28e2b" if any(k in str(v).lower() for k in ("medium", "moderate", "partial", "pending"))
                    else "#59a14f"
                    for v in cnt.index
                ]
                fig = go.Figure(go.Pie(
                    labels=cnt.index.tolist(), values=cnt.values.tolist(),
                    hole=0.35, textinfo="label+percent+value",
                    marker_colors=sev_colors,
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(
                        text=f"{tname} — {st_col} Distribution",
                        font=dict(size=16, color="#1a237e"),
                    ),
                    **base_layout,
                )
                safe_st = st_col.replace(" ", "_")
                key = f"chart__{safe_tname}__pie_{safe_st}"
                chart_data_store[key] = _fig_to_data(fig)
                charts[key] = _to_html(fig)
            except Exception as exc:
                log.error("%s status pie (%s) failed: %s", tname, st_col, exc)

        # ── Chart E: All exception/flag columns → one summary pie ────────────────
        if len(flag_cols) >= 2:
            try:
                exc_counts = {c: _flag_count(df[c]) for c in flag_cols}
                exc_counts = {k: v for k, v in exc_counts.items() if v > 0}
                if len(exc_counts) >= 2:
                    fig = go.Figure(go.Pie(
                        labels=list(exc_counts.keys()),
                        values=list(exc_counts.values()),
                        hole=0.35, textinfo="label+percent+value",
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=dict(
                            text=f"{tname} — Exception Flags Distribution ({sum(exc_counts.values())} total)",
                            font=dict(size=16, color="#1a237e"),
                        ),
                        **base_layout,
                    )
                    key = f"chart__{safe_tname}__exception_flags_pie"
                    chart_data_store[key] = _fig_to_data(fig)
                    charts[key] = _to_html(fig)
            except Exception as exc:
                log.error("%s exception flags pie failed: %s", tname, exc)

    return charts, chart_data_store


def generate_all_charts(
    agent_results: list[AgentResult],
    config: Config | None = None,
    raw_data: dict | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    """Generate exactly 5 charts combining ALL data from agents + raw tables.

    Always produces exactly these 5 keys:
      chart__bar     — flagged-row counts + signal types (all agents, all tables)
      chart__pie     — signal type distribution (all agents combined)
      chart__line    — monthly financial trend (raw data, best table)
      chart__hist    — numeric value distribution (raw data, highest variance col)
      chart__heatmap — column correlations (raw data, table with most numeric cols)
    """
    import pandas as pd
    import warnings

    if config is None:
        config = get_config()

    chart_dir = (
        os.path.join(config.chart_output_dir, run_id)
        if run_id
        else config.chart_output_dir
    )
    os.makedirs(chart_dir, exist_ok=True)

    template = _THEME_MAP.get(config.chart_theme.lower(), "plotly_white")
    base_layout = dict(
        template=template,
        width=_px(config.chart_width),
        height=_px(config.chart_height),
        margin=dict(l=60, r=40, t=70, b=90),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    _PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    charts: dict[str, str] = {}
    chart_data_store: dict[str, dict] = {}
    all_results = [r for r in agent_results if not r.get("error")]

    # ── 1. BAR — flagged rows per table + signal type counts (all agents) ────────
    try:
        table_counts: dict[str, int] = {}
        for r in all_results:
            t = r.get("table", "unknown")
            table_counts[t] = table_counts.get(t, 0) + 1

        sig_counts: dict[str, int] = {}
        for r in all_results:
            for sig in r.get("analysis", {}).get("all_signals", []):
                p = sig.split(":")[0] if ":" in sig else "other"
                sig_counts[p] = sig_counts.get(p, 0) + 1

        fig = go.Figure()
        if table_counts:
            t_sorted = sorted(table_counts.items(), key=lambda x: -x[1])
            fig.add_trace(go.Bar(
                name="Flagged Rows per Table",
                x=[t for t, _ in t_sorted],
                y=[v for _, v in t_sorted],
                marker_color=_PALETTE[0],
                hovertemplate="<b>%{x}</b><br>Flagged rows: %{y}<extra></extra>",
            ))
        if sig_counts:
            s_sorted = sorted(sig_counts.items(), key=lambda x: -x[1])
            fig.add_trace(go.Bar(
                name="Signal Type Count",
                x=[s for s, _ in s_sorted],
                y=[v for _, v in s_sorted],
                marker_color=_PALETTE[1],
                hovertemplate="<b>%{x}</b><br>Signals: %{y}<extra></extra>",
            ))
        fig.update_layout(
            title=dict(
                text=f"Audit Flags — {len(all_results)} Flagged Rows Across All Tables",
                font=dict(size=16, color="#1a237e"),
            ),
            barmode="group", xaxis_tickangle=-30, yaxis_title="Count",
            **base_layout,
        )
        chart_data_store["chart__bar"] = _fig_to_data(fig)
        charts["chart__bar"] = _to_html(fig)
    except Exception as exc:
        log.error("combined bar chart failed: %s", exc)

    # ── 2. PIE — all audit signals combined across every agent ───────────────────
    try:
        sig_counts2: dict[str, int] = {}
        for r in all_results:
            for sig in r.get("analysis", {}).get("all_signals", []):
                p = sig.split(":")[0] if ":" in sig else "other"
                sig_counts2[p] = sig_counts2.get(p, 0) + 1
        if sig_counts2:
            items = sorted(sig_counts2.items(), key=lambda x: -x[1])
            fig = go.Figure(go.Pie(
                labels=[i[0] for i in items],
                values=[i[1] for i in items],
                hole=0.35, textinfo="label+percent+value",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text=f"Audit Signal Distribution — {sum(sig_counts2.values())} Total Signals",
                    font=dict(size=16, color="#1a237e"),
                ),
                **base_layout,
            )
            chart_data_store["chart__pie"] = _fig_to_data(fig)
            charts["chart__pie"] = _to_html(fig)
    except Exception as exc:
        log.error("combined pie chart failed: %s", exc)

    # ── 3, 4, 5 — line / hist / heatmap from raw data ────────────────────────────
    if raw_data:
        _DT_THRESH = 0.8
        hist_cands:    list[tuple] = []
        line_cands:    list[tuple] = []
        heatmap_cands: list[tuple] = []

        for tname, df in raw_data.items():
            if df is None or df.empty:
                continue

            num_cols: list[str] = []
            for c in df.select_dtypes(include="number").columns:
                s = df[c].dropna()
                if len(s) < 5 or s.nunique() <= 2 or s.nunique() == len(s):
                    continue
                num_cols.append(c)
            num_cols.sort(
                key=lambda c: float(pd.to_numeric(df[c], errors="coerce").std() or 0),
                reverse=True,
            )

            dt_cols: list[str] = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
            for c in df.select_dtypes(include="object").columns:
                if c in dt_cols:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(df[c].dropna().head(20), errors="coerce")
                    if parsed.notna().mean() >= _DT_THRESH:
                        dt_cols.append(c)
                except Exception:
                    pass

            if num_cols:
                std = float(pd.to_numeric(df[num_cols[0]], errors="coerce").std() or 0)
                hist_cands.append((std, tname, num_cols[0], df))
            if dt_cols and num_cols:
                line_cands.append((len(df), tname, dt_cols[0], num_cols[0], df))
            if len(num_cols) >= 3:
                heatmap_cands.append((len(num_cols), tname, num_cols[:10], df))

        # 3. LINE
        if line_cands:
            _, tname, dc, nc, df = max(line_cands, key=lambda x: x[0])
            try:
                df_t = df[[dc, nc]].copy()
                df_t[nc] = pd.to_numeric(df_t[nc], errors="coerce")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_t[dc] = pd.to_datetime(df_t[dc], errors="coerce")
                df_t = df_t.dropna()
                if len(df_t) >= 3:
                    df_t["_p"] = df_t[dc].dt.to_period("M").astype(str)
                    monthly = df_t.groupby("_p")[nc].sum().sort_index()
                    fig = go.Figure(go.Scatter(
                        x=monthly.index.tolist(), y=monthly.values.tolist(),
                        mode="lines+markers",
                        line=dict(width=2, color=_PALETTE[0]), marker=dict(size=6),
                        hovertemplate=f"<b>%{{x}}</b><br>{nc} total: %{{y:,.0f}}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=dict(
                            text=f"Monthly {nc} Trend — {tname}",
                            font=dict(size=16, color="#1a237e"),
                        ),
                        xaxis_tickangle=-30, yaxis_title=f"Total {nc}",
                        xaxis_title="Period", **base_layout,
                    )
                    chart_data_store["chart__line"] = _fig_to_data(fig)
                    charts["chart__line"] = _to_html(fig)
            except Exception as exc:
                log.error("combined line chart failed: %s", exc)

        # 4. HISTOGRAM
        if hist_cands:
            _, tname, nc, df = max(hist_cands, key=lambda x: x[0])
            try:
                vals = pd.to_numeric(df[nc], errors="coerce").dropna()
                mean_val = float(vals.mean())
                fig = go.Figure(go.Histogram(
                    x=vals.tolist(), nbinsx=30, marker_color=_PALETTE[0],
                    hovertemplate=f"{nc}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                ))
                fig.add_vline(x=mean_val, line_dash="dash", line_color="#e15759",
                              annotation_text=f"Mean: {mean_val:,.2f}",
                              annotation_position="top right")
                fig.update_layout(
                    title=dict(
                        text=f"{nc} Distribution — {len(vals):,} rows ({tname})",
                        font=dict(size=16, color="#1a237e"),
                    ),
                    xaxis_title=nc, yaxis_title="Count", **base_layout,
                )
                chart_data_store["chart__hist"] = _fig_to_data(fig)
                charts["chart__hist"] = _to_html(fig)
            except Exception as exc:
                log.error("combined histogram failed: %s", exc)

        # 5. HEATMAP
        if heatmap_cands:
            _, tname, hm_cols, df = max(heatmap_cands, key=lambda x: x[0])
            try:
                corr = df[hm_cols].apply(pd.to_numeric, errors="coerce").corr()
                z    = [[round(float(corr.loc[a, b]), 4) for b in hm_cols] for a in hm_cols]
                text = [[f"{v:.2f}" for v in row] for row in z]
                fig = go.Figure(go.Heatmap(
                    x=hm_cols, y=hm_cols, z=z,
                    colorscale="RdBu", zmin=-1, zmax=1,
                    text=text, texttemplate="%{text}",
                    hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.4f}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(
                        text=f"Numeric Column Correlations — {tname}",
                        font=dict(size=16, color="#1a237e"),
                    ),
                    **base_layout,
                )
                chart_data_store["chart__heatmap"] = _fig_to_data(fig)
                charts["chart__heatmap"] = _to_html(fig)
            except Exception as exc:
                log.error("combined heatmap failed: %s", exc)

    # ── Table-specific data-insight charts (dynamic, one set per table) ──────────
    if raw_data:
        try:
            table_charts, table_data = _build_table_specific_charts(raw_data, config)
            charts.update(table_charts)
            chart_data_store.update(table_data)
            log.info("Table-specific charts added: %d", len(table_charts))
        except Exception as exc:
            log.error("Table-specific chart generation failed: %s", exc)

    for name, html in charts.items():
        try:
            with open(os.path.join(chart_dir, f"{name}.html"), "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as exc:
            log.error("Failed to save chart %s: %s", name, exc)
        if name in chart_data_store:
            try:
                with open(os.path.join(chart_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                    json.dump(chart_data_store[name], f)
            except Exception as exc:
                log.error("Failed to save chart data %s: %s", name, exc)

    log.info("Charts generated: %d total", len(charts))
    return charts


def make_all_charts_page(charts: dict[str, str]) -> str:
    """Wrap all chart HTML strings into a single scrollable HTML page using iframes."""
    sections = []
    for name, chart_html in charts.items():
        # Embed each self-contained chart HTML inside an iframe via srcdoc
        escaped = _html_lib.escape(chart_html, quote=True)
        sections.append(
            f"<h2 style='font-family:sans-serif;padding:16px 0 4px;color:#1a237e'>{name}</h2>"
            f"<iframe srcdoc=\"{escaped}\" "
            f"style='width:100%;height:550px;border:1px solid #ddd;border-radius:4px;"
            f"background:#fff;margin-bottom:24px;' frameborder='0'></iframe>"
        )

    body = "\n".join(sections)
    chart_count = len(charts)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>4sight v3 — Audit Analysis Charts</title>
  <style>
    body {{ background:#f8f9fa; padding:24px; font-family:Arial,sans-serif; }}
    h1 {{ color:#1a237e; }}
    h2 {{ color:#1a237e; }}
    iframe {{ display:block; }}
  </style>
</head>
<body>
  <h1>4sight v3 — Financial Audit Analysis Charts ({chart_count} charts)</h1>
  {body}
</body>
</html>"""
