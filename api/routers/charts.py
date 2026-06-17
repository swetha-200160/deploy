from __future__ import annotations

import json
import os
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from config import get_config
from reports.visualization.chart_generator import (
    _DISPATCH,
    make_all_charts_page,
)

router = APIRouter(prefix="/api/charts", tags=["Charts"])

# "hist" in filename ↔ "histogram" in endpoint
_TYPE_NORMALIZE = {"hist": "histogram", "histogram": "hist"}


def _chart_matches(name: str, chart_type: str, table: Optional[str], column: Optional[str]) -> bool:
    """Return True if a chart name matches the given type / table / column filters.

    Handles both naming conventions:
      New: chart__bar | chart__tablename__subtype
      Old: bar__table__col  (from agent results)
    """
    parts = name.split("__")

    if parts[0] == "chart":
        if len(parts) == 2:
            # Summary chart: chart__bar, chart__pie, chart__hist …
            file_type = _TYPE_NORMALIZE.get(parts[1], parts[1])
            file_table: Optional[str] = None
            file_subtype: Optional[str] = None
        elif len(parts) >= 3:
            # Table-specific: chart__tablename__subtype
            file_table = parts[1]
            file_subtype = "__".join(parts[2:])
            file_type = "pie" if "pie" in file_subtype.lower() else "bar"
        else:
            return False
    else:
        # Old agent-result naming: bar__table__col
        file_type = parts[0]
        file_table = parts[1] if len(parts) > 1 else None
        file_subtype = "__".join(parts[2:]) if len(parts) > 2 else None

    if file_type != chart_type and _TYPE_NORMALIZE.get(file_type) != chart_type:
        return False
    if table is not None and file_table != table:
        return False
    if column is not None and file_subtype is not None and column not in file_subtype:
        return False
    return True


def _get_state(request: Request, run_id: Optional[str]) -> dict:
    orch = request.app.state.orchestrator
    rid = run_id or orch.latest_run_id()
    if not rid:
        raise HTTPException(status_code=404, detail="No pipeline runs found.")
    state = orch.get_state(rid)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found.")
    charts_ready = bool(state.get("charts"))
    if state.get("status") not in ("done", "running") and not charts_ready:
        raise HTTPException(status_code=202, detail=f"Run status: {state.get('status')}")
    return state


def _find_chart_html(state: dict, chart_type: str, table: Optional[str], column: Optional[str]) -> Optional[str]:
    """Find chart HTML — first from in-memory state, then from disk files."""
    config = get_config()

    # 1. In-memory (available before RAM is cleared)
    for name, html in state.get("charts", {}).items():
        if _chart_matches(name, chart_type, table, column):
            return html

    # 2. Disk fallback — charts are saved to CHART_OUTPUT_DIR/{run_id}/ by chart_generator
    run_id = state.get("run_id")
    chart_dir = (
        os.path.join(config.chart_output_dir, run_id)
        if run_id and os.path.isdir(os.path.join(config.chart_output_dir, run_id))
        else config.chart_output_dir
    )
    if os.path.exists(chart_dir):
        for fname in sorted(os.listdir(chart_dir)):
            if not fname.endswith(".html"):
                continue
            name = fname[:-5]  # strip .html
            if not _chart_matches(name, chart_type, table, column):
                continue
            try:
                with open(os.path.join(chart_dir, fname), encoding="utf-8") as f:
                    return f.read()
            except Exception:
                continue
    return None


def _find_chart_data(state: dict, chart_type: str, table: Optional[str], column: Optional[str]) -> dict:
    """Find raw chart_data dict for JSON response — from RAM, results file, or disk JSON sidecars."""
    all_results = list(
        state.get("universal_results", []) +
        state.get("enterprise_row_results", []) +
        state.get("wow_row_results", [])
    )

    # Results cleared from RAM after pipeline — load from disk
    if not any(all_results):
        results_file = state.get("results_file")
        if results_file and os.path.exists(results_file):
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            for agent_type in ("universal", "enterprise_row", "wow_row"):
                all_results.extend(data.get(agent_type, []))

    for r in all_results:
        if r.get("chart_type") != chart_type:
            continue
        if table and r.get("table") != table:
            continue
        if column and r.get("column") != column:
            continue
        return r.get("chart_data", {})

    # Disk JSON sidecar fallback — saved by chart_generator alongside .html files
    config = get_config()
    run_id = state.get("run_id")
    chart_dir = (
        os.path.join(config.chart_output_dir, run_id)
        if run_id and os.path.isdir(os.path.join(config.chart_output_dir, run_id))
        else config.chart_output_dir
    )
    if os.path.exists(chart_dir):
        for fname in sorted(os.listdir(chart_dir)):
            if not fname.endswith(".json"):
                continue
            name = fname[:-5]  # strip .json
            if not _chart_matches(name, chart_type, table, column):
                continue
            try:
                with open(os.path.join(chart_dir, fname), encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {}


def _chart_endpoint(
    chart_type: str,
    table: Optional[str],
    column: Optional[str],
    request: Request,
    run_id: Optional[str],
    fmt: Literal["html", "json", "both"],
):
    state = _get_state(request, run_id)

    if fmt == "json":
        chart_data = _find_chart_data(state, chart_type, table, column)
        if not chart_data:
            raise HTTPException(status_code=404, detail=f"No '{chart_type}' chart data found.")
        return JSONResponse(content=chart_data)

    # Resolve HTML (shared by "html" and "both")
    html = _find_chart_html(state, chart_type, table, column)
    if not html:
        config = get_config()
        fn = _DISPATCH.get(chart_type)
        chart_data = _find_chart_data(state, chart_type, table, column)
        if fn and chart_data:
            title = f"{table or ''} — {column or ''} ({chart_type})"
            html = fn(chart_data, title, config)
        else:
            raise HTTPException(status_code=404, detail=f"No '{chart_type}' chart found.")

    if fmt == "both":
        chart_data = _find_chart_data(state, chart_type, table, column)
        return JSONResponse(content={"html": html, "chart_data": chart_data})

    return HTMLResponse(content=html)


def _load_all_results(state: dict) -> list:
    all_results = (
        state.get("universal_results", []) +
        state.get("enterprise_row_results", []) +
        state.get("wow_row_results", [])
    )
    if not any(all_results):
        results_file = state.get("results_file")
        if results_file and os.path.exists(results_file):
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            for agent_type in ("universal", "enterprise_row", "wow_row"):
                all_results.extend(data.get(agent_type, []))
    return all_results


def _load_all_charts_html(state: dict) -> dict:
    charts = state.get("charts", {})
    if not charts:
        config = get_config()
        run_id = state.get("run_id")
        chart_dir = (
            os.path.join(config.chart_output_dir, run_id)
            if run_id and os.path.isdir(os.path.join(config.chart_output_dir, run_id))
            else config.chart_output_dir
        )
        if os.path.exists(chart_dir):
            for fname in sorted(os.listdir(chart_dir)):
                if fname.endswith(".html"):
                    try:
                        with open(os.path.join(chart_dir, fname), encoding="utf-8") as f:
                            charts[fname[:-5]] = f.read()
                    except Exception:
                        continue
    return charts


def _load_all_charts_json(state: dict) -> list[dict]:
    """Load all chart JSON sidecars from disk — the source of truth for chart data."""
    config = get_config()
    run_id = state.get("run_id")
    chart_dir = (
        os.path.join(config.chart_output_dir, run_id)
        if run_id and os.path.isdir(os.path.join(config.chart_output_dir, run_id))
        else config.chart_output_dir
    )
    results = []
    if os.path.exists(chart_dir):
        for fname in sorted(os.listdir(chart_dir)):
            if not fname.endswith(".json"):
                continue
            name = fname[:-5]
            try:
                with open(os.path.join(chart_dir, fname), encoding="utf-8") as f:
                    chart_data = json.load(f)
                results.append({"name": name, "chart_data": chart_data})
            except Exception:
                continue
    return results


def _list_chart_filenames(state: dict) -> list[str]:
    """Return chart names from disk without reading file contents into RAM."""
    in_memory = state.get("charts", {})
    if in_memory:
        return sorted(in_memory.keys())
    config = get_config()
    run_id = state.get("run_id")
    chart_dir = (
        os.path.join(config.chart_output_dir, run_id)
        if run_id and os.path.isdir(os.path.join(config.chart_output_dir, run_id))
        else config.chart_output_dir
    )
    names = []
    if os.path.exists(chart_dir):
        for fname in sorted(os.listdir(chart_dir)):
            if fname.endswith(".html"):
                names.append(fname[:-5])
    return names


def _make_chart_index_page(chart_names: list[str], run_id: str) -> str:
    """Lightweight index page — each chart opens in a new tab via /api/charts/file."""
    links = "\n".join(
        f'<li><a href="/api/charts/file?run_id={run_id}&name={name}" target="_blank">{name}</a></li>'
        for name in chart_names
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>4sight v3 — Charts ({len(chart_names)})</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f8f9fa; padding: 24px; }}
    h1 {{ color: #1a237e; }}
    ul {{ line-height: 2; }}
    a {{ color: #1565c0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Charts — {len(chart_names)} available</h1>
  <ul>{links}</ul>
</body>
</html>"""


@router.get("/all")
async def get_all_charts(request: Request, run_id: Optional[str] = None, format: Literal["html", "json", "both"] = "html"):
    state = _get_state(request, run_id)

    if format == "json":
        charts_json = _load_all_charts_json(state)
        if not charts_json:
            raise HTTPException(status_code=404, detail="No chart data found.")
        return JSONResponse(content=charts_json)

    if format == "both":
        all_results = _load_all_results(state)
        charts_html = _load_all_charts_html(state)
        charts_json = [
            {
                "chart_type": r.get("chart_type"),
                "table": r.get("table"),
                "column": r.get("column"),
                "chart_data": r.get("chart_data", {}),
            }
            for r in all_results
            if r.get("chart_data")
        ]
        if not charts_json and not charts_html:
            raise HTTPException(status_code=404, detail="No chart data found.")
        return JSONResponse(content={
            "html": make_all_charts_page(charts_html) if charts_html else None,
            "charts": charts_json,
        })

    # HTML format — return a lightweight index page; each chart opens in a new tab
    rid = run_id or state.get("run_id", "")
    names = _list_chart_filenames(state)
    if not names:
        return HTMLResponse(content="<html><body><h2>No charts available. Run the pipeline first.</h2></body></html>")
    return HTMLResponse(content=_make_chart_index_page(names, rid))


# @router.get("/file")
# async def get_chart_file(run_id: str, name: str):
#     """Serve a single chart HTML file directly from disk."""
#     config = get_config()
#     chart_dir = (
#         os.path.join(config.chart_output_dir, run_id)
#         if os.path.isdir(os.path.join(config.chart_output_dir, run_id))
#         else config.chart_output_dir
#     )
#     safe_name = os.path.basename(name)
#     if not safe_name.endswith(".html"):
#         safe_name += ".html"
#     path = os.path.join(chart_dir, safe_name)
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail=f"Chart '{safe_name}' not found.")
#     return FileResponse(path, media_type="text/html")


@router.get("/countplot")
async def get_countplot(
    request: Request,
    table: Optional[str] = None,
    column: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("countplot", table, column, request, run_id, format)


@router.get("/bar")
async def get_bar_chart(
    request: Request,
    table: Optional[str] = None,
    column: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("bar", table, column, request, run_id, format)


@router.get("/pie")
async def get_pie_chart(
    request: Request,
    table: Optional[str] = None,
    column: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("pie", table, column, request, run_id, format)


@router.get("/stacked_bar")
async def get_stacked_bar(
    request: Request,
    table: Optional[str] = None,
    column: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("stacked_bar", table, column, request, run_id, format)


@router.get("/heatmap")
async def get_heatmap(
    request: Request,
    table: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("heatmap", table, None, request, run_id, format)


@router.get("/line")
async def get_line_chart(
    request: Request,
    table: Optional[str] = None,
    column: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("line", table, column, request, run_id, format)


@router.get("/histogram")
async def get_histogram(
    request: Request,
    table: Optional[str] = None,
    column: Optional[str] = None,
    run_id: Optional[str] = None,
    format: Literal["html", "json", "both"] = "html",
):
    return _chart_endpoint("histogram", table, column, request, run_id, format)
