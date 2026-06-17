from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from api.schemas import InsightsResponse

router = APIRouter(prefix="/api", tags=["Insights"])


def _get_state(request: Request, run_id: Optional[str]) -> dict:
    orch = request.app.state.orchestrator
    rid = run_id or orch.latest_run_id()
    if not rid:
        raise HTTPException(status_code=404, detail="No pipeline runs found.")
    state = orch.get_state(rid)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found.")
    if state.get("status") not in ("done", "running"):
        raise HTTPException(status_code=202, detail=f"Run status: {state.get('status')}")
    return state


@router.get("/insights", response_model=InsightsResponse)
async def get_insights(request: Request, run_id: Optional[str] = None):
    state = _get_state(request, run_id)
    insights = state.get("insights", {})
    if not insights:
        status = state.get("status", "unknown")
        code = 202 if status == "running" else 404
        raise HTTPException(status_code=code, detail="Insights not ready yet — pipeline still running." if status == "running" else "No insights available. Run the pipeline first.")
    narrative = insights.get("overall_narrative", "")
    if isinstance(narrative, list):
        narrative = " ".join(narrative)

    return InsightsResponse(
        key_findings=insights.get("key_findings", []),
        risk_signals=insights.get("risk_signals", []),
        insights=insights.get("insights", {}),
        recommendations=insights.get("recommendations", []),
        data_quality_score=int(insights.get("data_quality_score", 0)),
        overall_narrative=narrative,
    )


@router.get("/report", response_class=HTMLResponse)
async def get_report_html(request: Request, run_id: Optional[str] = None):
    state = _get_state(request, run_id)
    report = state.get("report", {})
    html_path = report.get("_html_path")
    if html_path and os.path.exists(html_path):
        return FileResponse(path=html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="HTML report not found. Run the pipeline first.")


@router.get("/report/json")
async def get_report_json(request: Request, run_id: Optional[str] = None):
    state = _get_state(request, run_id)
    report = state.get("report", {})

    # Serve directly from saved file — avoids loading full report into RAM
    # and prevents Swagger UI from freezing on large responses
    json_path = report.get("_json_path")
    if json_path and os.path.exists(json_path):
        return FileResponse(path=json_path, media_type="application/json")

    if not report:
        raise HTTPException(status_code=404, detail="No report available.")
    return {k: v for k, v in report.items() if not k.startswith("_")}


@router.get("/report/download")
async def download_report(request: Request, run_id: Optional[str] = None):
    state = _get_state(request, run_id)
    report = state.get("report", {})
    html_path = report.get("_html_path")
    if html_path and os.path.exists(html_path):
        return FileResponse(
            path=html_path,
            filename=f"{state.get('run_id', 'report')}.html",
            media_type="text/html",
        )
    raise HTTPException(status_code=404, detail="Report file not found.")


@router.get("/reports")
async def list_saved_reports():
    """List all saved pipeline runs that have a report JSON on disk."""
    from config import get_config
    cfg = get_config()
    report_dir = cfg.report_output_dir
    runs = []
    if os.path.isdir(report_dir):
        for entry in sorted(os.listdir(report_dir)):
            json_file = os.path.join(report_dir, entry, f"{entry}.json")
            html_file = os.path.join(report_dir, entry, f"{entry}.html")
            if os.path.isfile(json_file):
                runs.append({
                    "run_id":    entry,
                    "json_path": json_file,
                    "html_path": html_file if os.path.isfile(html_file) else None,
                    "size_kb":   round(os.path.getsize(json_file) / 1024, 1),
                })
    return {"saved_reports": runs, "count": len(runs)}


@router.get("/report/file/{run_id}")
async def get_report_file(run_id: str):
    """Load a saved report JSON directly from disk for any run_id.

    Use this when the pipeline state is not in memory (e.g. after restart)
    or to avoid Swagger UI freezing on large in-memory responses.
    """
    from config import get_config
    cfg = get_config()
    json_path = os.path.join(cfg.report_output_dir, run_id, f"{run_id}.json")
    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail=f"No saved report found for run '{run_id}'.")
    return FileResponse(path=json_path, media_type="application/json")
