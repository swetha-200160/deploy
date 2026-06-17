from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    AllAgentsResponse,
    AgentResultResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
    StepRunResponse,
    StepsStatusResponse,
    TableSchemaResponse,
)
from db import DB

class PipelineStep(str, Enum):
    load_data      = "load_data"
    route_schema   = "route_schema"
    universal      = "universal"
    enterprise_row = "enterprise_row"
    wow_row        = "wow_row"
    charts         = "charts"
    insights       = "insights"
    story          = "story"
    report         = "report"
    faiss          = "faiss"


router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

log = logging.getLogger(__name__)


def _get_orchestrator(request: Request):
    return request.app.state.orchestrator


def _resolve_run_id(request: Request, run_id: Optional[str]) -> str:
    orch = _get_orchestrator(request)
    if run_id:
        return run_id
    latest = orch.latest_run_id()
    if not latest:
        raise HTTPException(status_code=404, detail="No pipeline runs found.")
    return latest


async def _refresh_query_agent(request: Request, run_id: str | None = None) -> None:
    """Load the FAISS index for run_id and register it in app.state.query_agents."""
    try:
        from query_agent.faiss_store import FAISSStore
        from query_agent.query_agent import QueryAgent
        from config import get_config
        cfg = get_config()
        if run_id is None:
            run_id = request.app.state.orchestrator.latest_run_id()
        if run_id is None:
            return
        store = FAISSStore(cfg)
        if store.index_exists(run_id=run_id):
            store.load(run_id=run_id)
            request.app.state.query_agents[run_id] = QueryAgent(store, cfg)
            log.info("Query agent registered for run '%s'.", run_id)
    except Exception as exc:
        log.warning("Could not refresh query agent: %s", exc)


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: Request, body: PipelineRunRequest):
    import asyncio
    orch = _get_orchestrator(request)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    started_at = datetime.now().isoformat()

    tables = await asyncio.to_thread(orch.initialize, run_id, body.table_prefix)

    return PipelineRunResponse(
        run_id=run_id,
        status="initialized",
        tables_found=tables,
        started_at=started_at,
    )


@router.post("/process/{run_id}", response_model=PipelineStatusResponse)
async def process_pipeline(run_id: str, request: Request):
    import asyncio
    orch = _get_orchestrator(request)
    if not orch.get_state(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    await asyncio.to_thread(orch.process, run_id)
    await _refresh_query_agent(request, run_id)

    state = orch.get_state(run_id) or {}
    return PipelineStatusResponse(
        run_id=run_id,
        status=state.get("status", "done"),
        started_at=state.get("started_at", ""),
        completed_at=state.get("completed_at"),
        error=state.get("error"),
    )


@router.post("/resume/{run_id}", response_model=PipelineStatusResponse)
async def resume_pipeline(run_id: str, request: Request, start_from: str = "categorical"):
    """Resume a pipeline run from a specific agent, preserving earlier agents' results.

    Query param `start_from` accepts: numerical, categorical, datetime, text,
    enterprise, charts, insights, report, faiss.
    """
    import asyncio
    orch = _get_orchestrator(request)
    if not orch.get_state(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    try:
        await asyncio.to_thread(orch.process, run_id, start_from)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await _refresh_query_agent(request, run_id)

    state = orch.get_state(run_id) or {}
    return PipelineStatusResponse(
        run_id=run_id,
        status=state.get("status", "done"),
        started_at=state.get("started_at", ""),
        completed_at=state.get("completed_at"),
        error=state.get("error"),
    )


@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
async def get_status(run_id: str, request: Request):
    orch = _get_orchestrator(request)
    state = orch.get_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return PipelineStatusResponse(
        run_id=run_id,
        status=state.get("status", "unknown"),
        started_at=state.get("started_at", ""),
        completed_at=state.get("completed_at"),
        error=state.get("error"),
    )


@router.get("/results/{run_id}", response_model=AllAgentsResponse)
async def get_results(run_id: str, request: Request):
    orch = _get_orchestrator(request)
    state = orch.get_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if state.get("status") != "done":
        raise HTTPException(status_code=202, detail=f"Run '{run_id}' status: {state.get('status')}")

    universal = state.get("universal_results") or state.get("numerical_results", [])

    # Results are cleared from RAM after pipeline completes — load from disk if needed
    if not universal:
        results_file = state.get("results_file")
        if results_file and os.path.exists(results_file):
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            universal = data.get("universal", []) or data.get("numerical", [])

    def to_response(results):
        return [AgentResultResponse(**{k: v for k, v in r.items() if k != "chart_data"}) for r in results]

    return AllAgentsResponse(
        run_id=run_id,
        numerical=to_response(universal),
        categorical=[],
        datetime=[],
        text=[],
    )


@router.get("/steps/{run_id}", response_model=StepsStatusResponse)
async def get_steps(run_id: str, request: Request):
    """List all available steps and which ones have produced results so far."""
    orch = _get_orchestrator(request)
    state = orch.get_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    from orchestrator import Orchestrator
    steps = Orchestrator.GRANULAR_STEPS

    completed = {
        "load_data":      bool(state.get("raw_data") or state.get("tables_analyzed")),
        "route_schema":   bool(state.get("column_routes") or state.get("schema_info")),
        "universal":      bool(state.get("universal_results")),
        "enterprise_row": bool(state.get("enterprise_row_results")),
        "wow_row":        bool(state.get("wow_row_results")),
        "charts":         bool(state.get("charts")),
        "insights":       bool(state.get("insights")),
        "story":          bool(state.get("story")),
        "report":         bool(state.get("report")),
        "faiss":          state.get("status") == "done",
    }

    return StepsStatusResponse(
        run_id=run_id,
        available_steps=steps,
        completed_steps=completed,
    )


@router.post("/step/{run_id}", response_model=StepRunResponse)
async def run_single_step(run_id: str, request: Request, step: PipelineStep = PipelineStep.universal):
    """Run exactly one pipeline step and return a summary of its output.

    Use this for step-by-step testing: run a step, inspect the result via
    GET /results or GET /steps, fix any issues, then call this again with the
    next step name.

    Valid step names (in order):
      load_data, route_schema, universal, enterprise,
      wow_agents, charts, insights, story, report, faiss
    """
    import asyncio
    orch = _get_orchestrator(request)
    if not orch.get_state(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    try:
        state = await asyncio.to_thread(orch.process_single_step, run_id, step)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Build a lightweight result summary so the caller can see what was produced
    def _agent_summary(results: list) -> dict:
        return {
            "count": len(results),
            "results": [
                {
                    "table":         r.get("table", ""),
                    "column":        r.get("column", ""),
                    "chart_type":    r.get("chart_type", ""),
                    "insights_text": r.get("insights_text", ""),
                }
                for r in results
            ],
        }

    summary_keys = {
        "universal":      ("universal_results",      _agent_summary),
        "enterprise_row": ("enterprise_row_results", _agent_summary),
        "wow_row":        ("wow_row_results",         _agent_summary),
        "charts":         ("charts",                 lambda v: {"count": len(v)}),
        "insights":       ("insights",               lambda v: v),
        "story":          ("story",                  lambda v: {"keys": list(v.keys())}),
        "report":         ("report",                 lambda v: {"keys": list(v.keys())}),
        "load_data":      ("tables_analyzed",        lambda v: {"tables": v or []}),
        "route_schema":   ("schema_info",            lambda v: {"tables": list(v.keys())}),
        "faiss":          (None,                     lambda _: {"indexed": True}),
    }

    result_summary: dict = {}
    if step in summary_keys:
        key, summariser = summary_keys[step]
        val = state.get(key) if key else None
        result_summary = summariser(val) if val is not None else {}

    if state.get("status") == "failed":
        return StepRunResponse(
            run_id=run_id,
            step=step,
            status="failed",
            result_summary=result_summary,
            error=state.get("error"),
        )

    if step == "faiss":
        await _refresh_query_agent(request, run_id)

    return StepRunResponse(
        run_id=run_id,
        step=step,
        status="step_done",
        result_summary=result_summary,
    )


@router.get("/tables", response_model=list[str])
async def list_tables():
    return DB().tables()


@router.get("/schema", response_model=TableSchemaResponse)
async def get_schema(request: Request):
    orch = _get_orchestrator(request)
    run_id = orch.latest_run_id()
    if run_id:
        state = orch.get_state(run_id)
        schema = state.get("schema_info", {}) if state else {}
    else:
        from core.schema_router import classify_all, build_schema_info
        routes = classify_all()
        schema = build_schema_info(routes)
    return TableSchemaResponse(tables=schema)
