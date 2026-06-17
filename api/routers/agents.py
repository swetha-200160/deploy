from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from api.schemas import AgentResultResponse

router = APIRouter(prefix="/api/agents", tags=["Agents"])


def _load_results_if_needed(state: dict) -> dict:
    """Load agent results from disk if they were cleared from RAM after pipeline."""
    if any([
        state.get("universal_results"),
        state.get("enterprise_row_results"),
        state.get("wow_row_results"),
    ]):
        return state
    results_file = state.get("results_file")
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            state = dict(state)
            state["universal_results"]      = data.get("universal", [])
            state["enterprise_row_results"] = data.get("enterprise_row", [])
            state["wow_row_results"]        = data.get("wow_row", [])
        except Exception:
            pass
    return state


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


def _to_responses(results: list) -> list[AgentResultResponse]:
    return [AgentResultResponse(**{k: v for k, v in r.items() if k != "chart_data"}) for r in results]


@router.get("/universal", response_model=list[AgentResultResponse])
async def get_universal(request: Request, run_id: Optional[str] = None):
    """All results from the UniversalRowAgent (covers numerical, categorical, datetime, text)."""
    state = _load_results_if_needed(_get_state(request, run_id))
    return _to_responses(state.get("universal_results", []))


@router.get("/enterprise", response_model=list[AgentResultResponse])
async def get_enterprise(request: Request, run_id: Optional[str] = None):
    """All results from the EnterpriseRowAgent (multi-column, cross-table patterns)."""
    state = _load_results_if_needed(_get_state(request, run_id))
    return _to_responses(state.get("enterprise_row_results", []))


@router.get("/wow", response_model=list[AgentResultResponse])
async def get_wow(request: Request, run_id: Optional[str] = None):
    """All results from the WowRowAgent (week-over-week anomalies and surprises)."""
    state = _load_results_if_needed(_get_state(request, run_id))
    return _to_responses(state.get("wow_row_results", []))


@router.get("/{agent_name}/table/{table_name}", response_model=list[AgentResultResponse])
async def get_agent_results_for_table(
    agent_name: str,
    table_name: str,
    request: Request,
    run_id: Optional[str] = None,
):
    """Filter results for a specific agent and table.

    agent_name must be one of: universal, enterprise, wow
    """
    key_map = {
        "universal":  "universal_results",
        "enterprise": "enterprise_row_results",
        "wow":        "wow_row_results",
    }
    key = key_map.get(agent_name)
    if not key:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{agent_name}'. Valid values: {list(key_map)}",
        )
    state = _load_results_if_needed(_get_state(request, run_id))
    results = state.get(key, [])
    filtered = [r for r in results if r.get("table") == table_name]
    return _to_responses(filtered)
