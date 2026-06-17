from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from api.schemas import ChatRequest, ChatResponse, FeedbackRequest, QueryRequest, QueryResponse

router = APIRouter(prefix="/api/query", tags=["Query Agent"])


def _get_query_agent(request: Request, run_id: str | None) -> tuple:
    """Return (QueryAgent, resolved_run_id) for the given run_id (or latest if None).

    Lazily loads from disk if the run exists but wasn't loaded at startup.
    """
    from config import get_config
    from query_agent.faiss_store import FAISSStore
    from query_agent.query_agent import QueryAgent

    agents: dict = request.app.state.query_agents
    orch = request.app.state.orchestrator

    # Resolve run_id — default to latest finished run
    if run_id is None:
        if agents:
            run_id = max(agents.keys())
        else:
            run_id = orch.latest_run_id()

    if run_id is None:
        raise HTTPException(status_code=503, detail="No pipeline runs found. Run the pipeline first.")

    # Return cached agent
    if run_id in agents:
        agent = agents[run_id]
        # Always inject the current raw_store from app state — single attachment point.
        # This covers: startup loads, lazy-loads, and cache hits without repeating logic.
        agent.raw_store = request.app.state.raw_store
        return agent, run_id

    # Lazy-load from disk if the FAISS index exists for this run
    cfg = get_config()
    store = FAISSStore(cfg)
    if store.index_exists(run_id=run_id):
        try:
            store.load(run_id=run_id)
            agent = QueryAgent(store, cfg)
            agents[run_id] = agent
            # raw_store will be injected on the next line via the cache-hit path above
            agent.raw_store = request.app.state.raw_store
            return agent, run_id
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not load FAISS index for run '{run_id}': {exc}")

    raise HTTPException(
        status_code=404,
        detail=f"No FAISS index found for run '{run_id}'. The pipeline may still be running or the run_id is invalid.",
    )


@router.post("", response_model=QueryResponse)
async def submit_query(request: Request, body: QueryRequest):
    query_agent, resolved_run_id = _get_query_agent(request, body.run_id)

    session_id = body.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    result = query_agent.process_query(body.query, session_id)

    return QueryResponse(
        session_id=session_id,
        run_id=resolved_run_id,
        query=result["query"],
        preprocessed_query=result.get("preprocessed_query"),
        answer=result["answer"],
        confidence=result["confidence"],
        resolved=result["resolved"],
        query_intent=result["query_intent"],
        top_similarity=result["top_similarity"],
        num_docs_retrieved=result["num_docs_retrieved"],
    )


@router.post("/feedback")
async def submit_feedback(request: Request, body: FeedbackRequest, run_id: Optional[str] = None):
    query_agent, _ = _get_query_agent(request, run_id)
    if query_agent and query_agent.feedback_tracker:
        feedback_str = "helpful" if body.helpful else "not_helpful"
        query_agent.feedback_tracker.log_feedback(body.session_id, feedback_str)
        return {"success": True, "session_id": body.session_id}
    return {"success": False, "detail": "Feedback tracker not available."}


@router.get("/sessions", response_model=list[dict])
async def list_sessions(request: Request, run_id: Optional[str] = None, limit: int = 100):
    query_agent, _ = _get_query_agent(request, run_id)
    if query_agent and query_agent.feedback_tracker:
        return query_agent.feedback_tracker.get_all_sessions(limit=limit)
    return []


_CHAT_HISTORY_MAX = 8   # keep last 8 messages (4 user+assistant exchanges)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Multi-turn audit chatbot.

    Frontend sends only { query, session_id, run_id }.
    Backend manages conversation history — last 8 messages kept per session.
    """
    query_agent, resolved_run_id = _get_query_agent(request, body.run_id)

    session_id = body.session_id or f"sess_{uuid.uuid4().hex[:8]}"

    # Load existing history for this session from server-side store
    histories: dict = request.app.state.chat_histories
    history = histories.get(session_id, [])

    result = query_agent.process_query(body.query, session_id, chat_history=history)

    # Save updated history back, capped at last 8 messages
    updated_history = result["chat_history"][-_CHAT_HISTORY_MAX:]
    histories[session_id] = updated_history

    return ChatResponse(
        session_id=session_id,
        run_id=resolved_run_id,
        query=result["query"],
        preprocessed_query=result.get("preprocessed_query"),
        answer=result["answer"],
        history=[{"role": t["role"], "content": t["content"]} for t in updated_history],
        confidence=result["confidence"],
        resolved=result["resolved"],
        query_intent=result["query_intent"],
        num_docs_retrieved=result["num_docs_retrieved"],
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request, run_id: Optional[str] = None):
    query_agent, _ = _get_query_agent(request, run_id)
    if query_agent and query_agent.feedback_tracker:
        session = query_agent.feedback_tracker.get_session(session_id)
        if session:
            return session
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
