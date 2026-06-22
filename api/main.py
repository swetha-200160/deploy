from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_config
from api.routers import pipeline, agents, charts, insights, story as story_router, query as query_router, delete as delete_router

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

config = get_config()


_MAX_STORED_RUNS = 20


def _table_checksum(db, table: str) -> int:
    """Row count — cheap proxy for detecting inserts/deletes without a full table load."""
    try:
        return db.count(table)
    except Exception:
        return 0


async def _poll_table_changes(app: FastAPI) -> None:
    """
    Background task: polls tables every 60 s using checksums (row count + value hash).
    Detects inserts, deletes, and row-level updates. Skips trigger if a pipeline is
    already running. Evicts old runs to keep pipeline_store bounded.
    """
    from db import DB
    from orchestrator import pipeline_store

    last_checksums: dict[str, int] = {}
    db = DB()
    try:
        while True:
            await asyncio.sleep(60)
            try:
                running = any(s.get("status") == "running" for s in pipeline_store.values())
                if running:
                    log.debug("Pipeline running — skipping table change poll.")
                    continue

                # Only watch the source tables that the last pipeline run actually analyzed.
                # This permanently excludes internal tables (query_sessions, feedback, etc.)
                # because the pipeline never analyzes those — no hardcoded suffix lists needed.
                last_run = max(
                    (s for s in pipeline_store.values() if s.get("tables_analyzed")),
                    key=lambda s: s.get("started_at", ""),
                    default=None,
                )
                if last_run:
                    tables = last_run["tables_analyzed"]
                else:
                    # No completed run yet — watch all prefixed tables for the very first run,
                    # but exclude internal tracking tables (query_sessions, feedback, etc.)
                    from db import INTERNAL_TABLE_SUFFIXES
                    tables = [
                        t for t in db.tables(prefix=config.table_prefix)
                        if not t.endswith(INTERNAL_TABLE_SUFFIXES)
                    ]
                current_checksums = {t: _table_checksum(db, t) for t in tables}

                if last_checksums and current_checksums != last_checksums:
                    changed = [t for t in current_checksums if current_checksums[t] != last_checksums.get(t)]
                    new_tables = [t for t in current_checksums if t not in last_checksums]
                    dropped = [t for t in last_checksums if t not in current_checksums]
                    log.info(
                        "Table changes detected — changed=%s, new=%s, dropped=%s.",
                        changed, new_tables, dropped,
                    )

                    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_auto"
                    log.info("Auto-triggering pipeline: %s", run_id)
                    orch = app.state.orchestrator
                    asyncio.create_task(asyncio.to_thread(orch.run, run_id, None))

                    # Evict oldest runs beyond the cap
                    if len(pipeline_store) > _MAX_STORED_RUNS:
                        oldest = sorted(
                            pipeline_store,
                            key=lambda k: pipeline_store[k].get("started_at", ""),
                        )[:len(pipeline_store) - _MAX_STORED_RUNS]
                        for k in oldest:
                            del pipeline_store[k]

                last_checksums = current_checksums
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Table change polling error: %s", exc)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from orchestrator import Orchestrator, load_pipeline_store
    from query_agent.faiss_store import FAISSStore
    from query_agent.query_agent import QueryAgent

    # Always attach orchestrator and restore any previous runs from disk
    app.state.orchestrator = Orchestrator(config)
    load_pipeline_store()

    # Load per-run FAISS indexes — scan for run subdirectories
    query_agents: dict[str, QueryAgent] = {}
    app.state.query_agents = query_agents
    app.state.chat_histories = {}   # { session_id: [{"role": ..., "content": ...}] }
    faiss_base = config.faiss_index_path
    if os.path.exists(faiss_base):
        for entry in sorted(os.listdir(faiss_base)):
            run_dir = os.path.join(faiss_base, entry)
            if not os.path.isdir(run_dir):
                continue
            index_file = os.path.join(run_dir, config.faiss_index_file)
            if not os.path.exists(index_file):
                continue
            try:
                store = FAISSStore(config)
                store.load(run_id=entry)
                app.state.query_agents[entry] = QueryAgent(store, config)
                log.info("FAISS index loaded for run '%s'.", entry)
            except Exception as exc:
                log.warning("Could not load FAISS index for run '%s': %s", entry, exc)
    log.info("Query agents ready: %d run(s) indexed.", len(app.state.query_agents))

    # Load shared raw index (faiss_index/raw_index.faiss) — not per-run
    raw_store = FAISSStore(config)
    if raw_store.raw_index_exists():
        try:
            raw_store.load_raw()
            log.info("Shared raw index loaded: %d raw rows.", len(raw_store.documents))
        except Exception as exc:
            log.warning("Could not load raw index: %s", exc)
    else:
        log.info("No raw index found yet — watcher will build it on first poll.")
    app.state.raw_store = raw_store

    # Start table change detection in background
    poll_task = asyncio.create_task(_poll_table_changes(app))
    log.info("Table change polling started (interval: 60s).")

    # Start raw-data watcher — keeps FAISS raw-row chunks in sync with the DB
    from query_agent.raw_data_watcher import RawDataWatcher
    watcher = RawDataWatcher(app, interval=30)
    watcher_task = asyncio.create_task(watcher.start())
    log.info("RawDataWatcher started (interval: 30s).")

    yield

    watcher.stop()
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
    log.info("4sight v3 API shutting down.")


app = FastAPI(
    title="4sight v3",
    version="3.0.0",
    lifespan=lifespan,
    root_path="/4sight",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(pipeline.router)
app.include_router(agents.router)
app.include_router(charts.router)
app.include_router(insights.router)
app.include_router(story_router.router)
app.include_router(query_router.router)
app.include_router(delete_router.router)


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "model": config.llm_model,
        "prefix": config.table_prefix,
        "version": "3.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
        reload_excludes=["reports/*", "charts/*", "faiss_index/*"],
    )
