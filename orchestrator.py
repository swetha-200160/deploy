from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from config import Config, get_config
from core import schema_router
from core.groq_rate_limiter import reset_token_usage, log_token_summary

from agents.universal_row_agent import UniversalRowAgent
from agents.enterprise_row_agent import EnterpriseRowAgent
from agents.wow_row_agent import WowRowAgent
from reports.visualization.chart_generator import generate_all_charts
from output.insight_agent import InsightAgent
from output.story_agent import StoryAgent
from output.reporting_agent import ReportingAgent
from query_agent.faiss_store import FAISSStore
from db import DB

log = logging.getLogger(__name__)

# In-memory store for pipeline runs: { run_id: PipelineState }
pipeline_store: dict[str, dict] = {}

_STORE_FILENAME = "pipeline_store.json"
# Keys that hold large in-memory objects — excluded from disk persistence
_VOLATILE_KEYS = {"raw_data", "column_routes"}


def _store_path() -> str:
    from config import get_config
    cfg = get_config()
    return os.path.join(cfg.report_output_dir, _STORE_FILENAME)


def _persist_store() -> None:
    """Save pipeline_store metadata to disk so runs survive restarts."""
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {
            run_id: {k: v for k, v in state.items() if k not in _VOLATILE_KEYS}
            for run_id, state in pipeline_store.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, default=str, indent=2)
    except Exception as exc:
        log.warning("Could not persist pipeline_store: %s", exc)


def load_pipeline_store() -> None:
    """Restore previously persisted pipeline_store from disk on startup."""
    path = _store_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pipeline_store.update(data)
        log.info("Restored %d pipeline run(s) from disk.", len(data))
    except Exception as exc:
        log.warning("Could not load pipeline_store: %s", exc)


class Orchestrator:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._graph = self._build_graph()

    # ── Nodes ──────────────────────────────────────────────────────────────────

    def _load_data_node(self, state: dict) -> dict:
        log.info("Loading data from database...")
        try:
            db = DB()
            state["raw_data"] = db.load_all(prefix=state.get("table_prefix") or self.config.table_prefix)
            # Save table names before DataFrames are cleared later
            state["tables_analyzed"] = list(state["raw_data"].keys())
            log.info("Loaded %d tables", len(state["raw_data"]))
        except Exception as exc:
            log.error("load_data_node failed: %s", exc)
            state["status"] = "failed"
            state["error"] = str(exc)
        return state

    def _route_schema_node(self, state: dict) -> dict:
        log.info("Classifying column schema...")
        try:
            routes = schema_router.classify_all(self.config, preloaded=state.get("raw_data"))
            state["column_routes"] = routes
            state["schema_info"] = schema_router.build_schema_info(routes)
            log.info("Schema routing: %d columns classified", len(routes))
        except Exception as exc:
            log.error("route_schema_node failed: %s", exc)
            state["status"] = "failed"
            state["error"] = str(exc)
        return state

    @staticmethod
    def _filter_data(raw_data: dict, routes: list, column_type: str) -> tuple[dict, list]:
        """Return only the tables and routes needed for a specific agent type."""
        type_routes = [r for r in routes if r["detected_type"] == column_type]
        needed_tables = {r["table"] for r in type_routes}
        filtered_data = {t: df for t, df in raw_data.items() if t in needed_tables}
        return filtered_data, type_routes

    def _run_agents_parallel_node(self, state: dict) -> dict:
        """Run UniversalRowAgent, EnterpriseRowAgent, and WowRowAgent concurrently."""
        log.info("Running agents in parallel (universal + enterprise_row + wow_row)...")
        raw_data = state.get("raw_data", {})
        routes   = state.get("column_routes", [])

        def _call(agent_type: str, fn):
            try:
                result = fn()
                log.info("Agent '%s' done: %d result(s)", agent_type, len(result))
                return agent_type, result, None
            except Exception as exc:
                log.error("Agent '%s' failed: %s", agent_type, exc)
                return agent_type, None, str(exc)

        tasks = [
            ("universal",      lambda: UniversalRowAgent(self.config).run(raw_data, routes)),
            ("enterprise_row", lambda: EnterpriseRowAgent(self.config).run(raw_data)),
            ("wow_row",        lambda: WowRowAgent(self.config).run(raw_data, routes)),
        ]

        run_id = state.get("run_id")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_call, name, fn): name for name, fn in tasks}
            for future in concurrent.futures.as_completed(futures):
                agent_type, result, error = future.result()
                state[f"{agent_type}_results"] = [] if error else result
                # Immediately publish partial results so the API can serve them
                if run_id and run_id in pipeline_store:
                    pipeline_store[run_id][f"{agent_type}_results"] = state[f"{agent_type}_results"]
                    log.info("Agent '%s' results published to pipeline_store (%d)", agent_type, len(state[f"{agent_type}_results"]))

        log.info(
            "All agents complete — universal=%d, enterprise_row=%d, wow_row=%d",
            len(state.get("universal_results", [])),
            len(state.get("enterprise_row_results", [])),
            len(state.get("wow_row_results", [])),
        )
        return state

    # ── Individual agent nodes (for step-by-step testing) ─────────────────────

    def _run_universal_node(self, state: dict) -> dict:
        """Step-by-step testing entry point for UniversalRowAgent."""
        log.info("Running universal row agent...")
        try:
            results = UniversalRowAgent(self.config).run(
                state.get("raw_data", {}),
                state.get("column_routes", []),
            )
            state["universal_results"] = results
            log.info("Universal agent done: %d result(s)", len(results))
        except Exception as exc:
            log.error("universal agent failed: %s", exc)
            state["universal_results"] = []
        return state

    def _run_enterprise_row_node(self, state: dict) -> dict:
        log.info("Running enterprise row agent...")
        try:
            results = EnterpriseRowAgent(self.config).run(state.get("raw_data", {}))
            state["enterprise_row_results"] = results
            log.info("Enterprise row agent done: %d result(s)", len(results))
        except Exception as exc:
            log.error("enterprise_row agent failed: %s", exc)
            state["enterprise_row_results"] = []
        return state

    def _run_wow_row_node(self, state: dict) -> dict:
        log.info("Running WoW row agent...")
        try:
            results = WowRowAgent(self.config).run(
                state.get("raw_data", {}),
                state.get("column_routes", []),
            )
            state["wow_row_results"] = results
            log.info("WoW row agent done: %d result(s)", len(results))
        except Exception as exc:
            log.error("wow_row agent failed: %s", exc)
            state["wow_row_results"] = []
        return state

    # Required fields every AgentResult must have (from core.state.AgentResult)
    _REQUIRED_RESULT_FIELDS = {"agent_name", "table", "column", "analysis", "insights_text"}
    _MAX_AGENT_RETRIES = 2

    def _validate_agent_results_node(self, state: dict) -> dict:
        """Quality-gate between agents and downstream nodes.

        Checks:
        1. At least one agent returned non-empty results.
        2. Every result contains the required AgentResult fields.

        If validation fails, retries the agent run up to _MAX_AGENT_RETRIES times.
        Sets state["validation_score"] (0.0–1.0) and state["validation_warnings"]
        for observability regardless of outcome.
        """
        _MAX_RETRIES = self.__class__._MAX_AGENT_RETRIES

        for attempt in range(1, _MAX_RETRIES + 2):  # attempts: 1 .. MAX_RETRIES+1
            universal   = state.get("universal_results",      [])
            enterprise  = state.get("enterprise_row_results", [])
            wow         = state.get("wow_row_results",        [])
            all_results = universal + enterprise + wow

            warnings: list[str] = []

            # ── Check 1: at least one agent returned data ─────────────────────
            empty_agents = []
            if not universal:
                empty_agents.append("universal")
            if not enterprise:
                empty_agents.append("enterprise_row")
            if not wow:
                empty_agents.append("wow_row")

            if empty_agents:
                warnings.append(f"Empty results from: {', '.join(empty_agents)}")

            # ── Check 2: required fields present in every result ──────────────
            malformed = 0
            for r in all_results:
                missing = self._REQUIRED_RESULT_FIELDS - set(r.keys())
                if missing:
                    malformed += 1
                    warnings.append(
                        f"Result missing fields {missing} "
                        f"(agent={r.get('agent_name','?')}, table={r.get('table','?')})"
                    )

            # ── Score ─────────────────────────────────────────────────────────
            total = len(all_results) or 1
            score = round(1.0 - (malformed / total) - (len(empty_agents) * 0.1), 2)
            score = max(0.0, min(1.0, score))

            passed = (len(empty_agents) < 3) and (malformed == 0)

            log.info(
                "Agent validation (attempt %d/%d): score=%.2f, passed=%s, warnings=%d",
                attempt, _MAX_RETRIES + 1, score, passed, len(warnings),
            )
            for w in warnings:
                log.warning("  [validator] %s", w)

            state["validation_score"]    = score
            state["validation_warnings"] = warnings

            if passed:
                log.info("Agent validation PASSED (score=%.2f)", score)
                break

            # ── Retry ─────────────────────────────────────────────────────────
            if attempt <= _MAX_RETRIES:
                log.warning(
                    "Agent validation FAILED — retrying agents (attempt %d/%d)...",
                    attempt, _MAX_RETRIES,
                )
                # Ensure raw_data is available for the retry
                if not state.get("raw_data"):
                    state = self._load_data_node(state)
                    if not state.get("column_routes"):
                        state = self._route_schema_node(state)
                state = self._run_agents_parallel_node(state)
            else:
                log.error(
                    "Agent validation FAILED after %d attempt(s) — "
                    "proceeding with partial results (score=%.2f).",
                    _MAX_RETRIES + 1, score,
                )

        return state

    def _generate_charts_node(self, state: dict) -> dict:
        log.info("Generating charts...")
        try:
            all_results = (
                state.get("universal_results", []) +
                state.get("enterprise_row_results", []) +
                state.get("wow_row_results", [])
            )
            state["charts"] = generate_all_charts(all_results, self.config, raw_data=state.get("raw_data"), run_id=state.get("run_id"))
            log.info("Charts generated: %d", len(state["charts"]))
        except Exception as exc:
            log.error("generate_charts_node failed: %s", exc)
            state["charts"] = {}

        # Save total row count before freeing DataFrames — used by StoryAgent later.
        # tables_analyzed was already saved in load_data_node.
        state["total_rows_analyzed"] = sum(
            len(df) for df in state.get("raw_data", {}).values()
        )
        # Free DataFrames from RAM — no longer needed after all agents and charts are done.
        state.pop("raw_data", None)
        return state

    def _generate_insights_node(self, state: dict) -> dict:
        log.info("Generating executive insights...")
        try:
            # Build authoritative table row counts from raw_data (already loaded from DB).
            # These are passed to InsightAgent as ground truth so total_flagged per table
            # is always the exact DB row count — independent of AgentResult metadata.
            # raw_data is popped after charts but insights runs before that — so it is
            # available here. If raw_data was already cleared (resume scenario), fall back
            # to tables_analyzed with no counts (AgentResult dedup takes over).
            raw_data = state.get("raw_data", {})
            table_row_counts = {tbl: len(df) for tbl, df in raw_data.items()} if raw_data else {}

            agent = InsightAgent(self.config)
            state["insights"] = agent.run(
                state.get("universal_results", []),
                state.get("enterprise_row_results", []),
                state.get("wow_row_results", []),
                table_row_counts=table_row_counts,
            )
        except Exception as exc:
            log.error("generate_insights_node failed: %s", exc)
            state["insights"] = {}
        return state

    def _generate_story_node(self, state: dict) -> dict:
        log.info("Generating data story narrative...")
        try:
            agent = StoryAgent(self.config)
            state["story"] = agent.run(state)
        except Exception as exc:
            log.error("generate_story_node failed: %s", exc)
            state["story"] = {}
        return state

    def _generate_report_node(self, state: dict) -> dict:
        log.info("Building report...")
        try:
            agent = ReportingAgent(self.config)
            state["report"] = agent.build(state)
        except Exception as exc:
            log.error("generate_report_node failed: %s", exc, exc_info=True)
            state["report"] = {}
        return state

    def _index_faiss_node(self, state: dict) -> dict:
        log.info("Indexing analysis results into FAISS...")
        _run_id = state.get("run_id", "run")
        try:
            all_results = (
                state.get("universal_results", []) +
                state.get("enterprise_row_results", []) +
                state.get("wow_row_results", [])
            )
            store = FAISSStore(self.config)
            store.index_analysis_results(all_results, state.get("insights", {}), run_id=_run_id)
            state["status"] = "done"
            state["completed_at"] = datetime.now().isoformat()
        except Exception as exc:
            log.error("index_faiss_node failed: %s", exc, exc_info=True)
            state["status"] = "done"  # Still mark done — FAISS index is non-critical
            state["completed_at"] = datetime.now().isoformat()

        # Persist agent results to disk, then free from RAM.
        # The pipeline_store only needs metadata after this point; results load from disk on demand.
        try:
            _run_dir = os.path.join(self.config.report_output_dir, _run_id)
            os.makedirs(_run_dir, exist_ok=True)
            results_file = os.path.join(_run_dir, f"{_run_id}_results.json")
            results_payload = {
                "universal":      state.get("universal_results", []),
                "enterprise_row": state.get("enterprise_row_results", []),
                "wow_row":        state.get("wow_row_results", []),
            }
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_payload, f, default=str)
            state["results_file"] = results_file
            log.info("Agent results saved to disk: %s", results_file)
        except Exception as exc:
            log.error("Could not save results to disk: %s", exc)

        # Free large objects from RAM — charts are on disk in CHART_OUTPUT_DIR,
        # agent results are on disk in results_file above.
        state["universal_results"]      = []
        state["enterprise_row_results"] = []
        state["wow_row_results"]        = []
        state["charts"]                 = {}
        return state

    # ── Graph ──────────────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(dict)

        graph.add_node("load_data",            self._load_data_node)
        graph.add_node("route_schema",         self._route_schema_node)
        graph.add_node("run_agents",           self._run_agents_parallel_node)
        graph.add_node("validate_agents",      self._validate_agent_results_node)
        graph.add_node("generate_charts",      self._generate_charts_node)
        graph.add_node("generate_insights",    self._generate_insights_node)
        graph.add_node("generate_story",       self._generate_story_node)
        graph.add_node("generate_report",      self._generate_report_node)
        graph.add_node("index_faiss",          self._index_faiss_node)

        graph.set_entry_point("load_data")
        graph.add_edge("load_data",            "route_schema")
        graph.add_edge("route_schema",         "run_agents")
        graph.add_edge("run_agents",           "validate_agents")
        graph.add_edge("validate_agents",      "generate_charts")
        graph.add_edge("generate_charts",      "generate_insights")
        graph.add_edge("generate_insights",  "generate_story")
        graph.add_edge("generate_story",     "generate_report")
        graph.add_edge("generate_report",    "index_faiss")
        graph.add_edge("index_faiss",        END)

        return graph.compile()

    # ── Public ─────────────────────────────────────────────────────────────────

    def initialize(self, run_id: str, table_prefix: str | None = None) -> list[str]:
        """Load DB tables and classify schema. Returns list of table names."""
        prefix = table_prefix or self.config.table_prefix
        state: dict = {
            "run_id": run_id,
            "table_prefix": prefix,
            "raw_data": {},
            "schema_info": {},
            "column_routes": [],
            "universal_results":      [],
            "enterprise_row_results": [],
            "wow_row_results":        [],
            "charts": {},
            "insights": {},
            "story": {},
            "report": {},
            "status": "initialized",
            "started_at": datetime.now().isoformat(),
            "error": None,
        }
        pipeline_store[run_id] = state

        state = self._load_data_node(state)
        if state.get("status") != "failed":
            state = self._route_schema_node(state)
        pipeline_store[run_id] = state
        _persist_store()
        return list(state.get("raw_data", {}).keys())

    # Ordered list of step names — used for start_from resume logic
    _AGENT_STEPS = ["agents", "validate_agents", "charts", "insights", "report", "faiss"]

    # Granular steps for one-at-a-time testing mode
    # Each step name maps to the node function that handles it.
    # "load_data" and "route_schema" are handled by initialize() but included here
    # so callers can query the full ordered list.
    GRANULAR_STEPS = [
        "load_data",
        "route_schema",
        "universal",
        "enterprise_row",
        "wow_row",
        "validate_agents",
        "charts",
        "insights",
        "story",
        "report",
        "faiss",
    ]

    def process(self, run_id: str, start_from: str | None = None) -> None:
        """Run all agents and output steps for an initialized pipeline run.

        Args:
            run_id: The pipeline run to process.
            start_from: Optional step name to resume from (skips earlier steps).
                        Valid values: agents, charts, insights, report, faiss.
        """
        state = pipeline_store.get(run_id)
        if not state:
            raise ValueError(f"Run '{run_id}' not found.")

        if not state.get("raw_data"):
            state = self._load_data_node(state)
            state = self._route_schema_node(state)

        state["status"] = "running"
        pipeline_store[run_id] = state
        reset_token_usage()

        all_steps = [
            ("agents",          self._run_agents_parallel_node),
            ("validate_agents", self._validate_agent_results_node),
            ("charts",          self._generate_charts_node),
            ("insights",        self._generate_insights_node),
            ("story",           self._generate_story_node),
            ("report",          self._generate_report_node),
            ("faiss",           self._index_faiss_node),
        ]

        if start_from:
            step_names = [name for name, _ in all_steps]
            if start_from not in step_names:
                raise ValueError(
                    f"Unknown start_from '{start_from}'. Valid values: {step_names}"
                )
            skip_until = step_names.index(start_from)
            steps = all_steps[skip_until:]
            log.info("Resuming run %s from '%s' (skipping %d earlier steps)", run_id, start_from, skip_until)
        else:
            steps = all_steps

        try:
            for _, node_fn in steps:
                state = node_fn(state)
                pipeline_store[run_id] = state
                if state.get("status") == "failed":
                    break
        except Exception as exc:
            log.error("Pipeline process %s failed: %s", run_id, exc)
            pipeline_store[run_id]["status"] = "failed"
            pipeline_store[run_id]["error"] = str(exc)
            pipeline_store[run_id]["completed_at"] = datetime.now().isoformat()
        finally:
            log_token_summary(label=run_id)
            _persist_store()

    def process_single_step(self, run_id: str, step: str) -> dict:
        """Run exactly one named step and return the updated state.

        Useful for step-by-step testing: run a step, inspect results, fix if
        needed, then run the next step.

        Valid step names (in order): load_data, route_schema, numerical,
        categorical, datetime, text, enterprise, wow_agents, charts, insights,
        story, report, faiss.

        "load_data" and "route_schema" re-run the initialize phase for that step.
        All other steps require the run to have been initialized first
        (i.e. raw_data and column_routes must be present in state).
        """
        if step not in self.GRANULAR_STEPS:
            raise ValueError(
                f"Unknown step '{step}'. Valid steps: {self.GRANULAR_STEPS}"
            )

        state = pipeline_store.get(run_id)
        if not state:
            raise ValueError(f"Run '{run_id}' not found. Call /run first.")

        # Ensure raw_data is available for agent steps
        agent_steps = {"universal", "enterprise_row", "wow_row", "validate_agents", "charts"}
        if step in agent_steps and not state.get("raw_data"):
            log.info("raw_data missing — reloading for step '%s'", step)
            state = self._load_data_node(state)
            if not state.get("column_routes"):
                state = self._route_schema_node(state)
            pipeline_store[run_id] = state

        node_map = {
            "load_data":       self._load_data_node,
            "route_schema":    self._route_schema_node,
            "universal":       self._run_universal_node,
            "enterprise_row":  self._run_enterprise_row_node,
            "wow_row":         self._run_wow_row_node,
            "validate_agents": self._validate_agent_results_node,
            "charts":          self._generate_charts_node,
            "insights":        self._generate_insights_node,
            "story":           self._generate_story_node,
            "report":          self._generate_report_node,
            "faiss":           self._index_faiss_node,
        }

        log.info("Running single step '%s' for run %s", step, run_id)
        state = node_map[step](state)
        pipeline_store[run_id] = state
        _persist_store()
        return state

    def get_step_result(self, run_id: str, step: str) -> Any:
        """Return the output data for a specific completed step."""
        state = pipeline_store.get(run_id)
        if not state:
            return None
        result_keys = {
            "universal":      "universal_results",
            "enterprise_row": "enterprise_row_results",
            "wow_row":        "wow_row_results",
            "charts":     "charts",
            "insights":   "insights",
            "story":      "story",
            "report":     "report",
            "schema":     "schema_info",
        }
        key = result_keys.get(step)
        return state.get(key) if key else None

    def run(self, run_id: str, table_prefix: str | None = None) -> None:
        """Execute the full pipeline. Updates pipeline_store[run_id] in place."""
        self.initialize(run_id, table_prefix)
        self.process(run_id)

    def get_state(self, run_id: str) -> dict | None:
        return pipeline_store.get(run_id)

    def list_runs(self) -> list[str]:
        return list(pipeline_store.keys())

    def latest_run_id(self) -> str | None:
        if not pipeline_store:
            return None
        return max(pipeline_store.keys(), key=lambda k: pipeline_store[k].get("started_at", ""))
