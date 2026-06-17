from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class ColumnRoute(TypedDict):
    table: str
    column: str
    detected_type: str          # "numerical" | "categorical" | "datetime" | "text"
    sample_values: list
    null_ratio: float
    unique_count: int


class AgentResult(TypedDict):
    agent_name: str             # "numerical" | "categorical" | "datetime" | "text"
    table: str
    column: str
    analysis: dict
    insights_text: str
    chart_data: dict
    chart_type: str
    error: Optional[str]


class PipelineState(TypedDict):
    run_id: str
    table_prefix: str
    raw_data: dict              # { table_name: DataFrame }
    schema_info: dict           # { table: { col: detected_type } }
    column_routes: list         # list[ColumnRoute]
    universal_results: list       # list[AgentResult] — row-wise anomalies (all column types)
    numerical_results: list       # list[AgentResult] — alias for universal_results (downstream compat)
    categorical_results: list     # list[AgentResult] — empty (downstream compat)
    datetime_results: list        # list[AgentResult] — empty (downstream compat)
    text_results: list            # list[AgentResult] — empty (downstream compat)
    enterprise_row_results: list  # list[AgentResult] — row-wise data quality + fraud signals
    wow_row_results: list         # list[AgentResult] — row-wise WoW pattern signals
    charts: dict                # { chart_name: html_string }
    insights: dict
    story: dict                 # { chapter_1..4 + one_liner } from StoryAgent
    report: dict
    status: str                 # "running" | "done" | "failed"
    error: Optional[str]
    started_at: str
    completed_at: Optional[str]


class QueryState(TypedDict):
    query: str
    session_id: str
    chat_history: list          # [{"role": "user"|"assistant", "content": str}, ...]
    preprocessed_query: Optional[str]
    retrieved_docs: list
    reranked_docs: list
    metadata_context: str
    llm_answer: str
    final_answer: str
    resolution_confidence: float
    can_resolve: bool
    query_intent: str           # "data_related" | "out_of_scope"
    top_similarity: float
    num_retrieved: int
    feedback: Optional[str]
