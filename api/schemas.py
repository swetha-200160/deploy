from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    table_prefix: Optional[str] = None


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    tables_found: list[str]
    started_at: str


class PipelineStatusResponse(BaseModel):
    run_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class AgentResultResponse(BaseModel):
    agent_name: str
    table: str
    column: str
    analysis: dict
    insights_text: str
    chart_type: str
    error: Optional[str] = None


class AllAgentsResponse(BaseModel):
    run_id: str
    numerical: list[AgentResultResponse]
    categorical: list[AgentResultResponse]
    datetime: list[AgentResultResponse]
    text: list[AgentResultResponse]


class InsightsResponse(BaseModel):
    key_findings: list[Any]
    risk_signals: list[Any]
    insights: dict[str, Any]
    recommendations: list[Any]
    data_quality_score: int
    overall_narrative: str


class ChartListResponse(BaseModel):
    charts: dict[str, str]


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    run_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    run_id: str
    query: str
    preprocessed_query: Optional[str] = None
    answer: str
    confidence: float
    resolved: bool
    query_intent: str
    top_similarity: float
    num_docs_retrieved: int


class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    run_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    run_id: str
    query: str
    preprocessed_query: Optional[str] = None
    answer: str
    history: list[ChatMessage]
    confidence: float
    resolved: bool
    query_intent: str
    num_docs_retrieved: int


class FeedbackRequest(BaseModel):
    session_id: str
    helpful: bool


class TableSchemaResponse(BaseModel):
    tables: dict[str, dict]


class StepRunResponse(BaseModel):
    run_id: str
    step: str
    status: str
    result_summary: dict[str, Any]
    error: Optional[str] = None


class StepsStatusResponse(BaseModel):
    run_id: str
    available_steps: list[str]
    completed_steps: dict[str, bool]


class AvailableRunsResponse(BaseModel):
    charts: list[str]
    faiss: list[str]
    reports: list[str]


class DeleteResponse(BaseModel):
    deleted: bool
    item: str
    type: str
