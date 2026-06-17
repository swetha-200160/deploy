# 4sight v3 — Full Dynamic Multi-Agent Financial Data Analysis System
## Complete Code Plan & Architecture Document

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [Full File Structure](#4-full-file-structure)
5. [Environment Configuration (.env)](#5-environment-configuration-env)
6. [Data Flow](#6-data-flow)
7. [Module-by-Module Code Plan](#7-module-by-module-code-plan)
   - [config.py](#71-configpy)
   - [core/state.py](#72-corestatepy)
   - [core/schema_router.py](#73-coreschema_routerpy)
   - [agents/numerical_agent.py](#74-agentsnumerical_agentpy)
   - [agents/categorical_agent.py](#75-agentscategorical_agentpy)
   - [agents/datetime_agent.py](#76-agentsdatetime_agentpy)
   - [agents/text_agent.py](#77-agentstext_agentpy)
   - [output/insight_agent.py](#78-outputinsight_agentpy)
   - [output/reporting_agent.py](#79-outputreporting_agentpy)
   - [visualization/chart_generator.py](#710-visualizationchart_generatorpy)
   - [query_agent/faiss_store.py](#711-query_agentfaiss_storepy)
   - [query_agent/query_preprocessor.py](#712-query_agentquery_preprocessorpy)
   - [query_agent/reranker.py](#713-query_agentrerankerpy)
   - [query_agent/feedback_tracker.py](#714-query_agentfeedback_trackerpy)
   - [query_agent/query_agent.py](#715-query_agentquery_agentpy)
   - [orchestrator.py](#716-orchestratorpy)
   - [api/schemas.py](#717-apischemaspy)
   - [api/routers/pipeline.py](#718-apirouterspipelinepy)
   - [api/routers/agents.py](#719-apiroutersagentspy)
   - [api/routers/charts.py](#720-apirouterschartspy)
   - [api/routers/insights.py](#721-apiroutersinsightspy)
   - [api/routers/query.py](#722-apiroutersquerypy)
   - [api/main.py](#723-apimainpy)
8. [All FastAPI Endpoints](#8-all-fastapi-endpoints)
9. [LangGraph Workflows](#9-langgraph-workflows)
10. [Query Agent Flow (Enhanced RAG)](#10-query-agent-flow-enhanced-rag)
11. [PyECharts Visualization Strategy](#11-pyecharts-visualization-strategy)
12. [Build Order & Dependencies](#12-build-order--dependencies)
13. [Package Requirements](#13-package-requirements)

---

## 1. Project Overview

**4sight v3** is a fully dynamic, multi-agent financial data analysis system built on top of:
- **PostgreSQL** as the data source (tables auto-discovered by prefix `foresight`)
- **Groq LLaMA-3.1-8B-Instant** as the LLM with tool use
- **LangGraph** for agent orchestration
- **FAISS + BM25** for hybrid vector search
- **PyECharts** for interactive chart generation
- **FastAPI** as the sole backend serving analysis results, charts, insights, and user queries

**No hardcoding anywhere.** Every threshold, model name, table prefix, weight, and path comes from `.env`.

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Analysis reasoning + tool use |
| Orchestration | LangGraph `StateGraph` | Multi-agent pipeline wiring |
| Database | PostgreSQL + SQLAlchemy | Source data (5+ tables) |
| DB Layer | Existing `db.py` (`DB` class) | Table discovery + DataFrame loading |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Text → vectors |
| Vector Store | FAISS (`faiss-cpu`) | Semantic similarity search |
| Keyword Search | `rank_bm25` (BM25Okapi) | Keyword-based retrieval |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Score fusion re-ranking |
| Visualization | `pyecharts` | Interactive HTML charts |
| API | FastAPI + Uvicorn | All endpoints |
| Config | `python-dotenv` | Zero hardcoding |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│  /pipeline  /agents  /charts  /insights  /query         │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    orchestrator.py   │
              │  (Master LangGraph)  │
              └──────────┬──────────┘
                         │
         ┌───────────────▼───────────────┐
         │       core/schema_router.py    │
         │  Auto-detect column types      │
         │  numerical / categorical /     │
         │  datetime / text               │
         └───┬───────┬───────┬───────┬───┘
             │       │       │       │
     ┌───────▼─┐ ┌───▼───┐ ┌▼─────┐ ┌▼──────┐
     │Agent 1  │ │Agent 2│ │Agent3│ │Agent 4│
     │Numerical│ │Categ. │ │Date  │ │Text   │
     └────┬────┘ └───┬───┘ └──┬───┘ └───┬───┘
          └──────────┴─────────┴─────────┘
                         │
              ┌──────────▼──────────┐
              │  visualization/      │
              │  chart_generator.py  │
              │  PyECharts HTML      │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  output/             │
              │  insight_agent.py    │  ← LLM executive summary
              │  reporting_agent.py  │  ← JSON + HTML report
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  query_agent/        │
              │  faiss_store.py      │  ← Index all analysis results
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  POST /api/query     │
              │  query_agent.py      │  ← Enhanced RAG (LangGraph)
              │  preprocess →        │
              │  hybrid_search →     │
              │  rerank →            │
              │  similarity_check →  │
              │  LLM answer          │
              └─────────────────────┘
```

---

## 4. Full File Structure

```
4sight_v3/
│
├── .env                              # All config — no hardcoding anywhere
├── requirements.txt                  # All Python packages
│
├── excel_to_postgres.py              # EXISTING — Excel → PostgreSQL loader
├── db.py                             # EXISTING — Dynamic DB access (DB class)
│
├── config.py                         # Central .env reader, typed config object
│
├── core/
│   ├── __init__.py
│   ├── state.py                      # All LangGraph TypedDict state schemas
│   └── schema_router.py             # Auto-detect column types, route to agents
│
├── agents/
│   ├── __init__.py
│   ├── numerical_agent.py            # Agent 1: stats, outliers, correlation, VaR
│   ├── categorical_agent.py          # Agent 2: frequency, cardinality, segmentation
│   ├── datetime_agent.py             # Agent 3: trend, seasonality, MoM/YoY
│   └── text_agent.py                 # Agent 4: sentiment, TF-IDF, keywords
│
├── output/
│   ├── __init__.py
│   ├── insight_agent.py              # Combines 4 agent results → LLM summary
│   └── reporting_agent.py            # Builds final JSON + HTML report
│
├── visualization/
│   ├── __init__.py
│   └── chart_generator.py            # PyECharts: all chart types → HTML strings
│
├── query_agent/
│   ├── __init__.py
│   ├── faiss_store.py                # FAISS index + BM25 + hybrid search
│   ├── query_preprocessor.py         # Query normalization, expansion, cleaning
│   ├── reranker.py                   # Cross-encoder re-ranking + score fusion
│   ├── feedback_tracker.py           # Log retrieval sessions → PostgreSQL
│   └── query_agent.py                # LangGraph query workflow (Enhanced RAG)
│
├── orchestrator.py                   # Master LangGraph pipeline
│
└── api/
    ├── __init__.py
    ├── main.py                       # FastAPI app, CORS, startup, routers
    ├── schemas.py                    # All Pydantic request/response models
    └── routers/
        ├── __init__.py
        ├── pipeline.py               # /api/pipeline/* endpoints
        ├── agents.py                 # /api/agents/* endpoints
        ├── charts.py                 # /api/charts/* endpoints
        ├── insights.py               # /api/insights + /api/report endpoints
        └── query.py                  # /api/query/* endpoints
```

---

## 5. Environment Configuration (.env)

```env
# ── PostgreSQL ──────────────────────────────────────────
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database_name
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
TABLE_PREFIX=foresight           # fixed — matches existing db.py prefix

# ── Groq LLM ────────────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# ── Embeddings & FAISS ──────────────────────────────────
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=./faiss_index
FAISS_INDEX_FILE=analysis_index.faiss
FAISS_META_FILE=analysis_meta.json

# ── Hybrid Search Weights ───────────────────────────────
SEMANTIC_WEIGHT=0.6
BM25_WEIGHT=0.4
SIMILARITY_THRESHOLD=0.35
MIN_TOP_K=3
MAX_TOP_K=10
USE_DYNAMIC_TOPK=true

# ── Re-ranking ──────────────────────────────────────────
USE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_ALPHA=0.5

# ── Query Agent Feature Flags ───────────────────────────
USE_QUERY_PREPROCESSING=true
USE_HYBRID_SEARCH=true
TRACK_FEEDBACK=true
CONFIDENCE_THRESHOLD=0.6
HIGH_PRIORITY_THRESHOLD=0.3

# ── Numerical Analysis ──────────────────────────────────
OUTLIER_METHOD=iqr
ZSCORE_THRESHOLD=3.0
IQR_MULTIPLIER=1.5
CORRELATION_METHOD=pearson
DISTRIBUTION_BINS=20

# ── Categorical Analysis ─────────────────────────────────
MAX_CATEGORICAL_CARDINALITY=50
RARE_CATEGORY_THRESHOLD=0.05
IMBALANCE_RATIO_THRESHOLD=10.0

# ── Datetime Analysis ────────────────────────────────────
SEASONALITY_PERIOD=12
ROLLING_WINDOW=7
FORECAST_PERIODS=6

# ── Text / NLP Analysis ──────────────────────────────────
MAX_KEYWORDS=20
MIN_KEYWORD_FREQ=2
SENTIMENT_BATCH_SIZE=32
TFIDF_MAX_FEATURES=500

# ── Visualization ────────────────────────────────────────
CHART_OUTPUT_DIR=./charts
CHART_THEME=westeros
CHART_WIDTH=900px
CHART_HEIGHT=500px

# ── Output / Reports ─────────────────────────────────────
REPORT_OUTPUT_DIR=./reports

# ── FastAPI ──────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
CORS_ORIGINS=*
```

---

## 6. Data Flow

```
Step 1 — DB Discovery
  .env TABLE_PREFIX
      → DB().tables()          discovers all matching tables dynamically
      → DB().load_all()        loads every table as a pandas DataFrame
      → DB().columns(table)    reads column names per table
      → state["tables_analyzed"] saved immediately (list of table names)

Step 2 — Schema Routing
  schema_router.py
      → per-column type detection:
          is_numeric()         → "numerical"
          is_object/low-card() → "categorical"
          is_datetime()        → "datetime"
          is_text/high-len()   → "text"
      → returns ColumnRoute: { col_name: type, table: str }
      → groups columns into 4 agent payloads

Step 3 — 4 Sequential Analysis Agents (LangGraph nodes)
  Agent 1 (Numerical)     → AgentResult: { stats, outliers, correlation, financial_kpis }
  Agent 2 (Categorical)   → AgentResult: { freq, cardinality, segmentation, chi_square }
  Agent 3 (Datetime)      → AgentResult: { trend, seasonality, mom_yoy, rolling_avg }
  Agent 4 (Text)          → AgentResult: { sentiment, keywords, tfidf, topics }

  NOTE — LLM prompts use row-wise data format:
      Instead of: { "price": [150, 2800, 280], "company": ["A","B","C"] }
      Uses:       [{"company":"A","price":150}, {"company":"B","price":2800}, ...]
      This lets the LLM understand which values belong to the same record.

Step 4 — Visualization
  chart_generator.py
      receives each AgentResult
      → auto-selects chart type from data shape
      → renders PyECharts → HTML string per chart
      → saves HTML files to CHART_OUTPUT_DIR (always on disk)
      → *** raw_data (DataFrames) freed from RAM after this step ***

Step 5 — Output Layer
  insight_agent.py
      → sends all 4 AgentResults to Groq LLM
      → structured prompt → executive summary JSON
  reporting_agent.py
      → merges all agent results + charts + insights
      → outputs: {run_id}.json + {run_id}.html to REPORT_OUTPUT_DIR

Step 6 — FAISS Indexing + Memory Cleanup
  faiss_store.py
      → takes all analysis text (insights, summaries, stats)
      → embeds with sentence-transformers
      → builds FAISS index + BM25 corpus
      → saves to FAISS_INDEX_PATH
      → ready for query agent RAG
  Memory cleanup (after FAISS done):
      → agent results saved to {run_id}_results.json on disk
      → numerical_results, categorical_results, datetime_results,
        text_results, charts all cleared from pipeline_store RAM
      → only metadata (run_id, status, timestamps, results_file path) kept in RAM

Step 7 — Background: Table Change Detection
  api/main.py background task (_poll_table_changes):
      → polls DB().count(table) for every matching table every 60 s
      → compares against counts from last check
      → if any table row count changed (insert/update/delete) or
        tables added/removed → auto-triggers new pipeline run (run_id_auto)

Step 8 — FastAPI serves everything
  GET  /api/pipeline/results/{run_id}  → agent results (RAM or disk fallback)
  GET  /api/charts/{chart_type}        → PyECharts HTML (RAM or disk fallback)
  GET  /api/charts/{chart_type}?format=json → raw chart_data as JSON
  GET  /api/insights                   → LLM executive summary
  GET  /api/report                     → full HTML report
  POST /api/query                      → Enhanced RAG answer
```

---

## 7. Module-by-Module Code Plan

---

### 7.1 `config.py`

**Purpose:** Single source of truth. Reads `.env`, exposes a typed `Config` dataclass. Every other module imports from here — no direct `os.getenv()` calls elsewhere.

**Key class:**
```python
@dataclass
class Config:
    # DB
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    table_prefix: str

    # LLM
    groq_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # FAISS
    embedding_model: str
    faiss_index_path: str
    faiss_index_file: str
    faiss_meta_file: str

    # Search
    semantic_weight: float
    bm25_weight: float
    similarity_threshold: float
    min_top_k: int
    max_top_k: int
    use_dynamic_topk: bool

    # Reranking
    use_reranking: bool
    reranker_model: str
    reranker_alpha: float

    # Query Agent flags
    use_query_preprocessing: bool
    use_hybrid_search: bool
    track_feedback: bool
    confidence_threshold: float
    high_priority_threshold: float

    # Analysis
    outlier_method: str          # "iqr" | "zscore" | "isolation_forest"
    zscore_threshold: float
    iqr_multiplier: float
    correlation_method: str      # "pearson" | "spearman"
    distribution_bins: int
    max_categorical_cardinality: int
    rare_category_threshold: float
    imbalance_ratio_threshold: float
    seasonality_period: int
    rolling_window: int
    forecast_periods: int
    max_keywords: int
    min_keyword_freq: int
    sentiment_batch_size: int
    tfidf_max_features: int

    # Charts
    chart_output_dir: str
    chart_theme: str
    chart_width: str
    chart_height: str

    # Output
    report_output_dir: str

    # API
    api_host: str
    api_port: int
    api_reload: bool
    cors_origins: list[str]

def get_config() -> Config:
    """Load and return singleton Config from .env"""
    ...
```

**Singleton pattern:** `_config = None; def get_config() → Config` — loads once, cached.

---

### 7.2 `core/state.py`

**Purpose:** All LangGraph `TypedDict` state schemas used across the pipeline.

**States defined:**

```python
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
    analysis: dict              # Full analysis output (dynamic structure)
    insights_text: str          # LLM-generated plain-text insight for this column
    chart_data: dict            # Data payload for chart_generator
    chart_type: str             # Recommended chart type
    error: Optional[str]

class PipelineState(TypedDict):
    run_id: str
    table_prefix: str
    raw_data: dict[str, Any]            # { table_name: DataFrame }
    schema_info: dict[str, Any]         # { table: { col: detected_type } }
    column_routes: list[ColumnRoute]
    numerical_results: list[AgentResult]
    categorical_results: list[AgentResult]
    datetime_results: list[AgentResult]
    text_results: list[AgentResult]
    charts: dict[str, str]              # { chart_name: html_string }
    insights: dict[str, Any]            # Master LLM summary
    report: dict[str, Any]              # Final report
    status: str                         # "running" | "done" | "failed"
    error: Optional[str]
    started_at: str
    completed_at: Optional[str]

class QueryState(TypedDict):
    query: str
    session_id: str
    preprocessed_query: Optional[str]
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    metadata_context: str
    llm_answer: str
    final_answer: str
    resolution_confidence: float
    can_resolve: bool
    query_intent: str           # "data_related" | "out_of_scope"
    top_similarity: float
    num_retrieved: int
    feedback: Optional[str]
```

---

### 7.3 `core/schema_router.py`

**Purpose:** Reads every column from every table, classifies it dynamically, returns `ColumnRoute` list.

**Logic (fully dynamic — no hardcoded column names):**

```
For each table in DB().tables():
    df = DB().load(table)
    For each column in df.columns:

        1. Check pandas dtype:
           - int64 / float64                    → "numerical"
           - datetime64                          → "datetime"

        2. If dtype == object:
           - Try parse as datetime               → "datetime"
           - unique_ratio = nunique / len
           - avg_str_len = mean string length
           - if avg_str_len > TEXT_LENGTH_THRESHOLD and unique_ratio > 0.5 → "text"
           - elif nunique <= MAX_CATEGORICAL_CARDINALITY               → "categorical"
           - else                                                       → "text"

        3. If dtype == bool                      → "categorical"

        Produce ColumnRoute with: table, column, detected_type,
                                  sample_values[:5], null_ratio, unique_count
```

**Output:** Groups into 4 lists → passed into agent nodes.

---

### 7.4 `agents/numerical_agent.py`

**Purpose:** Full numerical analysis for every numeric column in every table.
Uses Groq tool use — Python functions registered as LLM tools.

**Groq Tool Definitions (dynamic, bound to actual data):**

```python
tools = [
    {
        "name": "calculate_descriptive_stats",
        "description": "Calculate mean, median, mode, std, variance, skewness, kurtosis, percentiles",
        "parameters": { "column_data": "array of numbers" }
    },
    {
        "name": "detect_outliers",
        "description": "Detect outliers using configured method (iqr/zscore/isolation_forest)",
        "parameters": { "column_data": "array", "method": "string" }
    },
    {
        "name": "analyze_distribution",
        "description": "Histogram bins, KDE, detect normal/skewed/bimodal/heavy-tail distribution",
        "parameters": { "column_data": "array", "bins": "int" }
    },
    {
        "name": "calculate_correlation",
        "description": "Pearson/Spearman correlation + VIF for multicollinearity",
        "parameters": { "dataframe_dict": "dict of arrays", "method": "string" }
    },
    {
        "name": "calculate_financial_kpis",
        "description": "Revenue trend, VaR, CVaR, Sharpe ratio, anomaly flags",
        "parameters": { "column_data": "array", "column_name": "string" }
    },
    {
        "name": "detect_missing_values",
        "description": "Null count, null%, MCAR/MAR/MNAR classification, imputation suggestion",
        "parameters": { "column_data": "array" }
    }
]
```

**Agent flow per column:**
1. Load column data from DataFrame
2. Build **row-wise sample** — all numeric columns per row: `[{"company":"A","price":150,"volume":1M}, ...]`
   so the LLM understands which values belong to the same record
3. Call LLM with tool use → LLM decides which tools to invoke
4. Execute tools (Python functions) → return results to LLM
5. LLM synthesizes final `AgentResult` with `analysis` dict + `insights_text`
6. Auto-select chart type: histogram (distribution), heatmap (correlation), line (trend)

**Output per column:**
```json
{
    "agent_name": "numerical",
    "table": "foresight_cheque_clearing_67318",
    "column": "cheque_amount",
    "analysis": {
        "descriptive_stats": { "mean": 45230.5, "median": 38000, "std": 22100, ... },
        "outliers": { "method": "iqr", "count": 23, "outlier_indices": [...], "anomaly_flag": true },
        "distribution": { "shape": "right_skewed", "bins": [...], "kde": [...] },
        "financial_kpis": { "var_95": 75000, "cvar_95": 92000, "sharpe_ratio": 1.23 },
        "missing": { "null_count": 5, "null_pct": 0.02, "classification": "MCAR" }
    },
    "insights_text": "The cheque_amount column shows strong right-skew with 23 outliers...",
    "chart_data": { ... },
    "chart_type": "histogram"
}
```

---

### 7.5 `agents/categorical_agent.py`

**Purpose:** Full categorical analysis for every low-cardinality text/object column.

**Groq Tool Definitions:**

```python
tools = [
    {
        "name": "analyze_frequency",
        "description": "Value counts, category dominance %, rare category detection",
        "parameters": { "column_data": "array", "rare_threshold": "float" }
    },
    {
        "name": "analyze_cardinality",
        "description": "Unique count, cardinality ratio, low/high cardinality classification",
        "parameters": { "column_data": "array", "max_cardinality": "int" }
    },
    {
        "name": "analyze_imbalance",
        "description": "Class imbalance ratio, majority/minority class detection",
        "parameters": { "column_data": "array", "threshold": "float" }
    },
    {
        "name": "analyze_relationships",
        "description": "Chi-square test between categorical columns, Cramér's V",
        "parameters": { "col_a": "array", "col_b": "array" }
    },
    {
        "name": "analyze_segmentation",
        "description": "Group-by aggregation, customer/branch/product segmentation",
        "parameters": { "column_data": "array", "numeric_col": "array", "agg_func": "string" }
    },
    {
        "name": "detect_missing_categories",
        "description": "Missing value count, mode imputation suggestion, unknown category detection",
        "parameters": { "column_data": "array" }
    }
]
```

**Chart type auto-selection:**
- `nunique <= 6` → pie chart
- `nunique <= 20` → bar chart
- With a numeric target column → stacked bar chart
- Frequency distribution → countplot

**Output per column:**
```json
{
    "agent_name": "categorical",
    "table": "foresight_cheque_clearing_67318",
    "column": "cheque_status",
    "analysis": {
        "frequency": { "Cleared": 4521, "Rejected": 1203, "Pending": 876 },
        "cardinality": { "unique_count": 3, "type": "low_cardinality" },
        "imbalance": { "ratio": 5.16, "dominant_class": "Cleared", "imbalanced": true },
        "missing": { "null_count": 0, "null_pct": 0.0 },
        "segmentation": { "by_branch": { "Branch_A": {...}, "Branch_B": {...} } }
    },
    "insights_text": "cheque_status is dominated by Cleared (69.2%) ...",
    "chart_data": { "labels": ["Cleared","Rejected","Pending"], "values": [4521,1203,876] },
    "chart_type": "pie"
}
```

---

### 7.6 `agents/datetime_agent.py`

**Purpose:** Full time-series analysis for every datetime column.

**Groq Tool Definitions:**

```python
tools = [
    {
        "name": "validate_datetime",
        "description": "Parse datetime, detect format, find missing dates, coverage range",
        "parameters": { "column_data": "array" }
    },
    {
        "name": "analyze_trend",
        "description": "Linear trend, moving average, trend direction (up/down/flat)",
        "parameters": { "dates": "array", "values": "array", "window": "int" }
    },
    {
        "name": "analyze_seasonality",
        "description": "Seasonal decomposition, peak months, trough months",
        "parameters": { "dates": "array", "values": "array", "period": "int" }
    },
    {
        "name": "calculate_rolling_stats",
        "description": "Rolling mean, rolling std, rolling sum with configurable window",
        "parameters": { "values": "array", "window": "int" }
    },
    {
        "name": "analyze_mom_yoy",
        "description": "Month-over-month % change, year-over-year % change",
        "parameters": { "dates": "array", "values": "array" }
    },
    {
        "name": "detect_anomalous_dates",
        "description": "Spike detection, dip detection, date gaps",
        "parameters": { "dates": "array", "values": "array" }
    }
]
```

**Chart type:** Always line chart with optional seasonality overlay.

---

### 7.7 `agents/text_agent.py`

**Purpose:** NLP analysis for high-cardinality text columns (remarks, descriptions, notes).

**Groq Tool Definitions:**

```python
tools = [
    {
        "name": "clean_text",
        "description": "Remove punctuation, lowercase, strip whitespace",
        "parameters": { "texts": "array" }
    },
    {
        "name": "analyze_sentiment",
        "description": "Positive/negative/neutral classification per text",
        "parameters": { "texts": "array" }
    },
    {
        "name": "extract_keywords",
        "description": "Top-N keywords by TF-IDF score",
        "parameters": { "texts": "array", "max_features": "int", "top_n": "int" }
    },
    {
        "name": "analyze_topics",
        "description": "LDA topic modeling, top terms per topic",
        "parameters": { "texts": "array", "n_topics": "int" }
    },
    {
        "name": "analyze_complaints",
        "description": "Flag negative sentiment + keywords → complaint signals",
        "parameters": { "texts": "array", "sentiment_scores": "array" }
    }
]
```

**Chart type:** Bar chart (keyword frequencies), pie chart (sentiment distribution).

---

### 7.8 `output/insight_agent.py`

**Purpose:** Takes all 4 `AgentResult` lists → sends combined summary to Groq LLM → returns structured executive insight.

**Prompt strategy (dynamic):**
```
System: You are a senior financial data analyst. Synthesize the following analysis results.

User:
=== NUMERICAL ANALYSIS ===
{numerical_results_summary}

=== CATEGORICAL ANALYSIS ===
{categorical_results_summary}

=== DATETIME ANALYSIS ===
{datetime_results_summary}

=== TEXT/NLP ANALYSIS ===
{text_results_summary}

Generate a structured JSON executive summary with:
- key_findings: list of top 5 findings
- risk_signals: list of detected anomalies/risks
- recommendations: list of actionable recommendations
- data_quality_score: 0-100
- overall_narrative: 3-4 sentence paragraph
```

**Output:**
```json
{
    "key_findings": ["...", "...", "..."],
    "risk_signals": ["...", "..."],
    "recommendations": ["...", "..."],
    "data_quality_score": 78,
    "overall_narrative": "The financial data reveals ..."
}
```

---

### 7.9 `output/reporting_agent.py`

**Purpose:** Assembles all results into final deliverables. Fully dynamic — no column or table names hardcoded.

**Outputs:**
1. `report.json` — full structured report saved to `REPORT_OUTPUT_DIR`
2. `report.html` — HTML report with embedded PyECharts charts

**Report JSON structure:**
```json
{
    "run_id": "run_20260515_143022",
    "generated_at": "2026-05-15T14:30:22",
    "tables_analyzed": ["foresight_report_summary", ...],
    "total_columns_analyzed": 47,
    "agents": {
        "numerical": [ ... AgentResult list ... ],
        "categorical": [ ... ],
        "datetime": [ ... ],
        "text": [ ... ]
    },
    "insights": { ... InsightAgent output ... },
    "charts": { "countplot_cheque_status": "<html>...", ... }
}
```

---

### 7.10 `visualization/chart_generator.py`

**Purpose:** Receives `AgentResult.chart_data` + `chart_type` → generates PyECharts HTML string.
All chart options (theme, width, height, title) come from `config.py`.

**Chart generators (one function per type):**

```python
def make_countplot(data: dict, title: str, config: Config) -> str:
    """Bar chart showing value frequency counts (categorical distribution)"""
    # data = { "labels": [...], "values": [...] }
    bar = Bar(init_opts=opts.InitOpts(theme=config.chart_theme,
                                      width=config.chart_width,
                                      height=config.chart_height))
    bar.add_xaxis(data["labels"])
    bar.add_yaxis("Count", data["values"])
    bar.set_global_opts(title_opts=opts.TitleOpts(title=title))
    return bar.render_embed()

def make_bar_chart(data: dict, title: str, config: Config) -> str:
    """Grouped bar chart for categorical comparisons"""
    ...

def make_pie_chart(data: dict, title: str, config: Config) -> str:
    """Pie chart for distribution proportions"""
    pie = Pie(...)
    pie.add("", list(zip(data["labels"], data["values"])))
    ...
    return pie.render_embed()

def make_stacked_bar_chart(data: dict, title: str, config: Config) -> str:
    """Stacked bar chart for multi-category breakdown"""
    # data = { "x_axis": [...], "series": [{ "name": "...", "values": [...] }] }
    ...

def make_heatmap(data: dict, title: str, config: Config) -> str:
    """Correlation heatmap for numerical columns"""
    ...

def make_line_chart(data: dict, title: str, config: Config) -> str:
    """Time-series line chart for datetime analysis"""
    ...

def make_histogram(data: dict, title: str, config: Config) -> str:
    """Distribution histogram for numerical columns"""
    ...

def generate_all_charts(agent_results: list[AgentResult], config: Config) -> dict[str, str]:
    """
    Auto-dispatch: reads chart_type from each AgentResult,
    calls the right maker function, returns { chart_name: html_string }
    """
    dispatch = {
        "countplot": make_countplot,
        "bar": make_bar_chart,
        "pie": make_pie_chart,
        "stacked_bar": make_stacked_bar_chart,
        "heatmap": make_heatmap,
        "line": make_line_chart,
        "histogram": make_histogram,
    }
    charts = {}
    for result in agent_results:
        fn = dispatch.get(result["chart_type"])
        if fn:
            name = f"{result['chart_type']}_{result['table']}_{result['column']}"
            charts[name] = fn(result["chart_data"], name, config)
    return charts
```

---

### 7.11 `query_agent/faiss_store.py`

**Purpose:** FAISS + BM25 hybrid vector store. Indexes all analysis results after pipeline runs. Serves the query agent.

**Key methods:**

```python
class FAISSStore:
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.index = None           # faiss.IndexFlatIP (inner product = cosine with normalized vecs)
        self.bm25 = None            # BM25Okapi instance
        self.documents = []         # list of { id, text, metadata }

    def index_analysis_results(self, agent_results: list[AgentResult], insights: dict):
        """
        Convert all agent results + insights → text chunks → embed → build FAISS + BM25.
        Saves index to FAISS_INDEX_PATH.
        """
        # Build text chunks from analysis results
        chunks = self._build_chunks(agent_results, insights)
        self.documents = chunks

        # Embed all chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))

        # Build BM25 index
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        # Save to disk
        self._save()

    def hybrid_search(self, query: str, k: int = None) -> list[dict]:
        """
        Semantic search (FAISS) + keyword search (BM25) → fuse scores.
        Returns top-k docs sorted by fused score.
        """
        k = k or self.config.min_top_k
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)

        # Semantic: FAISS cosine similarity
        semantic_scores, indices = self.index.search(query_vec.astype("float32"), len(self.documents))
        semantic_map = { int(i): float(s) for i, s in zip(indices[0], semantic_scores[0]) }

        # Keyword: BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max   # normalize 0-1

        # Fuse: semantic_weight * semantic + bm25_weight * bm25
        fused = []
        for idx, doc in enumerate(self.documents):
            sem = semantic_map.get(idx, 0.0)
            bm  = float(bm25_norm[idx])
            score = self.config.semantic_weight * sem + self.config.bm25_weight * bm
            fused.append({ **doc, "similarity": score, "semantic_score": sem, "bm25_score": bm })

        fused.sort(key=lambda x: x["similarity"], reverse=True)
        return fused[:k]

    def dynamic_top_k_search(self, query: str) -> list[dict]:
        """
        Adaptive top-k: start with MIN_TOP_K, expand up to MAX_TOP_K
        while similarity score stays above threshold.
        """
        results = self.hybrid_search(query, k=self.config.max_top_k)
        filtered = [r for r in results if r["similarity"] >= self.config.similarity_threshold]
        if len(filtered) < self.config.min_top_k:
            return results[:self.config.min_top_k]
        return filtered

    def _build_chunks(self, agent_results, insights) -> list[dict]:
        """Convert agent results to searchable text chunks with metadata"""
        chunks = []
        for result in agent_results:
            text = f"Table: {result['table']} | Column: {result['column']} | "
            text += f"Agent: {result['agent_name']} | Analysis: {result['insights_text']}"
            chunks.append({
                "id": f"{result['table']}_{result['column']}",
                "text": text,
                "metadata": {
                    "table": result["table"],
                    "column": result["column"],
                    "agent": result["agent_name"],
                    "chart_type": result["chart_type"]
                }
            })
        # Add insight chunks
        if insights:
            for finding in insights.get("key_findings", []):
                chunks.append({ "id": f"insight_{len(chunks)}", "text": finding, "metadata": { "type": "insight" } })
        return chunks

    def _save(self):
        os.makedirs(self.config.faiss_index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.config.faiss_index_path, self.config.faiss_index_file))
        with open(os.path.join(self.config.faiss_index_path, self.config.faiss_meta_file), "w") as f:
            json.dump(self.documents, f)

    def load(self):
        index_path = os.path.join(self.config.faiss_index_path, self.config.faiss_index_file)
        meta_path  = os.path.join(self.config.faiss_index_path, self.config.faiss_meta_file)
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.documents = json.load(f)
        tokenized = [d["text"].lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def format_context(self, docs: list[dict]) -> str:
        """Format retrieved docs into LLM prompt context string"""
        lines = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"[{i}] (score={doc['similarity']:.3f}) {doc['text']}")
        return "\n".join(lines)
```

---

### 7.12 `query_agent/query_preprocessor.py`

**Purpose:** Normalize and expand user queries before retrieval. Mirrors your existing NFC/IVR pattern.

**Operations (all toggleable from .env):**

```python
class QueryPreprocessor:
    def preprocess(self, query: str) -> str:
        query = self._normalize_whitespace(query)
        query = self._lowercase(query)
        query = self._expand_abbreviations(query)   # e.g. "chq" → "cheque"
        query = self._remove_stop_noise(query)      # remove filler words
        return query.strip()

    def _expand_abbreviations(self, text: str) -> str:
        # Abbreviation map loaded dynamically from config or a JSON file
        # Not hardcoded — reads from abbrev_map.json if present
        ...
```

---

### 7.13 `query_agent/reranker.py`

**Purpose:** Cross-encoder re-ranking + score fusion. Mirrors your existing NFC/IVR pattern.

```python
class Reranker:
    def __init__(self, config: Config):
        self.model = CrossEncoder(config.reranker_model)
        self.alpha = config.reranker_alpha   # 0.5 = equal weight

    def rerank_with_fusion(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Run cross-encoder on (query, doc_text) pairs.
        Fuse: alpha * cross_score + (1-alpha) * original_similarity
        Re-sort descending.
        """
        pairs = [(query, doc["text"]) for doc in documents]
        cross_scores = self.model.predict(pairs)
        cross_max = max(cross_scores) if max(cross_scores) > 0 else 1.0

        for doc, cs in zip(documents, cross_scores):
            norm_cs = float(cs) / cross_max
            doc["cross_score"] = norm_cs
            doc["fused_score"] = (
                self.alpha * norm_cs + (1 - self.alpha) * doc["similarity"]
            )

        documents.sort(key=lambda x: x["fused_score"], reverse=True)
        return documents
```

---

### 7.14 `query_agent/feedback_tracker.py`

**Purpose:** Log every query session to PostgreSQL. Mirrors your existing NFC/IVR pattern.

**Table created dynamically on startup:**
```sql
CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}_query_sessions (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    original_query  TEXT,
    preprocessed_query TEXT,
    retrieved_doc_ids  TEXT[],
    similarity_scores  FLOAT[],
    final_answer    TEXT,
    confidence_score FLOAT,
    query_intent    TEXT,
    feedback        TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

**Methods:**
```python
def log_retrieval(self, session_id, query, preprocessed, doc_ids, scores, answer, confidence, intent): ...
def log_feedback(self, session_id, feedback: str): ...   # "helpful" | "not_helpful"
def get_session(self, session_id) -> dict: ...
def get_all_sessions(self, limit=100) -> list[dict]: ...
```

---

### 7.15 `query_agent/query_agent.py`

**Purpose:** Full LangGraph query workflow. Same architecture as your `qa_agent_enhanced.py`, adapted for financial data domain.

**LangGraph Nodes:**

```
Node 1: preprocess_query_node
    if USE_QUERY_PREPROCESSING:
        query → QueryPreprocessor.preprocess() → preprocessed_query
    else:
        preprocessed_query = query

Node 2: retrieve_docs_node
    query = preprocessed_query
    if USE_HYBRID_SEARCH and USE_DYNAMIC_TOPK:
        docs = FAISSStore.dynamic_top_k_search(query)
    elif USE_HYBRID_SEARCH:
        docs = FAISSStore.hybrid_search(query, k=MIN_TOP_K)
    else:
        docs = FAISSStore.semantic_only_search(query, k=MIN_TOP_K)
    state["retrieved_docs"] = docs
    state["top_similarity"] = docs[0]["similarity"] if docs else 0.0

Node 3: rerank_docs_node
    if USE_RERANKING and retrieved_docs:
        docs = Reranker.rerank_with_fusion(query, retrieved_docs)
        state["reranked_docs"] = docs

Node 4: check_similarity_node
    top_sim = reranked_docs[0]["fused_score"] if reranked_docs else 0.0
    if top_sim < SIMILARITY_THRESHOLD:
        state["query_intent"] = "out_of_scope"
        state["final_answer"] = "I can only answer questions related to the financial data analysis."
        state["resolution_confidence"] = 0.0
        state["can_resolve"] = False
    else:
        state["query_intent"] = "data_related"
    → route: out_of_scope → track_feedback | continue → generate_answer

Node 5: generate_answer_node
    context = FAISSStore.format_context(reranked_docs)
    prompt = f"""
    You are a financial data analyst assistant.
    Answer ONLY using the context below. Do NOT use general knowledge.
    If context is insufficient, say: "I don't have enough data to answer this."

    CONTEXT:
    {context}

    QUESTION: {query}

    Answer:"""
    response = Groq(llm).invoke(prompt)
    state["final_answer"] = response.content

Node 6: assess_confidence_node
    prompt = "Rate this answer 0.0-1.0: query={query}, answer={answer}, docs={n}"
    confidence = float(Groq(llm).invoke(prompt).content)
    state["resolution_confidence"] = confidence
    state["can_resolve"] = confidence >= CONFIDENCE_THRESHOLD

Node 7: track_feedback_node
    if TRACK_FEEDBACK:
        FeedbackTracker.log_retrieval(...)
```

**Conditional edges:**
```
classify → route_after_similarity:
    "out_of_scope" → track_feedback → END
    "data_related" → generate_answer → assess_confidence → track_feedback → END
```

---

### 7.16 `orchestrator.py`

**Purpose:** Master LangGraph pipeline that wires all 4 analysis agents + visualization + output.

**Nodes in the master graph:**

```
Node 1: load_data_node
    DB().load_all() → state["raw_data"]
    state["tables_analyzed"] = list(raw_data.keys())   ← saved before DataFrames are freed

Node 2: route_schema_node
    schema_router.classify_all() → state["column_routes"]

Node 3: run_numerical_agent_node
    filter column_routes where detected_type == "numerical"
    → NumericalAgent.run(df, routes) → state["numerical_results"]
    LLM prompt uses row-wise format: [{"col_a": v1, "col_b": v2}, ...]

Node 4: run_categorical_agent_node
    filter column_routes where detected_type == "categorical"
    → CategoricalAgent.run(df, routes) → state["categorical_results"]
    LLM prompt uses row-wise format: [{"cat_col": "Rejected", "amount": 75000}, ...]

Node 5: run_datetime_agent_node
    filter column_routes where detected_type == "datetime"
    → DatetimeAgent.run(df, routes) → state["datetime_results"]
    LLM prompt uses row-wise format: [{"date": "2023-01-15", "amount": 75000}, ...]

Node 6: run_text_agent_node
    filter column_routes where detected_type == "text"
    → TextAgent.run(df, routes) → state["text_results"]
    LLM prompt uses row-wise format: [{"remarks": "delayed", "status": "Pending"}, ...]

Node 7: generate_charts_node
    all_results = numerical + categorical + datetime + text
    charts = chart_generator.generate_all_charts(all_results, config)
    state["charts"] = charts
    *** state.pop("raw_data") — DataFrames freed from RAM here ***
    Charts are already saved to CHART_OUTPUT_DIR as .html files.

Node 8: generate_insights_node
    InsightAgent.run(all_results) → state["insights"]

Node 9: generate_report_node
    ReportingAgent.build(state) → state["report"]
    saves {run_id}.json + {run_id}.html to REPORT_OUTPUT_DIR
    Uses state["tables_analyzed"] (not raw_data.keys()) since DataFrames are freed

Node 10: index_faiss_node
    FAISSStore.index_analysis_results(all_results, state["insights"])
    FAISS index is now ready for query agent
    state["status"] = "done"
    *** Disk persistence + RAM cleanup ***
    → agent results written to {REPORT_OUTPUT_DIR}/{run_id}_results.json
    → state["results_file"] = path to that file
    → numerical_results, categorical_results, datetime_results,
      text_results, charts all set to [] / {} (freed from RAM)
    Only lightweight metadata stays in pipeline_store (run_id, status, timestamps, results_file).
```

**Edge flow:**
```
load_data → route_schema →
    numerical_agent → categorical_agent → datetime_agent → text_agent
        → generate_charts → generate_insights → generate_report → index_faiss → END
```

**Run storage:** `pipeline_store = { run_id: PipelineState }` in memory.
After run completes, only metadata + `results_file` path are kept in RAM.
Full agent results are on disk; loaded on demand by API endpoints.

---

### 7.17 `api/schemas.py`

All Pydantic models for request/response validation:

```python
class PipelineRunRequest(BaseModel):
    table_prefix: Optional[str] = None   # uses .env default if not given

class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    tables_found: list[str]
    started_at: str

class PipelineStatusResponse(BaseModel):
    run_id: str
    status: str           # "running" | "done" | "failed"
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]

class AgentResultResponse(BaseModel):
    agent_name: str
    table: str
    column: str
    analysis: dict
    insights_text: str
    chart_type: str

class AllAgentsResponse(BaseModel):
    run_id: str
    numerical: list[AgentResultResponse]
    categorical: list[AgentResultResponse]
    datetime: list[AgentResultResponse]
    text: list[AgentResultResponse]

class InsightsResponse(BaseModel):
    key_findings: list[str]
    risk_signals: list[str]
    recommendations: list[str]
    data_quality_score: int
    overall_narrative: str

class ChartListResponse(BaseModel):
    charts: dict[str, str]   # { chart_name: html_string }

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None   # auto-generated if not provided

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    confidence: float
    resolved: bool
    query_intent: str          # "data_related" | "out_of_scope"
    top_similarity: float
    num_docs_retrieved: int

class FeedbackRequest(BaseModel):
    session_id: str
    helpful: bool

class TableSchemaResponse(BaseModel):
    tables: dict[str, dict]    # { table_name: { col_name: detected_type } }
```

---

### 7.18 `api/routers/pipeline.py`

```python
router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Trigger full analysis pipeline asynchronously.
    Returns run_id immediately. Use /status/{run_id} to poll.
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state = PipelineState(run_id=run_id, status="running", ...)
    pipeline_store[run_id] = state
    background_tasks.add_task(orchestrator.run, run_id, request.table_prefix)
    return PipelineRunResponse(run_id=run_id, status="running", ...)

@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
async def get_status(run_id: str):
    """Poll pipeline run status"""
    ...

@router.get("/results/{run_id}", response_model=AllAgentsResponse)
async def get_results(run_id: str):
    """
    Get all 4 agent results for a completed run.
    Results are loaded from RAM if present; otherwise loaded from
    {REPORT_OUTPUT_DIR}/{run_id}_results.json on disk (after RAM cleanup).
    """
    ...

@router.get("/tables", response_model=list[str])
async def list_tables():
    """Dynamically list all PostgreSQL tables matching TABLE_PREFIX"""
    return DB().tables()

@router.get("/schema", response_model=TableSchemaResponse)
async def get_schema():
    """Return full schema: tables → columns → auto-detected types"""
    ...
```

---

### 7.19 `api/routers/agents.py`

```python
router = APIRouter(prefix="/api/agents", tags=["Agents"])

@router.get("/numerical", response_model=list[AgentResultResponse])
async def get_numerical(run_id: Optional[str] = None):
    """Get numerical agent results (latest run if run_id not specified)"""
    ...

@router.get("/categorical", response_model=list[AgentResultResponse])
async def get_categorical(run_id: Optional[str] = None):
    ...

@router.get("/datetime", response_model=list[AgentResultResponse])
async def get_datetime(run_id: Optional[str] = None):
    ...

@router.get("/text", response_model=list[AgentResultResponse])
async def get_text(run_id: Optional[str] = None):
    ...

@router.get("/{agent_name}/table/{table_name}", response_model=list[AgentResultResponse])
async def get_agent_results_for_table(agent_name: str, table_name: str, run_id: Optional[str] = None):
    """Filter agent results by specific table name"""
    ...
```

---

### 7.20 `api/routers/charts.py`

> **Chart strategy:** Every chart endpoint supports two response formats via `?format=` query param:
> - `?format=html` (default) — PyECharts interactive chart rendered directly in the browser
> - `?format=json` — raw `chart_data` dict returned as JSON (for custom frontends)
>
> **Disk fallback:** After pipeline RAM cleanup, charts are served from `CHART_OUTPUT_DIR/*.html`
> files automatically — no re-run needed.

```python
router = APIRouter(prefix="/api/charts", tags=["Charts"])

# ── All charts on one page ─────────────────────────────────────────────────────

@router.get("/all", response_class=HTMLResponse)
async def get_all_charts(run_id: Optional[str] = None):
    """
    Renders ALL charts stacked on a single HTML page.
    Falls back to reading .html files from CHART_OUTPUT_DIR if RAM is cleared.
    """
    ...

# ── Individual chart endpoints ─────────────────────────────────────────────────
# All accept: ?table=...  &column=...  &run_id=...  &format=html|json
#
# format=html (default) → HTMLResponse — interactive PyECharts in browser
# format=json           → JSONResponse — raw chart_data dict
#
# Lookup order:
#   1. state["charts"] dict (in-memory, available during run)
#   2. CHART_OUTPUT_DIR/{chart_type}__{table}__{column}.html (disk, after RAM cleanup)
#   3. Regenerate on the fly from chart_data if file missing

@router.get("/countplot")   # ?format=html|json
@router.get("/bar")         # ?format=html|json
@router.get("/pie")         # ?format=html|json
@router.get("/stacked_bar") # ?format=html|json
@router.get("/heatmap")     # ?format=html|json
@router.get("/line")        # ?format=html|json
@router.get("/histogram")   # ?format=html|json
```

**Usage examples:**

```
# Browser — interactive chart (default)
GET /api/charts/pie?table=foresight_cheque_clearing&column=cheque_status
→ 200 text/html — full interactive PyECharts pie chart

# JSON — raw data for custom frontend
GET /api/charts/pie?table=foresight_cheque_clearing&column=cheque_status&format=json
→ 200 application/json
→ { "labels": ["Cleared","Rejected","Pending"], "values": [4521,1203,876] }

# All charts on one page
GET /api/charts/all
→ 200 text/html — all charts stacked vertically
```

---

### 7.21 `api/routers/insights.py`

```python
router = APIRouter(prefix="/api", tags=["Insights"])

@router.get("/insights", response_model=InsightsResponse)
async def get_insights(run_id: Optional[str] = None):
    """Return LLM executive summary from all 4 agents"""
    ...

@router.get("/report", response_class=HTMLResponse)
async def get_report_html(run_id: Optional[str] = None):
    """Return full HTML report (browser-renderable)"""
    ...

@router.get("/report/json")
async def get_report_json(run_id: Optional[str] = None):
    """Return full structured report as JSON"""
    ...

@router.get("/report/download")
async def download_report(run_id: Optional[str] = None):
    """Download report.html as file attachment"""
    ...
```

---

### 7.22 `api/routers/query.py`

```python
router = APIRouter(prefix="/api/query", tags=["Query Agent"])

@router.post("", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Submit a user query. Enhanced RAG pipeline:
    preprocess → hybrid_search(FAISS+BM25) → rerank → similarity_check → LLM answer
    Returns answer + confidence + intent (data_related | out_of_scope)
    """
    session_id = request.session_id or f"sess_{uuid4().hex[:8]}"
    result = query_agent.process_query(request.query, session_id)
    return QueryResponse(**result, session_id=session_id)

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Record user feedback (helpful / not_helpful) for a session"""
    feedback_tracker.log_feedback(request.session_id, "helpful" if request.helpful else "not_helpful")
    return { "success": True, "session_id": request.session_id }

@router.get("/sessions", response_model=list[dict])
async def list_sessions(limit: int = 100):
    """List all query sessions with retrieval stats"""
    return feedback_tracker.get_all_sessions(limit=limit)

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full details of a specific query session"""
    return feedback_tracker.get_session(session_id)
```

---

### 7.23 `api/main.py`

```python
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import get_config
from api.routers import pipeline, agents, charts, insights, query

config = get_config()

async def _poll_table_changes(app: FastAPI) -> None:
    """
    Background task started at server startup.
    Every 60 s: checks DB().count(table) for all tables matching TABLE_PREFIX.
    If any row count changed (insert/update/delete) or tables added/removed
    since the last check → auto-triggers a new pipeline run (run_id ends with _auto).
    Cancelled cleanly on server shutdown.
    """
    last_counts: dict[str, int] = {}
    while True:
        await asyncio.sleep(60)
        # ... compare counts, trigger orch.run() via asyncio.to_thread if changed

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.orchestrator = Orchestrator(config)
    # Load FAISS index if it exists → query agent ready
    store = FAISSStore(config)
    if os.path.exists(index_path):
        store.load()
        app.state.faiss_store = store
        app.state.query_agent = QueryAgent(store, config)
    # Start table change polling background task
    poll_task = asyncio.create_task(_poll_table_changes(app))

    yield

    # Shutdown: cancel polling task cleanly
    poll_task.cancel()
    await poll_task

app = FastAPI(title="4sight v3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, ...)
app.include_router(pipeline.router)
app.include_router(agents.router)
app.include_router(charts.router)
app.include_router(insights.router)
app.include_router(query.router)

@app.get("/health")
async def health():
    return { "status": "ok", "model": config.llm_model, "prefix": config.table_prefix }
```

---

## 8. All FastAPI Endpoints

| Method | Endpoint | Response | Description |
|---|---|---|---|
| GET | `/health` | JSON | Server health check |
| POST | `/api/pipeline/run` | JSON | Trigger full pipeline (async) |
| GET | `/api/pipeline/status/{run_id}` | JSON | Poll run status |
| GET | `/api/pipeline/results/{run_id}` | JSON | All 4 agent results |
| GET | `/api/tables` | JSON array | Discovered tables |
| GET | `/api/schema` | JSON | Tables → columns → detected types |
| GET | `/api/agents/numerical` | JSON | Numerical agent results |
| GET | `/api/agents/categorical` | JSON | Categorical agent results |
| GET | `/api/agents/datetime` | JSON | Datetime agent results |
| GET | `/api/agents/text` | JSON | Text/NLP agent results |
| GET | `/api/agents/{name}/table/{tbl}` | JSON | Agent results filtered by table |
| GET | `/api/charts/all` | **HTML** | All charts on one page — renders in browser (disk fallback) |
| GET | `/api/charts/countplot` | **HTML or JSON** | `?format=html` (default) renders in browser; `?format=json` returns chart_data |
| GET | `/api/charts/bar` | **HTML or JSON** | `?format=html` renders in browser; `?format=json` returns chart_data |
| GET | `/api/charts/pie` | **HTML or JSON** | `?format=html` renders in browser; `?format=json` returns chart_data |
| GET | `/api/charts/stacked_bar` | **HTML or JSON** | `?format=html` renders in browser; `?format=json` returns chart_data |
| GET | `/api/charts/heatmap` | **HTML or JSON** | `?format=html` renders in browser; `?format=json` returns chart_data |
| GET | `/api/charts/line` | **HTML or JSON** | `?format=html` renders in browser; `?format=json` returns chart_data |
| GET | `/api/charts/histogram` | **HTML or JSON** | `?format=html` renders in browser; `?format=json` returns chart_data |
| GET | `/api/insights` | JSON | LLM executive summary |
| GET | `/api/report` | HTML | Full HTML report |
| GET | `/api/report/json` | JSON | Full structured report |
| GET | `/api/report/download` | File | Download report.html |
| POST | `/api/query` | JSON | Submit user query (RAG) |
| POST | `/api/query/feedback` | JSON | Submit helpful/not-helpful |
| GET | `/api/query/sessions` | JSON | All query sessions |
| GET | `/api/query/sessions/{id}` | JSON | Specific session details |

---

## 9. LangGraph Workflows

### Master Pipeline Graph

```
[START]
  │
  ▼
load_data_node              (DB().load_all() → raw DataFrames; saves tables_analyzed list)
  │
  ▼
route_schema_node           (schema_router → ColumnRoute list)
  │
  ▼
run_numerical_agent_node    (row-wise LLM prompt: [{col_a: v, col_b: v}, ...])
  │
  ▼
run_categorical_agent_node  (row-wise LLM prompt: [{cat_col: "X", amount: 75000}, ...])
  │
  ▼
run_datetime_agent_node     (row-wise LLM prompt: [{date: "2023-01-15", amount: 75000}, ...])
  │
  ▼
run_text_agent_node         (row-wise LLM prompt: [{remarks: "delayed", status: "Pending"}, ...])
  │
  ▼
generate_charts_node        (PyECharts HTML → disk; raw_data freed from RAM here)
  │
  ▼
generate_insights_node      (Groq LLM executive summary)
  │
  ▼
generate_report_node        ({run_id}.json + {run_id}.html saved to REPORT_OUTPUT_DIR)
  │
  ▼
index_faiss_node            (FAISS + BM25 index built;
                             agent results → {run_id}_results.json on disk;
                             numerical/categorical/datetime/text/charts cleared from RAM)
  │
  ▼
[END]

Background (always running):
  _poll_table_changes()     (every 60s: DB count check → auto-trigger run if tables changed)
```

### Query Agent Graph

```
[START]
  │
  ▼
preprocess_query_node       (QueryPreprocessor.preprocess)
  │
  ▼
retrieve_docs_node          (FAISS hybrid search + dynamic top-k)
  │
  ▼
rerank_docs_node            (CrossEncoder + score fusion)
  │
  ▼
check_similarity_node       (similarity < threshold?)
  │
  ├── [out_of_scope] ──────────────────────┐
  │                                        ▼
  │                              track_feedback_node → [END]
  │
  ▼ [data_related]
generate_answer_node        (Groq LLM + RAG context)
  │
  ▼
assess_confidence_node      (Groq LLM confidence 0.0-1.0)
  │
  ▼
track_feedback_node         (FeedbackTracker.log_retrieval)
  │
  ▼
[END]
```

---

## 10. Query Agent Flow (Enhanced RAG)

### Step-by-step with all feature flags

```
User POST /api/query { "query": "Which branch has highest rejection rate?" }

1. preprocess_query_node
   USE_QUERY_PREPROCESSING=true
   → "which branch has highest rejection rate"
   → "which branch has highest cheque rejection rate"   (abbreviation expansion)

2. retrieve_docs_node
   USE_HYBRID_SEARCH=true, USE_DYNAMIC_TOPK=true
   → FAISS cosine search: top-10 candidate docs
   → BM25 keyword search: scores for "branch rejection rate"
   → Fuse: 0.6 * semantic + 0.4 * bm25
   → dynamic_top_k: keep docs where fused_score >= 0.35
   → returns 5 docs

3. rerank_docs_node
   USE_RERANKING=true
   → CrossEncoder("which branch highest rejection rate", doc_text) × 5 pairs
   → cross_score normalized 0-1
   → fused_score = 0.5 * cross_score + 0.5 * original_similarity
   → re-sorted

4. check_similarity_node
   top_doc.fused_score = 0.78 >= 0.35 threshold
   → query_intent = "data_related"
   → proceed to generate_answer

5. generate_answer_node
   context = "[1] (score=0.78) Table: foresight_cheque_clearing | Column: branch_name |
              Categorical | Frequency: Branch_A=1203 rejected, Branch_B=876 rejected..."
   prompt → Groq LLaMA-3.1-8B-Instant
   answer = "Branch_A has the highest cheque rejection rate at 34.2% (1,203 out of 3,512 cheques)..."

6. assess_confidence_node
   confidence = 0.87 → can_resolve = True

7. track_feedback_node
   → INSERT into foresight_query_sessions

Response:
{
    "session_id": "sess_a1b2c3",
    "answer": "Branch_A has the highest cheque rejection rate at 34.2%...",
    "confidence": 0.87,
    "resolved": true,
    "query_intent": "data_related",
    "top_similarity": 0.78,
    "num_docs_retrieved": 5
}
```

### Out-of-scope example

```
User POST /api/query { "query": "What is the weather in Dubai?" }

→ retrieve_docs: top similarity = 0.12 < 0.35 threshold
→ check_similarity: query_intent = "out_of_scope"
→ final_answer = "I can only answer questions related to the financial data analysis."
→ track_feedback (logs the rejection)

Response:
{
    "answer": "I can only answer questions related to the financial data analysis.",
    "confidence": 0.0,
    "resolved": false,
    "query_intent": "out_of_scope",
    "top_similarity": 0.12
}
```

---

## 11. PyECharts Visualization Strategy

### Chart Auto-Selection Logic

```
AgentResult.chart_type is set BY the agent (dynamic, based on data):

Numerical agent:
  - Single column distribution        → "histogram"
  - Multiple numeric columns          → "heatmap" (correlation matrix)
  - Time-indexed numeric              → "line"

Categorical agent:
  - nunique <= 6                       → "pie"
  - 6 < nunique <= 20                  → "bar" or "countplot"
  - Has numeric target column          → "stacked_bar"
  - Pure frequency distribution        → "countplot"

Datetime agent:
  - Always                             → "line"

Text agent:
  - Keyword frequencies                → "bar"
  - Sentiment distribution             → "pie"
```

### Chart Rendering — How It Works

All `/api/charts/{type}` endpoints support two formats via `?format=` query param:
- **`?format=html`** (default) — `text/html` response, PyECharts chart renders in browser
- **`?format=json`** — `application/json` response, raw `chart_data` dict for custom frontends

Charts are saved to `CHART_OUTPUT_DIR` as `.html` files by `chart_generator.py`.
After RAM cleanup, endpoints automatically fall back to reading from disk.

```
# Browser — interactive chart
GET /api/charts/pie?table=foresight_cheque_clearing&column=cheque_status
→ 200 text/html
→ Full interactive pie chart: Cleared / Rejected / Pending
   (hover = tooltip, click legend = toggle slice, zoom supported)

# JSON — raw data for custom frontend or API client
GET /api/charts/pie?table=foresight_cheque_clearing&column=cheque_status&format=json
→ 200 application/json
→ { "labels": ["Cleared","Rejected","Pending"], "values": [4521,1203,876] }

GET /api/charts/all
→ 200 text/html
→ Single page showing ALL charts stacked vertically
   (reads from CHART_OUTPUT_DIR if in-memory charts dict was cleared)
```

---

## 12. Build Order & Dependencies

| Step | File | Depends On | Est. Complexity |
|---|---|---|---|
| 1 | `config.py` | `.env` | Low |
| 2 | `core/state.py` | `config.py` | Low |
| 3 | `core/schema_router.py` | `db.py`, `config.py` | Medium |
| 4 | `agents/numerical_agent.py` | `core/state.py`, `config.py`, Groq | High |
| 5 | `agents/categorical_agent.py` | `core/state.py`, `config.py`, Groq | High |
| 6 | `agents/datetime_agent.py` | `core/state.py`, `config.py`, Groq | High |
| 7 | `agents/text_agent.py` | `core/state.py`, `config.py`, Groq | High |
| 8 | `visualization/chart_generator.py` | `core/state.py`, `config.py`, pyecharts | Medium |
| 9 | `output/insight_agent.py` | all agents, Groq | Medium |
| 10 | `output/reporting_agent.py` | `insight_agent.py`, `chart_generator.py` | Medium |
| 11 | `query_agent/faiss_store.py` | FAISS, BM25, sentence-transformers | High |
| 12 | `query_agent/query_preprocessor.py` | `config.py` | Low |
| 13 | `query_agent/reranker.py` | cross-encoder, `config.py` | Medium |
| 14 | `query_agent/feedback_tracker.py` | `db.py`, `config.py` | Medium |
| 15 | `query_agent/query_agent.py` | steps 11-14, LangGraph, Groq | High |
| 16 | `orchestrator.py` | steps 3-10, LangGraph | High |
| 17 | `api/schemas.py` | Pydantic | Low |
| 18 | `api/routers/pipeline.py` | `orchestrator.py`, `schemas.py` | Medium |
| 19 | `api/routers/agents.py` | `orchestrator.py`, `schemas.py` | Medium |
| 20 | `api/routers/charts.py` | `chart_generator.py`, `schemas.py` | Medium |
| 21 | `api/routers/insights.py` | `orchestrator.py`, `schemas.py` | Low |
| 22 | `api/routers/query.py` | `query_agent.py`, `schemas.py` | Medium |
| 23 | `api/main.py` | all routers | Low |

---

## 13. Package Requirements

```txt
# requirements.txt

# Core
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.26.0
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.0

# LLM & Agents
langchain-groq>=0.2.0
langchain>=0.3.0
langgraph>=0.2.0

# Embeddings & Vector Search
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
rank-bm25>=0.2.2

# Re-ranking
# (cross-encoder is included in sentence-transformers)

# NLP / Text Analysis
scikit-learn>=1.4.0
nltk>=3.8.0
gensim>=4.3.0          # LDA topic modeling

# Visualization
pyecharts>=2.0.0

# API
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0

# Excel loader (existing)
openpyxl>=3.1.0
xlrd>=2.0.1

# Utilities
python-multipart>=0.0.9
httpx>=0.27.0
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your PostgreSQL credentials and GROQ_API_KEY

# 3. Load Excel data (if not already loaded)
python excel_to_postgres.py your_data.xlsx --prefix foresight

# 4. Start the API server
python -m api.main
# OR
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Trigger the analysis pipeline
curl -X POST http://localhost:8000/api/pipeline/run

# 6. Poll status
curl http://localhost:8000/api/pipeline/status/{run_id}

# 7. View charts in browser
open http://localhost:8000/api/charts/pie
open http://localhost:8000/api/charts/countplot
open http://localhost:8000/api/charts/bar

# 8. Get insights
curl http://localhost:8000/api/insights

# 9. Query the data
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which branch has the highest rejection rate?"}'

# 10. View full API docs
open http://localhost:8000/docs
```

---

*Document version: 1.0 | Generated: 2026-05-15 | Project: 4sight v3*
