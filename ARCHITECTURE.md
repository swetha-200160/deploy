# 4sight v3 — System Architecture & LLM Flow

## Overview

4sight v3 is a multi-agent financial data analysis system built on LangGraph. It has **two independent flows**:
1. **Analysis Pipeline** — reads PostgreSQL tables, runs type-specific agents, generates charts + insights
2. **Query Flow (RAG)** — answers natural-language questions against the indexed analysis results

Two LLMs are used throughout:
- **Gemini** (`ChatGoogleGenerativeAI`) — code generation + per-column insights (high quality, structured output)
- **Groq LLaMA** (`ChatGroq`) — executive insights, RAG answers, confidence scoring (fast, cost-efficient)

---

## 1. Main Analysis Pipeline

```mermaid
flowchart TD
    A([User / Auto-trigger\nPOST /api/pipeline/run]) --> B

    subgraph INGEST["① Data Ingestion"]
        B[load_data_node\nDB.load_all → raw_data dict]
        B --> C[route_schema_node\nSchema Router]
        C --> C1{Classify each column}
        C1 -->|dtype + heuristics| C2[numerical / categorical\ndatetime / text]
    end

    C2 --> D

    subgraph AGENTS["② Type-Specific Agents  —  run sequentially"]
        D[NumericalAgent] --> E[CategoricalAgent]
        E --> F[DatetimeAgent]
        F --> G[TextAgent]
        G --> H[EnterpriseAgent\nfraud · quality · drift]
    end

    subgraph AGENT_DETAIL["Inside each Type Agent  (Numerical shown)"]
        direction TB
        AD1["① pandas_analysis_agent\nGemini Code-Gen"]
        AD2["② NumericalAgent\nGemini Insight"]
        AD1 -->|analysis dict| AD2
    end

    D -.->|per column| AGENT_DETAIL

    AGENTS --> I

    subgraph OUTPUT["③ Output Generation"]
        I[generate_charts_node\nPyECharts HTML files]
        I --> J[generate_insights_node\nInsightAgent]
        J --> K[generate_report_node\nReportingAgent]
        K --> L[index_faiss_node\nFAISS + BM25 index]
    end

    L --> M([Pipeline Done\nstatus = done])

    style INGEST fill:#e8f4fd,stroke:#2196F3
    style AGENTS fill:#fff3e0,stroke:#FF9800
    style AGENT_DETAIL fill:#fce4ec,stroke:#E91E63
    style OUTPUT fill:#e8f5e9,stroke:#4CAF50
```

---

## 2. LLM Call — Gemini Code Generation (pandas_analysis_agent)

Called once per column, per agent type.

```mermaid
sequenceDiagram
    participant A as Type Agent<br/>(Numerical / Categorical /<br/>Datetime / Text)
    participant P as pandas_analysis_agent
    participant G as Gemini<br/>(ChatGoogleGenerativeAI)
    participant E as Safe Exec

    A->>P: run_pandas_analysis(df, col, type)
    Note over P: Build prompt:<br/>DataFrame schema (rows, dtypes)<br/>+ per-type analysis spec

    P->>G: SystemMessage(_SYSTEM)<br/>HumanMessage(schema + spec prompt)
    Note over G: INPUT:<br/>• System: "pandas expert" rules<br/>• Human: df schema + column info<br/>  + full analysis keys to compute
    G-->>P: OUTPUT: ```python<br/>result = {<br/>  'descriptive_stats': {...},<br/>  'outliers_iqr': {...},<br/>  'financial_kpis': {...},<br/>  'fraud_signals': {...},<br/>  ...<br/>}```

    P->>E: ast.parse() + exec(code)
    E-->>P: result dict

    alt SyntaxError (up to 2 retries)
        P->>G: same prompt + error message
        G-->>P: corrected code
    end

    P-->>A: analysis dict
```

**Input to Gemini:**
```
DATAFRAME SCHEMA — 5000 rows
Columns: invoice_amount (float64), vendor_id (object), date (datetime64)
TARGET COLUMN: `invoice_amount`  |  dtype: float64  |  non-null: 4987/5000
------------------------------------------------------------
Analyze column `invoice_amount` in DataFrame `df`. Build `result` with:
  descriptive_stats — dict: mean, median, std, min, max, skewness ...
  outliers_iqr — dict: count, pct  ...
  financial_kpis — dict: total_sum, trend_direction, var_95, sharpe_ratio ...
  fraud_signals — dict: round_number_pct, negative_value_count, fraud_risk_level ...
```

**Output from Gemini:**
```python
result = {
    'descriptive_stats': {'mean': 45230.5, 'median': 38000.0, ...},
    'outliers_iqr': {'count': 47, 'pct': 0.94},
    'financial_kpis': {'total_sum': 225678900.0, 'trend_direction': 'up', ...},
    'fraud_signals': {'round_number_pct': 12.3, 'fraud_risk_level': 'medium'},
}
```

---

## 3. LLM Call — Gemini Insight Text (per column)

After the analysis dict is computed, the same agent makes a second Gemini call to generate human-readable insight.

```mermaid
sequenceDiagram
    participant A as NumericalAgent
    participant G as Gemini

    A->>G: SystemMessage("financial data analyst, 2-3 sentence insight")<br/>HumanMessage(compact stats summary)
    Note over G: INPUT:<br/>Column 'invoice_amount' | Table 'invoices'<br/>Stats: mean=45230, median=38000, std=18900<br/>Outliers IQR: 47 (0.94%), Z-score: 12 (0.24%)<br/>Financial: trend=up, VaR95=8200, sharpe=1.23<br/>Fraud risk: medium | round_numbers=12.3%
    G-->>A: OUTPUT: "invoice_amount shows a right-skewed distribution<br/>with 47 IQR outliers. The 12.3% round-number rate<br/>and medium fraud risk warrant further investigation..."
```

> Same pattern applies to **CategoricalAgent**, **DatetimeAgent**, and **TextAgent** — each uses Gemini for both code generation and insight text.

---

## 4. LLM Call — Groq Executive Insight (InsightAgent)

Runs once after all type-specific agents complete.

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant I as InsightAgent
    participant G as Groq LLaMA

    O->>I: run(numerical_results, categorical_results,<br/>datetime_results, text_results, enterprise_results)
    Note over I: Concatenate up to 3000 chars<br/>of per-column insight summaries

    I->>G: SystemMessage("senior financial analyst, return JSON only")<br/>HumanMessage(all agent summaries)
    Note over G: INPUT:<br/>NUMERICAL: Table invoices | Column invoice_amount |<br/>  Insight: right-skewed, 47 outliers...<br/>CATEGORICAL: Table vendors | Column status |<br/>  Insight: 23% rejected...<br/>DATETIME: Table invoices | Column date | ...<br/>ENTERPRISE: quality=87 (B), fraud_risk=medium, drift=low

    G-->>I: OUTPUT JSON:<br/>{<br/>  "key_findings": ["47 invoice outliers detected", ...],<br/>  "risk_signals": ["12.3% round numbers in amounts", ...],<br/>  "recommendations": ["Investigate vendor invoice rounding", ...],<br/>  "data_quality_score": 84,<br/>  "overall_narrative": "The financial dataset shows..."<br/>}
```

---

## 5. RAG Query Flow (QueryAgent)

Separate LangGraph pipeline, available after pipeline completes.

```mermaid
flowchart TD
    U([User\nPOST /api/query\n{"query": "..."}]) --> QP

    subgraph QG["LangGraph — QueryAgent"]
        QP[preprocess_query\nQueryPreprocessor\nexpand + normalize] --> RD
        RD[retrieve_docs\nFAISS + BM25\nhybrid search] --> RR
        RR[rerank_docs\nReciprocal Rank Fusion] --> CS

        CS{check_similarity\ntop score ≥ threshold?}
        CS -->|NO — out of scope| TF
        CS -->|YES — data related| GA

        GA[generate_answer\nGroq LLaMA] --> AC
        AC[assess_confidence\nGroq LLaMA] --> TF
        TF[track_feedback\nlog to FeedbackTracker]
    end

    TF --> R([Response\n{answer, confidence, resolved}])

    style QG fill:#f3e5f5,stroke:#9C27B0
```

### Query LLM Calls

```mermaid
sequenceDiagram
    participant Q as QueryAgent
    participant F as FAISSStore
    participant G as Groq LLaMA

    Q->>F: hybrid_search(preprocessed_query, k=10)
    F-->>Q: top-k docs with similarity scores

    Note over Q: Reciprocal Rank Fusion reranking

    Q->>G: [LLM CALL 1 — Answer Generation]<br/>SystemMessage("financial analyst, use context only")<br/>HumanMessage(CONTEXT: top-k docs + QUESTION: query)
    Note over G: INPUT:<br/>CONTEXT: [analysis result chunks from FAISS]<br/>  "invoice_amount: mean=45230, 47 outliers..."<br/>  "vendor status: 23% rejected vendors..."<br/>QUESTION: "Which tables have the highest fraud risk?"
    G-->>Q: OUTPUT: "The invoices table has the highest fraud risk<br/>with a medium risk level, driven by 12.3% round<br/>numbers in invoice_amount..."

    Q->>G: [LLM CALL 2 — Confidence Scoring]<br/>HumanMessage("Rate quality 0.0-1.0. Query:... Answer:...")
    Note over G: INPUT: query text + first 300 chars of answer<br/>+ number of documents retrieved
    G-->>Q: OUTPUT: "0.82"
```

---

## 6. Excel Tool Agent Flow

Used for direct questions against uploaded Excel/CSV files.

```mermaid
flowchart LR
    U([User Question\n+ session_id]) --> LS
    LS[Load Session Files\nfrom PostgreSQL Documents] --> CD
    CD[Clean DataFrames\nnumeric / date parsing] --> BA
    BA[Build Agent Prefix\nlist DFs + shapes + instructions] --> AG

    subgraph LA["LangChain Pandas Agent  —  Groq LLaMA"]
        AG[create_pandas_dataframe_agent\ntool-calling mode] --> LLM
        LLM[Groq LLaMA] -->|generates pandas code| TL
        TL[Python REPL Tool\nexecutes code on df] -->|result| LLM
    end

    LLM -->|final answer| R([ExcelAgentResult\nanswer + source_files])
```

---

## 7. Complete System Map

```mermaid
flowchart TB
    subgraph EXTERNAL["External Inputs"]
        XL[Excel / CSV Upload]
        PG[(PostgreSQL\nTables)]
        API_IN[REST API\nPOST /api/pipeline/run]
        Q_IN[REST API\nPOST /api/query]
    end

    XL -->|excel_to_postgres.py| PG
    API_IN --> ORCH

    subgraph ORCH["Orchestrator  (LangGraph)"]
        N1[load_data] --> N2[route_schema]
        N2 --> N3[numerical_agent]
        N3 --> N4[categorical_agent]
        N4 --> N5[datetime_agent]
        N5 --> N6[text_agent]
        N6 --> N7[enterprise_agent]
        N7 --> N8[generate_charts]
        N8 --> N9[generate_insights]
        N9 --> N10[generate_report]
        N10 --> N11[index_faiss]
    end

    PG -->|DB.load_all| N1

    subgraph LLM_CALLS["LLM Calls  (inside agents)"]
        GEM1["Gemini\nCode Gen\n×N columns"]
        GEM2["Gemini\nInsight Text\n×N columns"]
        GRQ1["Groq LLaMA\nExecutive JSON\n×1"]
    end

    N3 & N4 & N5 & N6 -.->|pandas_analysis_agent| GEM1
    N3 & N4 & N5 & N6 -.->|insight call| GEM2
    N9 -.->|InsightAgent| GRQ1

    subgraph STORAGE["Outputs"]
        CHARTS[PyECharts HTML\nCHART_OUTPUT_DIR]
        REPORT[Report JSON\nREPORT_OUTPUT_DIR]
        FAISS_IDX[(FAISS Index\n+ BM25)]
    end

    N8 --> CHARTS
    N10 --> REPORT
    N11 --> FAISS_IDX

    Q_IN --> QA

    subgraph QA["QueryAgent  (LangGraph)"]
        QN1[preprocess] --> QN2[retrieve]
        QN2 --> QN3[rerank]
        QN3 --> QN4{similarity check}
        QN4 -->|pass| QN5[generate_answer]
        QN5 --> QN6[assess_confidence]
        QN4 & QN6 --> QN7[track_feedback]
    end

    FAISS_IDX -->|hybrid search| QN2

    subgraph QA_LLM["Query LLM Calls"]
        GRQ2["Groq LLaMA\nAnswer Gen"]
        GRQ3["Groq LLaMA\nConfidence Score"]
    end

    QN5 -.-> GRQ2
    QN6 -.-> GRQ3

    QN7 --> RESP([JSON Response\nanswer · confidence · resolved])

    style EXTERNAL fill:#eceff1,stroke:#607D8B
    style ORCH fill:#e3f2fd,stroke:#1565C0
    style LLM_CALLS fill:#fce4ec,stroke:#AD1457
    style STORAGE fill:#e8f5e9,stroke:#2E7D32
    style QA fill:#f3e5f5,stroke:#6A1B9A
    style QA_LLM fill:#fce4ec,stroke:#AD1457
```

---

## 8. LLM Summary Table

| Call Site | Model | Input | Output | When |
|-----------|-------|-------|--------|------|
| `pandas_analysis_agent` | **Gemini** | DataFrame schema + per-type analysis spec | Python/pandas code block | Once per column (×N) |
| `NumericalAgent._run_column` | **Gemini** | Compact stats summary (text) | 2-3 sentence insight | Once per numerical column |
| `CategoricalAgent._run_column` | **Gemini** | Category freq summary | 2-3 sentence insight | Once per categorical column |
| `DatetimeAgent._run_column` | **Gemini** | Date range / trend summary | 2-3 sentence insight | Once per datetime column |
| `TextAgent._run_column` | **Gemini** | Sentiment / top-words summary | 2-3 sentence insight | Once per text column |
| `InsightAgent.run` | **Groq LLaMA** | All agent summaries concatenated | JSON executive summary | Once per pipeline run |
| `QueryAgent._generate_answer_node` | **Groq LLaMA** | FAISS context + user question | Answer text | Once per query |
| `QueryAgent._assess_confidence_node` | **Groq LLaMA** | Query + answer + doc count | Confidence float (0–1) | Once per query |
| `ExcelToolAgent._invoke_agent_sync` | **Groq LLaMA** | DataFrame(s) + user question | Answer (multi-turn tool-calling) | Once per Excel query |

---

## 9. State Flow Through Pipeline

```mermaid
stateDiagram-v2
    [*] --> initialized: POST /api/pipeline/run
    initialized --> running: process() called
    running --> running: each agent node updates PipelineState

    state running {
        [*] --> load_data
        load_data --> route_schema: raw_data dict
        route_schema --> numerical_agent: column_routes list
        numerical_agent --> categorical_agent: numerical_results []
        categorical_agent --> datetime_agent: categorical_results []
        datetime_agent --> text_agent: datetime_results []
        text_agent --> enterprise_agent: text_results []
        enterprise_agent --> generate_charts: enterprise_results {}
        generate_charts --> generate_insights: charts dict + RAM freed
        generate_insights --> generate_report: insights JSON
        generate_report --> index_faiss: report dict
        index_faiss --> [*]: results saved to disk + RAM freed
    }

    running --> done: index_faiss completes
    running --> failed: any node raises unhandled exception
    done --> [*]
    failed --> [*]
```
