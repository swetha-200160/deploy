from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def _detect_gpu(val: str | None) -> bool:
    if val and val.strip().lower() in ("false", "0", "no"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import faiss
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


@dataclass
class Config:
    # DB
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    table_prefix: str

    # LLM — Groq (row agents: universal, enterprise, wow — many calls, keep fast)
    groq_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # LLM — Groq output agents (insight, story — 1-2 calls per run, best quality)
    output_llm_model: str
    output_llm_max_tokens: int

    # LLM — Gemini (Categorical agent)
    gemini_api_key: str
    gemini_model: str

    # FAISS
    embedding_model: str
    faiss_index_path: str
    faiss_index_file: str
    faiss_meta_file: str
    raw_index_file: str
    raw_meta_file: str

    # Search
    semantic_weight: float
    bm25_weight: float
    similarity_threshold: float
    min_top_k: int
    max_top_k: int
    final_top_k: int
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

    # Numerical Analysis
    outlier_method: str
    zscore_threshold: float
    iqr_multiplier: float
    correlation_method: str
    distribution_bins: int

    # Categorical Analysis
    max_categorical_cardinality: int
    rare_category_threshold: float
    imbalance_ratio_threshold: float

    # Datetime Analysis
    seasonality_period: int
    rolling_window: int
    forecast_periods: int

    # Text / NLP
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

    # Hardware
    use_gpu: bool

    # API
    api_host: str
    api_port: int
    api_reload: bool
    cors_origins: list


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is not None:
        return _config

    cors_raw = os.getenv("CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_raw.split(",")] if cors_raw != "*" else ["*"]

    _config = Config(
        postgres_host=os.getenv("POSTGRES_HOST"),
        postgres_port=int(os.getenv("POSTGRES_PORT")),
        postgres_db=os.getenv("POSTGRES_DB"),
        postgres_user=os.getenv("POSTGRES_USER"),
        postgres_password=os.getenv("POSTGRES_PASSWORD"),
        table_prefix=os.getenv("TABLE_PREFIX"),

        groq_api_key=os.getenv("GROQ_API_KEY"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS")),
        output_llm_model=os.getenv("OUTPUT_LLM_MODEL", os.getenv("LLM_MODEL")),
        output_llm_max_tokens=int(os.getenv("OUTPUT_LLM_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "3500"))),

        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL"),

        embedding_model=os.getenv("EMBEDDING_MODEL"),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH"),
        faiss_index_file=os.getenv("FAISS_INDEX_FILE"),
        faiss_meta_file=os.getenv("FAISS_META_FILE"),
        raw_index_file=os.getenv("RAW_INDEX_FILE", "raw_index.faiss"),
        raw_meta_file=os.getenv("RAW_META_FILE", "raw_meta.json"),

        semantic_weight=float(os.getenv("SEMANTIC_WEIGHT")),
        bm25_weight=float(os.getenv("BM25_WEIGHT")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD")),
        min_top_k=int(os.getenv("MIN_TOP_K")),
        max_top_k=int(os.getenv("MAX_TOP_K")),
        final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
        use_dynamic_topk=_bool(os.getenv("USE_DYNAMIC_TOPK")),

        use_reranking=_bool(os.getenv("USE_RERANKING")),
        reranker_model=os.getenv("RERANKER_MODEL"),
        reranker_alpha=float(os.getenv("RERANKER_ALPHA")),

        use_query_preprocessing=_bool(os.getenv("USE_QUERY_PREPROCESSING")),
        use_hybrid_search=_bool(os.getenv("USE_HYBRID_SEARCH")),
        track_feedback=_bool(os.getenv("TRACK_FEEDBACK")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD")),
        high_priority_threshold=float(os.getenv("HIGH_PRIORITY_THRESHOLD")),

        outlier_method=os.getenv("OUTLIER_METHOD"),
        zscore_threshold=float(os.getenv("ZSCORE_THRESHOLD")),
        iqr_multiplier=float(os.getenv("IQR_MULTIPLIER")),
        correlation_method=os.getenv("CORRELATION_METHOD"),
        distribution_bins=int(os.getenv("DISTRIBUTION_BINS")),

        max_categorical_cardinality=int(os.getenv("MAX_CATEGORICAL_CARDINALITY", "20")),
        rare_category_threshold=float(os.getenv("RARE_CATEGORY_THRESHOLD")),
        imbalance_ratio_threshold=float(os.getenv("IMBALANCE_RATIO_THRESHOLD")),

        seasonality_period=int(os.getenv("SEASONALITY_PERIOD")),
        rolling_window=int(os.getenv("ROLLING_WINDOW")),
        forecast_periods=int(os.getenv("FORECAST_PERIODS")),

        max_keywords=int(os.getenv("MAX_KEYWORDS")),
        min_keyword_freq=int(os.getenv("MIN_KEYWORD_FREQ")),
        sentiment_batch_size=int(os.getenv("SENTIMENT_BATCH_SIZE")),
        tfidf_max_features=int(os.getenv("TFIDF_MAX_FEATURES")),

        chart_output_dir=os.getenv("CHART_OUTPUT_DIR"),
        chart_theme=os.getenv("CHART_THEME"),
        chart_width=os.getenv("CHART_WIDTH"),
        chart_height=os.getenv("CHART_HEIGHT"),

        report_output_dir=os.getenv("REPORT_OUTPUT_DIR"),

        use_gpu=_detect_gpu(os.getenv("USE_GPU")),

        api_host=os.getenv("API_HOST"),
        api_port=int(os.getenv("API_PORT")),
        api_reload=_bool(os.getenv("API_RELOAD")),
        cors_origins=cors_origins,
    )
    return _config
