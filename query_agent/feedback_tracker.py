from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text

from config import Config, get_config

log = logging.getLogger(__name__)


class FeedbackTracker:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        url = (
            f"postgresql+psycopg2://{self.config.postgres_user}:{self.config.postgres_password}"
            f"@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}"
        )
        self._engine = create_engine(url, pool_size=2, max_overflow=1)
        self._table = f"{self.config.table_prefix}_query_sessions"
        self._ensure_table()

    def _ensure_table(self) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self._table} (
            id                SERIAL PRIMARY KEY,
            session_id        TEXT NOT NULL,
            original_query    TEXT,
            preprocessed_query TEXT,
            retrieved_doc_ids  TEXT[],
            similarity_scores  FLOAT[],
            final_answer      TEXT,
            confidence_score  FLOAT,
            query_intent      TEXT,
            feedback          TEXT,
            created_at        TIMESTAMP DEFAULT NOW()
        )
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(text(ddl))
            log.info("Feedback table ready: %s", self._table)
        except Exception as exc:
            log.error("Could not create feedback table: %s", exc)

    def log_retrieval(
        self,
        session_id: str,
        query: str,
        preprocessed: str,
        doc_ids: list[str],
        scores: list[float],
        answer: str,
        confidence: float,
        intent: str,
    ) -> None:
        if not self.config.track_feedback:
            return
        sql = f"""
        INSERT INTO {self._table}
            (session_id, original_query, preprocessed_query, retrieved_doc_ids,
             similarity_scores, final_answer, confidence_score, query_intent)
        VALUES
            (:sid, :oq, :pq, :dids, :scores, :ans, :conf, :intent)
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(text(sql), {
                    "sid": session_id,
                    "oq": query,
                    "pq": preprocessed,
                    "dids": doc_ids,
                    "scores": scores,
                    "ans": answer,
                    "conf": confidence,
                    "intent": intent,
                })
        except Exception as exc:
            log.error("log_retrieval failed: %s", exc)

    def log_feedback(self, session_id: str, feedback: str) -> None:
        sql = f"UPDATE {self._table} SET feedback = :fb WHERE session_id = :sid"
        try:
            with self._engine.begin() as conn:
                conn.execute(text(sql), {"fb": feedback, "sid": session_id})
        except Exception as exc:
            log.error("log_feedback failed: %s", exc)

    def get_session(self, session_id: str) -> dict:
        sql = f"SELECT * FROM {self._table} WHERE session_id = :sid ORDER BY created_at DESC LIMIT 1"
        try:
            import pandas as pd
            with self._engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn, params={"sid": session_id})
            if df.empty:
                return {}
            return df.iloc[0].to_dict()
        except Exception as exc:
            log.error("get_session failed: %s", exc)
            return {}

    def get_all_sessions(self, limit: int = 100) -> list[dict]:
        sql = f"SELECT * FROM {self._table} ORDER BY created_at DESC LIMIT :lim"
        try:
            import pandas as pd
            with self._engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn, params={"lim": limit})
            return df.to_dict(orient="records")
        except Exception as exc:
            log.error("get_all_sessions failed: %s", exc)
            return []
