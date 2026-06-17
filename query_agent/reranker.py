from __future__ import annotations

import logging

from config import Config, get_config

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.alpha = config.reranker_alpha if config else 0.5
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                device = "cuda" if self.config.use_gpu else "cpu"
                self._model = CrossEncoder(self.config.reranker_model, device=device)
            except Exception as exc:
                log.warning("CrossEncoder not available: %s. Falling back to similarity-only ranking.", exc)
        return self._model

    def rerank_with_fusion(self, query: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return documents

        model = self._get_model()
        if model is None:
            # Fallback: set fused_score = similarity
            for doc in documents:
                doc["fused_score"] = doc.get("similarity", 0.0)
                doc["cross_score"] = 0.0
            return documents

        try:
            pairs = [(query, doc["text"]) for doc in documents]
            cross_scores = model.predict(pairs)

            # Min-max normalization: maps any range (including all-negative) to [0, 1]
            cs_min = float(min(cross_scores))
            cs_max = float(max(cross_scores))
            cs_range = cs_max - cs_min
            if cs_range < 1e-6:
                norm_scores = [0.5] * len(cross_scores)
            else:
                norm_scores = [(float(cs) - cs_min) / cs_range for cs in cross_scores]

            for doc, norm_cs in zip(documents, norm_scores):
                doc["cross_score"] = norm_cs
                doc["fused_score"] = self.alpha * norm_cs + (1 - self.alpha) * doc.get("similarity", 0.0)

            documents.sort(key=lambda x: x["fused_score"], reverse=True)
        except Exception as exc:
            log.error("Reranking failed: %s", exc)
            for doc in documents:
                doc["fused_score"] = doc.get("similarity", 0.0)
                doc["cross_score"] = 0.0

        return documents
