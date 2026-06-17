from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import Config, get_config
from core.state import AgentResult

log = logging.getLogger(__name__)


class FAISSStore:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        device = "cuda" if self.config.use_gpu else "cpu"
        self.embedding_model = SentenceTransformer(self.config.embedding_model, device=device)
        self._gpu_res = None
        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self.documents: list[dict] = []

    def _move_index_to_gpu(self) -> None:
        if not self.config.use_gpu or self.index is None:
            return
        try:
            self._gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self.index)
            log.info("FAISS index moved to GPU.")
        except Exception as exc:
            log.warning("Failed to move FAISS index to GPU, staying on CPU: %s", exc)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_analysis_results(self, agent_results: list[AgentResult], insights: dict, run_id: str | None = None) -> None:
        chunks = self._build_chunks(agent_results, insights)

        if not chunks:
            log.warning("No chunks to index — FAISS index not built.")
            return

        self.documents = chunks
        texts = [c["text"] for c in chunks]

        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)
        int_ids = np.array([self._chunk_int_id(c["id"]) for c in chunks], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype("float32"), int_ids)
        self._move_index_to_gpu()

        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        self._save(run_id=run_id)
        log.info("FAISS index built: %d documents, dim=%d (run_id=%s)", len(chunks), dim, run_id)

    # ── Search ────────────────────────────────────────────────────────────────

    def hybrid_search(self, query: str, k: int | None = None) -> list[dict]:
        if self.index is None or not self.documents:
            log.warning("FAISS index not loaded — returning empty results.")
            return []

        k = k or self.config.min_top_k
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)

        n = len(self.documents)
        semantic_scores, indices = self.index.search(query_vec.astype("float32"), n)
        semantic_map = {int(i): float(s) for i, s in zip(indices[0], semantic_scores[0])}

        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        fused = []
        for idx, doc in enumerate(self.documents):
            sem = semantic_map.get(idx, 0.0)
            bm = float(bm25_norm[idx])
            score = self.config.semantic_weight * sem + self.config.bm25_weight * bm
            fused.append({**doc, "similarity": score, "semantic_score": sem, "bm25_score": bm})

        fused.sort(key=lambda x: x["similarity"], reverse=True)
        return fused[:k]

    def dynamic_top_k_search(self, query: str) -> list[dict]:
        results = self.hybrid_search(query, k=self.config.max_top_k)
        filtered = [r for r in results if r["similarity"] >= self.config.similarity_threshold]
        if len(filtered) < self.config.min_top_k:
            return results[:self.config.min_top_k]
        return filtered

    def semantic_only_search(self, query: str, k: int | None = None) -> list[dict]:
        if self.index is None or not self.documents:
            return []
        k = k or self.config.min_top_k
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_vec.astype("float32"), k)
        results = []
        for i, s in zip(indices[0], scores[0]):
            if i < len(self.documents):
                results.append({**self.documents[i], "similarity": float(s), "semantic_score": float(s), "bm25_score": 0.0})
        return results

    def format_context(self, docs: list[dict]) -> str:
        lines = []
        for i, doc in enumerate(docs, 1):
            score = doc.get("fused_score", doc.get("similarity", 0))
            lines.append(f"[{i}] (score={score:.3f}) {doc['text']}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    # ── Incremental update helpers ────────────────────────────────────────────

    @staticmethod
    def _chunk_int_id(chunk_id: str) -> int:
        """Stable int64 ID from a string chunk ID — used by IndexIDMap2."""
        return int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2 ** 62)

    def _ensure_incremental_index(self) -> None:
        """Upgrade IndexFlatIP → IndexIDMap2 in-place if needed.

        Existing indexes saved before this change are loaded as IndexFlatIP.
        This converts them transparently so add_chunk/remove_chunk work without
        requiring a pipeline re-run.
        """
        if self.index is None or isinstance(self.index, faiss.IndexIDMap2):
            return
        log.info("Upgrading index to IndexIDMap2 for incremental updates …")
        n = self.index.ntotal
        d = self.index.d
        vectors = np.zeros((n, d), dtype="float32")
        self.index.reconstruct_n(0, n, vectors)
        base = faiss.IndexFlatIP(d)
        new_index = faiss.IndexIDMap2(base)
        int_ids = np.array([self._chunk_int_id(doc["id"]) for doc in self.documents], dtype=np.int64)
        new_index.add_with_ids(vectors, int_ids)
        self.index = new_index
        log.info("Index upgraded: %d vectors re-indexed.", n)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index after incremental changes."""
        tokenized = [d["text"].lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def get_raw_row_hashes(self) -> dict[str, str]:
        """Return {chunk_id: row_hash} for every indexed raw-row chunk."""
        return {
            d["id"]: d["metadata"]["row_hash"]
            for d in self.documents
            if d.get("metadata", {}).get("type") == "raw_row"
        }

    def add_chunk(self, chunk: dict) -> None:
        """Insert or replace one chunk in the live index (no full rebuild)."""
        # Remove stale copy if it exists
        if any(d["id"] == chunk["id"] for d in self.documents):
            self.remove_chunk(chunk["id"])
        self.documents.append(chunk)
        vec = self.embedding_model.encode([chunk["text"]], normalize_embeddings=True)
        # Initialize index on very first add (raw store starts with index=None)
        if self.index is None:
            dim = vec.shape[1]
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap2(base)
            self._move_index_to_gpu()
        else:
            self._ensure_incremental_index()
        int_id = np.array([self._chunk_int_id(chunk["id"])], dtype=np.int64)
        self.index.add_with_ids(vec.astype("float32"), int_id)

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove one chunk from the live index by its string ID."""
        self._ensure_incremental_index()
        self.documents = [d for d in self.documents if d["id"] != chunk_id]
        int_id = np.array([self._chunk_int_id(chunk_id)], dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(int_id.shape[0], faiss.swig_ptr(int_id)))

    # ── Shared raw-data index (fixed path, not per-run) ───────────────────────

    def _raw_dir(self) -> str:
        return self.config.faiss_index_path

    def raw_index_exists(self) -> bool:
        return os.path.exists(os.path.join(self._raw_dir(), self.config.raw_index_file))

    def save_raw(self) -> None:
        """Save raw-data index to the shared top-level folder (not per-run)."""
        path = self._raw_dir()
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, self.config.raw_index_file))
        with open(os.path.join(path, self.config.raw_meta_file), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)
        log.info("Raw index saved to %s", path)

    def load_raw(self) -> None:
        """Load raw-data index from the shared top-level folder."""
        path = self._raw_dir()
        self.index = faiss.read_index(os.path.join(path, self.config.raw_index_file))
        self._move_index_to_gpu()
        with open(os.path.join(path, self.config.raw_meta_file), encoding="utf-8") as f:
            self.documents = json.load(f)
        tokenized = [d["text"].lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        log.info("Raw index loaded: %d documents", len(self.documents))

    def _build_raw_data_chunks(self, tables: list[str]) -> list[dict]:
        """Build one FAISS chunk per raw DB row.

        Each chunk text is:
            [RAW ROW] Table: <table> | Row: <n> | col_a=val_a | col_b=val_b | ...

        A SHA-256 row_hash is stored in metadata so the change watcher can
        detect modifications/deletions without re-reading the full table.
        """
        try:
            from db import DB
            db = DB()
        except Exception as exc:
            log.warning("Could not connect to DB for raw data indexing: %s", exc)
            return []

        chunks: list[dict] = []
        skip_cols = {"__db_row_num__"}  # internal bookkeeping column

        for table in tables:
            try:
                df = db.load(table)
            except Exception as exc:
                log.warning("Could not load table %s for raw indexing: %s", table, exc)
                continue

            for idx, row in df.iterrows():
                # Build "col=val" pairs, skipping internal columns and nulls
                pairs = []
                for col in df.columns:
                    if col in skip_cols:
                        continue
                    val = row[col]
                    if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "NaTType"):
                        continue
                    pairs.append(f"{col}={val}")

                if not pairs:
                    continue

                row_num = int(row.get("__db_row_num__", idx))
                text = f"[RAW ROW] Table: {table} | Row: {row_num} | " + " | ".join(pairs)

                # Hash for change detection
                row_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

                chunks.append({
                    "id": f"raw__{table}__row{row_num}",
                    "text": text,
                    "metadata": {
                        "type":     "raw_row",
                        "table":    table,
                        "row_num":  row_num,
                        "row_hash": row_hash,
                    },
                })

            log.info("Raw indexed: %s — %d rows", table, len(df))

        return chunks

    def _build_chunks(self, agent_results: list[AgentResult], insights: dict) -> list[dict]:
        chunks = []
        for result in agent_results:
            if result.get("error"):
                continue

            # Extract risk level dynamically from insights_text
            it = result.get("insights_text", "") or ""
            risk_level = "UNKNOWN"
            for level in ("HIGH RISK", "MEDIUM RISK", "LOW RISK"):
                if level in it.upper():
                    risk_level = level
                    break

            # Extract signals dynamically — works for all agent types
            analysis   = result.get("analysis", {})
            signals    = (
                analysis.get("all_signals")      # universal
                or analysis.get("signals")        # enterprise_row
                or analysis.get("wow_signals")    # wow_row
                or []
            )
            signal_count = analysis.get("signal_count", len(signals))
            sig_str = " | ".join(signals[:5]) if signals else "none"

            # Human-readable prefix ensures natural-language queries like
            # "flagged transaction" or "anomaly in cheque" match correctly.
            risk_adj = {
                "HIGH RISK": "high-risk flagged",
                "MEDIUM RISK": "medium-risk flagged",
                "LOW RISK": "low-risk flagged",
            }.get(risk_level, "flagged")
            text = (
                f"Flagged audit record — {risk_adj} entry with {signal_count} anomaly signal(s). "
                f"[{risk_level}] "
                f"Table: {result['table']} | Row: {result['column']} | "
                f"Agent: {result['agent_name']} | "
                f"Signal count: {signal_count} | Signals: {sig_str} | "
                f"Audit finding: {it}"
            )
            chunks.append({
                "id": f"{result['table']}__{result['column']}__{result['agent_name']}",
                "text": text,
                "metadata": {
                    "table":        result["table"],
                    "column":       result["column"],
                    "agent":        result["agent_name"],
                    "risk_level":   risk_level,
                    "signal_count": signal_count,
                    "chart_type":   result.get("chart_type", ""),
                },
            })

        # Index executive-level audit findings
        for finding in insights.get("key_findings", []):
            if isinstance(finding, dict):
                desc  = finding.get("description") or str(finding)
                table = finding.get("table", "")
                col   = finding.get("column", "")
                text  = f"[AUDIT FINDING] {desc} (Table: {table}, Column: {col})"
            else:
                text = f"[AUDIT FINDING] {finding}"
            chunks.append({
                "id": f"audit_finding_{len(chunks)}",
                "text": text,
                "metadata": {"type": "audit_finding"},
            })
        for risk in insights.get("risk_signals", []):
            if isinstance(risk, dict):
                desc   = risk.get("description") or str(risk)
                table  = risk.get("table", "")
                col    = risk.get("column", "")
                stype  = risk.get("signal_type", "")
                text   = f"[RISK SIGNAL] {desc} Signal type: {stype}. (Table: {table}, Column: {col})"
            else:
                text = f"[RISK SIGNAL] {risk}"
            chunks.append({
                "id": f"risk_signal_{len(chunks)}",
                "text": text,
                "metadata": {"type": "risk_signal"},
            })
        for rec in insights.get("recommendations", []):
            if isinstance(rec, dict):
                action = rec.get("action") or rec.get("description") or str(rec)
                table  = rec.get("table", "")
                team   = rec.get("responsible_team", "")
                text   = f"[AUDIT RECOMMENDATION] {action} Responsible team: {team}. (Table: {table})"
            else:
                text = f"[AUDIT RECOMMENDATION] {rec}"
            chunks.append({
                "id": f"recommendation_{len(chunks)}",
                "text": text,
                "metadata": {"type": "recommendation"},
            })
        return chunks

    def _index_dir(self, run_id: str | None = None) -> str:
        """Return the directory where the index for a given run_id is stored."""
        if run_id:
            return os.path.join(self.config.faiss_index_path, run_id)
        return self.config.faiss_index_path

    def _save(self, run_id: str | None = None) -> None:
        path = self._index_dir(run_id)
        os.makedirs(path, exist_ok=True)
        faiss.write_index(
            self.index,
            os.path.join(path, self.config.faiss_index_file),
        )
        with open(os.path.join(path, self.config.faiss_meta_file), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)
        log.info("FAISS index saved to %s", path)

    def load(self, run_id: str | None = None) -> None:
        path = self._index_dir(run_id)
        index_path = os.path.join(path, self.config.faiss_index_file)
        meta_path = os.path.join(path, self.config.faiss_meta_file)
        self.index = faiss.read_index(index_path)
        self._move_index_to_gpu()
        with open(meta_path, encoding="utf-8") as f:
            self.documents = json.load(f)
        tokenized = [d["text"].lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        log.info("FAISS index loaded: %d documents (run_id=%s)", len(self.documents), run_id)

    def index_exists(self, run_id: str | None = None) -> bool:
        path = self._index_dir(run_id)
        return os.path.exists(os.path.join(path, self.config.faiss_index_file))

    def is_loaded(self) -> bool:
        return self.index is not None and len(self.documents) > 0
