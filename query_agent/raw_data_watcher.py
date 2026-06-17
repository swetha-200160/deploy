"""
raw_data_watcher.py — Background task that keeps the shared raw FAISS index in sync with the DB.

Architecture:
  - Analysis data  → per-run folder  faiss_index/run_XXXX/analysis_index.faiss
  - Raw row data   → shared folder   faiss_index/raw_index.faiss   (this file manages)

How it works:
  Every `interval` seconds:
    1. Cheap row-count pre-check per table (1 COUNT query, no row data loaded).
    2. Only if count changed → full table load + hash comparison.
    3. add_chunk / remove_chunk for changed rows only.
    4. Rebuild BM25 + save_raw() → raw_index.faiss updated.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

_SKIP_COLS = {"__db_row_num__"}


def _build_raw_chunk(table: str, row_num: int, row, columns) -> dict:
    """Build a single raw-row chunk dict from a DataFrame row."""
    pairs = []
    for col in columns:
        if col in _SKIP_COLS:
            continue
        val = row[col]
        if val is None:
            continue
        if hasattr(val, "__class__") and val.__class__.__name__ == "NaTType":
            continue
        pairs.append(f"{col}={val}")

    text = f"[RAW ROW] Table: {table} | Row: {row_num} | " + " | ".join(pairs)
    row_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    return {
        "id": f"raw__{table}__row{row_num}",
        "text": text,
        "metadata": {
            "type":     "raw_row",
            "table":    table,
            "row_num":  row_num,
            "row_hash": row_hash,
        },
    }


class RawDataWatcher:
    """Watches the DB for row-level changes and updates the shared raw FAISS index."""

    def __init__(self, app: "FastAPI", interval: int = 30) -> None:
        self.app = app
        self.interval = interval
        self._running = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_raw_store(self):
        """Return the shared raw FAISSStore from app state, or None."""
        return getattr(self.app.state, "raw_store", None)

    def _bootstrap(self, raw_store) -> None:
        """First-time indexing — reads all DB tables and builds the shared raw index."""
        from db import DB
        from config import get_config
        db = DB()
        try:
            cfg = get_config()
            from db import INTERNAL_TABLE_SUFFIXES
            tables = {
                t for t in db.tables(prefix=cfg.table_prefix)
                if not t.endswith(INTERNAL_TABLE_SUFFIXES)
            }
        except Exception as exc:
            log.warning("RawDataWatcher bootstrap: could not list DB tables — %s", exc)
            return
        finally:
            db.close()

        if not tables:
            log.info("RawDataWatcher bootstrap: no tables found in DB.")
            return

        # Reset any partial state left by a previous failed bootstrap attempt
        raw_store.documents = []
        raw_store.index = None
        raw_store.bm25 = None

        log.info("RawDataWatcher bootstrap: indexing raw rows for %d tables …", len(tables))
        raw_chunks = raw_store._build_raw_data_chunks(list(tables))
        if not raw_chunks:
            log.warning("RawDataWatcher bootstrap: no raw chunks built.")
            return

        for chunk in raw_chunks:
            raw_store.add_chunk(chunk)

        raw_store._rebuild_bm25()
        raw_store.save_raw()
        log.info("RawDataWatcher bootstrap complete: %d raw rows → raw_index.faiss", len(raw_chunks))

    def _check_and_update(self, raw_store) -> None:
        from db import DB

        stored_hashes = raw_store.get_raw_row_hashes()
        if not stored_hashes:
            # No raw rows yet — do full initial indexing
            self._bootstrap(raw_store)
            return

        # Derive table names from chunk IDs  "raw__{table}__row{n}"
        # Always exclude internal tables — filters out query_sessions, feedback, etc.
        # even if old chunks exist in the index (they get evicted via the removal loop below).
        from db import INTERNAL_TABLE_SUFFIXES
        tables: set[str] = set()
        for cid in stored_hashes:
            parts = cid.split("__")
            if len(parts) >= 3:
                tbl = parts[1]
                if not tbl.endswith(INTERNAL_TABLE_SUFFIXES):
                    tables.add(tbl)

        if not tables:
            return

        db = DB()
        added = changed = removed = 0

        try:
            for table in tables:
                # ── Step 1: cheap row-count pre-check ────────────────────────
                # Only do a full table load if the row count differs.
                # Avoids loading 6000-row tables every 30s when nothing changed.
                try:
                    live_count = db.count(table)
                except Exception as exc:
                    log.warning("RawDataWatcher: count(%s) failed — %s", table, exc)
                    continue

                stored_count = sum(1 for cid in stored_hashes if f"__{table}__" in cid)

                if live_count == stored_count:
                    log.debug("RawDataWatcher: %s unchanged (%d rows) — skipped.", table, live_count)
                    continue

                # ── Step 2: row count changed — full load + hash comparison ──
                log.info("RawDataWatcher: %s changed (%d → %d rows) — scanning …", table, stored_count, live_count)
                try:
                    df = db.load(table)
                except Exception as exc:
                    log.warning("RawDataWatcher: cannot load %s — %s", table, exc)
                    continue

                current: dict[str, dict] = {}
                for idx, row in df.iterrows():
                    row_num = int(row.get("__db_row_num__", idx))
                    chunk = _build_raw_chunk(table, row_num, row, df.columns)
                    current[chunk["id"]] = chunk

                table_stored = {k: v for k, v in stored_hashes.items() if f"__{table}__" in k}

                for chunk_id, chunk in current.items():
                    if chunk_id not in table_stored:
                        raw_store.add_chunk(chunk)
                        added += 1
                    elif chunk["metadata"]["row_hash"] != table_stored[chunk_id]:
                        raw_store.add_chunk(chunk)
                        changed += 1

                for chunk_id in table_stored:
                    if chunk_id not in current:
                        raw_store.remove_chunk(chunk_id)
                        removed += 1

        finally:
            db.close()

        if added or changed or removed:
            raw_store._rebuild_bm25()
            raw_store.save_raw()
            log.info("RawDataWatcher: +%d added | %d updated | -%d removed → raw_index.faiss", added, changed, removed)
        else:
            log.debug("RawDataWatcher: all tables unchanged.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        log.info("RawDataWatcher started — polling every %ds.", self.interval)
        while self._running:
            await asyncio.sleep(self.interval)
            raw_store = self._get_raw_store()
            if raw_store is None:
                continue
            try:
                await asyncio.to_thread(self._check_and_update, raw_store)
            except Exception as exc:
                log.warning("RawDataWatcher error: %s", exc)

    def stop(self) -> None:
        self._running = False
