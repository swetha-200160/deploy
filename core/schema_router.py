from __future__ import annotations

import logging

import pandas as pd

from config import Config, get_config
from core.state import ColumnRoute
from db import DB

log = logging.getLogger(__name__)


def classify_all(
    config: Config | None = None,
    exclude: list[str] | None = None,
    preloaded: dict | None = None,
) -> list[ColumnRoute]:
    """Classify every column in every table using pure pandas dtype rules.

    Pass ``preloaded`` (a {table_name: DataFrame} dict) to skip the DB fetch.
    """
    if config is None:
        config = get_config()

    excluded = set(exclude or [])

    # ── Load DataFrames ───────────────────────────────────────────────────────
    if preloaded is not None:
        table_dfs = {t: df for t, df in preloaded.items() if t not in excluded}
    else:
        db = DB()
        try:
            tables = [t for t in db.tables() if t not in excluded]
        except Exception as exc:
            log.error("Failed to list tables: %s", exc)
            return []
        table_dfs = {}
        for table in tables:
            try:
                table_dfs[table] = db.load(table)
            except Exception as exc:
                log.error("Failed to load table %s: %s", table, exc)

    if not table_dfs:
        return []

    # ── Classify columns ──────────────────────────────────────────────────────
    from agents.type_detection_agent import TypeDetectionAgent
    agent = TypeDetectionAgent(config)

    routes: list[ColumnRoute] = []

    for table, df in table_dfs.items():
        if df.empty:
            log.warning("Table %s is empty, skipping.", table)
            continue

        # Strip internal metadata columns (e.g. __db_row_num__ injected by db.load)
        # before type classification — they are not real data columns.
        analysis_df = df[[c for c in df.columns if not c.startswith("__")]]
        classifications = agent.classify_dataframe(analysis_df, table)

        for col in analysis_df.columns:
            series = df[col]
            try:
                unique_count = int(series.nunique())
            except TypeError:
                unique_count = -1

            routes.append(ColumnRoute(
                table=table,
                column=col,
                detected_type=classifications.get(f"{table}.{col}", "text"),
                sample_values=series.dropna().head(5).tolist(),
                null_ratio=float(series.isna().mean()),
                unique_count=unique_count,
            ))
            log.debug("  %s.%s → %s", table, col, classifications.get(f"{table}.{col}", "text"))

    log.info(
        "Schema routing complete: %d columns across %d tables",
        len(routes), len(table_dfs),
    )
    return routes


def build_schema_info(routes: list[ColumnRoute]) -> dict[str, dict[str, str]]:
    """Convert flat ColumnRoute list to { table: { col: type } } dict."""
    info: dict[str, dict[str, str]] = {}
    for r in routes:
        info.setdefault(r["table"], {})[r["column"]] = r["detected_type"]
    return info


def filter_by_type(routes: list[ColumnRoute], type_name: str) -> list[ColumnRoute]:
    return [r for r in routes if r["detected_type"] == type_name]
