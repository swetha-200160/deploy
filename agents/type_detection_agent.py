"""TypeDetectionAgent — pure pandas column type classification. No LLM."""
from __future__ import annotations

import logging

import pandas as pd

from config import Config, get_config

log = logging.getLogger(__name__)


class TypeDetectionAgent:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

    def classify_dataframe(self, df: pd.DataFrame, table: str) -> dict[str, str]:
        """Classify every column using pure pandas dtype rules.

        Returns { "table.column": "numerical|categorical|datetime|text" }
        """
        df = df.copy()

        # Try datetime conversion on object columns.
        # Use errors='coerce' so partial parses don't raise; require ≥50% of
        # non-null values to be valid dates before reclassifying the column.
        # This prevents all-null columns (and mostly-text columns) from being
        # mis-labelled as datetime.
        for col in df.columns:
            if df[col].dtype == object:
                non_null = int(df[col].notna().sum())
                if non_null == 0:
                    continue  # all-null → leave as object
                try:
                    converted = pd.to_datetime(df[col], format="mixed", dayfirst=False, errors="coerce")
                    valid_dt = int(converted.notna().sum())
                    if valid_dt / non_null >= 0.5:
                        df[col] = converted
                except Exception:
                    pass

        numerical_cols   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        datetime_cols    = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        object_cols      = df.select_dtypes(include=["object", "string"]).columns.tolist()
        categorical_cols = [col for col in object_cols if df[col].nunique() < self.config.max_categorical_cardinality]
        text_cols        = [col for col in object_cols if col not in categorical_cols]

        log.info(
            "Table '%s': %d numerical, %d datetime, %d categorical, %d text",
            table, len(numerical_cols), len(datetime_cols), len(categorical_cols), len(text_cols),
        )

        result: dict[str, str] = {}
        for col in numerical_cols:   result[f"{table}.{col}"] = "numerical"
        for col in datetime_cols:    result[f"{table}.{col}"] = "datetime"
        for col in categorical_cols: result[f"{table}.{col}"] = "categorical"
        for col in text_cols:        result[f"{table}.{col}"] = "text"
        return result
