"""
db.py — Dynamic PostgreSQL Data Access Layer
=============================================
Central module for reading data from PostgreSQL.
All other parts of the project import from here.

Usage examples:
    from db import DB

    db = DB()

    # List all available tables
    db.tables()

    # Load one full table as DataFrame
    df = db.load("foresight_report_summary")

    # Load with filters
    df = db.load("foresight_cheque_clearing_67318", where="cheque_status = 'Rejected'")

    # Load specific columns
    df = db.load("foresight_missing_grn_67222", columns=["po_no_c", "vendor_c", "severity"])

    # Load all tables at once as a dict
    all_data = db.load_all()

    # Run a raw SQL query
    df = db.query("SELECT * FROM foresight_report_summary WHERE severity = 'High'")

    # Get row count
    count = db.count("foresight_cheque_clearing_67318")

    # Get column names of a table
    cols = db.columns("foresight_price_variance_67227")
"""

import os
import logging
import warnings
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Suffixes that identify internal/tracking tables — never loaded by the pipeline
# and never watched by the auto-trigger poller.
INTERNAL_TABLE_SUFFIXES: tuple[str, ...] = (
    "_query_sessions",
    "_feedback",
)


# ---------------------------------------------------------------------------
# Data cleaning  (ported from excel_tool_agent.py)
# ---------------------------------------------------------------------------

def _clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert a string column to numeric.

    Handles:
    - "3,50,000"  → 350000.0   (comma thousand-separators)
    - "(500)"     → -500.0     (accounting negative notation)
    Already-numeric columns are returned unchanged.
    """
    if series.dtype in ("int64", "float64"):
        return series

    def _try_parse(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        if not s:
            return pd.NA
        negative = False
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
            negative = True
        s = s.replace(",", "").strip()
        try:
            result = float(s)
            return -result if negative else result
        except (ValueError, TypeError):
            return pd.NA

    converted = series.apply(_try_parse)
    return pd.to_numeric(converted, errors="coerce")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a DataFrame loaded from PostgreSQL.

    Applies the same logic used by excel_tool_agent.py:
    1. Strip whitespace from string columns
    2. Convert object columns to numeric where >50% of values parse as numbers
    3. Parse date columns where >50% of values parse as dates
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue

        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})

        numeric_attempt = _clean_numeric_column(df[col])
        non_null_numeric = numeric_attempt.notna().sum()
        total_non_null = df[col].notna().sum()

        if total_non_null > 0 and non_null_numeric / total_non_null > 0.5:
            df[col] = numeric_attempt
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    date_attempt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                date_non_null = date_attempt.notna().sum()
                if total_non_null > 0 and date_non_null / total_non_null > 0.5:
                    df[col] = date_attempt
            except Exception:
                pass

    return df


class DB:
    """
    Dynamic data access layer for PostgreSQL.
    Reads credentials and prefix from .env automatically.
    """

    def __init__(self):
        self._host     = os.getenv("POSTGRES_HOST", "localhost")
        self._port     = int(os.getenv("POSTGRES_PORT", 5432))
        self._dbname   = os.getenv("POSTGRES_DB")
        self._user     = os.getenv("POSTGRES_USER")
        self._password = os.getenv("POSTGRES_PASSWORD")
        self._prefix   = os.getenv("TABLE_PREFIX", "").strip()
        self._schema   = "public"

        # SQLAlchemy engine (clean pandas integration, no warnings)
        url = (
            f"postgresql+psycopg2://{self._user}:{self._password}"
            f"@{self._host}:{self._port}/{self._dbname}"
        )
        self._engine = create_engine(url, pool_size=5, max_overflow=2)
        log.info("DB connected → %s:%s/%s", self._host, self._port, self._dbname)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params=None) -> pd.DataFrame:
        """Run any SQL and return result as a DataFrame."""
        with self._engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tables(self, prefix: Optional[str] = None) -> list[str]:
        """
        List all table names in the database.
        If prefix is not given, uses TABLE_PREFIX from .env.
        Returns sorted list of table names.
        """
        pfx = prefix if prefix is not None else self._prefix
        sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema
              AND table_type = 'BASE TABLE'
              AND table_name LIKE :pattern
            ORDER BY table_name
        """
        df = self._execute(sql, {"schema": self._schema, "pattern": f"{pfx}%"})
        return df["table_name"].tolist()

    def columns(self, table: str) -> list[str]:
        """Return column names for a given table."""
        sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
        """
        df = self._execute(sql, {"schema": self._schema, "table": table})
        return df["column_name"].tolist()

    def count(self, table: str) -> int:
        """Return total row count of a table."""
        df = self._execute(f'SELECT COUNT(*) AS cnt FROM "{self._schema}"."{table}"')
        return int(df["cnt"].iloc[0])

    def load(
        self,
        table: str,
        columns: Optional[list[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a table into a DataFrame.

        Args:
            table    : Table name (e.g. 'foresight_report_summary')
            columns  : List of column names to select (None = all columns)
            where    : SQL WHERE clause string (e.g. "severity = 'High'")
            limit    : Max rows to return
            order_by : ORDER BY clause (e.g. "created_date DESC")

        Returns:
            pandas DataFrame
        """
        col_part  = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        # Inject __db_row_num__: Excel-compatible row number so audit output row
        # references always match the original Excel source file.
        #
        # ROW_NUMBER() OVER (ORDER BY ctid) = 1-based physical insertion order.
        # Adding +1 accounts for the Excel header row (row 1 = header, row 2 = first data row).
        # Result: output "row_33" means Excel row 33 — no mental arithmetic needed.
        #
        # Safe because:
        #   • Audit tables are imported from Excel and never updated (ctid is stable).
        #   • load_all() loads full tables (ROW_NUMBER counts all rows, not a filtered subset).
        #
        # Verification: SELECT * FROM (
        #   SELECT ROW_NUMBER() OVER (ORDER BY ctid)+1 AS rn, * FROM "table") t WHERE rn = 33;
        sql       = (
            f'SELECT ROW_NUMBER() OVER (ORDER BY ctid) + 1 AS __db_row_num__, '
            f'{col_part} FROM "{self._schema}"."{table}"'
        )

        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"

        log.info("load: %s  [where=%s, limit=%s]", table, where, limit)
        df = self._execute(sql)
        return clean_dataframe(df)

    def load_all(
        self,
        prefix: Optional[str] = None,
        exclude: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load ALL tables (matching prefix) into a dict of DataFrames.
        Internal tracking tables (query_sessions, feedback, etc.) are always
        excluded — see INTERNAL_TABLE_SUFFIXES.

        Args:
            prefix  : Override prefix (default: TABLE_PREFIX from .env)
            exclude : Additional table names to skip

        Returns:
            { table_name: DataFrame, ... }
        """
        exclude   = set(exclude or [])
        all_tables = self.tables(prefix=prefix)
        result    = {}

        for tbl in all_tables:
            if tbl in exclude:
                log.info("Skipping excluded table: %s", tbl)
                continue
            if tbl.endswith(INTERNAL_TABLE_SUFFIXES):
                log.debug("Skipping internal table: %s", tbl)
                continue
            try:
                df = self.load(tbl)
                result[tbl] = df
                log.info("Loaded %-45s → %d rows, %d cols", tbl, len(df), len(df.columns))
            except Exception as exc:
                log.error("Failed to load %s: %s", tbl, exc)

        return result

    def query(self, sql: str, params=None) -> pd.DataFrame:
        """
        Run any raw SQL query and return result as DataFrame.

        Args:
            sql    : Full SQL string
            params : Optional tuple/list of bind parameters

        Returns:
            pandas DataFrame
        """
        log.info("query: %s", sql[:120])
        return self._execute(sql, params=params)

    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame with table name, row count, column count
        for all tables matching the prefix.
        """
        rows = []
        for tbl in self.tables():
            try:
                col_count = len(self.columns(tbl))
                row_count = self.count(tbl)
                rows.append({
                    "table":   tbl,
                    "rows":    row_count,
                    "columns": col_count,
                })
            except Exception as exc:
                rows.append({"table": tbl, "rows": "ERROR", "columns": str(exc)})
        return pd.DataFrame(rows)

    def close(self):
        """Dispose the SQLAlchemy engine and all connections."""
        self._engine.dispose()
        log.info("DB connection pool closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
