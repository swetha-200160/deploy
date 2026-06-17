"""
Excel to PostgreSQL Dynamic Loader
------------------------------------
Reads any Excel file and loads all sheets into PostgreSQL.
Table names are created as: {TABLE_PREFIX}_{sheet_name}
TABLE_PREFIX is configured in the .env file.

Usage:
    python excel_to_postgres.py <path_to_excel_file> [--prefix myprefix] [--drop]

Arguments:
    <path_to_excel_file>   Path to the Excel file (.xlsx / .xls)
    --prefix               Override TABLE_PREFIX from .env
    --drop                 Drop existing tables before creating (default: replace data)
    --schema               PostgreSQL schema to use (default: public)
"""

import os
import re
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Convert any string to a safe PostgreSQL identifier."""
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)   # replace non-alphanumeric with _
    name = re.sub(r"_+", "_", name)        # collapse multiple underscores
    name = name.strip("_")
    # Postgres identifiers must not start with a digit
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "unnamed"


def infer_pg_type(series: pd.Series) -> str:
    """Infer a PostgreSQL column type from a pandas Series."""
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    # For object columns, peek at non-null values to detect dates/numbers
    non_null = series.dropna()
    if non_null.empty:
        return "TEXT"
    sample = non_null.iloc[0]
    if isinstance(sample, datetime):
        return "TIMESTAMP"
    if isinstance(sample, bool):
        return "BOOLEAN"
    if isinstance(sample, int):
        return "BIGINT"
    if isinstance(sample, float):
        return "DOUBLE PRECISION"
    return "TEXT"


def build_create_table_sql(table_name: str, df: pd.DataFrame, schema: str = "public") -> str:
    """Build a CREATE TABLE IF NOT EXISTS statement from a DataFrame."""
    col_defs = []
    for col in df.columns:
        pg_col  = slugify(col)
        pg_type = infer_pg_type(df[col])
        col_defs.append(f'    "{pg_col}" {pg_type}')
    cols_sql = ",\n".join(col_defs)
    return (
        f'CREATE TABLE IF NOT EXISTS "{schema}"."{table_name}" (\n'
        f"{cols_sql}\n"
        f");"
    )


def sanitize_value(val):
    """Convert pandas NA/NaT/nan to None for psycopg2."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return val


def load_sheet(conn, df: pd.DataFrame, table_name: str, schema: str, drop_first: bool):
    """Create table and insert all rows for one sheet."""
    # Rename DataFrame columns to safe Postgres identifiers
    col_map = {col: slugify(col) for col in df.columns}
    df = df.rename(columns=col_map)

    pg_cols = list(df.columns)

    with conn.cursor() as cur:
        # Optionally drop existing table
        if drop_first:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                    sql.Identifier(schema),
                    sql.Identifier(table_name),
                )
            )
            log.info("  Dropped existing table: %s.%s", schema, table_name)

        # Create table
        create_sql = build_create_table_sql(table_name, df, schema)
        cur.execute(create_sql)
        log.info("  Table ready: %s.%s", schema, table_name)

        # Truncate if table already existed and we are NOT dropping
        if not drop_first:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {}.{}").format(
                    sql.Identifier(schema),
                    sql.Identifier(table_name),
                )
            )

        # Build INSERT
        quoted_cols = [f'"{c}"' for c in pg_cols]
        insert_sql = (
            f'INSERT INTO "{schema}"."{table_name}" ({", ".join(quoted_cols)}) VALUES %s'
        )

        # Convert rows
        rows = [
            tuple(sanitize_value(v) for v in row)
            for row in df.itertuples(index=False, name=None)
        ]

        if rows:
            execute_values(cur, insert_sql, rows, page_size=500)
            log.info("  Inserted %d rows into %s.%s", len(rows), schema, table_name)
        else:
            log.warning("  Sheet is empty — no rows inserted into %s.%s", schema, table_name)

    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Load Excel sheets into PostgreSQL dynamically.")
    parser.add_argument("excel_file", help="Path to the Excel file (.xlsx / .xls)")
    parser.add_argument("--prefix",  default=None, help="Table prefix (overrides TABLE_PREFIX in .env)")
    parser.add_argument("--schema",  default="public", help="PostgreSQL schema (default: public)")
    parser.add_argument("--drop",    action="store_true", help="Drop tables before recreating")
    args = parser.parse_args()

    # Resolve prefix
    prefix = args.prefix or os.getenv("TABLE_PREFIX", "excel")
    prefix = slugify(prefix)

    # Resolve DB credentials
    db_config = {
        "host":     os.getenv("POSTGRES_HOST", "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname":   os.getenv("POSTGRES_DB"),
        "user":     os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    excel_path = args.excel_file
    if not os.path.isfile(excel_path):
        log.error("File not found: %s", excel_path)
        sys.exit(1)

    log.info("Loading Excel file: %s", excel_path)

    # Read all sheets (keep header as first row)
    try:
        all_sheets: dict = pd.read_excel(excel_path, sheet_name=None, dtype=str)
    except Exception as exc:
        log.error("Failed to read Excel file: %s", exc)
        sys.exit(1)

    log.info("Found %d sheet(s): %s", len(all_sheets), list(all_sheets.keys()))

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(**db_config)
        log.info(
            "Connected to PostgreSQL at %s:%s / %s",
            db_config["host"], db_config["port"], db_config["dbname"],
        )
    except Exception as exc:
        log.error("Could not connect to PostgreSQL: %s", exc)
        sys.exit(1)

    summary = []

    for sheet_name, df in all_sheets.items():
        log.info("Processing sheet: '%s'", sheet_name)

        if df.empty:
            log.warning("  Sheet '%s' is empty — skipping.", sheet_name)
            summary.append((sheet_name, "SKIPPED", 0))
            continue

        # Drop completely empty rows/columns
        df = df.dropna(how="all").reset_index(drop=True)

        # Build table name: {prefix}_{sheet_name}
        table_name = f"{prefix}_{slugify(sheet_name)}"

        try:
            load_sheet(conn, df, table_name, args.schema, args.drop)
            summary.append((sheet_name, "OK", len(df)))
        except Exception as exc:
            conn.rollback()
            log.error("  Failed to load sheet '%s': %s", sheet_name, exc)
            summary.append((sheet_name, "FAILED", 0))

    conn.close()

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  LOAD SUMMARY  |  Prefix: {prefix}  |  Schema: {args.schema}")
    print("=" * 60)
    print(f"  {'Sheet':<30}  {'Status':<8}  {'Rows':>6}")
    print("-" * 60)
    for sheet, status, count in summary:
        table = f"{prefix}_{slugify(sheet)}"
        print(f"  {sheet:<30}  {status:<8}  {count:>6}")
        print(f"    -> Table: {args.schema}.{table}")
    print("=" * 60)


if __name__ == "__main__":
    main()
