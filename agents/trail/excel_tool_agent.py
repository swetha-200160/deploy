"""LangChain Pandas DataFrame Agent for Excel/CSV analysis."""
import asyncio
import logging
import os
import uuid
import warnings
from dataclasses import dataclass, field

import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class ExcelAgentResult:
    """Structured result from the Excel/CSV tool agent."""
    success: bool
    answer: str
    source_files: list[str] = field(default_factory=list)
    dataframe_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    error: str | None = None


# --- Data Cleaning ---


def _clean_numeric_column(series: pd.Series) -> pd.Series:
    """Attempt to convert a column to numeric.

    Handles common number formats without altering currency:
    - "3,50,000" -> 350000.0  (comma separators)
    - "(500)" -> -500.0  (accounting negative notation)
    Currency symbols are NOT stripped — values are kept as-is.
    """
    if series.dtype in ("int64", "float64"):
        return series

    def _try_parse(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        if not s:
            return pd.NA

        # Handle accounting negatives: (500) -> -500
        negative = False
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
            negative = True

        # Remove thousand separators (commas)
        s = s.replace(",", "")
        s = s.strip()

        try:
            result = float(s)
            return -result if negative else result
        except (ValueError, TypeError):
            return pd.NA

    converted = series.apply(_try_parse)
    return pd.to_numeric(converted, errors="coerce")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a DataFrame for analysis.

    1. Strip whitespace from string columns
    2. Try to convert object columns to numeric where possible
    3. Parse date columns
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})

            # Attempt numeric conversion -- only if >50% of non-null values convert
            numeric_attempt = _clean_numeric_column(df[col])
            non_null = numeric_attempt.notna().sum()
            total_non_null = df[col].notna().sum()
            if total_non_null > 0 and non_null / total_non_null > 0.5:
                df[col] = numeric_attempt
            else:
                # Try date parsing (suppress format inference warnings)
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


# --- File Loading ---


def _load_dataframe(file_path: str, file_type: str) -> dict[str, pd.DataFrame]:
    """Load file into one or more DataFrames.

    Returns dict of {sheet_name: DataFrame}.
    For CSV, the key is the filename stem.
    For Excel, one key per sheet.
    """
    if file_type == "csv":
        encodings = ["utf-8", "cp1252", "latin1", "iso-8859-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                name = os.path.basename(file_path).rsplit(".", 1)[0]
                return {name: _clean_dataframe(df)}
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not decode CSV at {file_path}")

    elif file_type == "xlsx":
        xls = pd.ExcelFile(file_path)
        result = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            result[sheet] = _clean_dataframe(df)
        return result

    raise ValueError(f"Unsupported file type: {file_type}")


async def _load_session_dataframes(
    db: AsyncSession,
    session_id: uuid.UUID,
    prioritized_filenames: list[str] | None = None,
) -> tuple[list[pd.DataFrame], list[str], dict[str, tuple[int, int]]]:
    """Load all Excel/CSV DataFrames for a session.

    Returns:
        - dataframes: list of DataFrames (ordered: prioritized files first)
        - filenames: corresponding list of source file labels
        - shapes: dict of filename -> (rows, cols)
    """
    result = await db.execute(
        select(Document).where(
            Document.session_id == session_id,
            Document.file_type.in_(["xlsx", "csv"]),
        )
    )
    documents = result.scalars().all()

    if not documents:
        return [], [], {}

    # Sort documents: prioritized filenames first
    if prioritized_filenames:
        priority_set = {f.lower() for f in prioritized_filenames}
        documents = sorted(
            documents,
            key=lambda d: (0 if d.filename.lower() in priority_set else 1, d.filename),
        )

    all_dfs: list[pd.DataFrame] = []
    all_names: list[str] = []
    all_shapes: dict[str, tuple[int, int]] = {}

    for doc in documents:
        if not os.path.exists(doc.file_path):
            logger.warning(f"File not found: {doc.file_path} ({doc.filename})")
            continue
        try:
            sheets = _load_dataframe(doc.file_path, doc.file_type)
            for sheet_label, df in sheets.items():
                if doc.file_type == "xlsx" and len(sheets) > 1:
                    label = f"{doc.filename} / {sheet_label}"
                else:
                    label = doc.filename

                all_dfs.append(df)
                all_names.append(label)
                all_shapes[label] = df.shape
        except Exception as e:
            logger.error(f"Error loading {doc.filename}: {e}")
            continue

    return all_dfs, all_names, all_shapes


# --- Agent Construction & Invocation ---


def _build_agent_prefix(df_names: list[str], shapes: dict[str, tuple[int, int]], multi: bool = False) -> str:
    """Build a prefix prompt for the agent.

    LangChain variable naming:
    - Single DF (passed directly): `df`
    - Multiple DFs (passed as list): `df1`, `df2`, `df3`, ... (1-indexed, no `df`)
    """
    lines = [
        "You are a data analysis agent for an audit/finance team.",
        "You have access to the following pandas DataFrames (already loaded in memory):",
        "",
    ]
    for i, name in enumerate(df_names):
        rows, cols = shapes.get(name, (0, 0))
        # LangChain naming: single DF = `df`; list = `df1`, `df2`, ... (1-indexed)
        if multi:
            df_var = f"df{i + 1}"
        else:
            df_var = "df"
        lines.append(f"- `{df_var}`: '{name}' ({rows} rows, {cols} columns)")

    lines.extend([
        "",
        "IMPORTANT INSTRUCTIONS:",
        "- The DataFrames are already loaded in memory. Use them directly.",
        "- Do NOT try to load files from disk or import pandas.",
        "- When computing totals, sums, averages, etc., operate on ALL rows.",
        "- Numeric columns are already cleaned to float type.",
        "- Always provide the specific numeric result.",
        "- State which file and column(s) you used in your answer.",
    ])
    return "\n".join(lines)


def _invoke_agent_sync(
    dataframes: list[pd.DataFrame],
    df_names: list[str],
    shapes: dict[str, tuple[int, int]],
    question: str,
) -> ExcelAgentResult:
    """Synchronous function that creates and invokes the LangChain agent.

    LangChain variable naming convention (from source code):
    - Single DF (passed directly): `df`
    - Multiple DFs (passed as list): `df1`, `df2`, `df3`, ... (1-indexed)

    Runs in a thread via asyncio.to_thread().
    """
    try:
        llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.EXCEL_AGENT_TEMPERATURE,
            max_tokens=settings.EXCEL_AGENT_MAX_TOKENS,
        )

        multi = len(dataframes) > 1
        prefix = _build_agent_prefix(df_names, shapes, multi=multi)

        # Single DF → pass directly (variable: df); Multiple → pass as list (variables: df1, df2, ...)
        agent_input = dataframes[0] if len(dataframes) == 1 else dataframes

        agent_executor = create_pandas_dataframe_agent(
            llm=llm,
            df=agent_input,
            agent_type="tool-calling",
            prefix=prefix,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=settings.EXCEL_AGENT_MAX_ITERATIONS,
        )

        result = agent_executor.invoke({"input": question})
        output_text = result.get("output", str(result))

        return ExcelAgentResult(
            success=True,
            answer=output_text,
            source_files=df_names,
            dataframe_shapes=shapes,
        )

    except Exception as e:
        logger.error(f"Excel agent error: {e}", exc_info=True)
        return ExcelAgentResult(
            success=False,
            answer="",
            source_files=df_names,
            dataframe_shapes=shapes,
            error=str(e),
        )


# --- Public async entry point ---


async def run_excel_tool_agent(
    db: AsyncSession,
    question: str,
    session_id: uuid.UUID,
    prioritized_filenames: list[str] | None = None,
) -> ExcelAgentResult:
    """Run the LangChain pandas agent on Excel/CSV files in the session.

    Single DF → passed directly (variable: df).
    Multiple DFs → passed as list (variables: df1, df2, ...).
    The router's prioritized_filenames ensures the most relevant file is first (df1).

    Args:
        db: Async database session
        question: User's question
        session_id: Session UUID for document isolation
        prioritized_filenames: Optional list of filenames to prioritize
            (from router's target_excel_filenames)
    """
    dataframes, df_names, shapes = await _load_session_dataframes(
        db, session_id, prioritized_filenames
    )

    if not dataframes:
        return ExcelAgentResult(
            success=False,
            answer="No Excel or CSV files found in this session.",
            error="no_files",
        )

    logger.info(
        f"Excel agent: loaded {len(dataframes)} DataFrames for session {session_id}: "
        + ", ".join(f"{n} {s}" for n, s in shapes.items())
    )

    # Run the synchronous LangChain agent in a separate thread
    result = await asyncio.to_thread(
        _invoke_agent_sync,
        dataframes,
        df_names,
        shapes,
        question,
    )

    return result
