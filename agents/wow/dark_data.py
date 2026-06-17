from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class DarkDataAgent(BaseWowAgent):
    """
    Finds columns that are collected and stored but never contribute to
    any analysis — they are either mostly empty or completely uncorrelated
    with everything else.
    Also catches: approval fields that are always null (no approval happening),
    document reference fields that are empty (no document attached),
    GST/TDS fields that are never populated (compliance gap).
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Dark Data"

    @property
    def finding_type(self) -> str:
        return "dark_data"

    def system_prompt(self) -> str:
        return (
            "You are the Dark Data agent for a financial audit platform. "
            "Your job is to find columns that are collected but never used — "
            "either because they are mostly empty (null) or because they carry "
            "no statistical signal (uncorrelated with everything). "
            "This is critical for audit: empty approval fields mean approvals "
            "are not being recorded; empty document fields mean no supporting "
            "documents; empty GST/TDS fields mean compliance data is missing. "
            "Use get_schema to see all columns. "
            "Use get_column_coverage to find columns with very high null_pct "
            "or very low unique_count — these are dark/dead columns. "
            "Use get_correlation_matrix to find numeric columns that appear in "
            "no high-correlation pairs (max correlation < 0.1) — they carry no signal. "
            "Call submit_finding listing the dark columns and their business implication."
        )

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Get column names, dtypes and null% for a table.",
                    "parameters": {
                        "type": "object",
                        "properties": {"table": {"type": "string"}},
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_column_coverage",
                    "description": (
                        "Return null_pct, filled_pct and unique_count for every column. "
                        "High null_pct = dark/unused field. Low unique_count = no variation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"table": {"type": "string"}},
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_correlation_matrix",
                    "description": (
                        "All numeric column-pair correlations. "
                        "Columns absent from all high-correlation pairs "
                        "carry no signal and are dark data."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"table": {"type": "string"}},
                        "required": ["table"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict:
        if name == "get_schema":
            return st.get_schema(raw_data, args["table"])
        if name == "get_column_coverage":
            return st.get_column_coverage(raw_data, args["table"])
        if name == "get_correlation_matrix":
            return st.get_correlation_matrix(raw_data, args["table"])
        return {"error": f"Unknown tool: {name}"}
