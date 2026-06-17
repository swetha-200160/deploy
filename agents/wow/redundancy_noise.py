from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class RedundancyNoiseAgent(BaseWowAgent):
    """
    Finds KPIs and columns that carry zero informational value — they are
    either completely uncorrelated with everything else or duplicate
    another column. Helps clean up dashboards and audit reports.
    Also catches: dark/unfilled fields that should have data (indicating
    missing audit controls).
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Redundancy and Noise"

    @property
    def finding_type(self) -> str:
        return "redundancy_noise"

    def system_prompt(self) -> str:
        return (
            "You are the Redundancy and Noise agent for a financial audit platform. "
            "Your job is to identify two things: "
            "1. Zero-value KPIs — numeric columns with near-zero correlation with "
            "every other column (they tell you nothing useful). "
            "2. Redundant columns — pairs of columns that are nearly identical "
            "(correlation > 0.95), meaning one is completely unnecessary. "
            "Use get_schema to see all columns. "
            "Use get_correlation_matrix — look for columns that appear in NO high "
            "correlation pairs (max absolute correlation < 0.1) and pairs with "
            "correlation > 0.95. "
            "Use get_column_coverage to find columns with very high null% (never filled). "
            "Call submit_finding listing which columns should be removed and why."
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
                    "name": "get_correlation_matrix",
                    "description": (
                        "All numeric column-pair correlations. "
                        "Columns absent from high-correlation pairs are zero-value KPIs. "
                        "Pairs with correlation near 1.0 are redundant."
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
                    "name": "get_column_coverage",
                    "description": "Null%, filled% and unique count for every column.",
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
        if name == "get_correlation_matrix":
            return st.get_correlation_matrix(raw_data, args["table"])
        if name == "get_column_coverage":
            return st.get_column_coverage(raw_data, args["table"])
        return {"error": f"Unknown tool: {name}"}
