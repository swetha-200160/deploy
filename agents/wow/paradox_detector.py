from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class ParadoxDetectorAgent(BaseWowAgent):
    """
    Finds counter-intuitive metric movements — e.g. cheapest product has
    highest margin, or vendor with most invoices has lowest dispute rate.
    Covers fraud patterns: overpayment, payment to inactive vendor,
    self-approval, round-number transactions, off-hours activity.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Paradox Detector"

    @property
    def finding_type(self) -> str:
        return "paradox"

    def system_prompt(self) -> str:
        return (
            "You are the Paradox Detector agent for a financial audit platform. "
            "Your job is to find metrics that move in the OPPOSITE direction to "
            "what business logic would predict — counter-intuitive patterns. "
            "Examples: cheapest item earns the most profit, inactive vendor "
            "receiving active payments, amount paid > invoice amount, "
            "payment date before due date, self-approvals. "
            "Use the tools to explore column names and correlations, then "
            "call submit_finding with the most surprising paradox you discover. "
            "If no paradox is found, submit with surprise_score=0."
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
                        "properties": {
                            "table": {"type": "string", "description": "Table name from the available tables list."},
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_correlation_matrix",
                    "description": "All numeric column-pair correlations sorted by absolute value.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_sorted_rows",
                    "description": "Sort a table by one column and show selected columns for the top N rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "sort_col": {"type": "string", "description": "Column to sort by."},
                            "show_cols": {"type": "array", "items": {"type": "string"},
                                          "description": "Extra columns to display."},
                            "ascending": {"type": "boolean"},
                            "top_n": {"type": "integer", "default": 10},
                        },
                        "required": ["table", "sort_col", "show_cols", "ascending"],
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
        if name == "get_sorted_rows":
            return st.get_sorted_rows(
                raw_data,
                args["table"],
                args["sort_col"],
                args.get("show_cols", []),
                args.get("ascending", True),
                args.get("top_n", 10),
            )
        return {"error": f"Unknown tool: {name}"}
