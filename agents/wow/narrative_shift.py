from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class NarrativeShiftAgent(BaseWowAgent):
    """
    Detects when the key driver of a business outcome changes over time.
    Example: until Q2 price was the top driver of customer conversion,
    from Q3 delivery speed became number one.
    Covers: vendor master data manipulation (when did bank account change?),
    payroll rate manipulation (temporary change and reversion),
    expense pattern changing at year-end.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Narrative Shift"

    @property
    def finding_type(self) -> str:
        return "narrative_shift"

    def system_prompt(self) -> str:
        return (
            "You are the Narrative Shift agent for a financial audit platform. "
            "Your job is to find cases where the key driver of a metric CHANGED "
            "over time — what drove outcomes in the early period is no longer "
            "the main driver in the recent period. "
            "Examples: price used to drive revenue, now delivery speed does; "
            "a vendor's invoice volume used to predict payments, now it no longer does; "
            "a payroll field that changed and reverted (temporary manipulation). "
            "Use get_schema to find numeric columns. "
            "Use get_correlation_matrix to identify potential driver columns. "
            "Use get_driver_shift to compare early vs recent correlation between "
            "a target column and its potential drivers — high 'shift' value means "
            "the relationship changed. "
            "Call submit_finding describing which driver changed and when."
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
                    "description": "All numeric column-pair correlations — identifies potential drivers.",
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
                    "name": "get_driver_shift",
                    "description": (
                        "For each column in driver_cols, compare its correlation with "
                        "target_col in the first N rows (early) vs the last N rows (recent). "
                        "A high 'shift' value means the relationship changed significantly."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "target_col": {"type": "string", "description": "The outcome column to explain."},
                            "driver_cols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of potential driver columns.",
                            },
                            "window": {"type": "integer", "default": 30,
                                       "description": "Number of rows to use for early and recent windows."},
                        },
                        "required": ["table", "target_col", "driver_cols"],
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
        if name == "get_driver_shift":
            return st.get_driver_shift(
                raw_data, args["table"],
                args["target_col"], args["driver_cols"],
                args.get("window", 30),
            )
        return {"error": f"Unknown tool: {name}"}
