from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class SimpsonParadoxAgent(BaseWowAgent):
    """
    Detects cases where an overall trend completely reverses when data is
    split by a subgroup — the Simpson's Paradox. Critical for fraud detection:
    e.g. overall tender looks competitive but one vendor always wins within
    each category; overall approval rate looks fine but one approver rubber-stamps.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Simpsons Paradox"

    @property
    def finding_type(self) -> str:
        return "simpsons_paradox"

    def system_prompt(self) -> str:
        return (
            "You are the Simpson's Paradox agent for a financial audit platform. "
            "Your job is to find cases where a trend in the overall data reverses "
            "when you split the data into subgroups. "
            "Examples: overall vendor performance looks average but within each "
            "category one vendor always wins; overall expense approval rate looks "
            "fine but one approver has an abnormally high approval rate in every "
            "sub-category. "
            "Use get_schema to find categorical (group) columns and numeric columns. "
            "Use get_group_vs_aggregate_correlation — if overall_correlation and "
            "by_group correlations have OPPOSITE signs, a paradox is present. "
            "Use get_segment_breakdown to quantify the imbalance per group. "
            "Call submit_finding with the strongest paradox you find."
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
                    "name": "get_group_vs_aggregate_correlation",
                    "description": (
                        "Returns the overall correlation between x_col and y_col "
                        "AND the per-group correlation for each value in group_col. "
                        "A paradox exists when overall and per-group signs differ."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "group_col": {"type": "string", "description": "Categorical column to split by."},
                            "x_col": {"type": "string"},
                            "y_col": {"type": "string"},
                        },
                        "required": ["table", "group_col", "x_col", "y_col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_segment_breakdown",
                    "description": "Count and value % per segment — shows concentration.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "group_col": {"type": "string"},
                            "value_col": {"type": "string"},
                        },
                        "required": ["table", "group_col", "value_col"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict:
        if name == "get_schema":
            return st.get_schema(raw_data, args["table"])
        if name == "get_group_vs_aggregate_correlation":
            return st.get_group_vs_aggregate_correlation(
                raw_data, args["table"],
                args["group_col"], args["x_col"], args["y_col"],
            )
        if name == "get_segment_breakdown":
            return st.get_segment_breakdown(
                raw_data, args["table"], args["group_col"], args["value_col"],
            )
        return {"error": f"Unknown tool: {name}"}
