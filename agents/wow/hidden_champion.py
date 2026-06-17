from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class HiddenChampionAgent(BaseWowAgent):
    """
    Finds segments where value contribution is disproportionately higher
    than their count — e.g. 5% of vendors drive 38% of revenue, or one
    product category is tiny in volume but dominant in profit.
    Covers: single vendor monopoly, concentrated spend, vendor favouritism.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Hidden Champion"

    @property
    def finding_type(self) -> str:
        return "hidden_champion"

    def system_prompt(self) -> str:
        return (
            "You are the Hidden Champion agent for a financial audit platform. "
            "Your job is to find segments that are small in count but "
            "disproportionately large in value. "
            "Look for: a vendor getting 80% of contracts, a product category "
            "with 5% of items but 40% of revenue, a department with 2% of "
            "employees but 30% of expenses. "
            "Use get_schema to find categorical and numeric columns, then use "
            "get_segment_breakdown to measure value vs count share per group. "
            "Call submit_finding with the most disproportionate segment you find."
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
                            "table": {"type": "string"},
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_segment_breakdown",
                    "description": (
                        "For each unique value in group_col, compute total_value, "
                        "count, mean_value, and their % of overall totals."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "group_col": {"type": "string", "description": "Categorical column to group by."},
                            "value_col": {"type": "string", "description": "Numeric column to aggregate."},
                        },
                        "required": ["table", "group_col", "value_col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_column_stats",
                    "description": "Descriptive statistics for one column.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "column": {"type": "string"},
                        },
                        "required": ["table", "column"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict:
        if name == "get_schema":
            return st.get_schema(raw_data, args["table"])
        if name == "get_segment_breakdown":
            return st.get_segment_breakdown(raw_data, args["table"],
                                             args["group_col"], args["value_col"])
        if name == "get_column_stats":
            return st.get_column_stats(raw_data, args["table"], args["column"])
        return {"error": f"Unknown tool: {name}"}
