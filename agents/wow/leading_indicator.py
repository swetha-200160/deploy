from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class LeadingIndicatorAgent(BaseWowAgent):
    """
    Finds metrics that consistently predict another metric N days ahead.
    Useful for: predicting revenue from early signals, detecting that
    a vendor's invoice frequency predicts payment disputes N weeks later,
    or employee leave patterns predicting payroll anomalies.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Leading Indicator"

    @property
    def finding_type(self) -> str:
        return "leading_indicator"

    def system_prompt(self) -> str:
        return (
            "You are the Leading Indicator agent for a financial audit platform. "
            "Your job is to find which metric consistently predicts another metric "
            "N days or periods ahead — a time-lag relationship. "
            "Examples: Tuesday website sessions predict Friday revenue, "
            "invoice frequency at lag-14 predicts payment disputes, "
            "expense claim submissions predict audit flags 30 days later. "
            "Use get_schema to find numeric columns. "
            "Use get_correlation_matrix to find correlated columns. "
            "Use get_lag_correlation on promising pairs to find the strongest "
            "predictive lag. "
            "Call submit_finding with the most actionable leading indicator found."
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
                    "description": "All numeric column-pair correlations sorted by absolute value.",
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
                    "name": "get_lag_correlation",
                    "description": (
                        "Compute correlation between col_a and col_b at each lag offset. "
                        "Positive lag means col_a shifted forward N periods. "
                        "A high correlation at lag=14 means col_a predicts col_b 14 periods later."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "col_a": {"type": "string", "description": "Potential leading indicator column."},
                            "col_b": {"type": "string", "description": "Outcome column to predict."},
                            "max_lag": {"type": "integer", "default": 30},
                        },
                        "required": ["table", "col_a", "col_b"],
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
        if name == "get_lag_correlation":
            return st.get_lag_correlation(
                raw_data, args["table"],
                args["col_a"], args["col_b"],
                args.get("max_lag", 30),
            )
        return {"error": f"Unknown tool: {name}"}
