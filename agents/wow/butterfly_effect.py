from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class ButterflyEffectAgent(BaseWowAgent):
    """
    Finds small operational changes that caused large downstream impacts.
    Examples: 2% warehouse packing drop → 19% complaint spike,
    a small vendor data change → payment routing failures,
    a minor policy tweak → significant expense claim surge.
    Covers: fictitious invoice chain, payment without documents,
    threshold splitting patterns.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Butterfly Effect"

    @property
    def finding_type(self) -> str:
        return "butterfly_effect"

    def system_prompt(self) -> str:
        return (
            "You are the Butterfly Effect agent for a financial audit platform. "
            "Your job is to find small changes in one metric that caused large "
            "downstream impacts in other metrics. "
            "Examples: a small drop in packing speed caused a huge spike in complaints, "
            "a minor change in a vendor field triggered a cascade of payment failures, "
            "a small policy change caused an outsized expense surge. "
            "Use get_schema to identify numeric columns. "
            "Use get_change_points to find where a column changed significantly. "
            "Use get_correlation_matrix to find which other columns are correlated "
            "with the changed column — those are the downstream impacts. "
            "Call submit_finding describing the cause and its downstream effect."
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
                    "name": "get_change_points",
                    "description": (
                        "Find positions in a column where the rolling mean shifts "
                        "by more than 20%. Returns before/after means and % change."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "col": {"type": "string", "description": "Numeric column to scan for changes."},
                            "window": {"type": "integer", "default": 10},
                        },
                        "required": ["table", "col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_correlation_matrix",
                    "description": "All numeric column-pair correlations — identifies downstream columns.",
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
        if name == "get_change_points":
            return st.get_change_points(
                raw_data, args["table"], args["col"], args.get("window", 10),
            )
        if name == "get_correlation_matrix":
            return st.get_correlation_matrix(raw_data, args["table"])
        return {"error": f"Unknown tool: {name}"}
