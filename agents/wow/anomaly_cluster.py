from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class AnomalyClusterAgent(BaseWowAgent):
    """
    Detects when outliers form patterns rather than being random noise.
    One outlier = noise. Ten outliers clustered in time = a real signal.
    Covers: duplicate payments (same amount clustering), round-number
    transactions, bulk approvals in impossibly short time, threshold-split
    transactions clustering just below a limit.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Anomaly Cluster"

    @property
    def finding_type(self) -> str:
        return "anomaly_cluster"

    def system_prompt(self) -> str:
        return (
            "You are the Anomaly Cluster agent for a financial audit platform. "
            "Your job is to find outliers that CLUSTER together — not random noise "
            "but a real signal that something systematic is happening. "
            "Examples: multiple payments of the same amount clustering in a short "
            "period (duplicate payment fraud), multiple approvals in seconds "
            "(bulk rubber-stamping), many transactions just below a limit "
            "(threshold splitting), expense claims clustering on the same dates. "
            "Use get_schema to find numeric and date columns. "
            "Use get_outlier_stats to find how many outliers exist in a column. "
            "Use get_outlier_time_clustering to check if those outliers are "
            "concentrated in time — if pct_within_7_days is high, it is a cluster. "
            "Call submit_finding with the most suspicious outlier cluster."
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
                    "name": "get_outlier_stats",
                    "description": (
                        "Return outlier count, %, bounds and sample values "
                        "using the IQR method."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "col": {"type": "string", "description": "Numeric column to check for outliers."},
                            "iqr_multiplier": {"type": "number", "default": 1.5},
                        },
                        "required": ["table", "col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_outlier_time_clustering",
                    "description": (
                        "Check whether outliers in a column are concentrated "
                        "in time — returns temporal_clustering (bool) and "
                        "pct_within_7_days_of_each_other."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "col": {"type": "string"},
                            "date_col": {"type": "string"},
                            "iqr_multiplier": {"type": "number", "default": 1.5},
                        },
                        "required": ["table", "col", "date_col"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict:
        if name == "get_schema":
            return st.get_schema(raw_data, args["table"])
        if name == "get_outlier_stats":
            return st.get_outlier_stats(
                raw_data, args["table"], args["col"],
                args.get("iqr_multiplier", 1.5),
            )
        if name == "get_outlier_time_clustering":
            return st.get_outlier_time_clustering(
                raw_data, args["table"], args["col"], args["date_col"],
                args.get("iqr_multiplier", 1.5),
            )
        return {"error": f"Unknown tool: {name}"}
