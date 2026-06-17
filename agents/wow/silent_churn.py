from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class SilentChurnAgent(BaseWowAgent):
    """
    Detects entities whose activity is declining significantly in the last
    60-90 days — early churn warning before it becomes visible.
    Also catches: ghost employees (zero activity), salary after exit,
    inactive vendor re-activation patterns.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Silent Churn Predictor"

    @property
    def finding_type(self) -> str:
        return "silent_churn"

    def system_prompt(self) -> str:
        return (
            "You are the Silent Churn Predictor agent for a financial audit platform. "
            "Your job is to find entities (customers, vendors, employees) whose "
            "activity level has dropped significantly in the recent period (last 60 days) "
            "compared to before — a pre-churn signal. "
            "Also look for: entities with zero recent activity despite historical "
            "presence (ghost patterns), or sudden drops in transaction frequency. "
            "Use get_schema to find date, ID, and activity columns. "
            "Use get_time_trend for overall trend. "
            "Use get_entity_activity_change to find entities with the largest decline. "
            "Call submit_finding with your most important churn or ghost-pattern finding."
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
                    "name": "get_time_trend",
                    "description": "Compare recent N-row window mean vs prior window mean for a value column.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "date_col": {"type": "string"},
                            "value_col": {"type": "string"},
                            "window": {"type": "integer", "default": 30},
                        },
                        "required": ["table", "date_col", "value_col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_entity_activity_change",
                    "description": (
                        "For each entity (id_col), compare total activity_col "
                        "in the last 60 days vs before. Returns change_pct per entity."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "id_col": {"type": "string", "description": "Entity identifier column."},
                            "date_col": {"type": "string"},
                            "activity_col": {"type": "string", "description": "Numeric activity column."},
                            "top_n": {"type": "integer", "default": 10},
                        },
                        "required": ["table", "id_col", "date_col", "activity_col"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict:
        if name == "get_schema":
            return st.get_schema(raw_data, args["table"])
        if name == "get_time_trend":
            return st.get_time_trend(raw_data, args["table"],
                                      args["date_col"], args["value_col"],
                                      args.get("window", 30))
        if name == "get_entity_activity_change":
            return st.get_entity_activity_change(
                raw_data, args["table"],
                args["id_col"], args["date_col"],
                args["activity_col"], args.get("top_n", 10),
            )
        return {"error": f"Unknown tool: {name}"}
