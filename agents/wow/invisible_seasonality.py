from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class InvisibleSeasonalityAgent(BaseWowAgent):
    """
    Finds non-calendar cyclical patterns — cycles that have nothing to do
    with months, quarters or holidays. Also detects end-of-period manipulation
    (recurring spikes at reporting dates) and off-hours transaction patterns.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Invisible Seasonality"

    @property
    def finding_type(self) -> str:
        return "invisible_seasonality"

    def system_prompt(self) -> str:
        return (
            "You are the Invisible Seasonality agent for a financial audit platform. "
            "Your job is to find recurring patterns in time-series data that are NOT "
            "explained by standard calendar events (months, quarters, holidays). "
            "Examples: sales drop every 11th week for unknown reasons, transaction "
            "spikes at every period end (window dressing), recurring off-hours activity. "
            "Use get_schema to find date and numeric columns. "
            "Use get_autocorrelation to find lags with high correlation — if the "
            "dominant lag is not 7, 30, 90 or 365 it may be invisible seasonality. "
            "Use get_fft_dominant_period for cycle length. "
            "Call submit_finding with the most surprising non-calendar cycle found."
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
                    "name": "get_autocorrelation",
                    "description": (
                        "Compute autocorrelation at each lag for a time-ordered column. "
                        "High autocorrelation at lag N means the series repeats every N periods."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "date_col": {"type": "string"},
                            "value_col": {"type": "string"},
                            "max_lag": {"type": "integer", "default": 52},
                        },
                        "required": ["table", "date_col", "value_col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fft_dominant_period",
                    "description": (
                        "Find the dominant cycle period using FFT. "
                        "Returns dominant_period_units and cycle_strength (0-1)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "date_col": {"type": "string"},
                            "value_col": {"type": "string"},
                        },
                        "required": ["table", "date_col", "value_col"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict:
        if name == "get_schema":
            return st.get_schema(raw_data, args["table"])
        if name == "get_autocorrelation":
            return st.get_autocorrelation(
                raw_data, args["table"],
                args["date_col"], args["value_col"],
                args.get("max_lag", 52),
            )
        if name == "get_fft_dominant_period":
            return st.get_fft_dominant_period(
                raw_data, args["table"], args["date_col"], args["value_col"],
            )
        return {"error": f"Unknown tool: {name}"}
