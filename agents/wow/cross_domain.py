from __future__ import annotations

import pandas as pd

from agents.wow.base_wow_agent import BaseWowAgent
from agents.wow.tools import statistical as st
from config import Config


class CrossDomainCorrelationAgent(BaseWowAgent):
    """
    Finds surprising statistical correlations BETWEEN unrelated tables.
    Examples: employee satisfaction scores predict customer NPS 45 days later,
    vendor bank account changes correlate with payment spikes,
    HR data correlates with vendor payment patterns (kickback signal),
    employee records share fields with vendor records (conflict of interest).
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @property
    def agent_name(self) -> str:
        return "Cross-Domain Correlation"

    @property
    def finding_type(self) -> str:
        return "cross_domain_correlation"

    def system_prompt(self) -> str:
        return (
            "You are the Cross-Domain Correlation agent for a financial audit platform. "
            "Your job is to find surprising correlations BETWEEN different tables — "
            "datasets that appear unrelated but share a statistical relationship. "
            "Examples: HR employee data correlating with vendor payment data "
            "(possible conflict of interest), procurement data correlating with "
            "employee bank data (kickback signal), asset disposal values correlating "
            "with buyer employee IDs. "
            "Use get_schema on multiple tables to discover their columns. "
            "Use get_cross_table_correlation to test correlations between columns "
            "from different tables. Focus on columns that are conceptually unrelated "
            "but numerically correlated. "
            "Call submit_finding with the most surprising cross-domain relationship."
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
                    "name": "get_cross_table_correlation",
                    "description": (
                        "Compute Pearson correlation between one column in table_a "
                        "and one column in table_b. Useful for finding unexpected "
                        "links between unrelated datasets."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_a": {"type": "string"},
                            "col_a": {"type": "string"},
                            "table_b": {"type": "string"},
                            "col_b": {"type": "string"},
                        },
                        "required": ["table_a", "col_a", "table_b", "col_b"],
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
        if name == "get_cross_table_correlation":
            return st.get_cross_table_correlation(
                raw_data,
                args["table_a"], args["col_a"],
                args["table_b"], args["col_b"],
            )
        if name == "get_column_stats":
            return st.get_column_stats(raw_data, args["table"], args["column"])
        return {"error": f"Unknown tool: {name}"}
