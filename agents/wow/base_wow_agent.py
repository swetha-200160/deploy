from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import pandas as pd
from groq import Groq

from agents.wow.models import WowFinding
from config import Config

log = logging.getLogger(__name__)

_WOW_MODEL = "llama-3.1-8b-instant"
_MAX_ROUNDS = 8
_MAX_TOOL_RESULT_CHARS = 2000

# Shared submit_finding tool — appended to every agent's tool list by the base class.
_SUBMIT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_finding",
        "description": (
            "Submit your final finding once you have gathered enough evidence. "
            "Call this when you have identified a concrete insight."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "One-sentence headline describing the finding.",
                },
                "surprise_score": {
                    "type": "number",
                    "description": "How surprising / counter-intuitive is this finding? 0 = not at all, 10 = very surprising.",
                },
                "actionability_score": {
                    "type": "number",
                    "description": "How actionable is this for the business? 0 = no action possible, 10 = clear immediate action.",
                },
                "evidence": {
                    "type": "string",
                    "description": "Key numbers and facts from the tool results that support this finding.",
                },
                "recommended_action": {
                    "type": "string",
                    "description": "What specific business action should be taken based on this finding?",
                },
            },
            "required": [
                "headline", "surprise_score", "actionability_score",
                "evidence", "recommended_action",
            ],
        },
    },
}


class BaseWowAgent(ABC):
    """
    Abstract base for all 12 wow-factor agents.

    Subclasses define:
      - agent_name  : str property
      - finding_type: str property
      - system_prompt() -> str
      - tool_schemas() -> list[dict]   (JSON schemas for Groq tool-use)
      - execute_tool(name, args, raw_data) -> dict  (Python dispatcher)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = Groq(api_key=config.groq_api_key)

    # ── Abstract interface ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def agent_name(self) -> str: ...

    @property
    @abstractmethod
    def finding_type(self) -> str: ...

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def tool_schemas(self) -> list[dict]: ...

    @abstractmethod
    def execute_tool(self, name: str, args: dict,
                     raw_data: dict[str, pd.DataFrame]) -> dict: ...

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, raw_data: dict[str, pd.DataFrame],
            schema_info: dict) -> list[WowFinding]:
        """
        Run the agentic tool-calling loop.
        - Builds a text schema context from raw_data (LLM never sees raw rows).
        - Loops: LLM calls tool → Python executes on raw_data → result back to LLM.
        - Stops when LLM calls submit_finding or MAX_ROUNDS reached.
        Returns a list of WowFinding objects (usually 0 or 1).
        """
        data_context = self._build_data_context(raw_data)
        all_tools    = self.tool_schemas() + [_SUBMIT_TOOL]

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": (
                    f"{data_context}\n\n"
                    "Analyze this data to find insights relevant to your role. "
                    "Use the tools to explore the data, then call submit_finding "
                    "with your most important finding."
                ),
            },
        ]

        findings: list[WowFinding] = []

        for round_num in range(_MAX_ROUNDS):
            try:
                resp = self._client.chat.completions.create(
                    model=_WOW_MODEL,
                    messages=messages,
                    tools=all_tools,
                    tool_choice="auto",
                    max_tokens=2048,
                )
            except Exception as exc:
                log.error("[%s] Groq API error on round %d: %s", self.agent_name, round_num, exc)
                break

            msg = resp.choices[0].message

            # No tool calls → LLM is done
            if not msg.tool_calls:
                break

            # Append assistant message to history
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool call
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                try:
                    if tc.function.name == "submit_finding":
                        finding = WowFinding(
                            agent_name=self.agent_name,
                            headline=str(args.get("headline", "")),
                            surprise_score=_clamp(args.get("surprise_score", 0)),
                            actionability_score=_clamp(args.get("actionability_score", 0)),
                            finding_type=self.finding_type,
                            evidence={"summary": str(args.get("evidence", ""))},
                            recommended_action=str(args.get("recommended_action", "")),
                            tables_analyzed=list(raw_data.keys()),
                        )
                        findings.append(finding)
                        result: dict = {"status": "finding recorded"}
                    else:
                        result = self.execute_tool(tc.function.name, args, raw_data)
                except Exception as exc:
                    log.warning("[%s] Tool '%s' raised: %s",
                                self.agent_name, tc.function.name, exc)
                    result = {"error": str(exc)}

                # Truncate oversized tool results so they fit in the context window
                result_str = json.dumps(result, default=str)
                if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                    result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + "... (truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        return findings

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_data_context(raw_data: dict[str, pd.DataFrame]) -> str:
        """
        Convert raw_data into a compact text description for the LLM.
        The LLM never receives the actual DataFrame rows — only this summary.
        """
        lines = ["Available tables (use these exact names in tool calls):"]
        for tbl, df in raw_data.items():
            col_parts = []
            for col in df.columns[:25]:          # cap at 25 cols per table
                dtype    = str(df[col].dtype)
                null_pct = round(float(df[col].isna().mean() * 100), 1)
                col_parts.append(f"{col}({dtype},{null_pct}%null)")
            lines.append(f"  {tbl} [{len(df)} rows]: {', '.join(col_parts)}")
        return "\n".join(lines)


def _clamp(value, lo: float = 0.0, hi: float = 10.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 0.0
