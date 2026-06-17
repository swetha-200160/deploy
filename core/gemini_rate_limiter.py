"""Simple RPM-only rate limiter for Gemini API with token usage tracking.

Gemini free-tier limits:
  Flash models: 15 RPM, 1,000,000 TPM
  TPM is effectively unlimited — only RPM needs throttling.

Token tracking
--------------
Every call through gemini_invoke() accumulates totals in a thread-safe
counter. Use these helpers to read / reset the counters:

    from core.gemini_rate_limiter import get_token_usage, reset_token_usage

    usage = get_token_usage()
    # {'calls': 4, 'input_tokens': 6200, 'output_tokens': 3100,
    #  'total_tokens': 9300, 'by_agent': {'universal': {...}, ...}}

    reset_token_usage()   # call at the start of each pipeline run
"""
from __future__ import annotations

import threading
import time
import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

_RPM_LIMIT = 14   # 14/15 — small safety buffer
_WINDOW    = 60.0


class _GeminiRPMLimiter:
    """Thread-safe sliding-window RPM limiter."""

    def __init__(self, rpm_limit: int = _RPM_LIMIT) -> None:
        self._rpm_limit = rpm_limit
        self._lock  = threading.Lock()
        self._calls: list[float] = []   # monotonic timestamps of recent calls

    def acquire(self) -> None:
        """Block until an RPM slot is available."""
        while True:
            with self._lock:
                now    = time.monotonic()
                cutoff = now - _WINDOW
                # evict calls outside the window
                self._calls = [t for t in self._calls if t > cutoff]
                if len(self._calls) < self._rpm_limit:
                    self._calls.append(now)
                    return
                wait = self._calls[0] + _WINDOW - now + 0.1

            log.debug("gemini_rate_limiter — waiting %.1fs (%d/%d RPM used)",
                      wait, len(self._calls), self._rpm_limit)
            time.sleep(max(0.1, wait))


# ── Token usage tracker ────────────────────────────────────────────────────────

class _TokenTracker:
    """Thread-safe accumulator for Gemini token usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.calls         = 0
        self.input_tokens  = 0
        self.output_tokens = 0
        self.total_tokens  = 0
        # per-agent breakdown: agent_name → same four counters
        self.by_agent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )

    def record(self, response: Any, agent_name: str = "unknown") -> None:
        """Extract usage_metadata from a LangChain Gemini response and accumulate."""
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return

        # LangChain returns usage_metadata as a dict or an object
        if isinstance(meta, dict):
            inp  = int(meta.get("input_tokens",  0))
            out  = int(meta.get("output_tokens", 0))
            tot  = int(meta.get("total_tokens",  inp + out))
        else:
            inp  = int(getattr(meta, "input_tokens",  0))
            out  = int(getattr(meta, "output_tokens", 0))
            tot  = int(getattr(meta, "total_tokens",  inp + out))

        with self._lock:
            self.calls         += 1
            self.input_tokens  += inp
            self.output_tokens += out
            self.total_tokens  += tot

            ag = self.by_agent[agent_name]
            ag["calls"]         += 1
            ag["input_tokens"]  += inp
            ag["output_tokens"] += out
            ag["total_tokens"]  += tot

        log.info(
            "gemini_token_usage — agent=%-20s call_input=%5d call_output=%5d "
            "| run_total=%d",
            agent_name, inp, out, self.total_tokens,
        )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls":         self.calls,
                "input_tokens":  self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens":  self.total_tokens,
                "by_agent":      {k: dict(v) for k, v in self.by_agent.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._reset()


# Shared singletons — one per process
_limiter = _GeminiRPMLimiter()
_tracker = _TokenTracker()


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_token_usage() -> dict:
    """Return a snapshot of accumulated token usage for the current run.

    Returns::

        {
          'calls': 8,
          'input_tokens': 12400,
          'output_tokens': 6200,
          'total_tokens': 18600,
          'by_agent': {
              'universal':  {'calls': 4, 'input_tokens': 6200, ...},
              'enterprise': {'calls': 2, 'input_tokens': 3100, ...},
              'wow':        {'calls': 2, 'input_tokens': 3100, ...},
          }
        }
    """
    return _tracker.snapshot()


def reset_token_usage() -> None:
    """Reset all token counters — call once at the start of each pipeline run."""
    _tracker.reset()
    log.debug("gemini_token_usage — counters reset")


def log_token_summary(label: str = "pipeline") -> None:
    """Log a formatted token usage summary — call at end of pipeline run."""
    u = get_token_usage()
    log.info(
        "━━ Gemini token summary [%s] ━━  calls=%d  input=%d  output=%d  total=%d",
        label, u["calls"], u["input_tokens"], u["output_tokens"], u["total_tokens"],
    )
    for agent, ag in u["by_agent"].items():
        log.info(
            "   %-20s calls=%d  input=%6d  output=%6d  total=%6d",
            agent, ag["calls"], ag["input_tokens"], ag["output_tokens"], ag["total_tokens"],
        )


def gemini_invoke(llm, messages, agent_name: str = "unknown"):
    """Drop-in replacement for ``llm.invoke(messages)`` with RPM throttling
    and automatic token usage tracking.

    Usage::

        from core.gemini_rate_limiter import gemini_invoke
        response = gemini_invoke(llm, [SystemMessage(...), HumanMessage(...)],
                                 agent_name="universal")
    """
    _limiter.acquire()
    response = llm.invoke(messages)
    _tracker.record(response, agent_name=agent_name)
    return response
