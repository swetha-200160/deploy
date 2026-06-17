"""Proactive sliding-window rate limiter for Groq API.

Prevents 429 errors by throttling BEFORE API calls instead of reacting to them.

Conservative targets (80% of actual Groq on_demand/free-tier limits):
  llama-3.1-8b-instant   : TPM=6K, RPM=30 → target 4 800 TPM, 24 RPM
  llama-3.3-70b-versatile: TPM=6K, RPM=30 → target 4 800 TPM, 24 RPM

All Groq agents share one module-level singleton so token budgets are
coordinated across the full pipeline run, not per-agent.

Burst protection strategy
-------------------------
Two independent constraints are enforced in groq_invoke():

  1. Sliding-window RPM + TPM  — prevents overuse across a 60-second window.
  2. Minimum inter-call gap    — measured from when the LAST RESPONSE arrived
     to when the NEXT REQUEST is sent. This is what Groq's burst limiter sees.
     Set to _MIN_INTERVAL = 3.0 s (well under 1 req/2s = 30 RPM max).

Why response-to-request matters
--------------------------------
If we track time from acquire() to acquire(), the actual gap Groq sees is:

  acquire gap  =  response_latency  +  our processing time  ≈  very short
  (e.g. call 2 acquired at 12:58:12, responds at 12:58:14, call 3 fires at
   12:58:14.22 → Groq sees only 0.22 s between its response and next request)

By tracking the gap from RESPONSE received to REQUEST sent we match what
Groq's rate-limiter actually measures.

Token tracking (mirrors gemini_rate_limiter)
--------------------------------------------
Every call through groq_invoke() accumulates input/output/total token counts
in a thread-safe counter, broken down per agent. Use these helpers:

    from core.groq_rate_limiter import get_token_usage, reset_token_usage

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

_WINDOW       = 60.0    # seconds — matches Groq's rolling window

# ── Per-model rate limit profiles ──────────────────────────────────────────────
# Groq free-tier TPM = 6 000 for all models.
# Larger models produce longer outputs per call, consuming TPM faster.
# _MIN_INTERVAL is the minimum gap between last RESPONSE and next REQUEST.
# Formula: 60 / (MIN_INTERVAL + avg_latency) × avg_tokens_per_call ≤ TPM_LIMIT
#
#  8b model:  avg ~600 tokens/call → 60/(10+1)×600 ≈ 3 300 tpm  ✓
#  70b model: avg ~1 500 tokens/call → 60/(20+2)×1500 ≈ 4 100 tpm  ✓
_MODEL_PROFILES: dict[str, dict] = {
    # Source: https://console.groq.com/docs/rate-limits (on_demand / free tier, 2026-06)
    #
    # llama-3.1-8b-instant: TPM=6 000, RPM=30 (free/on_demand tier)
    #   Target 80% TPM = 4 800. With 3 parallel agents averaging ~600 tok/call,
    #   8 calls/min max → min_interval=8s serialises calls safely under the window.
    "llama-3.1-8b-instant": {
        "tpm_limit":    4_800,   # 80% of 6 000 TPM
        "rpm_limit":    24,      # 80% of 30 RPM
        "min_interval": 8.0,
    },
    # llama-3.3-70b-versatile: TPM=6 000, RPM=30 (free/on_demand tier)
    #   Only 1-2 calls per pipeline run (InsightAgent + StoryAgent).
    "llama-3.3-70b-versatile": {
        "tpm_limit":    4_800,   # 80% of 6 000 TPM
        "rpm_limit":    24,      # 80% of 30 RPM
        "min_interval": 8.0,
    },
}
_DEFAULT_PROFILE = {
    "tpm_limit":    4_800,
    "rpm_limit":    24,
    "min_interval": 8.0,
}

def _get_profile(model_env_var: str = "LLM_MODEL") -> dict:
    """Return the rate-limit profile for the configured model env var."""
    import os
    model = os.getenv(model_env_var, "").strip()
    return _MODEL_PROFILES.get(model, _DEFAULT_PROFILE)

# Row agent profile (LLM_MODEL — many calls, needs fast+low-token model)
_profile      = _get_profile("LLM_MODEL")
_TPM_LIMIT    = _profile["tpm_limit"]
_RPM_LIMIT    = _profile["rpm_limit"]
_MIN_INTERVAL = _profile["min_interval"]

# Pre-call estimate: chars/3.5 covers input only; multiply by 2.5 to roughly
# account for output tokens (2-3 sentences per row/col ≈ 50-100% of input).
_CHARS_PER_TOKEN = 3.5
_OUTPUT_OVERHEAD = 2.5


# ── Sliding-window RPM + TPM limiter ──────────────────────────────────────────

class _SlidingWindowLimiter:
    """Thread-safe sliding-window limiter tracking RPM and TPM simultaneously."""

    def __init__(self, tpm_limit: int = _TPM_LIMIT, rpm_limit: int = _RPM_LIMIT) -> None:
        self._tpm_limit = tpm_limit
        self._rpm_limit = rpm_limit
        self._lock  = threading.Lock()
        # Each entry: [monotonic_timestamp, tokens] — tokens corrected after response.
        self._calls: list[list[float | int]] = []

    def _evict(self, now: float) -> None:
        cutoff = now - _WINDOW
        i = 0
        while i < len(self._calls) and self._calls[i][0] <= cutoff:
            i += 1
        if i:
            self._calls = self._calls[i:]

    def acquire(self, estimated_tokens: int) -> list:
        """Block until RPM + TPM capacity is available.

        Returns the entry object (by reference) so the caller can correct the
        token count after the response arrives.  Using a reference instead of
        an index avoids stale-index bugs: _evict() re-slices _calls, shifting
        every integer index, but Python object references remain valid regardless
        of where the list is in the deque.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)

                reqs = len(self._calls)
                toks = sum(e[1] for e in self._calls)

                # Cap the effective estimate so a single oversized call can still
                # proceed once the window is clear (prevents infinite deadlock).
                effective = min(estimated_tokens, self._tpm_limit)
                if reqs < self._rpm_limit and toks + effective <= self._tpm_limit:
                    entry: list[float | int] = [now, effective]
                    self._calls.append(entry)
                    return entry          # ← return the object, not an index

                wait = (self._calls[0][0] + _WINDOW - now + 0.1) if self._calls else 1.0

            log.debug(
                "groq_rate_limiter — waiting %.1fs  (window: %d reqs / %d toks, adding ~%d toks)",
                wait, reqs, toks, estimated_tokens,
            )
            time.sleep(max(0.1, min(wait, 10.0)))

    def update(self, entry: list, actual_tokens: int) -> None:
        """Replace the estimated token count with the real value.

        Mutates the entry list in place — no lock needed because CPython's GIL
        makes a single list-item assignment atomic, and the entry is only ever
        written once (here) after being read by acquire().
        """
        entry[1] = actual_tokens


# ── Token usage tracker ────────────────────────────────────────────────────────

class _TokenTracker:
    """Thread-safe accumulator for Groq token usage (mirrors gemini_rate_limiter)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.calls         = 0
        self.input_tokens  = 0
        self.output_tokens = 0
        self.total_tokens  = 0
        self.by_agent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )

    def record(self, response: Any, agent_name: str = "unknown") -> tuple[int, int, int]:
        """Extract usage from a LangChain Groq response and accumulate.

        Returns (input_tokens, output_tokens, total_tokens) for the call.
        """
        inp, out, tot = _extract_token_counts(response)
        if tot == 0:
            return 0, 0, 0

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
            "groq_token_usage  — agent=%-20s call_input=%5d call_output=%5d "
            "| run_total=%d",
            agent_name, inp, out, self.total_tokens,
        )
        return inp, out, tot

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


# ── Token extraction helpers ───────────────────────────────────────────────────

def _extract_token_counts(response: Any) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, total_tokens) from a Groq response."""
    try:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            if isinstance(usage, dict):
                inp = int(usage.get("input_tokens",  0))
                out = int(usage.get("output_tokens", 0))
            else:
                inp = int(getattr(usage, "input_tokens",  0))
                out = int(getattr(usage, "output_tokens", 0))
            tot = inp + out
            if tot > 0:
                return inp, out, tot

        meta = getattr(response, "response_metadata", {})
        tu   = meta.get("token_usage", {})
        inp  = int(tu.get("prompt_tokens",     0))
        out  = int(tu.get("completion_tokens", 0))
        tot  = int(tu.get("total_tokens",      inp + out))
        if tot > 0:
            return inp, out, tot
    except Exception:
        pass
    return 0, 0, 0


def _estimate_tokens(messages) -> int:
    """Conservative pre-call estimate: input chars + assumed output overhead."""
    chars = sum(len(str(getattr(m, "content", m))) for m in messages)
    input_est = max(100, int(chars / _CHARS_PER_TOKEN))
    return int(input_est * _OUTPUT_OVERHEAD)


# ── Per-model singletons — one limiter + burst tracker per model ──────────────
# Each Groq model has its own TPM/RPM quota — they must be tracked independently.
# Sharing one limiter caused 8b row-agent calls to fill the burst gap, then the
# 70b InsightAgent would fire immediately into its own separate (full) quota.

class _ModelLimiter:
    """Bundles a sliding-window limiter + burst tracker for one Groq model."""
    def __init__(self, profile: dict) -> None:
        self.limiter   = _SlidingWindowLimiter(
            tpm_limit=profile["tpm_limit"],
            rpm_limit=profile["rpm_limit"],
        )
        self.min_interval:      float = profile["min_interval"]
        self.last_response_time: float = 0.0
        self.lock = threading.Lock()

_model_limiters: dict[str, _ModelLimiter] = {}
_model_limiters_lock = threading.Lock()


def _get_model_limiter(model: str) -> _ModelLimiter:
    """Return (creating if needed) the per-model limiter for the given model name."""
    with _model_limiters_lock:
        if model not in _model_limiters:
            profile = _MODEL_PROFILES.get(model, _DEFAULT_PROFILE)
            _model_limiters[model] = _ModelLimiter(profile)
        return _model_limiters[model]


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
          }
        }
    """
    return _tracker.snapshot()


def reset_token_usage() -> None:
    """Reset all token counters — call once at the start of each pipeline run."""
    _tracker.reset()
    log.debug("groq_token_usage — counters reset")


def log_token_summary(label: str = "pipeline") -> None:
    """Log a formatted token usage summary — call at end of pipeline run."""
    u = get_token_usage()
    log.info(
        "━━ Groq token summary [%s] ━━  calls=%d  input=%d  output=%d  total=%d",
        label, u["calls"], u["input_tokens"], u["output_tokens"], u["total_tokens"],
    )
    for agent, ag in u["by_agent"].items():
        log.info(
            "   %-20s calls=%d  input=%6d  output=%6d  total=%6d",
            agent, ag["calls"], ag["input_tokens"], ag["output_tokens"], ag["total_tokens"],
        )


_RETRY_WAIT_DEFAULT = 20.0   # seconds to back off when a 429 slips through (free-tier resets are short)
_MAX_RETRIES        = 3      # max attempts per groq_invoke call


def _parse_groq_duration(val: str) -> float | None:
    """Parse a Groq duration string like '7.66s', '120ms', '1m30s' into seconds.

    Groq returns x-ratelimit-reset-tokens / x-ratelimit-reset-requests as
    human-readable durations, not raw seconds.  Only retry-after is raw seconds.
    Returns None if the format is unrecognised.
    """
    import re as _re
    val = val.strip()
    # plain number → already seconds
    try:
        return float(val)
    except ValueError:
        pass
    # e.g. "120ms"
    m = _re.fullmatch(r"([\d.]+)ms", val)
    if m:
        return float(m.group(1)) / 1000.0
    # e.g. "7.66s"
    m = _re.fullmatch(r"([\d.]+)s", val)
    if m:
        return float(m.group(1))
    # e.g. "1m30s" or "1m30.5s" or "1m"
    m = _re.fullmatch(r"(?:([\d.]+)m)?(?:([\d.]+)s)?", val)
    if m and (m.group(1) or m.group(2)):
        mins = float(m.group(1) or 0)
        secs = float(m.group(2) or 0)
        return mins * 60 + secs
    return None


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from a 429 exception, if present.

    Only the standard retry-after header is used for backoff — it is always
    raw seconds per HTTP spec and Groq docs.  The x-ratelimit-reset-* headers
    use Groq's duration-string format ('7.66s', '1m30s') and refer to TPM/RPD
    window resets, not the actual wait time needed before retrying.

    LangChain may wrap the raw httpx/requests response inside various
    exception attributes — we check the most common ones.
    """
    for attr in ("response", "original_error", "__cause__", "__context__"):
        try:
            r = getattr(exc, attr, None)
            if r is None:
                continue
            headers = getattr(r, "headers", None) or {}
            val = headers.get("retry-after")
            if val:
                parsed = _parse_groq_duration(str(val))
                if parsed is not None:
                    return parsed
        except Exception:
            pass
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc)
    return (
        "429" in s
        or "413" in s
        or "503" in s
        or "rate limit" in s.lower()
        or "too many requests" in s.lower()
        or "over capacity" in s.lower()
        or "payload too large" in s.lower()
        or "request too large" in s.lower()
    )


_FALLBACK_THRESHOLD = 60.0   # seconds — if retry-after exceeds this, switch to fallback model immediately


def groq_invoke(llm, messages, agent_name: str = "unknown", fallback_llm=None):
    """Drop-in replacement for ``llm.invoke(messages)`` with proactive rate limiting,
    automatic token tracking, automatic retry on 429, and optional model fallback.

    Two-layer defence
    -----------------
    Layer 1 — Proactive (prevents almost all 429s):
      • Burst gap: enforces _MIN_INTERVAL between last response received and
        next request sent (what Groq's burst limiter actually measures).
      • Sliding-window RPM + TPM: blocks if the 60-second window is full.

    Layer 2 — Reactive (catches the rare slip-through):
      • If llm.invoke() raises a 429/RateLimitError, check retry-after:
        – retry-after ≤ _FALLBACK_THRESHOLD (60s): back off and retry same model.
        – retry-after > _FALLBACK_THRESHOLD OR no fallback wait acceptable:
          if fallback_llm is provided, switch to it immediately (no wait).
          Otherwise back off and retry same model up to _MAX_RETRIES.

    fallback_llm
    ------------
    Pass a secondary ChatGroq instance (e.g. llama-3.1-8b-instant) when calling
    output agents that prefer a large model but must not stall the pipeline on a
    long quota reset.  On activation the fallback is used for the remainder of
    this groq_invoke call only — the caller's self.llm is unchanged.

    Usage::

        from core.groq_rate_limiter import groq_invoke
        response = groq_invoke(llm, [SystemMessage(...), HumanMessage(...)],
                               agent_name="insight", fallback_llm=fallback_llm)
    """
    active_llm = llm

    for attempt in range(1, _MAX_RETRIES + 1):
        # Resolve model name from whichever LLM is currently active
        model_name = getattr(active_llm, "model_name", None) or getattr(active_llm, "model", None) or "unknown"
        ml = _get_model_limiter(model_name)

        # ── 1. Burst gap: per-model, atomically wait + claim the slot ─────────
        while True:
            with ml.lock:
                now   = time.monotonic()
                since = now - ml.last_response_time
                if since >= ml.min_interval:
                    ml.last_response_time = now + ml.min_interval
                    break
                wait = ml.min_interval - since
            log.info("groq_rate_limiter — burst gap: waiting %.2fs (agent=%s, model=%s)", wait, agent_name, model_name)
            time.sleep(wait)

        # ── 2. RPM + TPM sliding-window check (per-model) ─────────────────────
        estimated = _estimate_tokens(messages)
        entry = ml.limiter.acquire(estimated)

        # ── 3. API call ────────────────────────────────────────────────────────
        try:
            response = active_llm.invoke(messages)
        except Exception as exc:
            with ml.lock:
                ml.last_response_time = time.monotonic()
            ml.limiter.update(entry, 0)

            if _is_rate_limit_error(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_after(exc) or _RETRY_WAIT_DEFAULT

                # If the quota window is long AND a fallback model is available,
                # switch immediately rather than stalling the pipeline.
                if wait > _FALLBACK_THRESHOLD and fallback_llm is not None and active_llm is not fallback_llm:
                    fallback_model = getattr(fallback_llm, "model_name", None) or getattr(fallback_llm, "model", None) or "fallback"
                    log.warning(
                        "groq_rate_limiter — 429 on attempt %d/%d (agent=%s, model=%s); "
                        "retry-after=%.0fs exceeds threshold=%.0fs — switching to fallback model %s",
                        attempt, _MAX_RETRIES, agent_name, model_name, wait, _FALLBACK_THRESHOLD, fallback_model,
                    )
                    active_llm = fallback_llm
                    continue   # retry immediately with fallback, no sleep

                log.warning(
                    "groq_rate_limiter — 429 on attempt %d/%d (agent=%s, model=%s); "
                    "backing off %.0fs then retrying",
                    attempt, _MAX_RETRIES, agent_name, model_name, wait,
                )
                time.sleep(wait)
                continue
            raise

        # ── 4. Stamp ACTUAL response time (per-model) ─────────────────────────
        with ml.lock:
            ml.last_response_time = time.monotonic()

        _, _, tot = _tracker.record(response, agent_name=agent_name)
        if tot > 0:
            ml.limiter.update(entry, tot)
            log.debug("groq_rate_limiter — corrected entry: est=%d → actual=%d (model=%s)", estimated, tot, model_name)

        return response

    raise RuntimeError("groq_invoke: exhausted retries — should be unreachable")


# ── Chat-specific rate limiter (separate from pipeline) ───────────────────────
# Interactive chat/query agent calls are sequential and user-facing.
# They use a shorter burst gap (2s) so responses feel fast.
# Completely independent from the pipeline limiters — no shared state.

_CHAT_MIN_INTERVAL = 2.0   # seconds between chat LLM calls

_chat_model_limiters: dict[str, _ModelLimiter] = {}
_chat_model_limiters_lock = threading.Lock()

_CHAT_PROFILE = {
    "tpm_limit":    4_800,
    "rpm_limit":    24,
    "min_interval": _CHAT_MIN_INTERVAL,
}


def _get_chat_model_limiter(model: str) -> _ModelLimiter:
    with _chat_model_limiters_lock:
        if model not in _chat_model_limiters:
            _chat_model_limiters[model] = _ModelLimiter(_CHAT_PROFILE)
        return _chat_model_limiters[model]


def groq_invoke_chat(llm, messages, agent_name: str = "unknown"):
    """groq_invoke variant for interactive chat/query agents.

    Uses a separate per-model limiter with a 2s burst gap instead of the
    pipeline's 8s gap.  Pipeline agents are unaffected — they continue to
    use groq_invoke() with the original limiter.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        ml = _get_chat_model_limiter(model_name)

        # Burst gap (chat-specific, 2s)
        while True:
            with ml.lock:
                now   = time.monotonic()
                since = now - ml.last_response_time
                if since >= ml.min_interval:
                    ml.last_response_time = now + ml.min_interval
                    break
                wait = ml.min_interval - since
            log.info("groq_rate_limiter(chat) — burst gap: waiting %.2fs (agent=%s, model=%s)", wait, agent_name, model_name)
            time.sleep(wait)

        # RPM + TPM sliding-window check
        estimated = _estimate_tokens(messages)
        entry = ml.limiter.acquire(estimated)

        try:
            response = llm.invoke(messages)
        except Exception as exc:
            with ml.lock:
                ml.last_response_time = time.monotonic()
            ml.limiter.update(entry, 0)

            if _is_rate_limit_error(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_after(exc) or _RETRY_WAIT_DEFAULT
                log.warning(
                    "groq_rate_limiter(chat) — 429 on attempt %d/%d (agent=%s); backing off %.0fs",
                    attempt, _MAX_RETRIES, agent_name, wait,
                )
                time.sleep(wait)
                continue
            raise

        with ml.lock:
            ml.last_response_time = time.monotonic()

        _, _, tot = _tracker.record(response, agent_name=agent_name)
        if tot > 0:
            ml.limiter.update(entry, tot)

        return response

    raise RuntimeError("groq_invoke_chat: exhausted retries — should be unreachable")
