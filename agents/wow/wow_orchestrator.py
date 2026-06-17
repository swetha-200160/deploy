from __future__ import annotations

import concurrent.futures
import logging

import pandas as pd

from agents.wow.models import WowFinding
from agents.wow.paradox_detector import ParadoxDetectorAgent
from agents.wow.hidden_champion import HiddenChampionAgent
from agents.wow.silent_churn import SilentChurnAgent
from agents.wow.invisible_seasonality import InvisibleSeasonalityAgent
from agents.wow.leading_indicator import LeadingIndicatorAgent
from agents.wow.butterfly_effect import ButterflyEffectAgent
from agents.wow.simpsons_paradox import SimpsonParadoxAgent
from agents.wow.redundancy_noise import RedundancyNoiseAgent
from agents.wow.cross_domain import CrossDomainCorrelationAgent
from agents.wow.anomaly_cluster import AnomalyClusterAgent
from agents.wow.narrative_shift import NarrativeShiftAgent
from agents.wow.dark_data import DarkDataAgent
from config import Config

log = logging.getLogger(__name__)


def run_wow_agents(
    raw_data: dict[str, pd.DataFrame],
    schema_info: dict,
    config: Config,
) -> list[WowFinding]:
    """
    Instantiate all 12 wow-factor agents and run them in parallel.
    Each agent receives the full raw_data dict and schema_info.
    Returns all findings sorted by combined_score descending.
    """
    agent_classes = [
        ParadoxDetectorAgent,
        HiddenChampionAgent,
        SilentChurnAgent,
        InvisibleSeasonalityAgent,
        LeadingIndicatorAgent,
        ButterflyEffectAgent,
        SimpsonParadoxAgent,
        RedundancyNoiseAgent,
        CrossDomainCorrelationAgent,
        AnomalyClusterAgent,
        NarrativeShiftAgent,
        DarkDataAgent,
    ]

    def _run_one(agent_cls) -> list[WowFinding]:
        agent = agent_cls(config)
        try:
            findings = agent.run(raw_data, schema_info)
            log.info("[WowOrchestrator] %s → %d finding(s)", agent.agent_name, len(findings))
            return findings
        except Exception as exc:
            log.error("[WowOrchestrator] %s failed: %s", agent_cls.__name__, exc)
            return []

    all_findings: list[WowFinding] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_run_one, cls): cls.__name__ for cls in agent_classes}
        for future in concurrent.futures.as_completed(futures):
            try:
                all_findings.extend(future.result())
            except Exception as exc:
                log.error("[WowOrchestrator] Future error: %s", exc)

    # Sort by combined score descending so the best findings come first
    all_findings.sort(key=lambda f: f.combined_score, reverse=True)

    promoted = [f for f in all_findings if f.promoted]
    log.info(
        "[WowOrchestrator] Complete — %d total findings, %d promoted to headline",
        len(all_findings), len(promoted),
    )
    return all_findings
