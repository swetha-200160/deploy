from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WowFinding:
    agent_name: str
    headline: str
    surprise_score: float        # 0–10
    actionability_score: float   # 0–10
    finding_type: str
    evidence: dict
    recommended_action: str
    tables_analyzed: list[str] = field(default_factory=list)

    @property
    def combined_score(self) -> float:
        return self.surprise_score + self.actionability_score

    @property
    def promoted(self) -> bool:
        return self.combined_score >= 14.0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "headline": self.headline,
            "surprise_score": self.surprise_score,
            "actionability_score": self.actionability_score,
            "combined_score": self.combined_score,
            "finding_type": self.finding_type,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "tables_analyzed": self.tables_analyzed,
            "promoted": self.promoted,
        }
