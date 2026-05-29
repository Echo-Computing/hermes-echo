"""Dataclass models for debates, tournaments, and code execution results."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import uuid
import math


# --- Hypothesis ---

@dataclass
class Hypothesis:
    """A research hypothesis with ELO rating and lifecycle tracking."""
    title: str
    description: str
    mechanism: str = ""
    evidence: List[str] = field(default_factory=list)
    elo_rating: float = 1500.0
    round_created: int = 0
    status: str = "alive"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    merged_from: List[str] = field(default_factory=list)
    critiques: List[dict] = field(default_factory=list)
    refinement_history: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "mechanism": self.mechanism,
            "evidence": self.evidence,
            "elo_rating": self.elo_rating,
            "round_created": self.round_created,
            "status": self.status,
            "merged_from": self.merged_from,
            "critiques": self.critiques,
            "refinement_history": self.refinement_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        return cls(
            id=d.get("id", uuid.uuid4().hex[:12]),
            title=d.get("title", ""),
            description=d.get("description", ""),
            mechanism=d.get("mechanism", ""),
            evidence=d.get("evidence", []),
            elo_rating=d.get("elo_rating", 1500.0),
            round_created=d.get("round_created", 0),
            status=d.get("status", "alive"),
            merged_from=d.get("merged_from", []),
            critiques=d.get("critiques", []),
            refinement_history=d.get("refinement_history", []),
        )


# --- ELO Tournament ---

@dataclass
class DebateResult:
    """Outcome of a single head-to-head hypothesis debate."""
    hypothesis_a_id: str
    hypothesis_b_id: str
    winner: str                      # "A", "B", or "tie"
    reasoning: str
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float
    elo_b_after: float


# ELO constants
ELO_K_FACTOR = 32
ELO_STARTING_RATING = 1500.0


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def elo_update(rating: float, expected: float, actual: float, k: float = ELO_K_FACTOR) -> float:
    """Update an ELO rating given expected score, actual score, and K-factor."""
    return rating + k * (actual - expected)


def select_debate_pairs(hypotheses: List[Hypothesis], max_pairs: int) -> List[tuple]:
    """Select debate pairs by ELO proximity — pair closest-rated hypotheses.

    Returns list of (hypothesis_a, hypothesis_b) tuples.
    """
    alive = [h for h in hypotheses if h.status == "alive"]
    if len(alive) < 2:
        return []

    sorted_by_elo = sorted(alive, key=lambda h: h.elo_rating)
    pairs = []
    seen = set()

    # Pair adjacent hypotheses in ELO ranking (closest-rated fight each other)
    for i in range(0, len(sorted_by_elo) - 1, 2):
        a, b = sorted_by_elo[i], sorted_by_elo[i + 1]
        pair_key = tuple(sorted([a.id, b.id]))
        if pair_key not in seen:
            pairs.append((a, b))
            seen.add(pair_key)

    # If odd number, pair the last unpaired one with the next closest
    if len(sorted_by_elo) % 2 == 1 and len(sorted_by_elo) >= 3:
        last = sorted_by_elo[-1]
        second_last = sorted_by_elo[-2]
        pair_key = tuple(sorted([last.id, second_last.id]))
        if pair_key not in seen:
            pairs.append((last, second_last))
            seen.add(pair_key)

    return pairs[:max_pairs]


# --- Code Execution ---

@dataclass
class CodeExecutionResult:
    """Result from a single Finch-style code execution instance."""
    instance_id: int
    code: str
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    findings: str                   # LLM's interpretation of results


@dataclass
class ConsensusResult:
    """Consensus across multiple parallel code executions."""
    total_instances: int
    agreeing_instances: int
    consensus_reached: bool         # True if >= majority agree
    majority_finding: Optional[str]
    all_findings: List[str]
    verdict: str                    # "accepted", "rejected", "inconclusive"
