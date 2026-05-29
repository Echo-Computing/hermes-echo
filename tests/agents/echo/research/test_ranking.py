"""Unit tests for ELO rating math and debate pair selection.

These are pure functions — no LLM calls, no mocking needed.
"""

import pytest
import math
from hermes_cli.agents.echo.research.models import (
    Hypothesis,
    DebateResult,
    elo_expected,
    elo_update,
    select_debate_pairs,
    ELO_K_FACTOR,
    ELO_STARTING_RATING,
)


class TestELOMath:
    """Test the ELO rating system math."""

    def test_equal_ratings_expected_score(self):
        """Two equally rated players should each have 0.5 expected score."""
        assert elo_expected(1500, 1500) == 0.5
        assert elo_expected(1000, 1000) == 0.5
        assert elo_expected(2000, 2000) == 0.5

    def test_higher_rated_expected_to_win(self):
        """Higher-rated player should have >0.5 expected score."""
        assert elo_expected(1600, 1400) > 0.5
        assert elo_expected(2000, 1000) > elo_expected(1600, 1400)
        # 400 point difference = ~0.91 expected
        assert elo_expected(1900, 1500) > 0.9

    def test_lower_rated_expected_to_lose(self):
        """Lower-rated player should have <0.5 expected score."""
        assert elo_expected(1400, 1600) < 0.5

    def test_expected_scores_sum_to_one(self):
        """Expected scores for A vs B and B vs A should sum to 1."""
        e_a = elo_expected(1520, 1480)
        e_b = elo_expected(1480, 1520)
        assert abs((e_a + e_b) - 1.0) < 0.001

    def test_elo_update_win(self):
        """Winning should increase rating."""
        old = 1500.0
        new = elo_update(old, 0.5, 1.0)  # Win against equal
        assert new > old
        assert new == 1516.0  # 1500 + 32 * (1.0 - 0.5) = 1516

    def test_elo_update_loss(self):
        """Losing should decrease rating."""
        old = 1500.0
        new = elo_update(old, 0.5, 0.0)  # Loss against equal
        assert new < old
        assert new == 1484.0  # 1500 + 32 * (0.0 - 0.5) = 1484

    def test_elo_update_tie(self):
        """Tie adjusts rating toward expectation."""
        old = 1500.0
        new = elo_update(old, 0.5, 0.5)  # Tie
        assert new == 1500.0  # No change when expected = actual

    def test_elo_update_upset(self):
        """Lower-rated player beating higher-rated gains more points."""
        # Underdog wins: expected 0.09, actual 1.0 → big gain
        old = 1400.0
        expected = elo_expected(1400, 1900)  # ~0.05
        new = elo_update(old, expected, 1.0)
        gain = new - old
        assert gain > 25  # Should be near 30

    def test_elo_k_factor_applied(self):
        """Custom K factor should scale the update."""
        old = 1500.0
        new_default = elo_update(old, 0.5, 1.0, k=32)
        new_custom = elo_update(old, 0.5, 1.0, k=16)
        assert new_default - old == 16.0
        assert new_custom - old == 8.0


class TestDebatePairSelection:
    """Test the debate pair selection algorithm."""

    def make_hypotheses(self, *ratings):
        """Create test hypotheses with given ELO ratings."""
        return [
            Hypothesis(
                title="H{}".format(i),
                description="Hypothesis {}".format(i),
                elo_rating=r,
            )
            for i, r in enumerate(ratings)
        ]

    def test_empty_returns_empty(self):
        """No hypotheses → no pairs."""
        assert select_debate_pairs([], 5) == []

    def test_single_returns_empty(self):
        """One hypothesis → no pairs (need 2)."""
        h = self.make_hypotheses(1500)
        assert select_debate_pairs(h, 5) == []

    def test_two_returns_one_pair(self):
        """Two hypotheses → one pair."""
        hyps = self.make_hypotheses(1500, 1520)
        pairs = select_debate_pairs(hyps, 5)
        assert len(pairs) == 1
        assert pairs[0][0].elo_rating == 1500
        assert pairs[0][1].elo_rating == 1520

    def test_pairs_by_elo_proximity(self):
        """Pairs should match closest-rated hypotheses."""
        hyps = self.make_hypotheses(1400, 1500, 1600, 1700)
        pairs = select_debate_pairs(hyps, 5)
        assert len(pairs) == 2
        # Should pair 1400-1500 and 1600-1700
        ratings_set_1 = {pairs[0][0].elo_rating, pairs[0][1].elo_rating}
        ratings_set_2 = {pairs[1][0].elo_rating, pairs[1][1].elo_rating}
        assert ratings_set_1 in ({1400, 1500}, {1500, 1400})
        assert ratings_set_2 in ({1600, 1700}, {1700, 1600})

    def test_respects_max_pairs(self):
        """Should not exceed max_pairs limit."""
        hyps = self.make_hypotheses(1400, 1450, 1500, 1550, 1600, 1650)
        pairs = select_debate_pairs(hyps, max_pairs=2)
        assert len(pairs) <= 2

    def test_eliminated_excluded(self):
        """Eliminated hypotheses should not be paired."""
        hyps = self.make_hypotheses(1500, 1520, 1480)
        hyps[1].status = "eliminated"
        pairs = select_debate_pairs(hyps, 5)
        assert len(pairs) == 1
        # Only 1500 and 1480 should be paired
        ids_in_pair = {pairs[0][0].id, pairs[0][1].id}
        assert hyps[1].id not in ids_in_pair

    def test_no_duplicate_pairs(self):
        """Should not create duplicate pairings."""
        hyps = self.make_hypotheses(1500, 1520, 1480)
        pairs = select_debate_pairs(hyps, 10)
        pair_ids = {tuple(sorted([p[0].id, p[1].id])) for p in pairs}
        assert len(pair_ids) == len(pairs)  # No duplicates


class TestHypothesisLifecycle:
    """Test the Hypothesis dataclass methods."""

    def test_default_values(self):
        h = Hypothesis(title="Test", description="A test hypothesis")
        assert h.elo_rating == 1500.0
        assert h.status == "alive"
        assert h.round_created == 0
        assert h.evidence == []
        assert h.merged_from == []
        assert len(h.id) == 12  # UUID hex[:12]

    def test_to_dict(self):
        h = Hypothesis(
            title="Test",
            description="Description",
            mechanism="Mechanism",
            evidence=["source1", "source2"],
            elo_rating=1520.0,
            round_created=2,
            status="refined",
        )
        d = h.to_dict()
        assert d["title"] == "Test"
        assert d["elo_rating"] == 1520.0
        assert d["status"] == "refined"
        assert d["evidence"] == ["source1", "source2"]

    def test_from_dict(self):
        d = {
            "id": "abc123",
            "title": "From Dict",
            "description": "Created from dict",
            "mechanism": "Test mechanism",
            "elo_rating": 1480.0,
            "round_created": 1,
            "status": "alive",
            "evidence": ["ev1"],
            "merged_from": [],
            "critiques": [{"severity": "minor", "critique": "Needs more evidence"}],
            "refinement_history": ["Refined in round 1"],
        }
        h = Hypothesis.from_dict(d)
        assert h.id == "abc123"
        assert h.title == "From Dict"
        assert h.elo_rating == 1480.0
        assert len(h.critiques) == 1

    def test_from_dict_defaults(self):
        """from_dict should handle missing fields gracefully."""
        d = {"title": "Minimal", "description": "Min desc"}
        h = Hypothesis.from_dict(d)
        assert h.title == "Minimal"
        assert h.elo_rating == 1500.0
        assert h.status == "alive"
        assert len(h.id) == 12  # Generated UUID

    def test_roundtrip(self):
        """to_dict → from_dict should preserve all data."""
        original = Hypothesis(
            title="Roundtrip Test",
            description="Testing serialization",
            mechanism="Test mechanism",
            elo_rating=1530.0,
            round_created=3,
            status="refined",
        )
        restored = Hypothesis.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.elo_rating == original.elo_rating
        assert restored.id == original.id
        assert restored.status == original.status


class TestCodeExecutionModels:
    """Test CodeExecutionResult and ConsensusResult models."""

    def test_code_execution_result(self):
        from hermes_cli.agents.echo.research.models import CodeExecutionResult
        r = CodeExecutionResult(
            instance_id=1,
            code="print('hello')",
            stdout="hello\n",
            stderr="",
            exit_code=0,
            success=True,
            findings="Code executed successfully",
        )
        assert r.success
        assert r.exit_code == 0
        assert r.findings is not None

    def test_consensus_result(self):
        from hermes_cli.agents.echo.research.models import ConsensusResult
        c = ConsensusResult(
            total_instances=3,
            agreeing_instances=2,
            consensus_reached=True,
            majority_finding="Hypothesis supported",
            all_findings=["Hypothesis supported", "Hypothesis supported", "Inconclusive"],
            verdict="accepted",
        )
        assert c.consensus_reached
        assert c.verdict == "accepted"
        assert c.agreeing_instances == 2
