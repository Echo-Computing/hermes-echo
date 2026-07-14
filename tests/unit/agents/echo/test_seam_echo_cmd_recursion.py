"""Regression tests for the scaled recursion_limit + partial-state dump.

Kept in the seam-tests scratch
(.hermes-edit/seam-tests/tests/unit/agents/echo/) and namespaced ``test_seam_*``
so it survives a fresh WSL re-import + apply_seam.sh (it previously lived only
in the live clone and would have been lost). apply_seam.sh copies + runs this as
a post-deploy attestation. Hermetic: mocks the research graph, no Ollama/network.
Imports the LIVE seam (hermes_cli is editable-installed from the live clone)."""
import asyncio
import pytest
from hermes_cli.commands.echo_cmd import _research_recursion_limit, _dump_partial_research_state


class TestRecursionLimitScaling:
    def test_one_round_minimum_budget(self):
        assert _research_recursion_limit(1) == 1 * 12 + 20

    def test_ten_rounds_scales(self):
        assert _research_recursion_limit(10) == 10 * 12 + 20

    def test_twenty_rounds_not_exhausted(self):
        # 20 rounds must not hit the old hardcoded-100 ceiling mid-run.
        assert _research_recursion_limit(20) == 20 * 12 + 20
        assert _research_recursion_limit(20) > 100

    def test_zero_means_unlimited_capped_at_1000(self):
        assert _research_recursion_limit(0) == 1000

    def test_negative_treated_as_unlimited(self):
        assert _research_recursion_limit(-5) == 1000

    def test_none_falls_back_to_safe_default(self):
        assert _research_recursion_limit(None) == 120


class TestPartialStateDump:
    def test_dump_writes_crash_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        partial = {"hypotheses": [{"id": "h1", "title": "partial"}],
                   "errors": [{"node": "x", "error": "boom"}],
                   "final_report": {"leaderboard": []}}
        _dump_partial_research_state(partial, "test goal", "test-model", RuntimeError("crash"))
        crash_files = list((tmp_path / ".hermes" / "history").glob("research-*-CRASH.json"))
        assert len(crash_files) == 1
        import json
        data = json.loads(crash_files[0].read_text())
        assert data["partial"] is True
        assert "crash" in data["crash_reason"]
        assert data["hypotheses"] == partial["hypotheses"]
        assert data["model"] == "test-model"

    def test_dump_survives_bad_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        # non-dict state must not raise
        _dump_partial_research_state(None, "g", "m", RuntimeError("x"))


class TestPartialCaptureOnRecursionError:
    """The load-bearing path: when the graph raises mid-stream, the last
    yielded state is saved as a partial report (no data loss)."""

    def _fake_config(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            echo=SimpleNamespace(model="m",
                research=SimpleNamespace(max_rounds=5, debates_per_round=1,
                    hypotheses_per_round=1, parallel_instances=1, code_timeout=10,
                    search_results_per_query=1)),
            ollama=SimpleNamespace(api_url="http://x", model="m", timeout=10, retry=1),
        )

    def test_recursion_error_saves_partial_report(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        class _FakeGraph:
            async def astream(self, state, config):
                yield {"hypotheses": [{"id": "h1", "title": "partial-hyp"}],
                        "final_report": {"leaderboard": [{"rank": 1, "elo_rating": 1500, "title": "t"}]},
                        "errors": [], "tournament_results": [], "code_execution_results": [], "search_results": []}
                raise RecursionError("simulated GraphRecursionError")
        import hermes_cli.commands.echo_cmd as ec
        monkeypatch.setattr(ec, "create_research_graph", lambda: _FakeGraph(), raising=False)
        # The import inside _run_research is `from ...graph import create_research_graph`.
        import hermes_cli.agents.echo.research.graph as rg
        monkeypatch.setattr(rg, "create_research_graph", lambda: _FakeGraph())
        ec._run_research(self._fake_config(), "test goal")
        reports = list((tmp_path / ".hermes" / "history").glob("research-*.json"))
        assert reports, "no report saved on recursion error (data loss)"
        data = json.loads(reports[0].read_text())
        assert data.get("partial") is True
        assert data["report"]["leaderboard"][0]["title"] == "t"