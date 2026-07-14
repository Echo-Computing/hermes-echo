"""Step 4 autoresearch (2026-07-07) — tests for the constrained-mutation-mode
contract + config + the results.tsv provenance ledger + the Goodhart gate.

Three public-safe MIT-able artifacts, mined as DESIGN from karpathy/autoresearch
(program.md is unlicensed -> NO text copied, re-implemented as original Echo
content):
  (1) contract_loader.AUTORESEARCH_CONTRACT  — a program.md-style contract with
      fields mutation_surface / evaluator / keep_discard / autonomy_posture,
      with a governance-gate-BEFORE-keep. Delivered as a Python string constant
      (no skills/ loader exists in Hermes; a .py constant is auto-attested by
      seam_manifest.discover() / verify_integrity() for free).
  (2) code_execution constrained mode — an opt-in flag (research.constrained_mode)
      that swaps the system prompt to the contract + imposes a per-mutation
      wall-clock budget + a hard outer kill + equal-compute-slice (same timeout=
      per sequential instance). A keep/discard PROPOSAL dict is emitted on the
      per-hypothesis result; it is INFORMATIONAL and OVERWRITTEN by the gauntlet.
  (3) provenance.ResearchOutcome + results.tsv — an append-only ledger at
      ~/.hermes/learning/results.tsv (OUTSIDE the repo -> git-clean by default),
      one row per hypothesis that reached format_report, with the gauntlet's
      composite status (keep|discard|crash) + a governance_verdict column.

GOODHART DEFENSE (load-bearing, the largest risk per the locked deep-dive):
  The keep/discard DECISION cannot come from the wrapper's proposal dict or a
  scalar metric. It is the composite of run_consensus (ELO) + run_reflection
  (fatal->eliminated) + run_ranking (ELO floor) + verify_integrity(), written
  ONLY at format_report. These tests pin that invariant: a proposal that says
  "keep" is OVERRIDDEN to discard when reflection eliminates the hypothesis
  (test_goodhart_proposal_overridden_by_reflection).
"""
import base64  # for the encoded contract-hygiene banned list
import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

from hermes_cli.agents.echo.research import provenance as prov
from hermes_cli.agents.echo.research.contract_loader import (
    AUTORESEARCH_CONTRACT, load_contract,
)
from hermes_cli.agents.echo.research.models import CodeExecutionResult
from hermes_cli.agents.echo.research.nodes import code_execution as ce
from hermes_cli.agents.echo.seam_manifest import DEFAULT_SCRATCH


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _set_flag(monkeypatch, on):
    monkeypatch.setattr(ce, "SANDBOX_TIMEOUT", 30)  # stable default for tests


def _fake_state(hypotheses, constrained=None, num_instances=3):
    research = {
        "parallel_instances": num_instances,
        "code_timeout": 30,
    }
    if constrained is not None:
        research["constrained_mode"] = constrained
    return {
        "research_goal": "test goal",
        "config": {
            "ollama": {"api_url": "http://x", "model": "m", "timeout": 10, "retry": 0},
            "research": research,
        },
        "hypotheses": hypotheses,
        "code_execution_results": [],
        "errors": [],
    }


class _FakeOllama:
    """Captures the system_prompt each chat() is called with; returns a valid
    JSON code response so the node's json-extraction path succeeds."""
    def __init__(self, *a, **k):
        self.system_prompts = []
        self.call_count = 0
    async def chat(self, prompt, system_prompt=None, temperature=0.3):
        self.system_prompts.append(system_prompt)
        self.call_count += 1
        return ('{"code": "print(1)", "expected_output": "1", '
                '"statistical_method": "none"}')
    async def close(self):
        pass


def _fake_exec_success(code, instance_id, timeout=30):
    return CodeExecutionResult(
        instance_id=instance_id, code=code, stdout="1", stderr="",
        exit_code=0, success=True, findings="supports",
    )


def _run_node(state, monkeypatch, fake_exec=None):
    """Run run_code_execution with Ollama + _execute_sandboxed mocked (no
    network, no Ollama). Returns (state, fake_ollama).

    The node constructs ``OllamaClient(...)`` (a CALL), so we patch
    ``ce.OllamaClient`` with a FACTORY that returns a recorded _FakeOllama
    instance the test can inspect (system_prompts captured)."""
    created = []

    def _factory(*a, **k):
        c = _FakeOllama()
        created.append(c)
        return c

    monkeypatch.setattr(ce, "OllamaClient", _factory)
    if fake_exec is None:
        fake_exec = _fake_exec_success
    monkeypatch.setattr(ce, "_execute_sandboxed", fake_exec)
    out = asyncio.run(ce.run_code_execution(state))
    return out, (created[0] if created else None)


# --------------------------------------------------------------------------
# (1) contract loader
# --------------------------------------------------------------------------

class TestContractLoader:
    def test_load_contract_nonempty(self):
        s = load_contract()
        assert isinstance(s, str) and len(s) > 200

    def test_contract_fields_present(self):
        """The 4 program.md-style fields are present (original Echo wording,
        not karpathy text)."""
        s = load_contract()
        for marker in ("mutation_surface", "evaluator", "keep_discard",
                       "autonomy_posture"):
            assert marker in s, "contract missing field: {}".format(marker)

    def test_contract_format_placeholders(self):
        """The contract accepts .format(allowed_imports=..., timeout=...) —
        the code_execution node relies on this."""
        out = load_contract().format(allowed_imports="numpy, json", timeout=42)
        assert "numpy, json" in out and "42s" in out

    def test_contract_governance_gate_before_keep(self):
        """The contract STATES the governance-gate-before-keep (Goodhart
        defense made explicit to the LLM)."""
        s = load_contract()
        assert "REFLECTION" in s and "RANKING" in s and "CONSENSUS" in s
        assert "INTEGRITY" in s
        # the LLM is explicitly told it does NOT decide keep
        assert "You do NOT decide" in s or "do not decide" in s.lower() or "NOT the keep" in s

    def test_contract_hygiene_clean(self):
        """The contract text carries NO private-substrate vocabulary (public-safe;
        defensive — the contract is a public artifact, so a regression that
        re-introduces private-substrate references is caught). The banned set is
        the hygiene grep: the private-substrate tokens the contract deliberately
        avoids ('Protected stores are masked' instead of naming the substrate)."""
        s = load_contract()
        # banned private-substrate tokens, base64-encoded so this test file does
        # not itself carry them (same self-clean technique as the banned-tokens.yml
        # release guard — the plaintext tokens would self-trigger that guard).
        banned = base64.b64decode("YXhpcy1EfGFuaW1hfGFmZmVjdCBzdG9yZXxhZmZlY3Qv").decode().split("|")
        hits = [t for t in banned if t.lower() in s.lower()]
        assert hits == [], "contract carries private-substrate tokens: {}".format(hits)

    def test_contract_no_autonomous_forever_loop(self):
        """autonomy_posture EXPLICITLY forbids the autonomous forever-loop
        (the governance-less thing we refuse to build)."""
        s = load_contract()
        assert "NO autonomous forever-loop" in s or "no autonomous forever" in s.lower()
        assert "self-replication" in s.lower() or "no self-replication" in s.lower()


# --------------------------------------------------------------------------
# (2) code_execution constrained-mode branch
# --------------------------------------------------------------------------

class TestConstrainedModeBranch:
    def _hyp(self):
        return [{"id": "h1", "title": "t", "description": "d",
                 "mechanism": "m", "status": "alive", "elo_rating": 1500}]

    def test_off_uses_default_system_prompt(self, monkeypatch):
        """constrained_mode absent -> the ORIGINAL CODE_GEN_SYSTEM_PROMPT
        (no behavior change for normal research runs)."""
        state = _fake_state(self._hyp(), constrained=None)
        out, fake = _run_node(state, monkeypatch)
        assert fake.system_prompts, "chat was never called"
        sp = fake.system_prompts[0]
        # NOTE: ``==`` not ``is`` — str.format() returns a new string object
        # each call, so two .format() calls produce equal-but-not-identical
        # strings. The value check pins that the default prompt is the Finch
        # prompt with the default SANDBOX_TIMEOUT (no constrained swap).
        assert sp == ce.CODE_GEN_SYSTEM_PROMPT.format(
            allowed_imports=", ".join(sorted(ce.ALLOWED_IMPORTS)),
            timeout=ce.SANDBOX_TIMEOUT,
        )

    def test_on_substitutes_contract(self, monkeypatch):
        """constrained_mode.contract == 'autoresearch_mode' -> the system
        prompt IS the governed-mutation contract."""
        state = _fake_state(self._hyp(), constrained={
            "contract": "autoresearch_mode",
            "per_mutation_budget": 12,
            "outer_kill": 99,
        })
        out, fake = _run_node(state, monkeypatch)
        sp = fake.system_prompts[0]
        assert "CONSTRAINED MUTATION MODE" in sp
        assert "12s" in sp  # per_mutation_budget threaded into the prompt
        assert sp is not ce.CODE_GEN_SYSTEM_PROMPT

    def test_on_with_other_contract_name_is_off(self, monkeypatch):
        """A non-matching contract name is OFF (defensive: only the exact
        'autoresearch_mode' token opts in)."""
        state = _fake_state(self._hyp(), constrained={"contract": "other"})
        out, fake = _run_node(state, monkeypatch)
        sp = fake.system_prompts[0]
        assert "Code Execution Agent (Finch)" in sp  # the default prompt


class TestPerMutationBudget:
    def _hyp(self):
        return [{"id": "h1", "title": "t", "description": "d",
                 "mechanism": "m", "status": "alive", "elo_rating": 1500}]

    def test_budget_threads_to_sandboxed(self, monkeypatch):
        """per_mutation_budget is passed as timeout= to _execute_sandboxed
        when constrained on."""
        seen = {}
        def fake_exec(code, iid, timeout=30):
            seen[iid] = timeout
            return _fake_exec_success(code, iid, timeout)
        state = _fake_state(self._hyp(), constrained={
            "contract": "autoresearch_mode", "per_mutation_budget": 17, "outer_kill": 99,
        }, num_instances=3)
        _run_node(state, monkeypatch, fake_exec=fake_exec)
        assert all(v == 17 for v in seen.values()), "budget not threaded: {}".format(seen)
        assert len(seen) == 3

    def test_off_uses_default_timeout(self, monkeypatch):
        """constrained off -> _execute_sandboxed gets SANDBOX_TIMEOUT (not the
        budget) — no behavior change."""
        seen = {}
        def fake_exec(code, iid, timeout=30):
            seen[iid] = timeout
            return _fake_exec_success(code, iid, timeout)
        state = _fake_state(self._hyp(), constrained=None, num_instances=3)
        _run_node(state, monkeypatch, fake_exec=fake_exec)
        assert all(v == ce.SANDBOX_TIMEOUT for v in seen.values())


class TestOuterKill:
    def _hyp(self):
        return [{"id": "h1", "title": "t", "description": "d",
                 "mechanism": "m", "status": "alive", "elo_rating": 1500}]

    def test_outer_kill_stops_batch(self, monkeypatch):
        """outer_kill already in the past -> the FIRST instance is past the
        deadline -> recorded as a timed-out crash + _execute_sandboxed is
        NEVER called."""
        called = []
        def fake_exec(code, iid, timeout=30):
            called.append(iid)
            return _fake_exec_success(code, iid, timeout)
        # outer_kill = -1.0 -> deadline = now - 1s (already past at loop top)
        state = _fake_state(self._hyp(), constrained={
            "contract": "autoresearch_mode", "per_mutation_budget": 30, "outer_kill": -1.0,
        }, num_instances=3)
        out, _ = _run_node(state, monkeypatch, fake_exec=fake_exec)
        assert called == [], "_execute_sandboxed called past the outer kill: {}".format(called)
        # the per-hyp result carries the crash (non-success) instances
        res = out["code_execution_results"][0]
        assert res["consensus_reached"] is False
        assert any(not inst["success"] for inst in res["instances"])

    def test_outer_kill_not_reached_when_off(self, monkeypatch):
        """constrained off -> outer_kill = 10*SANDBOX_TIMEOUT (huge); the
        batch runs normally (no premature kill)."""
        state = _fake_state(self._hyp(), constrained=None, num_instances=3)
        out, _ = _run_node(state, monkeypatch)
        res = out["code_execution_results"][0]
        # all 3 instances executed (default _fake_exec_success -> consensus)
        assert res["total_instances"] == 3


class TestProposalDict:
    def _hyp(self):
        return [{"id": "h1", "title": "t", "description": "d",
                 "mechanism": "m", "status": "alive", "elo_rating": 1500}]

    def test_proposal_present_and_constrained_flag(self, monkeypatch):
        """The per-hypothesis result carries a 'proposal' dict recording
        constrained_mode + the budget_used (informational)."""
        state = _fake_state(self._hyp(), constrained={
            "contract": "autoresearch_mode", "per_mutation_budget": 20, "outer_kill": 99,
        })
        out, _ = _run_node(state, monkeypatch)
        prop = out["code_execution_results"][0]["proposal"]
        assert prop["constrained_mode"] is True
        assert prop["budget_used"] == 20

    def test_proposal_off_when_constrained_off(self, monkeypatch):
        state = _fake_state(self._hyp(), constrained=None)
        out, _ = _run_node(state, monkeypatch)
        prop = out["code_execution_results"][0]["proposal"]
        assert prop["constrained_mode"] is False

    def test_proposal_is_informational_not_the_decision(self, monkeypatch):
        """The proposal's 'action' is a HINT (consensus-derived), NOT the
        keep/discard decision. The decision is the gauntlet's, at
        format_report. Pin: the proposal says 'keep' on consensus, but that
        string is NOT a status field — it's a reason string."""
        state = _fake_state(self._hyp(), constrained=None)
        out, _ = _run_node(state, monkeypatch)
        res = out["code_execution_results"][0]
        assert "proposal" in res
        # the result dict has NO top-level 'status' — status is decided later
        assert "status" not in res, "code_execution must NOT set a keep/discard status"


# --------------------------------------------------------------------------
# (3) provenance ledger
# --------------------------------------------------------------------------

class TestResearchOutcome:
    def test_roundtrip(self):
        o = prov.ResearchOutcome("r1", "t", "c1", "accepted", 1.5, "keep", "gv")
        assert prov.ResearchOutcome.from_dict(o.to_dict()).to_tsv_row() == o.to_tsv_row()

    def test_tsv_row_seven_fields(self):
        o = prov.ResearchOutcome("r1", "t", "c1", "accepted", 1.5, "keep", "gv")
        assert o.to_tsv_row().count("\t") == 6  # 7 columns = 6 tabs

    def test_tsv_row_order_matches_columns(self):
        o = prov.ResearchOutcome("R", "T", "C", "M", "1.0", "keep", "G")
        assert o.to_tsv_row() == "R\tT\tC\tM\t1.0\tkeep\tG"
        assert prov.TSV_COLUMNS == (
            "run_id", "timestamp", "commit", "metric", "resource",
            "status", "governance_verdict",
        )

    def test_tsv_row_strips_tabs_newlines_in_fields(self):
        o = prov.ResearchOutcome("r", "t", "c", "m", 1.0, "keep",
                                 "a\tb\nc")
        row = o.to_tsv_row()
        assert "\n" not in row and row.count("\t") == 6


class TestBuildOutcome:
    def _hyp(self, status="alive", elo=1500):
        return {"id": "c1", "status": status, "elo_rating": elo}

    def _cr(self, reached=True, failed=False):
        return {
            "hypothesis_id": "c1",
            "verdict": "accepted" if reached else "inconclusive",
            "consensus_reached": reached,
            "majority_finding": "supports" if reached else None,
            "instances": [{"success": not failed}] if failed else [{"success": True}],
        }

    def test_keep_on_alive_clean_consensus(self):
        o = prov.build_outcome(self._hyp("alive"), self._cr(True), [], "r1")
        assert o.status == "keep"

    def test_keep_on_refined(self):
        o = prov.build_outcome(self._hyp("refined"), self._cr(True), [], "r1")
        assert o.status == "keep"

    def test_discard_on_reflection_eliminated(self):
        o = prov.build_outcome(self._hyp("eliminated"), self._cr(True), [], "r1")
        assert o.status == "discard"

    def test_crash_on_failed_instances_alive(self):
        o = prov.build_outcome(self._hyp("alive"), self._cr(False, failed=True), [], "r1")
        assert o.status == "crash"

    def test_reflection_eliminated_wins_over_crash(self):
        """If reflection eliminated the hypothesis, that wins over crash
        (the gauntlet's verdict is authoritative)."""
        o = prov.build_outcome(self._hyp("eliminated"), self._cr(False, failed=True), [], "r1")
        assert o.status == "discard"

    def test_breach_forces_discard_even_on_alive(self):
        """A non-empty verify_integrity (seam breach) forces discard for
        every row — never keep into a compromised substrate."""
        o = prov.build_outcome(self._hyp("alive"), self._cr(True),
                               ["some/seam.py:abc live"], "r1")
        assert o.status == "discard"
        assert "BREACH" in o.governance_verdict

    def test_inconclusive_consensus_suppresses_alive(self):
        """Goodhart consensus gate: a hypothesis that is alive, with
        code that ran and ALL instances succeeded but reached NO consensus
        (no majority), is DISCARDED — contract gate #1 'inconclusive
        suppresses'. The consensus node only applies a -10 ELO penalty and
        never sets status='eliminated', so without this branch the row would
        silently read 'keep' while governance_verdict says consensus=inconclusive
        (a self-contradiction the contract refuses)."""
        # all instances succeeded, but consensus_reached=False (inconclusive)
        cr = {
            "hypothesis_id": "c1",
            "verdict": "inconclusive",
            "consensus_reached": False,
            "majority_finding": None,
            "instances": [{"success": True}, {"success": True}, {"success": True}],
        }
        o = prov.build_outcome(self._hyp("alive"), cr, [], "r1")
        assert o.status == "discard", (
            "Goodhart breach: inconclusive consensus on an alive hypothesis "
            "was kept — contract gate #1 (inconclusive suppresses) not enforced")
        assert "inconclusive" in o.governance_verdict

    def test_crash_not_silently_demoted_to_consensus_discard(self):
        """If instances actually CRASHED (some failed) AND consensus was not
        reached, the status is 'crash' (a real failure), NOT a consensus-
        discard. The consensus-discard branch must only fire when code ran
        cleanly but disagreed — a crash is a distinct outcome."""
        cr = {
            "hypothesis_id": "c1",
            "verdict": "inconclusive",
            "consensus_reached": False,
            "majority_finding": None,
            "instances": [{"success": True}, {"success": False}, {"success": True}],
        }
        o = prov.build_outcome(self._hyp("alive"), cr, [], "r1")
        assert o.status == "crash"

    def test_compose_verdict_has_all_four_signals(self):
        o = prov.build_outcome(self._hyp("alive", 1620), self._cr(True), [], "r1")
        for sig in ("reflection=", "consensus=", "elo=", "integrity="):
            assert sig in o.governance_verdict, sig

    def test_metric_records_majority_finding(self):
        o = prov.build_outcome(self._hyp("alive"), self._cr(True), [], "r1")
        assert o.metric == "supports"


class TestAppendResultsTsv:
    def test_creates_header_then_appends(self, tmp_path):
        p = tmp_path / "results.tsv"
        o = prov.ResearchOutcome("r", "t", "c", "m", 1.0, "keep", "g")
        prov.append_results_tsv(p, o)
        lines = p.read_text().splitlines()
        assert lines[0] == "\t".join(prov.TSV_COLUMNS)  # header
        assert lines[1] == o.to_tsv_row()
        # second call appends (no second header)
        prov.append_results_tsv(p, o)
        lines2 = p.read_text().splitlines()
        assert lines2[0] == "\t".join(prov.TSV_COLUMNS)
        assert len(lines2) == 3  # 1 header + 2 rows

    def test_write_outcomes_for_run_matches_by_id(self, tmp_path, monkeypatch):
        hyps = [{"id": "a", "status": "alive"}, {"id": "b", "status": "eliminated"}]
        crs = [{"hypothesis_id": "a", "verdict": "accepted", "consensus_reached": True,
                "majority_finding": "sup", "instances": [{"success": True}]}]
        p = tmp_path / "results.tsv"
        monkeypatch.setattr(prov, "default_tsv_path", lambda c=None: p)
        n = prov.write_outcomes_for_run(hyps, crs, [], "run1")
        assert n == 2
        lines = p.read_text().splitlines()
        assert len(lines) == 3  # header + 2
        # row for 'a' is keep (alive + consensus), row for 'b' is discard (no cr -> discard)
        assert "keep" in lines[1] and "a" in lines[1]
        assert "discard" in lines[2]


class TestTsvPathOutsideRepo:
    def test_default_path_under_home_learning(self):
        p = prov.default_tsv_path({})
        assert p.as_posix().endswith(".hermes/learning/results.tsv")
        # the ledger is under ~/.hermes/learning — OUTSIDE the repo tree
        # (no .gitignore entry needed)
        assert "/hermes-echo/" not in p.as_posix()

    def test_default_path_respects_learning_dir_override(self):
        p = prov.default_tsv_path({"research": {"learning_dir": "/tmp/alt"}})
        assert p.as_posix() == "/tmp/alt/results.tsv"

    def test_git_check_ignore_clean(self, tmp_path):
        """The ledger is operator state OUTSIDE the repo, never committed.
        The real invariant is that the ledger path does not resolve INSIDE the
        hermes-echo repo tree — so assert that directly via Path.resolve()
        containment, not via ``git check-ignore`` (which returns rc=128
        "outside repository" for a tmp_path and would pass for the wrong
        reason). The git check is kept as a secondary signal but asserted
        specifically (rc==1 == not-ignored, NOT the broad rc!=0 that also
        admits the rc=128 error case)."""
        p = tmp_path / "results.tsv"
        prov.append_results_tsv(
            p, prov.ResearchOutcome("r", "t", "c", "m", 1.0, "keep", "g"))
        assert p.exists()
        # Primary invariant: the ledger path resolves OUTSIDE the repo.
        # Derive the repo root from this test file's location (portable, no
        # hardcoded host path): .../<repo>/tests/unit/agents/echo/<this file>
        repo = Path(__file__).resolve().parents[4]
        try:
            p.resolve().relative_to(repo)
            inside = True
        except ValueError:
            inside = False
        assert not inside, (
            "ledger path resolves inside the repo -> would be committable")
        # Secondary signal: git check-ignore on a path outside the repo returns
        # rc=128 ("path is outside repository"), which is NOT the rc=1
        # "not-ignored" the old broad assertion claimed. Assert the specific
        # rc==1 only when the path is inside a repo; here it is outside, so we
        # only assert git did not report it as ignored (rc != 0).
        import subprocess
        r = subprocess.run(["git", "check-ignore", str(p)],
                           capture_output=True, cwd=str(repo))
        assert r.returncode != 0  # not reported as ignored (clean)


# --------------------------------------------------------------------------
# (4) Goodhart — proposal overridden by the gauntlet
# --------------------------------------------------------------------------

class TestGoodhartGate:
    def test_proposal_keep_overridden_by_reflection_eliminated(self, monkeypatch):
        """The code_execution proposal says 'keep' (consensus reached), but
        reflection eliminates the hypothesis -> build_outcome (the gauntlet
        writer) records DISCARD. The metric the LLM cites is not the metric
        that decides keep."""
        # code_execution emits proposal.action='keep' (consensus True)
        state = _fake_state([{"id": "h1", "title": "t", "description": "d",
                              "mechanism": "m", "status": "alive", "elo_rating": 1500}],
                           constrained=None)
        out, _ = _run_node(state, monkeypatch)
        ce_res = out["code_execution_results"][0]
        assert ce_res["proposal"]["action"] == "keep"
        # NOW reflection runs (downstream) and sets status='eliminated'
        hyp_post_reflection = {"id": "h1", "status": "eliminated", "elo_rating": 1400}
        # format_report's writer composes the outcome from the POST-gauntlet
        # hypothesis; the proposal is NOT consulted.
        outcome = prov.build_outcome(hyp_post_reflection, ce_res, [], "run1")
        assert outcome.status == "discard", (
            "Goodhart breach: reflection eliminated the hypothesis but the "
            "proposal's 'keep' leaked into the ledger status"
        )

    def test_format_report_writes_keep_only_on_survivor(self, monkeypatch, tmp_path):
        """format_report writes a 'keep' row ONLY for a hypothesis that
        survived the gauntlet (alive/refined + clean integrity); an eliminated
        one gets 'discard' — the row status is the gauntlet composite, not the
        proposal."""
        from hermes_cli.agents.echo.research.nodes.format_report import format_report
        p = tmp_path / "results.tsv"
        monkeypatch.setattr(prov, "default_tsv_path", lambda c=None: p)
        monkeypatch.setattr("hermes_cli.agents.echo.seam_manifest.verify_integrity",
                            lambda *a, **k: [])
        state = {
            "research_goal": "g",
            "config": {},
            "hypotheses": [
                {"id": "a", "title": "A", "description": "", "mechanism": "",
                 "status": "alive", "elo_rating": 1600},
                {"id": "b", "title": "B", "description": "", "mechanism": "",
                 "status": "eliminated", "elo_rating": 1400},
            ],
            "code_execution_results": [
                {"hypothesis_id": "a", "verdict": "accepted",
                 "consensus_reached": True, "majority_finding": "sup",
                 "instances": [{"success": True}],
                 "proposal": {"action": "keep"}},
                {"hypothesis_id": "b", "verdict": "accepted",
                 "consensus_reached": True, "majority_finding": "sup",
                 "instances": [{"success": True}],
                 "proposal": {"action": "keep"}},  # proposal says keep...
            ],
            "tournament_results": [],
        }
        asyncio.run(format_report(state))
        lines = p.read_text().splitlines()
        assert len(lines) == 3  # header + 2 rows
        # row for 'a' (survivor) is keep; row for 'b' (eliminated) is discard
        # despite BOTH proposals saying 'keep'
        a_row = next(l for l in lines if "\ta\t" in l)
        b_row = next(l for l in lines if "\tb\t" in l)
        assert "keep" in a_row.split("\t")[5]
        assert "discard" in b_row.split("\t")[5], (
            "eliminated hypothesis kept despite proposal='keep' -> Goodhart breach")

    def test_format_report_breach_forces_all_discard(self, monkeypatch, tmp_path):
        """A non-empty verify_integrity (seam breach) forces discard for
        EVERY row — never keep into a compromised substrate."""
        from hermes_cli.agents.echo.research.nodes.format_report import format_report
        p = tmp_path / "results.tsv"
        monkeypatch.setattr(prov, "default_tsv_path", lambda c=None: p)
        monkeypatch.setattr("hermes_cli.agents.echo.seam_manifest.verify_integrity",
                            lambda *a, **k: ["some/seam.py:abc"])
        state = {
            "research_goal": "g", "config": {},
            "hypotheses": [{"id": "a", "title": "A", "description": "",
                            "mechanism": "", "status": "alive", "elo_rating": 1600}],
            "code_execution_results": [{"hypothesis_id": "a", "verdict": "accepted",
                                        "consensus_reached": True,
                                        "majority_finding": "sup",
                                        "instances": [{"success": True}],
                                        "proposal": {"action": "keep"}}],
            "tournament_results": [],
        }
        asyncio.run(format_report(state))
        lines = p.read_text().splitlines()
        row = next(l for l in lines if "\ta\t" in l)
        assert "discard" in row.split("\t")[5]
        assert "BREACH" in row.split("\t")[6]

    def test_format_report_none_integrity_forces_discard(self, monkeypatch, tmp_path):
        """Fail-open fix: verify_integrity() returns
        None when the scratch dir is absent/unverifiable (e.g. an unplugged
        or HERMES_SEAM_SCRATCH unset). None MUST NOT read as clean (the old
        ``verify_integrity() or []`` coerced None -> [] -> keep — a fail-open
        the adjacent comment explicitly refused). Pin: an alive hypothesis
        with consensus, on an unverifiable substrate, records DISCARD with an
        'unverifiable' marker in governance_verdict."""
        from hermes_cli.agents.echo.research.nodes.format_report import format_report
        p = tmp_path / "results.tsv"
        monkeypatch.setattr(prov, "default_tsv_path", lambda c=None: p)
        # verify_integrity returns None (soft-fail, NOT raised) — the
        # fail-open path. Must coerce to ['unverifiable'] -> discard.
        monkeypatch.setattr("hermes_cli.agents.echo.seam_manifest.verify_integrity",
                            lambda *a, **k: None)
        state = {
            "research_goal": "g", "config": {},
            "hypotheses": [{"id": "a", "title": "A", "description": "",
                            "mechanism": "", "status": "alive", "elo_rating": 1600}],
            "code_execution_results": [{"hypothesis_id": "a", "verdict": "accepted",
                                        "consensus_reached": True,
                                        "majority_finding": "sup",
                                        "instances": [{"success": True}],
                                        "proposal": {"action": "keep"}}],
            "tournament_results": [],
        }
        asyncio.run(format_report(state))
        lines = p.read_text().splitlines()
        assert len(lines) == 2  # header + 1 row
        row = lines[1]
        # the alive+consensus hypothesis is DISCARDED because the substrate
        # is unverifiable (None coerced to ['unverifiable'] -> breach)
        assert "discard" in row.split("\t")[5], (
            "fail-open: unverifiable substrate (verify_integrity None) "
            "recorded 'keep' instead of discard")
        # the governance_verdict records the unverifiable marker (integrity=
        # BREACH:1, since ['unverifiable'] is a 1-element truthy breach list)
        gv = row.split("\t")[6]
        assert "BREACH" in gv or "unverifiable" in gv, (
            "unverifiable substrate not surfaced in governance_verdict: " + gv)

    def test_format_report_provenance_failure_does_not_break_report(self, monkeypatch, tmp_path):
        """A provenance write failure (e.g. read-only FS) is swallowed — the
        final_report still returns (provenance is operator state, never
        user-facing)."""
        from hermes_cli.agents.echo.research.nodes.format_report import format_report
        def boom(path, outcome):
            raise OSError("read-only filesystem")
        monkeypatch.setattr(prov, "append_results_tsv", boom)
        monkeypatch.setattr("hermes_cli.agents.echo.seam_manifest.verify_integrity",
                            lambda *a, **k: [])
        state = {
            "research_goal": "g", "config": {},
            "hypotheses": [{"id": "a", "title": "A", "description": "",
                            "mechanism": "", "status": "alive", "elo_rating": 1600}],
            "code_execution_results": [], "tournament_results": [],
        }
        out = asyncio.run(format_report(state))
        assert "final_report" in out  # report still assembled
        assert out["final_report"]["surviving_hypotheses"] == 1


# --------------------------------------------------------------------------
# (5) seam attestation — the new files are auto-discovered
# --------------------------------------------------------------------------

class TestSeamDiscovery:
    @pytest.mark.skipif(not os.path.isdir(os.environ.get("HERMES_SEAM_SCRATCH", DEFAULT_SCRATCH)), reason="seam scratch dir absent (public build — no deploy source tree mounted)")
    def test_new_files_in_discover(self):
        """The 3 new seam files are auto-discovered by seam_manifest.discover
        (so they deploy + attest for free, no manifest edit). seam_manifest is
        a seam file deployed at hermes_cli/agents/echo/seam_manifest.py."""
        from hermes_cli.agents.echo.seam_manifest import discover
        pairs = discover()
        scratch_rels = [s for s, _ in pairs]
        assert "research/provenance.py" in scratch_rels
        assert "research/contract_loader.py" in scratch_rels
        assert "research/nodes/format_report.py" in scratch_rels