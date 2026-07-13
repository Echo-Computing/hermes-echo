"""P0-9 (2026-07-06 red-team) — tests for the seam manifest single-source-of-
truth module (deployed live at hermes_cli/agents/echo/seam_manifest.py).

Two roles: (1) apply_seam.sh imports discover() to auto-generate the cp +
attestation lists; (2) agent.py calls verify_integrity() at module load as the
runtime seam-integrity self-check. The most important test here is
test_verify_integrity_current_live_tree_is_clean — it is the DEADLOCK GUARD: if
verify_integrity() false-positives on a freshly-deployed clean tree, agent.py's
module-load check would raise on import, locking out the agent AND failing
apply_seam.sh's own import smoke (the one command that recovers it). That test
asserts the runtime check agrees with apply_seam.sh's deploy-time attestation
on a clean deploy."""
import os

import pytest

from hermes_cli.agents.echo.seam_manifest import (
    DEFAULT_SCRATCH,
    _is_helper,
    _live_rel,
    discover,
    verify_integrity,
)


class TestIsHelper:
    def test_excludes_single_underscore_lowercase_helpers(self):
        assert _is_helper("_run_probe.py")
        assert _is_helper("_verify_redteam.py")
        assert _is_helper("_sb_test")  # rule is name-based, extension-agnostic

    def test_keeps_dunder_init(self):
        # __init__.py has a double underscore -> second char is '_', not lowercase
        assert _is_helper("__init__.py") is False

    def test_keeps_normal_seam_files(self):
        assert _is_helper("agent.py") is False
        assert _is_helper("seam_manifest.py") is False


class TestLiveRelMapping:
    def test_echo_cmd_overridden_to_commands(self):
        assert _live_rel("echo_cmd.py") == "commands/echo_cmd.py"

    def test_default_mirrors_under_agents_echo(self):
        assert _live_rel("agent.py") == "agents/echo/agent.py"
        assert _live_rel("tools/shell_tools.py") == "agents/echo/tools/shell_tools.py"
        assert _live_rel("research/nodes/code_execution.py") == "agents/echo/research/nodes/code_execution.py"

    def test_seam_manifest_itself_uses_default_rule(self):
        assert _live_rel("seam_manifest.py") == "agents/echo/seam_manifest.py"


class TestDiscover:
    @pytest.mark.skipif(not os.path.isdir(os.environ.get("ANIMA_SEAM_SCRATCH", DEFAULT_SCRATCH)), reason="seam scratch dir absent (public build — no deploy source tree mounted)")
    def test_returns_the_known_seam_set_with_correct_mappings(self):
        pairs = dict(discover())
        # The public seam files + seam_manifest.py itself (added by P0-9).
        # Verified mappings (override table + default rule):
        assert pairs["echo_cmd.py"] == "commands/echo_cmd.py"
        assert pairs["agent.py"] == "agents/echo/agent.py"
        assert pairs["state.py"] == "agents/echo/state.py"
        assert pairs["system_prompt.py"] == "agents/echo/system_prompt.py"
        assert pairs["tools/shell_tools.py"] == "agents/echo/tools/shell_tools.py"
        assert pairs["tools/search_tools.py"] == "agents/echo/tools/search_tools.py"
        assert pairs["research/nodes/code_execution.py"] == "agents/echo/research/nodes/code_execution.py"
        assert pairs["seam_manifest.py"] == "agents/echo/seam_manifest.py"

    @pytest.mark.skipif(not os.path.isdir(os.environ.get("ANIMA_SEAM_SCRATCH", DEFAULT_SCRATCH)), reason="seam scratch dir absent (public build — no deploy source tree mounted)")
    def test_excludes_helper_scripts(self):
        pairs = dict(discover())
        for s_rel in pairs:
            assert not s_rel.startswith("_"), f"helper script leaked into seam set: {s_rel}"

    @pytest.mark.skipif(not os.path.isdir(os.environ.get("ANIMA_SEAM_SCRATCH", DEFAULT_SCRATCH)), reason="seam scratch dir absent (public build — no deploy source tree mounted)")
    def test_excludes_seam_tests_subtree(self):
        # seam-tests/ lives under scratch but is NOT seam source; it must not
        # appear in the source-seam discover() (else test_seam_*.py would be
        # cp'd into agents/echo/ as source).
        pairs = dict(discover())
        for s_rel in pairs:
            assert not s_rel.startswith("seam-tests/"), f"seam-tests leaked into source seam: {s_rel}"
            assert "seam-tests" not in s_rel

    @pytest.mark.skipif(not os.path.isdir(os.environ.get("ANIMA_SEAM_SCRATCH", DEFAULT_SCRATCH)), reason="seam scratch dir absent (public build — no deploy source tree mounted)")
    def test_excludes_pycache(self):
        pairs = dict(discover())
        for s_rel in pairs:
            assert "__pycache__" not in s_rel

    def test_raises_on_absent_scratch(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover(str(tmp_path / "does-not-exist"))

    def test_raises_on_empty_scratch(self, tmp_path):
        # a scratch with only helpers / __pycache__ -> no seam files -> fail closed
        (tmp_path / "_helper.py").write_text("x = 1\n")
        with pytest.raises(RuntimeError):
            discover(str(tmp_path))


class TestVerifyIntegrity:
    def test_clean_deploy_returns_empty(self, tmp_path):
        scratch = tmp_path / "scratch"
        live = tmp_path / "hermes_cli"
        scratch.mkdir()
        (scratch / "agent.py").write_text("SEAM\n", encoding="utf-8")
        (live / "agents" / "echo").mkdir(parents=True)
        (live / "agents" / "echo" / "agent.py").write_text("SEAM\n", encoding="utf-8")
        assert verify_integrity(scratch_dir=str(scratch), live_root=str(live)) == []

    def test_detects_content_drift(self, tmp_path):
        scratch = tmp_path / "scratch"
        live = tmp_path / "hermes_cli"
        scratch.mkdir()
        (scratch / "agent.py").write_text("FRESH SEAM\n", encoding="utf-8")
        (live / "agents" / "echo").mkdir(parents=True)
        (live / "agents" / "echo" / "agent.py").write_text("STALE SEAM\n", encoding="utf-8")
        mismatches = verify_integrity(scratch_dir=str(scratch), live_root=str(live))
        assert mismatches, "content drift not detected"
        assert mismatches[0][0] == "agents/echo/agent.py"
        assert mismatches[0][2] != "MISSING"  # it exists, just differs

    def test_detects_missing_live_file(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "agent.py").write_text("SEAM\n", encoding="utf-8")
        live = tmp_path / "hermes_cli"
        live.mkdir()  # no agents/echo/agent.py at all
        mismatches = verify_integrity(scratch_dir=str(scratch), live_root=str(live))
        assert mismatches
        assert mismatches[0][2] == "MISSING"

    def test_crlf_normalization_does_not_false_positive(self, tmp_path):
        """scratch (NTFS, CRLF) vs live (Linux, LF) with IDENTICAL content must
        NOT be a mismatch — the runtime check would otherwise false-positive
        after a clean apply_seam.sh deploy (lockout deadlock)."""
        scratch = tmp_path / "scratch"
        live = tmp_path / "hermes_cli"
        scratch.mkdir()
        (scratch / "agent.py").write_bytes(b"line1\r\nline2\r\n")
        (live / "agents" / "echo").mkdir(parents=True)
        (live / "agents" / "echo" / "agent.py").write_bytes(b"line1\nline2\n")
        assert verify_integrity(scratch_dir=str(scratch), live_root=str(live)) == []

    def test_scratch_absent_returns_none(self, tmp_path):
        # unverifiable (env), NOT a mismatch -> caller soft-fails
        assert verify_integrity(scratch_dir=str(tmp_path / "nope")) is None

    def test_verify_integrity_current_live_tree_is_clean(self):
        """DEADLOCK GUARD (critical): on the freshly-deployed live tree,
        verify_integrity() MUST return []. If it false-positives, agent.py's
        module-load check raises on import -> the agent is locked out AND
        apply_seam.sh's import smoke fails (the one recovery command). This
        test asserts the runtime hash agrees with apply_seam.sh's deploy-time
        hash on a clean deploy (same CRLF normalization, same 12-hex sha256)."""
        result = verify_integrity()  # DEFAULT_SCRATCH + __file__-derived live_root
        if result is None:
            pytest.skip("scratch dir absent (seam source tree not mounted) — unverifiable, not a mismatch")
        assert result == [], (
            "runtime seam check would false-positive on a clean deploy (deadlock): "
            f"{result}"
        )