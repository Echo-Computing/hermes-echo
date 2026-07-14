"""Tests for the confirm_destructive LIVE gate in execute_tools.

confirm_destructive was set in config (echo_cmd.py L114, default True; --yes
forces False) and promised in the system prompt but read by NOTHING —
documented-not-enforced. This makes it a live gate at the execute_tools
dispatch chokepoint (agent.py, after every integrity guard has passed and before
any tool executes): a CertifiedTool with destructive=True is REFUSED fail-closed
unless an operator confirmer approves it (interactive REPL) or
confirm_destructive is off. A pure guard_source_policy=='write' check would
MISS run_shell (policy 'none' — takes `command`, not `path`), so destructive is
a separate CertifiedTool metadata field, set on write_file / edit_file / run_shell.

The gate is fail-closed in every refusal mode: confirmer absent (headless
--prompt), confirmer returns False, confirmer raises, or confirmer yields a
non-bool. --yes / config set confirm_destructive=False -> gate skipped. The
confirmer is operator-supplied (echo_cmd), never LLM-supplied; it receives
(tool_name, params) only, never state.

These tests pin all six behaviors (a)-(f) on the REAL gate path (execute_tools
with a monkeypatched _build_registry routing a fake destructive/non-destructive
CertifiedTool through the metadata-driven gates), not a reimplementation.
"""
import pytest

from hermes_cli.agents.echo.agent import CertifiedTool, execute_tools, _build_registry
from hermes_cli.agents.echo.state import EchoState


# ---- clean handlers (handler-cert clean; the gate under test is confirm, not cert) ----

def _clean_handler(x: str = "") -> str:
    return "ok"


# ---- a minimal fake registry (same pattern as the cert-collapse tests) ----

class _FakeReg:
    def __init__(self, tool, call):
        self._tool = tool
        self._call = call
        self.executed = False

    def parse_tool_calls(self, _text):
        return [self._call]

    def has_tool_calls(self, text):
        return True

    def get(self, name):
        return self._tool if name == self._call["name"] else None

    def list_tools(self):
        return [{"name": self._call["name"], "description": "d", "parameters": []}]

    def execute(self, _n, _p):
        self.executed = True
        return {"name": self._call["name"], "success": True,
                "output": "RAN", "error": None}


def _destructive_tool():
    """A destructive, cert-clean, no-path CertifiedTool — passes the cert
    (requires_handler_cert=False skips it) + guard-source (policy 'none') +
    protected-store (params reference no store) gates, so the ONLY gate left is
    the confirm_destructive gate under test."""
    return CertifiedTool(
        name="my_destructive",
        description="destructive probe",
        parameters=[{"name": "x", "type": "string", "required": False}],
        handler=_clean_handler,
        guard_source_policy="none",   # no `path` param -> sanity permits "none"
        requires_handler_cert=False,   # skip the cert; the gate under test is confirm
        destructive=True,
    )


def _non_destructive_tool():
    """Same as above but destructive=False — the confirm gate must skip it."""
    return CertifiedTool(
        name="my_safe",
        description="safe probe",
        parameters=[{"name": "x", "type": "string", "required": False}],
        handler=_clean_handler,
        guard_source_policy="none",
        requires_handler_cert=False,
        destructive=False,
    )


def _run(confirm_destructive, confirmer, monkeypatch):
    """Run execute_tools with the destructive tool + the given cfg."""
    tool = _destructive_tool()
    call = {"name": "my_destructive", "parameters": {"x": "1"}}
    fake = _FakeReg(tool, call)
    # Patch _build_registry (execute_tools calls it) via monkeypatch so the
    # patch is cleaned up after the test (a bare module assignment would leak
    # the fake into other suites' _build_registry() calls).
    monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                        lambda _md=None: fake)
    cfg = {
        "memory_dir": "/tmp/echomem_inj3",
        "confirm_destructive": confirm_destructive,
    }
    if confirmer is not None:
        cfg["confirmer"] = confirmer
    state = EchoState(config=cfg, messages=[], user_input="")
    state["response"] = "<ignored>"
    return execute_tools(state), fake


# ---- confirmer variants (module-level so they have real source if needed) ----

def _confirm_true(name, params):
    return True

def _confirm_false(name, params):
    return False

def _confirm_raising(name, params):
    raise RuntimeError("confirmer crashed")

def _confirm_nonbool(name, params):
    return "yes"  # truthy non-bool string -> must be REFUSED (is True, not bool())


class TestConfirmDestructiveGate:
    def test_a_confirm_on_no_confirmer_refused(self, monkeypatch):
        """confirm_destructive=True + confirmer absent (headless --prompt) ->
        destructive tool REFUSED fail-closed; handler NOT executed."""
        out, fake = _run(True, None, monkeypatch)
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "requires confirmation" in r0["error"]
        assert not fake.executed

    def test_b_confirmer_true_executes(self, monkeypatch):
        """confirm_destructive=True + confirmer returns True -> executes."""
        out, fake = _run(True, _confirm_true, monkeypatch)
        r0 = out["tool_results"][0]
        assert r0["success"], f"expected execution, got {r0!r}"
        assert fake.executed

    def test_c_confirmer_false_refused(self, monkeypatch):
        """confirm_destructive=True + confirmer returns False -> REFUSED."""
        out, fake = _run(True, _confirm_false, monkeypatch)
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "requires confirmation" in r0["error"]
        assert not fake.executed

    def test_d_confirmer_raises_refused_fail_closed(self, monkeypatch):
        """A confirmer that raises is treated as refuse (fail-closed) — a
        confirmer fault never lets a destructive tool through."""
        out, fake = _run(True, _confirm_raising, monkeypatch)
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "requires confirmation" in r0["error"]
        assert not fake.executed

    def test_e_confirm_off_executes_regardless(self, monkeypatch):
        """confirm_destructive=False (--yes / config) -> gate skipped ->
        destructive tool EXECUTES regardless of confirmer (even None)."""
        out, fake = _run(False, None, monkeypatch)
        r0 = out["tool_results"][0]
        assert r0["success"], f"confirm off should allow, got {r0!r}"
        assert fake.executed

    def test_f_non_destructive_tool_executes_regardless(self, monkeypatch):
        """A non-destructive tool (destructive=False) EXECUTES regardless of
        confirm_destructive + confirmer — the gate only touches destructive
        tools."""
        tool = _non_destructive_tool()
        call = {"name": "my_safe", "parameters": {"x": "1"}}
        fake = _FakeReg(tool, call)
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        state = EchoState(
            config={"memory_dir": "/tmp/echomem_inj3",
                    "confirm_destructive": True},  # on, + no confirmer
            messages=[], user_input="",
        )
        state["response"] = "<ignored>"
        out = execute_tools(state)
        r0 = out["tool_results"][0]
        assert r0["success"], f"non-destructive should pass, got {r0!r}"
        assert fake.executed

    def test_confirmer_nonbool_is_refused(self, monkeypatch):
        """A confirmer returning a truthy NON-bool ("yes") is REFUSED — the gate
        requires the bool True (``_r is True``), not bool() coercion. This is the strict
        invariant: non-bool -> refuse. The sole real
        confirmer (echo_cmd's click.confirm) returns a real bool, so the strict
        check has no downside and closes the bool()-coercion overclaim. (test_b
        covers the bool-True-accepts path.)"""
        out, fake = _run(True, _confirm_nonbool, monkeypatch)
        r0 = out["tool_results"][0]
        assert not r0["success"], "truthy non-bool confirmer must be refused, not coerced"
        assert "requires confirmation" in r0["error"]
        assert not fake.executed


class TestDestructiveMetadataOnStandardTools:
    """Pin that the three destructive standard tools carry destructive=True at
    registration (a regression here would silently re-open the gate-miss the
    field was added to close — esp. run_shell, whose guard_source_policy='none'
    means a ==write check would miss it)."""

    def test_write_file_edit_file_run_shell_are_destructive(self):
        reg = _build_registry()
        for _name in ("write_file", "edit_file", "run_shell"):
            t = reg.get(_name)
            assert t is not None, f"{_name} missing from registry"
            assert getattr(t, "destructive", None) is True, (
                f"{_name} must be destructive=True (confirm gate); got "
                f"destructive={getattr(t, 'destructive', '<MISSING>')!r}"
            )

    def test_read_file_search_code_memory_are_not_destructive(self):
        """Read-only / benign tools stay non-destructive (the gate must not
        over-block reads or the agent's own memory store)."""
        reg = _build_registry()
        for _name in ("read_file", "search_code", "memory", "search_web", "fetch_url"):
            t = reg.get(_name)
            assert t is not None
            assert getattr(t, "destructive", False) is False, (
                f"{_name} should be destructive=False; got {getattr(t,'destructive',None)!r}"
            )