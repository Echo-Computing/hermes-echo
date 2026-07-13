"""P0-6 (2026-07-06 red-team, audit #10/L25) — universal affect-cert
enforcement: the _register_tool registration chokepoint + the import-time
_verify_registry_affect_cert attestation.

L25 finding (MEDIUM, extensibility): assert_function_reads_no_affect was applied
ONLY to a prefixed subset of tools, not every tool. P0-8a delivered a `requires_affect_cert`
field + a `__post_init__` construction cert (set True on all 9 handler-bearing
standard tools), but the SILENT OPT-OUT stayed open: a SeamedTool with
requires_affect_cert=False + a real (non-None) handler skips the cert, leaves
_affect_cert_ok=False, and execute_tools' cert gate (`if _req_cert:`) is skipped
entirely -> the handler runs uncertified. No standard tool does this today
(memory is the only False, and it is handler=None), but the surface exists.

P0-6 closes that surface at REGISTRATION: _register_tool refuses any SeamedTool
with a non-None handler whose _affect_cert_ok is False (the silent opt-out).
handler=None is the ONE sanctioned opt-out (memory). The check is isinstance-
gated on SeamedTool, so a plain upstream _Tool (e.g. a leak-probe fake) passes
through UNCHECKED here and is caught by execute_tools' MAX-RESTRICTIVE getattr
defaults (stricter than an isinstance refusal, per agent.py:1138-1144). An
import-time attestation re-fires the invariant over the live registry so a
hand-edited silent opt-out fails at IMPORT (P0-9c doctrine), not first turn.

HONEST SCOPE: P0-6 closes the "prefixed-only" half of L25, NOT the bare-string
sqlite3.connect path / bare-float dump-signature bypass half — that needs the
execution-sandbox ceiling P0-8 did NOT deliver (documented, not closed).
The cert stays an AST FLOOR.

These tests pin: (a) the chokepoint refuses the silent opt-out; (b) a certified
handler passes; (c) handler=None is exempt; (d) a plain _Tool passes through
unchecked (leak-probe compat); (e) the attestation passes the real registry;
(f) the attestation catches a silent opt-out in a fake registry; (g) the
attestation exempts handler=None; (h) memory is the one sanctioned opt-out in
the real registry; (i) a tainted handler is still refused at CONSTRUCTION (the
existing floor P0-6 does not weaken).
"""
import pytest

from hermes_cli.agents.echo.agent import (
    SeamedTool,
    _PROMPT_GUARD,
    _build_registry,
    _register_tool,
    _verify_registry_affect_cert,
)
from hermes_cli.agents.echo.tools.registry import Tool


# ---- module-level handlers (inspect.getsource needs a real source file) ----

def _clean_handler(x: str = "") -> str:
    """Affect-clean handler; reads no banned affect field name."""
    return "ok"


def _tainted_handler(state: dict) -> str:
    """Reads a banned affect field via subscript — trips the construction cert."""
    return state["valence"]


# ---- a minimal fake registry (register + list_tools + get) ----

class _Reg:
    """Fake ToolRegistry: records register() calls; serves list_tools/get for
    the attestation tests."""

    def __init__(self, tools=None):
        self.registered = []
        self._tools = tools or {}

    def register(self, tool):
        self.registered.append(tool)

    def list_tools(self):
        return [{"name": n, "description": "x", "parameters": []} for n in self._tools]

    def get(self, name):
        return self._tools.get(name)


def _no_path_params():
    return [{"name": "x", "type": "string", "required": False}]


class TestRegisterToolChokepoint:
    def test_a_chokepoint_refuses_silent_opt_out(self):
        """A handler-bearing SeamedTool with requires_affect_cert=False (cert
        skipped -> _affect_cert_ok False) is REFUSED at registration — the
        silent opt-out the L25 finding named is closed."""
        t = SeamedTool(
            name="t_silent",
            description="silent opt-out",
            parameters=_no_path_params(),
            handler=_clean_handler,
            guard_source_policy="none",
            requires_affect_cert=False,
        )
        assert t._affect_cert_ok is False  # cert skipped at construction
        reg = _Reg()
        with pytest.raises(RuntimeError, match="silent opt-out"):
            _register_tool(reg, t)
        assert reg.registered == []  # never reached reg.register

    def test_b_chokepoint_accepts_certified_handler(self):
        """A handler-bearing SeamedTool with requires_affect_cert=True (cert
        ran -> _affect_cert_ok True) passes the chokepoint -> reg.register
        called. (A handler-bearing tool with execution_sandbox='none' must
        carry a non-empty rationale — test_b supplies one.)"""
        t = SeamedTool(
            name="t_certified",
            description="certified",
            parameters=_no_path_params(),
            handler=_clean_handler,
            guard_source_policy="none",
            requires_affect_cert=True,
            execution_sandbox="none",
            execution_sandbox_rationale="test fixture: in-process handler",
        )
        assert t._affect_cert_ok is True
        reg = _Reg()
        _register_tool(reg, t)  # must not raise
        assert reg.registered == [t]

    def test_c_chokepoint_exempts_handler_none(self):
        """A SeamedTool with handler=None + requires_affect_cert=False (the
        memory pattern) is NOT refused — handler=None is the ONE sanctioned
        opt-out (no callable to scan)."""
        t = SeamedTool(
            name="t_no_handler",
            description="handler none",
            parameters=_no_path_params(),
            handler=None,
            guard_source_policy="none",
            requires_affect_cert=False,
        )
        assert t._affect_cert_ok is False  # cert skipped (handler None)
        reg = _Reg()
        _register_tool(reg, t)  # must not raise
        assert reg.registered == [t]

    def test_d_chokepoint_passes_plain_tool_unchecked(self):
        """A plain upstream _Tool (non-SeamedTool) with a handler passes through
        the chokepoint UNCHECKED (isinstance-gated) -> reg.register called. This
        is the leak-probe compat guarantee: plain-_Tool fakes are NOT refused
        here; they route through execute_tools' MAX-RESTRICTIVE getattr defaults
        (requires_affect_cert defaults True for a plain _Tool) — stricter than an
        isinstance refusal (agent.py:1138-1144). The chokepoint is registration-
        only and must never become an execute-time isinstance refusal."""
        plain = Tool(
            name="plain_tool",
            description="plain upstream tool",
            parameters=[],
            handler=_clean_handler,
        )
        assert not isinstance(plain, SeamedTool)
        reg = _Reg()
        _register_tool(reg, plain)  # must not raise
        assert reg.registered == [plain]


class TestVerifyRegistryAffectCertAttestation:
    def test_e_attestation_passes_real_registry(self):
        """The import-time attestation over the REAL registry does not raise —
        every handler-bearing standard tool is certified (_affect_cert_ok True)
        and memory is handler=None (exempt)."""
        _verify_registry_affect_cert(_build_registry())  # must not raise

    def test_f_attestation_catches_silent_opt_out(self):
        """A registry containing a handler-bearing tool with _affect_cert_ok
        False (the silent opt-out) -> attestation raises (belt-and-suspenders:
        the chokepoint would have refused it at register; this catches a tool
        that slipped in by another path)."""
        silent = SeamedTool(
            name="t_silent_attest",
            description="silent",
            parameters=_no_path_params(),
            handler=_clean_handler,
            guard_source_policy="none",
            requires_affect_cert=False,
        )
        assert silent._affect_cert_ok is False
        reg = _Reg(tools={"t_silent_attest": silent})
        with pytest.raises(RuntimeError, match="silent affect-cert opt-out"):
            _verify_registry_affect_cert(reg)

    def test_g_attestation_exempts_handler_none(self):
        """A registry containing a handler=None tool (memory pattern) does NOT
        raise — handler=None is exempt from the attestation too."""
        no_handler = SeamedTool(
            name="t_no_handler_attest",
            description="handler none",
            parameters=_no_path_params(),
            handler=None,
            guard_source_policy="none",
            requires_affect_cert=False,
        )
        reg = _Reg(tools={"t_no_handler_attest": no_handler})
        _verify_registry_affect_cert(reg)  # must not raise

    def test_h_memory_is_the_one_sanctioned_opt_out(self):
        """Pin memory's load-bearing opt-out in the real registry: handler is
        None AND requires_affect_cert is False. A future change flipping either
        would either brick memory at execute time (True + handler=None -> F8
        refuse) or brick registration (False + a real handler -> chokepoint
        refuse). This test is the change-detector for that contract."""
        mem = _build_registry().get("memory")
        assert mem is not None, "memory tool missing from registry"
        assert mem.handler is None
        assert mem.requires_affect_cert is False


class TestConstructionFloorUntouched:
    @pytest.mark.skipif(_PROMPT_GUARD is None, reason="affect-cert inert in the public build (private anima safety package not installed)")
    def test_i_tainted_handler_still_refused_at_construction(self):
        """P0-6 does NOT weaken the existing construction-time floor: a
        SeamedTool with requires_affect_cert=True whose handler reads a banned
        affect field (state['valence']) is REFUSED at __post_init__ (the cert
        raises before the tool is ever built, never mind registered)."""
        with pytest.raises(RuntimeError, match="affect"):
            SeamedTool(
                name="t_tainted",
                description="tainted handler",
                parameters=_no_path_params(),
                handler=_tainted_handler,
                guard_source_policy="none",
                requires_affect_cert=True,
            )