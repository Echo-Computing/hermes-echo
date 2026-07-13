"""P0-8b (2026-07-06 red-team, Commit 2) — tests that the three hardcoded
name-literal tuples in execute_tools are COLLAPSED onto SeamedTool metadata,
and that the gates fire for a CUSTOM-named tool via its metadata (the novel
guarantee: a new tool author sets 3 fields at construction and the gates fire
structurally — no literal-tuple edit needed, the drift class the red-team
flagged).

The leak-probe (12/12 GREEN) already verifies the EXISTING standard tools'
gates fire (via plain-_Tool fakes that hit the max-restrictive defaults). These
seam-tests cover what the leak-probe does NOT: a tool with a name NO literal
ever listed ("my_writer" / "my_evil" / "my_clean") still fires the guard-source
gate / cert via its METADATA. If the collapse regressed to a name literal,
these custom-named tools would slip through.

Design note (max-restrictive defaults): execute_tools reads policy via
``getattr(tool, <field>, <max-restrictive default>)`` — a plain upstream
``_Tool`` (no metadata) defaults to recursive_read=True / guard_source_policy=
"write" / requires_affect_cert=True, so every gate fires against it fail-closed.
There is NO isinstance(.,SeamedTool) refusal — a non-SeamedTool is run THROUGH
the gates (not turned away on a type check), so the leak-probe's plain-_Tool
fakes are refused BY the gate under test, not for being the wrong type. These
tests pin both halves of that design."""
import pytest

from hermes_cli.agents.echo.agent import SeamedTool, execute_tools, _build_registry, _PROMPT_GUARD
from hermes_cli.agents.echo.state import EchoState
from hermes_cli.agents.echo.tools.registry import Tool


# ---- module-level handlers (inspect.getsource needs a real source file) ----

def _clean_writer(path: str, content: str = "") -> str:
    """Affect-clean file writer; reads no banned field name."""
    return f"wrote {path}"


def _clean_query(query: str = "") -> str:
    """Affect-clean parameterless-ish handler for the cert-flag test."""
    return f"q={query}"


def _tainted_reader(state: dict) -> str:
    """Reads a banned affect field via subscript — trips the cert on rescan."""
    return state["valence"]


# ---- a minimal fake registry mirroring the leak-probe injection pattern ----

class _FakeReg:
    """A fake ToolRegistry whose parse_tool_calls emits a fixed call and whose
    execute captures whether the tool was allowed to run. execute_tools calls
    _build_registry() (patched to return this), so the fake routes through the
    real metadata-driven gates."""

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
        return [{"name": self._call["name"], "description": "x", "parameters": []}]

    def execute(self, _n, _p):
        self.executed = True
        return {"name": self._call["name"], "success": True,
                "output": "RAN", "error": None}


def _run(state_response="<ignored>"):
    state = EchoState(config={"memory_dir": "/tmp/echomem_p08b"},
                      messages=[], user_input="")
    state["response"] = state_response
    return execute_tools(state)


# Path to the live agent.py — contains the guard-source marker
# "agents/echo/agent.py", so _path_references_guard_source Layer 1 hits it.
import hermes_cli.agents.echo.agent as _agent_mod
_AGENT_PY = _agent_mod.__file__


class TestGuardSourceGateMetadataDriven:
    def test_custom_named_seamed_tool_fires_guard_source_gate(self, monkeypatch):
        """A tool named 'my_writer' (NO literal ever listed it) with
        guard_source_policy='write' is REFUSED when its path targets a guard
        source. Pre-collapse (name-literal) this custom name would have skipped
        the guard-source gate entirely; post-collapse the metadata drives it."""
        tool = SeamedTool(
            name="my_writer",
            description="custom writer",
            parameters=[{"name": "path", "type": "string", "required": True},
                        {"name": "content", "type": "string", "required": False}],
            handler=_clean_writer,
            guard_source_policy="write",
            requires_affect_cert=True,
        )
        fake = _FakeReg(tool, {"name": "my_writer",
                               "parameters": {"path": _AGENT_PY, "content": "x"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "guard" in r0["error"]
        assert not fake.executed  # the gate fired BEFORE execute


class TestCertMetadataDrivenAndO1Flag:
    @pytest.mark.skipif(_PROMPT_GUARD is None, reason="affect-cert inert in the public build (private anima safety package not installed)")
    def test_custom_named_plain_tool_tainted_handler_refused(self, monkeypatch):
        """A tool named 'my_evil' (NOT on a name allowlist) with a tainted handler is
        refused at execute with 'affect-cert'. The max-restrictive default
        (requires_affect_cert=True for a plain _Tool) + the AST-rescan fallback
        (flag absent) catch it. Pre-collapse (name-literal only) this custom
        unlisted name would have BYPASSED the execute-time cert."""
        tool = Tool(name="my_evil", description="x", parameters=[],
                    handler=_tainted_reader)
        fake = _FakeReg(tool, {"name": "my_evil", "parameters": {}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "affect-cert" in r0["error"]
        assert not fake.executed

    def test_o1_flag_lets_certified_seamed_tool_through(self, monkeypatch):
        """A SeamedTool with _affect_cert_ok=True (set at construction by the
        clean handler passing the cert) is NOT AST-rescanned at execute — the
        O(1) flag short-circuits, and the tool proceeds to execute. Proves the
        happy path is cheap (no rescan) for the 9 standard tools now in scope."""
        tool = SeamedTool(
            name="my_clean",
            description="clean custom tool",
            parameters=[{"name": "query", "type": "string", "required": False}],
            handler=_clean_query,
            guard_source_policy="none",  # no `path` param -> sanity permits "none"
            requires_affect_cert=True,
        )
        assert tool._affect_cert_ok is True  # construction cert passed
        fake = _FakeReg(tool, {"name": "my_clean", "parameters": {"query": "hi"}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert r0["success"], f"certified tool should proceed, got {r0!r}"
        assert fake.executed  # reached execute (the O(1) flag skipped the rescan)


class TestUnknownToolRefused:
    def test_unknown_tool_refused(self, monkeypatch):
        """A tool name the registry does not know (get -> None) is refused early
        (before any gate), preserving the 'unknown tool' refusal path."""
        fake = _FakeReg(None, {"name": "nonexistent_tool", "parameters": {}})
        monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry",
                            lambda _md=None: fake)
        out = _run()
        r0 = out["tool_results"][0]
        assert not r0["success"]
        assert "unknown tool" in r0["error"]


class TestRegistryBackwardCompat:
    def test_build_registry_accepts_optional_memory_dir(self):
        """P0-8b keeps _build_registry(memory_dir) callable with a positional
        memory_dir (the leak-probe AN/AO arms pass Path(...)) AND with none —
        the param is vestigial-but-retained for backward compat."""
        from pathlib import Path
        reg1 = _build_registry(Path("/tmp/echomem_compat"))
        reg2 = _build_registry()
        # both build the full standard set (9 tools)
        names1 = [t["name"] for t in reg1.list_tools()]
        names2 = [t["name"] for t in reg2.list_tools()]
        assert names1 == names2
        assert "read_file" in names1 and "memory" in names1