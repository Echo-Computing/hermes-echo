"""P0-8a (2026-07-06 red-team, Commit 1) — tests for the SeamedTool dataclass
subclass (deployed live at hermes_cli/agents/echo/agent.py).

SeamedTool carries three axis-D policy metadata fields and a construction-time
affect-cert fired in __post_init__. This is the registration invariant — the
floor; path gates are the ceiling. These tests pin the two REFUSAL
invariants that make the firewall survive 20+ extension modules (per-author
discipline converted to a structural gate):

  (a) a requires_affect_cert=True tool whose handler reads a banned affect
      field name (valence/arousal/bb_*/pe) is REFUSED at construction — the
      cert is universal now, not just a name prefix.
  (b) a tool taking a `path` parameter with guard_source_policy="none" is
      REFUSED at construction — a path-taking tool MUST be covered by the
      guard-source gate (stricter than today's hand-maintained literal).

Commit 2 (P0-8b) adds the execute_tools isinstance-backstop + collapses the
three name-literal tuples onto these fields; its tests
(non-SeamedTool-refused-at-execute, gate-fires-via-metadata, _get_registry
memoized) land with that commit. Commit 3 (P0-8c) adds the empty-extensions
no-op test. This file covers only what Commit 1 makes structural."""
import pytest

from hermes_cli.agents.echo.agent import SeamedTool, _PROMPT_GUARD


# ---- module-level handlers (inspect.getsource needs a real source file) ----

def _clean_reader(path: str) -> str:
    """Affect-clean: returns its arg, reads no banned field name."""
    return path


def _affect_reading_handler(state: dict) -> str:
    """Reads a banned affect field via subscript — must trip the cert."""
    return state["valence"]


class TestSeamedToolAffectCert:
    def test_clean_handler_certified_ok(self):
        t = SeamedTool(
            name="t_clean",
            description="clean handler",
            parameters=[{"name": "path", "type": "string", "required": True}],
            handler=_clean_reader,
            guard_source_policy="read",
            requires_affect_cert=True,
        )
        assert t._affect_cert_ok is True

    @pytest.mark.skipif(_PROMPT_GUARD is None, reason="affect-cert inert in the public build (private anima safety package not installed)")
    def test_affect_reading_handler_refused_at_construction(self):
        """requires_affect_cert=True + a handler that reads state['valence']
        MUST raise at construction (fail-closed). This is the universal cert —
        not gated on a name prefix."""
        with pytest.raises(RuntimeError):
            SeamedTool(
                name="t_bad_affect",
                description="reads affect",
                parameters=[{"name": "state", "type": "object", "required": True}],
                handler=_affect_reading_handler,
                guard_source_policy="none",  # no `path` param -> sanity passes
                requires_affect_cert=True,
            )

    @pytest.mark.skipif(_PROMPT_GUARD is None, reason="affect-cert inert in the public build (private anima safety package not installed)")
    def test_affect_cert_universal_not_name_prefixed(self):
        """The red-team's headline: the cert must fire for ANY tool with
        requires_affect_cert=True, not only a prefixed name. A custom-named tool
        that reads affect is still refused."""
        with pytest.raises(RuntimeError):
            SeamedTool(
                name="summarize_state",  # custom name — old gate would miss it
                description="no name prefix",
                parameters=[{"name": "s", "type": "object", "required": True}],
                handler=_affect_reading_handler,
                guard_source_policy="none",
                requires_affect_cert=True,
            )

    def test_handler_none_skips_cert(self):
        """handler=None (the memory special-case) skips the cert — no callable
        to scan. Must NOT raise; _affect_cert_ok stays False (not True)."""
        t = SeamedTool(
            name="t_none_handler",
            description="no handler",
            parameters=[],
            handler=None,
            requires_affect_cert=True,
        )
        assert t._affect_cert_ok is False  # skipped, not certified

    def test_cert_off_by_default_for_non_cert_tool(self):
        """requires_affect_cert defaults False -> cert skipped -> _affect_cert_ok
        stays False even if a handler were affect-reading (the cert is opt-in
        via the field; Commit 2's execute_tools backstop refuses non-SeamedTool
        and the metadata drives the gate, so an opt-out tool is still gated by
        guard_source_policy / recursive_read)."""
        t = SeamedTool(
            name="t_no_cert",
            description="cert off",
            parameters=[{"name": "path", "type": "string", "required": True}],
            handler=_clean_reader,
            guard_source_policy="read",
            requires_affect_cert=False,
        )
        assert t._affect_cert_ok is False


class TestSeamedToolPathParamSanity:
    def test_path_param_with_none_guard_refused(self):
        """A tool taking a `path` parameter with guard_source_policy='none' is
        REFUSED at construction — a path-taking tool must be covered by the
        guard-source gate (anti-tampering). Stricter than the old hand-
        maintained 4-tuple: the invariant is now structural."""
        with pytest.raises(RuntimeError):
            SeamedTool(
                name="t_path_none",
                description="path but no guard",
                parameters=[{"name": "path", "type": "string", "required": True}],
                handler=_clean_reader,
                guard_source_policy="none",
                requires_affect_cert=False,
            )

    def test_path_param_with_read_guard_ok(self):
        t = SeamedTool(
            name="t_path_read",
            description="path with read guard",
            parameters=[{"name": "path", "type": "string", "required": True}],
            handler=_clean_reader,
            guard_source_policy="read",
            requires_affect_cert=True,
        )
        assert t._affect_cert_ok is True
        assert t.guard_source_policy == "read"

    def test_path_param_with_write_guard_ok(self):
        t = SeamedTool(
            name="t_path_write",
            description="path with write guard",
            parameters=[{"name": "path", "type": "string", "required": True}],
            handler=_clean_reader,
            guard_source_policy="write",
            requires_affect_cert=True,
        )
        assert t._affect_cert_ok is True
        assert t.guard_source_policy == "write"

    def test_no_path_param_with_none_guard_ok(self):
        """A tool WITHOUT a `path` param is permitted to keep
        guard_source_policy='none' (e.g. run_shell, search_web). The sanity
        rule only couples `path`-taking tools to a non-none guard."""
        t = SeamedTool(
            name="t_no_path_none",
            description="no path, none guard",
            parameters=[{"name": "query", "type": "string", "required": True}],
            handler=_clean_reader,
            guard_source_policy="none",
            requires_affect_cert=True,
        )
        assert t._affect_cert_ok is True

    def test_guard_policy_without_path_param_refused(self):
        """The DUAL direction of the path-param sanity (red-team 4-lens B1,
        2026-07-06): guard_source_policy != 'none' REQUIRES a declared 'path'
        param. The guard-source gate inspects only params.get('path',''), so a
        tool that declares a guard policy but names its path param 'target'
        would have the gate fire on an always-empty string -> never match -> the
        LLM reads the axis-D guard source via the misnamed param. 'path' is the
        contract name for the path parameter; the metadata-driven gate only
        delivers its generality promise if this direction is enforced too."""
        with pytest.raises(RuntimeError, match="has no 'path' parameter"):
            SeamedTool(
                name="t_guard_no_path",
                description="guard but misnamed path param",
                parameters=[{"name": "target", "type": "string", "required": True}],
                handler=_clean_reader,
                guard_source_policy="read",
                requires_affect_cert=True,
            )


class TestSeamedToolMetadataDefaults:
    def test_defaults(self):
        t = SeamedTool(
            name="t_defaults",
            description="defaults",
            parameters=[],
            handler=_clean_reader,
        )
        assert t.requires_affect_cert is False
        assert t.recursive_read is False
        assert t.guard_source_policy == "none"
        assert t._affect_cert_ok is False  # cert off by default

    def test_seamed_tool_is_a_tool(self):
        """SeamedTool subclasses the upstream Tool so it registers in the
        upstream ToolRegistry unchanged (zero new seam files for P0-8a)."""
        from hermes_cli.agents.echo.tools.registry import Tool
        assert isinstance(SeamedTool(
            name="t_isa", description="d", parameters=[], handler=_clean_reader,
        ), Tool)