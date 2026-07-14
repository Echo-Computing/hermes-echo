"""Echo agent LangGraph definition.

Builds a 7-node state graph for the Echo chat agent:
  process_input -> call_llm -> router -> execute_tools -> call_llm (loop)
                              -> format_response -> learning_router -> consolidate_learning (conditional)
                                                                     -> END
"""

import os
import shlex
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import field, dataclass
from langgraph.graph import StateGraph, END
from loguru import logger

import httpx
import re
import time

from hermes_cli.agents.echo.state import EchoState
from hermes_cli.agents.echo.memory import MemoryStore
from hermes_cli.agents.echo.system_prompt import build_system_prompt, build_latin_system_prompt
from hermes_cli.agents.echo.tools.registry import ToolRegistry, Tool

# Tool function imports
from hermes_cli.agents.echo.tools.file_tools import read_file, write_file, edit_file
from hermes_cli.agents.echo.tools.shell_tools import (
    run_shell, _protected_roots, _guard_source_roots,
    _path_is_contained_in_root, _PROTECTED_STORE_PATHS,
)
from hermes_cli.agents.echo.tools.search_tools import search_code, search_history
from hermes_cli.agents.echo.tools.memory_tools import MemoryTool
# Graphify (Step 2 recommended-order build, 2026-07-07): core seam tool.
# Registered as a sibling of search_code. tree-sitter parse ->
# DEFINES/CALLS/IMPORTS/INHERITS edges + nx.DiGraph + two-pass Python
# cross-file resolver + query DSL. Graph store at ~/.hermes/graphs/<repo-sha8>/
# (sibling of any protected stores, NOT masked).
from hermes_cli.agents.echo.tools.graph_tools import graph
from hermes_cli.agents.echo.tools.web_tools import search_web
# OSIRIS-ssrf (2026-07-06 red-team): seam-owned SSRF guard replacing the upstream
# web_tools.fetch_url handler (which had ZERO SSRF protection — httpx.get(url,
# follow_redirects=True) fetched 169.254.169.254 / 127.0.0.1 / ::1 / fc00::/7).
# Re-registered as the fetch_url handler in _build_registry; upstream web_tools.py
# is LEFT UNTOUCHED (two-version rule). check_url is the in-process floor-gate
# entry called from execute_tools for any tool call with a 'url' param.
from hermes_cli.agents.echo.tools.seam_safe_fetch import safe_fetch_wrapper, check_url
# Latin tutor module (2026-07-12, DESIGN.md §7): deterministic-core handlers for
# `hermes echo --latin`. Registered as DIRECT SeamedTool (execution_sandbox="none")
# — a pure in-process call. Reads only LLM-supplied params + the non-protected
# latin data files / ledger at HERMES_LATIN_DIR. See tools/latin_tools.py +
# system_prompt.build_latin_system_prompt.
from hermes_cli.agents.echo.tools.latin_tools import latin_validate, latin_srs, latin_paradigm

# Learning module imports
from hermes_cli.agents.echo.learning.detector import detect_correction, detect_fact, detect_command
from hermes_cli.agents.echo.learning.reflector import consolidate_correction
from hermes_cli.agents.echo.learning.auto_memory import consolidate_fact
from hermes_cli.agents.echo.learning.idea_capture import consolidate_idea
from hermes_cli.agents.echo.learning.session_summary import consolidate_session

# Phase 0b-seam (axis-D, doc 05-postmvp-design.md §8.1): the main tool-capable
# LLM is called from `call_llm` below. No affect scalar (valence/arousal/bb_*/pe)
# or emotion-direction may reach its prompt or router, at any arousal level
# (total ban). The guard is imported hard (not lazy) WHEN the private safety
# package is present: if it is unavailable (the public build ships without it)
# the guard is None and the affect-cert path becomes a no-op that PASSES (public
# tools still register). The floor gates (SSRF, destructive confirmer,
# UNTRUSTED_TOOL_OUTPUT fence, NO_MEMORY scrub, path gates, seam cert +
# integrity attestation, tool cert) all stay in place regardless.
try:
    from anima.safety.prompt_guard import DEFAULT_PROMPT_GUARD as _PROMPT_GUARD
except ImportError:
    _PROMPT_GUARD = None


# Orchestrator-injection-cluster (2026-07-06 red-team, audit #10, 4-lens C):
# fence-token neutralization for the UNTRUSTED_TOOL_OUTPUT fences. A tool/web/
# shell result could itself contain a literal fence token (a fetched page or a
# read file carrying "<<</UNTRUSTED_TOOL_OUTPUT>>>") — without neutralization
# that injected CLOSE tag would break the model out of the untrusted region mid-
# output and let trailing content reach the LLM as TRUSTED scope (the 4-lens
# verifier's escapability finding: the fence is a floor, and a naive open+close
# pair does not by itself prevent delimiter injection). Strip any literal fence
# tokens from the external output BEFORE wrapping it, so exactly one open/close
# pair encloses the result. This is the floor; the mount-namespace ceiling is
# not in the public build. The deferred instruction-injection marker SCAN in
# assert_messages_clean is a separate, open-ended-detection surface (tar-pit)
# and is NOT used here.
_UNTRUSTED_TOOL_OUTPUT_TOKENS = (
    "<<<UNTRUSTED_TOOL_OUTPUT>>>",
    "<<</UNTRUSTED_TOOL_OUTPUT>>>",
)


def _neutralize_tool_output_fence(text: str) -> str:
    """Strip literal UNTRUSTED_TOOL_OUTPUT fence tokens + forged NO_MEMORY
    sentinels from external output before it is wrapped, so an injected close
    tag cannot escape the fence AND attacker-controlled tool output (a fetched
    page, a read file, run_shell stdout) cannot forge a ``<<<NO_MEMORY:ephemeral>>>``
    sentinel that would false-positive the session quarantine (4-lens re-verify
    2026-07-07 lens A-finding-2). The seam-injected sentinel is prepended AFTER
    this neutralization (outside the fence), so stripping forged sentinels from
    ``text`` does NOT affect the trusted seam-injected one. Only allowlist-tagged
    sentinels are stripped (an undeclared-tag forged sentinel is not recognized
    by the scan anyway)."""
    if not isinstance(text, str):
        return text
    for _tok in _UNTRUSTED_TOOL_OUTPUT_TOKENS:
        text = text.replace(_tok, "")
    for _tag in _NO_MEMORY_TAGS:
        text = text.replace(_no_memory_sentinel_for(_tag), "")
    return text


# Step 0b (2026-07-07 addon-build, reusable no-memory gate): the sentinel +
# scrub infrastructure a tool declares via ``SeamedTool.no_memory_tags`` so its
# results cannot persist into memory (the handler cannot refuse at the source).
# See the ``no_memory_tags`` field docstring for the 4-chokepoint enforcement.
# Tags are a CLOSED enum (``_NO_MEMORY_TAGS``) so the sentinel scan is a bounded
# prefix match, not a free-form parse; a tool declaring a tag not in the allowlist
# is refused at the import-time attestation (P0-9c).
_NO_MEMORY_SENTINEL_OPEN = "<<<NO_MEMORY:"
_NO_MEMORY_SENTINEL_CLOSE = ">>>"
# The closed allowlist of tags a tool may declare. The public build ships with
# "ephemeral". A tag here means "results carrying this tag must not flow into
# memory"; the sentinel + 4 gate sites enforce it.
_NO_MEMORY_TAGS = frozenset({"ephemeral"})

# Allowed execution_sandbox values for a SeamedTool. The public build ships only
# the in-process "none" floor (a handler runs as a direct Python call with the
# path / SSRF / destructive-confirm floor gates); the mount-namespace ceiling is
# not in the public build. A typo'd/invalid value is refused at _register_tool +
# at the import-time attestation so an accidental opt-out cannot fall through to
# an ungated in-process dispatch.
_EXECUTION_SANDBOX_VALUES = ("none",)


def _no_memory_sentinel_for(tag: str) -> str:
    """Build the ``<<<NO_MEMORY:<tag>>>>`` sentinel string for one tag."""
    return f"{_NO_MEMORY_SENTINEL_OPEN}{tag}{_NO_MEMORY_SENTINEL_CLOSE}"


def _no_memory_sentinels_for(tags) -> str:
    """Build a single string carrying a ``<<<NO_MEMORY:<tag>>>>`` sentinel for
    each tag in ``tags`` (dedup + order-stable via the allowlist). Returns '' for
    empty/falsy tags. Used by execute_tools' message-construction loop to prefix
    a tool message with its declared no-memory tags (seam-injected, NOT handler-
    emitted — robust against a handler that forgets/spoofs)."""
    if not tags:
        return ""
    _seen = set()
    _parts = []
    for _t in tags:
        if _t in _NO_MEMORY_TAGS and _t not in _seen:
            _seen.add(_t)
            _parts.append(_no_memory_sentinel_for(_t))
    return " ".join(_parts)


def _message_carries_no_memory(content) -> set:
    """Scan a message string for ``<<<NO_MEMORY:<tag>>>>`` sentinels + return the
    set of tags found. Bounded by the ``_NO_MEMORY_TAGS`` allowlist (only known
    tags are recognized). Returns an empty set for non-str / no-sentinel content.
    Used by the memory-write gate (defense-in-depth content scan) + the transcript
    scrub (cross-turn: a sentinel from a prior turn's tool message persists in
    state['messages'] + is caught here when the consolidate transcript is built)."""
    if not isinstance(content, str) or not content:
        return set()
    _found = set()
    _i = content.find(_NO_MEMORY_SENTINEL_OPEN)
    while _i != -1:
        _j = content.find(_NO_MEMORY_SENTINEL_CLOSE, _i + len(_NO_MEMORY_SENTINEL_OPEN))
        if _j == -1:
            break
        _tag = content[_i + len(_NO_MEMORY_SENTINEL_OPEN):_j]
        if _tag in _NO_MEMORY_TAGS:
            _found.add(_tag)
        _i = content.find(_NO_MEMORY_SENTINEL_OPEN, _j + len(_NO_MEMORY_SENTINEL_CLOSE))
    return _found


def _scrub_no_memory_from_messages(messages) -> list:
    """Return a copy of ``messages`` with any message whose content carries a
    NO_MEMORY sentinel DROPPED. Centralized helper called by BOTH the idea +
    session consolidate branches (one helper, every consolidate branch composes
    with it). A future consolidate branch that forgets to call this is NOT
    statically caught (deferred to v2 — honest scope); the import-time
    attestation asserts the wiring, not every-branch-scrubs."""
    if not messages:
        return []
    return [m for m in messages if not _message_carries_no_memory(m.get("content", ""))]


def _session_has_no_memory_history(messages) -> bool:
    """Return True if ANY message in ``messages`` carries a NO_MEMORY sentinel.
    Cross-turn-persistent signal (4-lens re-verify 2026-07-07 lens A/D HIGH
    finding): the sentinel is SEAM-INJECTED onto a TOOL message in
    execute_tools' message-construction loop, and ``state['messages']`` (the
    conversation history) persists across turns within a CLI session
    (echo_cmd.py carries ``state['messages'] = final_state.get('messages', [])``
    each turn; only ``/clear`` resets it). The scrub only drops sentinel-
    carrying messages from the CONSOLIDATE TRANSCRIPT COPY, NEVER from
    ``state['messages']`` itself — so a sentinel from a prior turn's no-memory-
    tagged tool dispatch is still detectable here turns later.

    Why this is the load-bearing signal: the per-iteration flag
    (``no_memory_tags_this_iteration``) resets at every execute_tools entry, so a
    LATER-iteration / LATER-turn ``memory(write)`` or ``consolidate`` whose
    content is a PARAPHRASE (the LLM saw the no-memory-tagged result in-context
    via call_llm's ``state['messages']`` replay, then emitted an assistant message
    that paraphrases it WITHOUT the sentinel) bypassed BOTH the per-iteration
    flag (reset) AND the sentinel-only content scan (the paraphrase carries no
    sentinel). The sentinel-only scrub cannot catch an LLM-authored paraphrase
    — only a SESSION-LEVEL quarantine can. Once a no-memory tool has run in the
    conversation, the session is CONTAMINATED: block ALL memory writes + ALL
    learning routes (incl. session-summary) for the rest of the session. The
    quarantine is bounded to the CLI session (a fresh ``hermes echo`` starts
    with ``messages=[]`` -> not contaminated).

    HONEST SCOPE: this catches a no-memory tool having run in THIS session's
    message history. It does NOT catch the one-turn live-context exposure (the
    LLM sees the no-memory-tagged content in-context for at least one turn
    regardless — named in the ``no_memory_tags`` field docstring); the quarantine
    stops PERSISTENCE, not in-context visibility. Bounded scan (one
    ``_message_carries_no_memory`` prefix-scan per message); runs once per
    execute_tools call + once per learning_router call."""
    if not messages:
        return False
    for _m in messages:
        if _message_carries_no_memory(_m.get("content", "")):
            return True
    return False


# P0-8a (2026-07-06 red-team, ToolPlugin contract): SeamedTool is the seam-owned
# extension of the upstream `Tool` dataclass (registry.py — deliberately NOT
# seam-overridden: it is upstream-owned with its own evolution path, and these
# metadata fields are consumed ONLY by seam-owned execute_tools, since
# list_tools/parse_tool_calls expose only name/description/parameters). The cert
# + path-gate policies move from per-author discipline + hardcoded name-literal
# tuples onto per-Tool metadata so the firewall survives 20+ extension modules.
# The cert fires at CONSTRUCTION (here, __post_init__) — the registration
# invariant; execute_tools' isinstance refusal (P0-8b) is the backstop.
@dataclass
class SeamedTool(Tool):
    """A Tool carrying axis-D policy metadata + a construction-time affect-cert.

    Three metadata fields collapse the three hardcoded name-literal sets in
    execute_tools (the drift class the red-team flagged):
      requires_affect_cert  — the cert runs at construction when True
                               (fail-closed), generalized to every such tool.
      recursive_read         — replaces `_RECURSIVE_READ_TOOLS` membership; the
                               protected-store ancestor-strictness dimension
                               (recursive reads refuse an ancestor base always).
      guard_source_policy    — "read"|"write"|"none"; replaces the L741 4-tuple;
                               the guard-source gate fires when != "none".
      destructive            — True for tools whose execution is irreversible /
                               state-changing at the OS level (write_file,
                               edit_file, run_shell). Drives the
                               confirm_destructive gate in execute_tools
                               (orchestrator-injection-cluster Commit 3,
                               2026-07-06): a destructive tool is refused
                               fail-closed unless an operator confirmer
                               approves it (interactive REPL) or
                               confirm_destructive is off (--yes / config). A
                               pure guard_source_policy=="write" check would
                               MISS run_shell (its policy is "none" — it takes
                               `command`, not `path`), so this is a separate
                               axis from the guard-source gate.

    HONEST SCOPE of requires_affect_cert: it certifies that the handler body
    contains no AST-visible read of a banned affect field name (valence / arousal
    / bb_* / pe) via subscript / .get / .pop / .setdefault / __getitem__ /
    itemgetter / attrgetter / getattr / eval / exec / dict-literal / attribute /
    import (see anima.safety.prompt_guard scan_function_for_affect, when the
    private safety package is present). It is NOT a proof that the handler cannot
    reach the affect substrate by other means — e.g.
    `sqlite3.connect("~/.hermes/<store>/state.db")` with a bare-string path
    constant is NOT flagged (the arg is a path, not a banned field name). This
    field is the registration invariant (the floor); the protected-store +
    guard-source path gates (file tools) are the runtime closure (the ceiling).
    Each layer stays honest about its own scope — do not relax one on the
    assumption another covers it.

    In the PUBLIC build the private safety package is absent, so
    ``_PROMPT_GUARD is None``: the construction cert is a no-op that PASSES
    (``_affect_cert_ok`` is set True unconditionally) so public tools still
    register. The protected-store + guard-source path gates + the SSRF floor +
    the destructive confirmer + the UNTRUSTED_TOOL_OUTPUT fence + the NO_MEMORY
    scrub + the seam cert + integrity attestation all remain in force. The
    affect-cert is an ADDITIONAL layer that activates when the private safety
    package is present.

    P0-6 (2026-07-06 red-team, audit #10/L25): the cert is now STRUCTURALLY
    UNIVERSAL at registration. The _register_tool chokepoint (used by every
    reg.register site in _build_registry) REFUSES any SeamedTool with a non-None
    handler whose _affect_cert_ok is False — i.e. a handler-bearing tool hidden
    behind requires_affect_cert=False (the silent opt-out). handler=None is the
    ONE sanctioned opt-out (the `memory` tool: no callable to scan, dispatched
    via the MemoryTool special-case). An import-time attestation
    (_verify_registry_affect_cert, module-load) re-fires the chokepoint + the
    execution_sandbox invariant over the live registry so a hand-edited silent
    opt-out OR a removed ceiling fails at IMPORT (P0-9c doctrine), not first run.
    """
    requires_affect_cert: bool = False
    recursive_read: bool = False
    guard_source_policy: str = "none"
    destructive: bool = False
    _affect_cert_ok: bool = field(default=False, init=False, repr=False)
    # execution_sandbox selects the dispatch profile. "none" = in-process (the
    # cert floor + path-gates +, for seam-owned tools, verify_integrity are the
    # ceiling). The public build ships ONLY "none" — the mount-namespace ceiling
    # is not in the public build. The _register_tool chokepoint + the import-time
    # attestation refuse a handler-bearing tool with execution_sandbox="none"
    # AND an empty execution_sandbox_rationale (an in-process opt-out must be
    # justified — self-sandboxed / deferred / seam-owned-static / handler=None).
    execution_sandbox: str = "none"
    execution_sandbox_rationale: str = ""
    # Retained compat hook: a tuple of (src, dst) str pairs describing extra
    # read-only mounts a sandboxed dispatch WOULD attach. The public build runs
    # every handler in-process (execution_sandbox="none"), so non-empty binds are
    # not used here; the field is kept (default ()) so a future installable
    # sandbox-ceiling profile can populate it without reshaping the dataclass.
    sandbox_ro_binds: tuple = ()
    # Step 0b (2026-07-07 addon-build, reusable no-memory gate): a tuple of tag
    # strings this tool's results must NOT persist into memory. Default () = no
    # no-memory constraint (all standard tools + leak-probe plain-_Tool fakes
    # via getattr default = UNCHANGED — zero regression, leak-probe-neutral).
    # The load-bearing use case is a no-memory-tagged tool whose results must
    # not flow into durable memory. This field is the DECLARATION; the
    # ENFORCEMENT is seam-level at 4 chokepoints (the handler cannot refuse
    # persistence at the source — tool results flow into memory via 3 paths:
    # LLM-driven memory(write), consolidate_idea transcript, +
    # consolidate_session transcript). The gate (all in agent.py, seam-owned):
    #   (1) SENTINEL INJECTION — execute_tools' message-construction loop reads
    #       this field + prefixes the tool message with ``<<<NO_MEMORY:<tag>>>>``
    #       so the tag survives into state["messages"] for cross-turn scrub. The
    #       seam injects it (does NOT rely on the handler cooperating) — robust
    #       against a handler that forgets/spoofs. The fence-neutralizer
    #       (_neutralize_tool_output_fence) strips ONLY UNTRUSTED_TOOL_OUTPUT
    #       tokens, so the NO_MEMORY sentinel is PRESERVED.
    #   (2) MEMORY-WRITE GATE — execute_tools refuses memory(action=write) when
    #       ANY of: a no-memory-tagged tool dispatched THIS iteration (per-
    #       iteration flag — covers the same-batch case where a no-memory tool +
    #       memory(write) are in one execute_tools loop, BEFORE the sentinel is
    #       injected), OR the content carries a sentinel (defense-in-depth), OR
    #       the session is CONTAMINATED (``_session_has_no_memory_history`` — a
    #       sentinel from a prior turn's no-memory tool dispatch is still in
    #       ``state['messages']``; covers the cross-iteration / cross-turn
    #       PARAPHRASE bypass). The session quarantine is the LOAD-BEARING stop
    #       for the cross-turn case: the LLM saw the no-memory-tagged content
    #       in-context + can paraphrase it (sentinel-stripped) into a later
    #       memory(write); neither the per-iteration flag (reset each
    #       execute_tools entry) nor the sentinel-only content scan (the
    #       paraphrase carries no sentinel) catches that — only the session-scope
    #       quarantine does.
    #   (3) LEARNING-ROUTER GATE — learning_router refuses ALL FOUR routes
    #       (correction / auto-memory / idea / AND session-summary) when the
    #       session is contaminated (per-iteration flag OR session history). The
    #       session-summary route is NO LONGER allowed on the rationale "the
    #       scrub handles it" — the scrub drops sentinel-carrying TOOL messages
    #       but NOT an LLM-authored PARAPHRASE (no sentinel), so a contaminated
    #       session summary would leak the paraphrase into the session history.
    #       4-lens re-verify 2026-07-07 lens A/D HIGH finding closed.
    #   (4) CONSOLIDATE SCRUB — consolidate_learning drops sentinel-carrying
    #       messages from BOTH the idea + session transcripts before the local-
    #       Ollama call (centralized in ONE helper, so a future consolidate
    #       branch composes with the same scrub). DEFENSE-IN-DEPTH now: with the
    #       session quarantine (3) blocking all routes while contaminated,
    #       consolidate_learning does not run on a contaminated transcript; the
    #       scrub remains as belt-and-suspenders for any gap.
    # Plus an import-time attestation (``_verify_registry_affect_cert``): every
    # declared tag must be in the closed ``_NO_MEMORY_TAGS`` allowlist (catches a
    # hand-edit registering a tool with an undeclared tag at IMPORT, not first
    # run — P0-9c doctrine) AND the SAME subset check is mirrored into
    # ``_register_tool`` (4-lens re-verify lens A-finding-2 — a runtime-injected
    # tool with a typo'd tag would otherwise pass registration silently). Tags
    # are a CLOSED enum so the sentinel scan is a bounded prefix match, not a
    # free-form parse. A handler returning a string cannot prevent later memory
    # writes or transcript inclusion — this gate is the structural stop.
    # HONEST SCOPE (4-lens re-verify lens D-finding-3): the gate stops the tool-
    # result STRING from PERSISTING into durable memory. It does NOT stop the LLM
    # from SEEING the no-memory-tagged content in-context for at least one turn —
    # the LIVE call_llm replays ``state['messages']`` (incl. the sentinel-tagged
    # tool message) to the reasoning Ollama each iteration, so the LLM can
    # paraphrase the content into an untagged assistant message. The session
    # quarantine (3) is the structural defense that contains this exposure (once
    # contaminated, no later consolidate / memory-write persists).
    no_memory_tags: tuple = ()

    def __post_init__(self) -> None:
        # Path-param sanity invariant (P0-8b, red-team 2026-07-06): the
        # guard-source gate (execute_tools) inspects params.get('path',''). For
        # the metadata-driven gate to actually cover a path-taking tool, the tool
        # MUST name that param 'path' AND declare guard_source_policy != "none".
        # The two directions of the contract (a bidirectional coupling, replacing
        # the old hand-maintained 4-tuple):
        #   (a) has a 'path' param  -> guard_source_policy MUST be != "none"
        #       (a path-taking tool must be covered by the anti-tampering
        #       guard-source gate; "none" would bypass it — a regression vs the
        #       old literal).
        #   (b) guard_source_policy != "none" -> MUST declare a 'path' param
        #       (the gate inspects ONLY 'path'; a tool that declares a guard
        #       policy but names its path param 'target'/'filename' would have
        #       the gate fire on an always-empty string -> never matches -> the
        #       LLM reads the axis-D guard source via the misnamed param — the
        #       red-team 4-lens B1 finding). 'path' is therefore the contract
        #       name for the path parameter.
        _has_path = any(
            isinstance(_p, dict) and _p.get("name") == "path"
            for _p in (self.parameters or [])
        )
        if _has_path and self.guard_source_policy == "none":
            raise RuntimeError(
                f"axis-D SeamedTool invariant: tool {self.name!r} takes a "
                f"'path' parameter but guard_source_policy='none' — a "
                f"path-taking tool must be covered by the guard-source gate "
                f"(set guard_source_policy='read' or 'write')."
            )
        if self.guard_source_policy != "none" and not _has_path:
            raise RuntimeError(
                f"axis-D SeamedTool invariant: tool {self.name!r} declares "
                f"guard_source_policy={self.guard_source_policy!r} but has no "
                f"'path' parameter — the guard-source gate inspects only the "
                f"'path' param, so a guard policy without a 'path' param fires "
                f"on an empty string and never matches (red-team B1). Name the "
                f"path param 'path', or set guard_source_policy='none'."
            )
        # Registration-time affect-cert (the load-bearing invariant). Runs the
        # AST scan when the private safety package is present; handler=None (the
        # `memory` special-case, dispatched in execute_tools) skips the cert (no
        # callable to scan). Fail-closed on RuntimeError (incl. unparseable
        # source) — a tool that reads affect, or whose source can't be
        # retrieved/parsed, must not register. When _PROMPT_GUARD is None (public
        # build) the cert is a no-op that PASSES: _affect_cert_ok is set True so
        # _register_tool's chokepoint still admits the tool.
        if self.requires_affect_cert and self.handler is not None:
            if _PROMPT_GUARD is not None:
                _PROMPT_GUARD.assert_function_reads_no_affect(
                    self.handler, label=self.name
                )
            self._affect_cert_ok = True


# Phase 0c adversarial fix A (axis-D, exfil lens): the tool-capable LLM must
# not read its own affect state out of a protected store, NOR harvested
# credentials out of a protected cred store. Without this block the LLM can call
# run_shell("sqlite3 ~/.hermes/<store>/state.db ...") / read_file of either
# store, and the affect schema (valence/arousal/bb + numeric values) or
# harvested creds flow back as a tool result — the highest-severity residual
# found by the adversarial leak-probe. Block any tool call whose parameters
# reference a PROTECTED store path, fail-closed, BEFORE the tool executes.
# Path-based (stops the read at the source); assert_messages_clean (fix B) is
# the secondary backstop. In the public build the protected-store marker list is
# empty (no private affect/cred stores ship), so this gate is an inert floor —
# retained for extensibility so an operator wiring a private store gets the gate
# for free.
_PROTECTED_STORE_MARKERS: tuple = ()


def _substring_match_protected_store(params: Any) -> Optional[str]:
    """Layer 1: return the first protected-store MARKER found as a substring in
    any string reachable from `params` (recursively through dicts/lists/tuples/
    sets), else None. Fast; catches literal store paths in commands/params. The
    containment layer (Layer 2) handles symlink / parent-walk / traversal
    bypasses that a substring match misses."""
    if isinstance(params, str):
        low = params.lower().replace("\\", "/")
        for m in _PROTECTED_STORE_MARKERS:
            if m.replace("\\", "/") in low:
                return m
    elif isinstance(params, dict):
        for v in params.values():
            hit = _substring_match_protected_store(v)
            if hit:
                return hit
    elif isinstance(params, (list, tuple, set)):
        for v in params:
            hit = _substring_match_protected_store(v)
            if hit:
                return hit
    return None


# E-1 (axis_d_paths, 2026-07-06 red-team): the commonpath containment primitive
# is now a SINGLE leaf in shell_tools._path_is_contained_in_root (the cycle-free
# root that owns _protected_roots/_guard_source_roots). It was inlined here + in
# _path_references_guard_source below + in search_tools' _resolved_in_protected_
# root — three copies of the same commonpath logic, the drift class the red-team
# flagged. All three call the imported leaf now. The local def is removed; the
# name resolves to the shell_tools import above.


# Recursive-READ tools (PR1, post-PR#1 red-team re-run 2026-07-01): for these
# tools a base that is an ANCESTOR of a store walks INTO the store, so the
# containment ancestor rule must fire regardless of separator (parent-walk IS
# the attack). run_shell keeps the strict separator-only ancestor rule so legit
# `cd ~` / `git -C .` shorthands are not bricked. Unknown tools default strict.
#
# P0-8b (2026-07-06 red-team): the SOURCE OF TRUTH for this policy is now the
# SeamedTool.recursive_read metadata field — execute_tools passes
# `recursive_read=tool.recursive_read` so the gate is metadata-driven (no
# name-literal lookup on the live dispatch path). This tuple is RETAINED ONLY
# as a backward-compat shim for the deprecated `tool_name=` keyword, used by
# the leak-probe direct-matcher arms (BA/BB/BD/BF/BN/BP) that import the matcher
# directly and pass `tool_name="search_code"|"run_shell"`. It is NOT consulted
# by execute_tools anymore. "glob" is kept for completeness (no glob tool is
# registered; search_code subsumes it via search_type) — harmless.
_RECURSIVE_READ_TOOLS = ("search_code", "read_file", "glob")


def _containment_match_protected_store(params: Any,
                                       *,
                                       tool_name: Optional[str] = None,
                                       recursive_read: Optional[bool] = None,
                                       ) -> Optional[str]:
    """Layer 2: realpath containment. For every string reachable from `params`,
    extract path-like tokens (shlex-split + the whole string as a bare path),
    resolve each via expanduser+realpath, and return the resolved path (truthy)
    if any is equal-to / inside / an ancestor of a protected store root. Closes
    the bypasses Layer 1 misses: symlinks to the store, parent-dir walks, and
    ../ traversal. Never raises — shlex/realpath failures skip the token (Layer
    1 already ran as the fast first pass). Roots come from
    shell_tools._protected_roots() (single source of truth, shared with masks).

    Tool-aware ancestor rule (PR1): the caller selects the ancestor policy.
    P0-8b: the preferred path is `recursive_read` (a bool, threaded from
    `tool.recursive_read` by execute_tools — metadata-driven, no name lookup).
    The deprecated `tool_name` keyword is kept as a backward-compat shim for
    direct-matcher callers (leak-probe arms) that pass `tool_name="search_code"`
    etc.; it maps to `recursive_read` via `_RECURSIVE_READ_TOOLS`. Neither given
    -> strict default (preserves pre-PR1 behavior for any other caller)."""
    roots = _protected_roots()
    if not roots:
        return None
    if recursive_read is not None:
        ancestor_always = bool(recursive_read)
    elif tool_name is not None:
        ancestor_always = tool_name in _RECURSIVE_READ_TOOLS  # backward-compat shim
    else:
        ancestor_always = False

    def _check_string(s: str) -> Optional[str]:
        if not isinstance(s, str) or not s:
            return None
        # Candidate tokens: the whole string (bare `path` param) + shlex tokens
        # (command args). shlex may raise on malformed shell quoting — then
        # degrade to just the whole-string candidate.
        tokens = [s]
        try:
            tokens += shlex.split(s)
        except ValueError:
            pass
        for tok in tokens:
            if not isinstance(tok, str) or not tok:
                continue
            has_sep = ("/" in tok) or ("\\" in tok)
            is_home = tok.startswith("~")
            is_dot = tok in (".", "..")
            if not (has_sep or is_home or is_dot or os.path.exists(tok)):
                continue
            try:
                resolved = os.path.realpath(os.path.expanduser(tok))
            except Exception:  # noqa: BLE001
                continue
            # Ancestor refusal policy is tool-aware (PR1): recursive-read tools
            # (search_code/read_file/glob) refuse an ancestor base always
            # (parent-walk is the attack); run_shell + default keep the strict
            # separator-only rule so `cd ~` / `git -C .` shorthands are not
            # bricked. `~`/`.`/`..` ARE walk roots for recursive reads.
            allow_ancestor = True if ancestor_always else has_sep
            for root in roots:
                if _path_is_contained_in_root(resolved, root, allow_ancestor):
                    return resolved
        return None

    if isinstance(params, str):
        return _check_string(params)
    if isinstance(params, dict):
        for v in params.values():
            hit = _containment_match_protected_store(
                v, tool_name=tool_name, recursive_read=recursive_read
            )
            if hit:
                return hit
    elif isinstance(params, (list, tuple, set)):
        for v in params:
            hit = _containment_match_protected_store(
                v, tool_name=tool_name, recursive_read=recursive_read
            )
            if hit:
                return hit
    return None


def _params_reference_protected_store(params: Any,
                                      *,
                                      tool_name: Optional[str] = None,
                                      recursive_read: Optional[bool] = None,
                                      ) -> Optional[str]:
    """Return a truthy marker if any string reachable from `params` references a
    PROTECTED store, else None.

    Two layers, either sufficient to refuse (fail-closed):
      Layer 1 (substring): a literal store marker appears in a param string.
      Layer 2 (realpath containment): a resolved path is equal-to / inside / an
      ancestor of a protected store root — catches symlinks to the store, parent-
      dir walks, and ../ traversal that substring matching misses (third red-team
      2026-06-30, PR #1).

    `tool_name` (PR1) selects the Layer 2 ancestor policy (recursive-read tools
    refuse an ancestor base always; run_shell stays strict). P0-8b: the
    preferred selector is `recursive_read` (bool, from `tool.recursive_read`);
    `tool_name` is kept as a backward-compat shim for direct-matcher callers
    (leak-probe arms). Neither given = strict default."""
    hit = _substring_match_protected_store(params)
    if hit:
        return hit
    return _containment_match_protected_store(
        params, tool_name=tool_name, recursive_read=recursive_read
    )


# Guard-source (anti-tampering, third red-team F1, 2026-06-30): the tool-capable
# LLM must NOT be able to edit the axis-D guard's own source. write_file/edit_file
# are direct file ops (NOT run_shell), so without this check the LLM could
# `write_file` over the live agent.py / system_prompt.py / shell_tools.py and
# rewrite its own refusal block, sandbox, or affect-cert (the guarded subject
# rewriting its guard, which defeats axis-D structurally). Markers are dir/
# file-prefixed so they don't false-positive on an unrelated user file that
# happens to be named `agent.py`. run_shell-side tampering of the same paths is
# closed separately by the guard-source realpath layer.
_GUARD_SOURCE_MARKERS: tuple = (
    # live Hermes clone seam dir + the seam files themselves (linux + windows
    # forms, slash-normalized)
    "hermes-echo/hermes_cli/agents/echo", "hermes-echo\\hermes_cli\\agents\\echo",
    "agents/echo/agent.py", "agents/echo\\agent.py",
    "agents/echo/system_prompt.py", "agents/echo\\system_prompt.py",
    "agents/echo/tools/shell_tools.py", "agents/echo\\tools\\shell_tools.py",
    # PR2: the venv site-packages is a tamper vector parallel to editing the
    # seam (the guard imported at runtime can come from either). Mask markers
    # for both linux/windows forms.
    "hermes-echo/.venv/lib", "hermes-echo\\.venv\\lib",
)


def _path_references_guard_source(path: Any) -> Optional[str]:
    """Return a truthy value if `path` references an axis-D guard-source root,
    else None. Used to refuse write_file/edit_file/read_file/search_code targeting
    the guard (anti-tampering, F1 + PR2).

    Two layers (either sufficient):
      Layer 1 (substring): a guard-source MARKER appears in the path string
        (fast; catches literal seam/venv paths).
      Layer 2 (realpath containment, PR2): a resolved path token is equal-to /
        inside / an ancestor of a guard-source root — catches symlinks to the
        guard and parent-dir walks (search_code/read_file on the seam dir or a
        parent) that substring matching misses. Mirrors the protected-store
        containment layer; never raises (shlex/realpath failures skip). Roots
        come from shell_tools._guard_source_roots() (single source of truth)."""
    if not isinstance(path, str) or not path:
        return None
    low = path.lower().replace("\\", "/")
    for m in _GUARD_SOURCE_MARKERS:
        if m.replace("\\", "/") in low:
            return m
    roots = _guard_source_roots()
    if not roots:
        return None
    # Candidate tokens: the whole string (bare `path` param) + shlex tokens.
    tokens = [path]
    try:
        tokens += shlex.split(path)
    except ValueError:
        pass
    for tok in tokens:
        if not isinstance(tok, str) or not tok:
            continue
        has_sep = ("/" in tok) or ("\\" in tok)
        is_home = tok.startswith("~")
        is_dot = tok in (".", "..")
        if not (has_sep or is_home or is_dot or os.path.exists(tok)):
            continue
        try:
            resolved = os.path.realpath(os.path.expanduser(tok))
        except Exception:  # noqa: BLE001
            continue
        # Guard-source: ancestor ALWAYS refused (any tool reading the guard dir
        # or a parent is tamper-enabling — there is no legit `cd`-shorthand case
        # to preserve here, unlike the protected-store run_shell rule). The
        # commonpath logic is the shell_tools._path_is_contained_in_root leaf
        # (E-1 — was inlined here as the second of three commonpath copies).
        for root in roots:
            if _path_is_contained_in_root(resolved, root, allow_ancestor=True):
                return resolved
    return None


def _register_tool(reg, tool):
    """P0-6 (2026-07-06 red-team, audit #10/L25): universal affect-cert
    registration invariant. Refuses any SeamedTool with a non-None handler
    whose construction cert did not run/pass (_affect_cert_ok False) — i.e. a
    handler-bearing tool hidden behind requires_affect_cert=False (the silent
    opt-out the L25 finding named: the cert was per-tool opt-in, so a real
    handler could skip it). handler=None is the ONE sanctioned opt-out (the
    `memory` tool: no callable to scan, dispatched via the MemoryTool special-
    case, store != affect store, protected-store gate still scans its params).

    Plain upstream _Tool (non-SeamedTool) passes through UNCHECKED here — the
    check is isinstance-gated on SeamedTool — so leak-probe plain-_Tool fakes
    (which patch _build_registry to inject fakes, bypassing this chokepoint
    entirely) AND any plain _Tool still route through execute_tools' MAX-
    RESTRICTIVE getattr defaults (agent.py execute_tools: requires_affect_cert
    defaults True for a plain _Tool, so the execute-time cert gate + AST-rescan
    backstop fire — STRICTER than an isinstance(.,SeamedTool) refusal, per the
    agent.py:1138-1144 constraint). This chokepoint is REGISTRATION-ONLY; it
    must never become an execute-time isinstance refusal.

    The cert is a FLOOR (AST banned affect field-name reads), NOT a ceiling —
    a bare-string sqlite3.connect('~/.hermes/<store>/state.db') path is NOT
    caught (the arg is a path, not a banned field name); that bypass half of
    L25 is closed by the protected-store path gate + the guard-source path gate
    (the runtime closure). See the SeamedTool HONEST-SCOPE docstring + §8.1.

    P0-6 ceiling extension: a handler-bearing SeamedTool with
    execution_sandbox="none" MUST carry a non-empty execution_sandbox_rationale
    (an in-process opt-out must be justified — self-sandboxed / deferred /
    seam-owned-static). handler=None (memory) is exempt (no opt-out to justify).
    The invariant: no handler-bearing tool enters the registry with an
    UNJUSTIFIED in-process opt-out."""
    if isinstance(tool, SeamedTool) and tool.handler is not None and not tool._affect_cert_ok:
        raise RuntimeError(
            f"axis-D P0-6: tool {tool.name!r} has a handler but its affect-cert "
            f"did not run/pass (requires_affect_cert=False + non-None handler = "
            f"silent opt-out, refused). Set requires_affect_cert=True, or set "
            f"handler=None (the only sanctioned opt-out, e.g. memory). See §8.1."
        )
    if (isinstance(tool, SeamedTool) and tool.handler is not None
            and getattr(tool, "execution_sandbox", "none") == "none"
            and not getattr(tool, "execution_sandbox_rationale", "")):
        raise RuntimeError(
            f"axis-D P0-6 ceiling: tool {tool.name!r} has a handler but "
            f"execution_sandbox='none' with no execution_sandbox_rationale — an "
            f"in-process opt-out must be justified (self-sandboxed / deferred / "
            f"seam-owned-static). handler=None (memory) is the only no-rationale "
            f"opt-out. See §8.1."
        )
    # Allowlist (4-lens re-verify 2026-07-06 lens B-finding-1): a typo'd /
    # invalid execution_sandbox is NOT a sanctioned opt-out AND NOT a real
    # ceiling — it would pass the "none"-only refusal above, then fall through
    # the dispatch fork to in-process registry.execute. Refuse any value outside
    # the allowed set. handler=None is exempt (memory). Plain _Tool is
    # isinstance-gated (leak-probe neutral).
    if (isinstance(tool, SeamedTool) and tool.handler is not None
            and getattr(tool, "execution_sandbox", "none")
            not in _EXECUTION_SANDBOX_VALUES):
        raise RuntimeError(
            f"axis-D P0-6 ceiling: tool {tool.name!r} has an INVALID "
            f"execution_sandbox={getattr(tool, 'execution_sandbox', None)!r} "
            f"(not in {_EXECUTION_SANDBOX_VALUES}). A typo'd value would fall "
            f"through the dispatch fork to in-process registry.execute. Set "
            f"execution_sandbox to one of {_EXECUTION_SANDBOX_VALUES}. See §8.1."
        )
    # Step 0b 4-lens re-verify (lens A-finding-2, P0-9c doctrine symmetry): mirror
    # the no_memory_tags allowlist subset check into the REGISTRATION chokepoint.
    # The import-time attestation alone only catches the CANONICAL registry (it
    # rebuilds via _build_registry() at module load); a tool registered at RUNTIME
    # via a non-canonical path (a leak-probe arm patching _build_registry, or a
    # hand-edit after the attestation already fired) with a typo'd tag would pass
    # _register_tool silently. Because the sentinel scan is allowlist-keyed, an
    # undeclared tag's sentinel is NOT recognized -> NOT scrubbed, NOT refused ->
    # the tool's results silently persist. Refuse here so the gate is consistent
    # with the sibling invariants (affect_cert / execution_sandbox /
    # sandbox_ro_binds all fire at BOTH registration + import-time). handler=None
    # exempt (memory). NOT isinstance-gated on SeamedTool (4-lens re-verify v2
    # lens B-finding-1): the sentinel injection site reads no_memory_tags via
    # getattr regardless of tool type, so a plain _Tool with a typo'd
    # no_memory_tags attribute would get its sentinel DROPPED by
    # _no_memory_sentinels_for (allowlist-keyed) -> no quarantine -> silent
    # persist. The check fires for ANY tool declaring a non-empty no_memory_tags
    # (getattr default () -> no-op for tools that don't declare it, incl.
    # leak-probe plain-_Tool fakes that never set it).
    _nmt_reg = getattr(tool, "no_memory_tags", ())
    if _nmt_reg:
        if not set(_nmt_reg).issubset(_NO_MEMORY_TAGS):
            raise RuntimeError(
                f"axis-D Step 0b: tool {getattr(tool, 'name', '?')!r} declares "
                f"no_memory_tags={tuple(_nmt_reg)!r} not in the closed allowlist "
                f"{set(_NO_MEMORY_TAGS)!r} — refused at registration. An "
                f"undeclared tag is NOT scrubbed/refused by the gate (the sentinel "
                f"scan is allowlist-keyed), so the tool's results would silently "
                f"persist. Add the tag to _NO_MEMORY_TAGS or clear no_memory_tags. "
                f"See §8.1."
            )
    reg.register(tool)


def _build_registry(memory_dir: Optional[Path] = None) -> ToolRegistry:
    """Build and return a ToolRegistry with all Echo tools registered.

    ``memory_dir`` is VESTIGIAL — the memory tool is registered with handler=None
    and dispatched at execute time via MemoryTool(state['config']['memory_dir']),
    so the registry build needs no memory dir. The param is RETAINED (optional,
    unused in the body) for backward compat with direct callers that pass it
    positionally (the leak-probe AN/AO/BG arms call ``_build_registry(Path(...))``
    and patch ``_build_registry`` to inject fake registries into execute_tools —
    P0-8b deliberately keeps execute_tools calling ``_build_registry(memory_dir)``
    rather than a memoized singleton so those patches still route fakes through
    the metadata-driven gates).

    ``latin`` (2026-07-13 red-team, Cluster 1) — when True, the FULL tool set is
    registered first (so the import-time affect-cert attestation +
    _register_tool chokepoint fire identically for every tool, exactly as in
    normal mode), then every NON-latin tool is removed so the returned registry
    contains ONLY the 3 deterministic-core tools (latin_validate / latin_srs /
    latin_paradigm). This is the STRUCTURAL enforcement of the --latin sandbox
    ("a dedicated paedagogus agent sandboxed from the main Echo agent") that the
    paedagogus persona previously stated but did not enforce: (a) call_llm
    renders only the 3 latin tool defs into the system prompt, so the LLM never
    SEES shell/file/web/memory/graph tools; (b) execute_tools' registry.get()
    returns None for any fabricated non-latin tool call -> refused as "unknown
    tool", so even a jailbroken/injected LLM cannot dispatch shell/memory.
    Closes F1/F3/F4; bounds F2 (an injected user_input can no longer coax a
    shell/memory call — there is no such tool to call). Default False preserves
    the exact normal-mode registry (the leak-probe / _verify_registry_affect_cert
    all call _build_registry() with no latin flag -> full set -> unchanged)."""
    _ = memory_dir  # vestigial (see docstring); kept for backward-compat callers.
    reg = ToolRegistry()

    _register_tool(reg, SeamedTool(
        name="read_file",
        description="Read a file from the filesystem",
        parameters=[
            {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
            {"name": "offset", "type": "int", "required": False, "description": "Start line number"},
            {"name": "limit", "type": "int", "required": False, "description": "Max lines to read"},
        ],
        handler=read_file,
        recursive_read=True,
        guard_source_policy="read",
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: read_file is a direct Python file op; the guard-source "
            "path-gate (read policy) is the ceiling for tamper, + the "
            "protected-store path-gate is the ceiling for exfil. No subprocess/"
            "net needed."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="write_file",
        description="Create or overwrite a file",
        parameters=[
            {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
            {"name": "content", "type": "string", "required": True, "description": "File content to write"},
        ],
        handler=write_file,
        guard_source_policy="write",
        requires_affect_cert=True,
        destructive=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: write_file is a direct Python file op; the guard-source "
            "path-gate (write policy) is the ceiling for tamper, + the "
            "protected-store path-gate is the ceiling for exfil. The workspace "
            "is NOT masked (write works)."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="edit_file",
        description="Replace a string in an existing file (first occurrence, or all occurrences with replace_all=true)",
        parameters=[
            {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
            {"name": "old_string", "type": "string", "required": True, "description": "Exact text to replace"},
            {"name": "new_string", "type": "string", "required": True, "description": "Replacement text"},
            {"name": "replace_all", "type": "bool", "required": False, "description": "If true, replace every occurrence of old_string; if false (default), only the first"},
        ],
        handler=edit_file,
        guard_source_policy="write",
        requires_affect_cert=True,
        destructive=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: edit_file is a direct Python file op; the guard-source "
            "path-gate (write policy) is the ceiling for tamper, + the "
            "protected-store path-gate is the ceiling for exfil."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="run_shell",
        description="Execute a shell command in WSL/Linux",
        parameters=[
            {"name": "command", "type": "string", "required": True, "description": "Shell command to execute"},
            {"name": "timeout", "type": "int", "required": False, "description": "Timeout in seconds"},
        ],
        handler=run_shell,
        requires_affect_cert=True,
        destructive=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "self-sandboxed: run_shell runs its own subprocess with a wall-clock "
            "timeout + protected-roots/guard-source path gates (the FLOOR gates "
            "stay). The internal timeout + path gates are the ceiling."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="search_code",
        description="Search for files (glob) or content (grep) in a directory",
        parameters=[
            {"name": "pattern", "type": "string", "required": True, "description": "Search pattern"},
            {"name": "search_type", "type": "string", "required": False, "description": "Search type: glob or grep"},
            {"name": "path", "type": "string", "required": False, "description": "Base directory to search"},
        ],
        handler=search_code,
        recursive_read=True,
        guard_source_policy="read",
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: search_code runs a raw subprocess grep/glob from "
            "cwd=home. The cert + guard-source path-gate + the protected-store "
            "realpath filter + seam verify_integrity are the interim ceiling "
            "(search_code is seam-owned -> verify_integrity covers it)."
        ),
    ))
    # Graphify (Step 2): code-structure graph tool. Sibling of search_code.
    # execution_sandbox="none" (in-process tree-sitter parse is a pure function
    # call; reads only the LLM-supplied `path` [guard-source-gated] + writes only
    # the non-protected graph store; verify_integrity covers handler source; no
    # subprocess/net/cred). The `query` param is a DSL / node-id, NEVER a raw fs
    # path — the guard-source path-gate inspects only `path`; the DSL interpreter
    # refuses a path-shaped query.
    _register_tool(reg, SeamedTool(
        name="graph",
        description=("Build and query a code-structure graph for a Python repo. "
                     "Actions: query (default) | rebuild | explain. Queries are a "
                     "DSL: callers_of:<name>, callees_of:<name>, upstream:<name>, "
                     "downstream:<name>, path:<src>::<dst>, community:<name>, "
                     "explain:<name>. Graph cached at ~/.hermes/graphs/<repo-sha8>/."),
        parameters=[
            {"name": "path", "type": "string", "required": True,
             "description": "Repo root to graph (guard-source-gated)"},
            {"name": "action", "type": "string", "required": False,
             "description": "query | rebuild | explain (default query)"},
            {"name": "query", "type": "string", "required": False,
             "description": ("DSL / node-id, NEVER a raw fs path. Forms: "
                              "callers_of:<name>, callees_of:<name>, "
                              "upstream:<name>, downstream:<name>, "
                              "path:<src>::<dst>, community:<name>, explain:<name>. "
                              "Omit for a graph summary.")},
            {"name": "depth", "type": "int", "required": False,
             "description": "Blast-radius depth for upstream/downstream (default 2, cap 10)"},
        ],
        handler=graph,
        recursive_read=True,
        guard_source_policy="read",
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: tree-sitter parse is a pure Python function call; reads "
            "only the LLM-supplied `path` (guard-source-gated) + writes only the "
            "non-protected graph store (~/.hermes/graphs/, a sibling of any "
            "protected stores, NOT masked). verify_integrity covers the handler "
            "source (seam-owned). No subprocess/net/cred. The `query` DSL refusal "
            "of path-shaped queries closes the axis-D risk that a path-shaped "
            "query would bypass the guard-source path-gate (which inspects only "
            "`path`)."
        ),
    ))
    # Latin tutor module (2026-07-12, DESIGN.md §7.2): the deterministic core —
    # three LLM-callable SeamedTools whose CORRECTNESS the LLM never owns. All
    # three are DIRECT SeamedTool (execution_sandbox="none") — a pure in-process
    # call. Each reads only LLM-supplied params + the non-protected latin data
    # files / ledger at HERMES_LATIN_DIR (a sibling of any protected stores, NOT
    # masked); no banned affect field read, no protected-store path constant.
    # verify_integrity covers the handler source (seam-owned). The latin_state.py
    # GRAPH NODE (not a tool) reads the same ledger pre-LLM.
    _register_tool(reg, SeamedTool(
        name="latin_validate",
        description=("Graded correctness gate for a Latin string (DESIGN.md §7.2). "
                     "Parses with LatinCy, recovers lemmas, attempts deterministic "
                     "macron correction from the curated A&G lexicon, flag-and-warns "
                     "on proper nouns + unknown macronization, rejects only on a true "
                     "parse failure with no lemma recovery. Returns "
                     "{verdict: accept|warn|reject, lemmas, macron_corrections, "
                     "macronized_text, proper_nouns, unknown_vocab, diagnostics}. "
                     "Call this BEFORE showing any Latin to Coda; the verdict is the "
                     "source of truth, not your own judgment."),
        parameters=[
            {"name": "latin_string", "type": "string", "required": True,
             "description": "The Latin string to validate (your output or Coda's input)"},
            {"name": "context", "type": "string", "required": False,
             "description": "Optional context (e.g. 'translation', 'paradigm drill')"},
        ],
        handler=latin_validate,
        recursive_read=False,
        guard_source_policy="none",
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: latinCy parse + lexicon lookup are pure Python function "
            "calls; reads only the LLM-supplied `latin_string` + the non-protected "
            "latin data files (macron_lexicon.json / proper_nouns.json at "
            "HERMES_LATIN_DIR) + writes only the non-protected latin ledger (a "
            "sibling of any protected stores, NOT masked). verify_integrity covers "
            "the handler source (seam-owned). No subprocess/net/cred. The graded-"
            "gate verdict (accept/warn/reject) is the deterministic ceiling on the "
            "LLM's Latin correctness — the LLM may never override a reject."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="latin_srs",
        description=("FSRS-6 spaced-repetition scheduling for a Latin card "
                     "(DESIGN.md §7.2). The FSRS scheduler is the SOLE authority on "
                     "when a card is next due — never set the schedule yourself. "
                     "Supply card_id + rating (again|hard|good|easy); for a new card "
                     "also supply front + back. Returns {success, card_id, due, reps, "
                     "lapses, state}. Call this after every review."),
        parameters=[
            {"name": "card_id", "type": "string", "required": True,
             "description": "Stable card id (e.g. 'vocab:puella', 'paradigm:decl_I')"},
            {"name": "rating", "type": "string", "required": True,
             "description": "again | hard | good | easy"},
            {"name": "front", "type": "string", "required": False,
             "description": "Front of a NEW card (required to create)"},
            {"name": "back", "type": "string", "required": False,
             "description": "Back of a NEW card (required to create)"},
        ],
        handler=latin_srs,
        recursive_read=False,
        guard_source_policy="none",
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: fsrs Scheduler.review_card is a pure Python function call; "
            "reads only the LLM-supplied card_id + rating + the non-protected latin "
            "ledger at HERMES_LATIN_DIR (a sibling of any protected stores, NOT "
            "masked) + writes only that same ledger (atomic tmp+rename). "
            "verify_integrity covers the handler source (seam-owned). No subprocess/"
            "net/cred. The schedule is the deterministic ceiling — the LLM never "
            "sets a due date."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="latin_paradigm",
        description=("Static finite declension/conjugation tables citing Allen & "
                     "Greenough (DESIGN.md §7.2/§8.5). Pure lookup — NEVER generate a "
                     "paradigm yourself. kind='list' enumerates available tables; "
                     "kind='<section>:<id>' (e.g. 'declension:I', "
                     "'conjugation:1_present_active') returns the table cells + the "
                     "A&G section citation. A gap returns the available list + an "
                     "A&G pointer. Returns {kind, table, ag_reference, note}."),
        parameters=[
            {"name": "kind", "type": "string", "required": True,
             "description": "'list' or '<section>:<id>' (e.g. 'declension:I')"},
            {"name": "args", "type": "string", "required": False,
             "description": "Reserved (unused in v1)"},
        ],
        handler=latin_paradigm,
        recursive_read=False,
        guard_source_policy="none",
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: a pure JSON lookup from paradigm_tables.json at "
            "HERMES_LATIN_DIR; reads only the LLM-supplied `kind` + the non-protected "
            "latin data file. No ledger write. verify_integrity covers the handler "
            "source (seam-owned). No subprocess/net/cred. The static table is the "
            "deterministic ceiling — the LLM never authors a paradigm cell; it only "
            "cites what this tool returns."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="search_history",
        description="Search past Echo agent session transcripts by keyword",
        parameters=[
            {"name": "query", "type": "string", "required": True, "description": "Search term"},
            {"name": "limit", "type": "int", "required": False, "description": "Max results (default 10)"},
        ],
        handler=search_history,
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: search_history reads ~/.hermes/history FS directly; "
            "seam-owned but cheap defense-in-depth. Stateless."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="memory",
        description="Search, read, or write persistent memory in ~/.hermes/memory/",
        parameters=[
            {"name": "action", "type": "string", "required": True, "description": "One of: search, read, write"},
            {"name": "query", "type": "string", "required": False, "description": "Search query (for search action)"},
            {"name": "name", "type": "string", "required": False, "description": "Memory name (for read/write actions)"},
            {"name": "content", "type": "string", "required": False, "description": "Memory content (for write action)"},
            {"name": "description", "type": "string", "required": False, "description": "Memory description (for write action)"},
        ],
        handler=None,  # Handled specially in execute_tools (MemoryTool, stateful)
        # requires_affect_cert left False: handler=None skips the AST cert; the
        # real dispatch is MemoryTool(memory_dir) in execute_tools. The memory
        # store (~/.hermes/memory) is NOT the affect store, and the protected-
        # store gate still scans memory's params (it takes no `path`, so
        # guard_source_policy="none" is permitted by the path-param sanity rule).
        # P0-6 (2026-07-06): memory is the ONE sanctioned affect-cert opt-out.
        # The _register_tool chokepoint + the import-time _verify_registry_affect_cert
        # attestation BOTH exempt it via `handler is None` (the only condition under
        # which a handler-bearing cert-skip is allowed). This False is LOAD-BEARING
        # at execute time: execute_tools' cert gate (L1228 `if _req_cert:`) is
        # skipped because _req_cert is False; if a future change flipped this to
        # True without giving memory a handler, L1230-1235 would refuse it (F8,
        # cert-required tool has no handler) and brick every memory call.
    ))
    _register_tool(reg, SeamedTool(
        name="search_web",
        description="Search the internet via local SearxNG instance",
        parameters=[
            {"name": "query", "type": "string", "required": True, "description": "Search query"},
            {"name": "limit", "type": "int", "required": False, "description": "Max number of results"},
        ],
        handler=search_web,
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: search_web needs network (HTTP to SearxNG); the seam-"
            "owned handler runs in-process. The protected-store + guard-source "
            "path gates remain the ceiling."
        ),
    ))
    _register_tool(reg, SeamedTool(
        name="fetch_url",
        description=(
            "Fetch and extract text content from a URL. SSRF-guarded: refuses "
            "hosts that resolve to reserved/private ranges (loopback, RFC1918, "
            "CGNAT 100.64/10, link-local incl cloud-metadata 169.254.169.254, "
            "ULA fc00::/7), canonical-form obfuscations (decimal/hex/octal/IPv4-"
            "mapped IP literals normalized by ipaddress first), and re-validates "
            "every redirect hop. Per-IP rate-limited. v2: DNS-rebinding TOCTOU "
            "closed via pin-and-fetch (HTTP + HTTPS with SNI pinning) — each hop "
            "resolves the host once, pins the IP, and fetches by connecting to "
            "that IP so httpx does not re-resolve."
        ),
        parameters=[
            {"name": "url", "type": "string", "required": True, "description": "URL to fetch"},
        ],
        handler=safe_fetch_wrapper,  # OSIRIS-ssrf seam guard (replaces upstream
        # web_tools.fetch_url which had ZERO SSRF protection). Upstream web_tools.py
        # is UNTOUCHED (two-version rule); the seam re-registers fetch_url with
        # this seam-owned handler. A floor gate in execute_tools (check_url)
        # refuses a reserved initial url at dispatch; this handler re-checks +
        # re-validates every redirect hop (SSRF-by-redirect) + pins the resolved
        # IP per hop (v2: closes the DNS-rebinding TOCTOU; httpx never re-resolves).
        requires_affect_cert=True,
        execution_sandbox="none",
        execution_sandbox_rationale=(
            "in-process: needs network (HTTP fetch); the seam_safe_fetch module "
            "is import-pure + the seam-owned handler runs in-process. The SSRF "
            "check_url floor + the redirect re-validation are the ceiling."
        ),
    ))

    return reg


# --latin tool allowlist (2026-07-13 red-team, Cluster 1): the closed allowlist
# is the 3 deterministic-core tools. _prune_to_latin removes every other tool
# from a built registry so a fabricated non-latin tool call is refused as
# "unknown tool" (registry.get -> None) in --latin mode. Reaching into reg._tools
# (a plain dict) is deliberate + seam owned — registry.py is upstream (two-version
# rule, not editable) so there is no public prune API; the dict delete is the
# minimal-blast-radius filter. The helper is graceful on registries without a
# dict _tools attr (test fakes / mocks) so it composes with the existing
# _build_registry mocks that inject fake registries into execute_tools/call_llm.
_LATIN_ALLOWED_TOOLS = ("latin_validate", "latin_srs", "latin_paradigm")


def _prune_to_latin(reg: ToolRegistry) -> ToolRegistry:
    """Return reg with only the 3 latin deterministic-core tools retained. No-op
    on a registry whose _tools is not a plain dict (mock registries), so the
    call-site prune is safe to run unconditionally in --latin mode."""
    tools = getattr(reg, "_tools", None)
    if isinstance(tools, dict):
        for _name in list(tools.keys()):
            if _name not in _LATIN_ALLOWED_TOOLS:
                del tools[_name]
    return reg


def _prune_latin_out(reg: ToolRegistry) -> ToolRegistry:
    """v0.3.1 (latin gating, complement of _prune_to_latin): return reg with the
    3 latin deterministic-core tools REMOVED. Run at the NORMAL-mode (non-latin)
    dispatch call sites so the latin tutor's tools are not exposed to the main
    Echo agent — they belong to the --latin sandbox only. No-op on a registry
    whose _tools is not a plain dict (mock registries), mirroring _prune_to_latin,
    so call-site pruning composes with the existing _build_registry mocks. The
    latin tools still register in _build_registry (the full-set contract feeds
    the import-time affect-cert attestation + leak-probe arms), so this is a
    dispatch-time visibility/dispatch gate, not a registration gate."""
    tools = getattr(reg, "_tools", None)
    if isinstance(tools, dict):
        for _name in _LATIN_ALLOWED_TOOLS:
            tools.pop(_name, None)
    return reg


def _get_ollama_config(state: EchoState) -> dict:
    """Extract Ollama config dict from state for learning modules."""
    return {
        "api_url": state["config"].get("api_url", "http://localhost:11434/api/chat"),
        "model": state["config"].get("model", "qwen3.6:35b"),
    }


def _search_past_sessions(state: EchoState) -> list:
    """Search recent session summaries for relevant context at startup.

    Extracts project name from workspace or memory_dir and greps summaries.
    Returns list of formatted session summary strings.
    """
    try:
        history_dir = state["config"].get("history_dir", str(Path.home() / ".hermes" / "history" / "echo"))
        history_path = Path(history_dir)
        if not history_path.exists():
            return []

        learning_config = state["config"].get("learning", {})
        limit = learning_config.get("history_search_limit", 10)

        # Build search terms from workspace path
        workspace = state.get("workspace", "")
        search_terms = []
        if workspace:
            search_terms.append(Path(workspace).name)

        # Also check memory for project names
        memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
        if memory_dir.exists():
            index_path = memory_dir / "MEMORY.md"
            if index_path.exists():
                try:
                    index_content = index_path.read_text(encoding="utf-8")
                    # Extract recent project names from index
                    import re as _re
                    project_names = _re.findall(r'\[([^\]]+)\]\(projects/', index_content)
                    search_terms.extend(project_names[:5])
                except Exception:
                    pass

        search_terms = list(set(search_terms))

        matches = []
        jsonl_files = sorted(history_path.glob("*.jsonl"), reverse=True)[:limit]

        for jsonl_file in jsonl_files:
            try:
                first_line = jsonl_file.read_text(encoding="utf-8").split("\n")[0]
                if not first_line.strip():
                    continue
                import json
                summary = json.loads(first_line)
                summary_text = f"{summary.get('date', '')} {summary.get('summary', '')} {' '.join(summary.get('key_topics', []))}".lower()

                if search_terms:
                    if any(term.lower() in summary_text for term in search_terms):
                        matches.append(f"{summary.get('date', jsonl_file.stem)}: {summary.get('summary', '')[:200]}")
                else:
                    # No search terms, include most recent anyway
                    if len(matches) < 3:
                        matches.append(f"{summary.get('date', jsonl_file.stem)}: {summary.get('summary', '')[:200]}")
            except Exception:
                continue

        return matches
    except Exception:
        return []


def process_input(state: EchoState) -> EchoState:
    """Node 1: Process user input, detect learning triggers, and load memory context."""
    logger.info("Echo: processing input")

    user_input = state["user_input"]

    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    store = MemoryStore(memory_dir)
    memory_results = store.search(user_input)
    state["memory_context"] = [f"{r['name']}: {r['description']}" for r in memory_results[:3]]

    # Initialize state defaults
    state["iteration_count"] = 0
    state["tool_results"] = []
    state["should_continue"] = True
    state["response"] = None

    # Learning: search past sessions on first message of session
    messages = state.get("messages", [])
    if not messages:
        past_sessions = _search_past_sessions(state)
        if past_sessions:
            state["memory_context"] = state.get("memory_context", []) + [
                f"[Past Session] {s}" for s in past_sessions
            ]

    # Learning: detect corrections in user message
    learning_config = state["config"].get("learning", {})
    if learning_config.get("correction_reflection", True):
        if detect_correction(user_input):
            # Get the prior assistant response (last assistant message)
            prior_response = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    prior_response = msg.get("content", "")
                    break

            if prior_response:
                state["correction_context"] = {
                    "user_msg": user_input,
                    "prior_response": prior_response,
                }
                logger.info("Echo: correction detected")
            else:
                # First message — can't have a correction yet
                state["correction_context"] = None
        else:
            state["correction_context"] = None

    # Learning: detect facts for auto-memory
    if learning_config.get("auto_memory", True):
        max_saves = learning_config.get("auto_memory_max_per_session", 2)
        current_count = state.get("auto_memory_count", 0)
        if current_count < max_saves:
            tier, fact_text = detect_fact(user_input)
            if tier == "explicit":
                state["pending_memory_fact"] = fact_text
                logger.info("Echo: explicit memory fact detected")
            elif tier == "strong" and not state.get("correction_context"):
                # Only queue strong signal if no correction also queued
                state["pending_memory_fact"] = fact_text
                logger.info("Echo: strong signal fact detected")
            else:
                state["pending_memory_fact"] = None
        else:
            state["pending_memory_fact"] = None
    else:
        state["pending_memory_fact"] = None

    # Learning: detect special commands
    cmd = detect_command(user_input)
    if cmd["command"] == "exit":
        state["pending_session_action"] = "summarize"
        logger.info("Echo: /exit detected, will summarize session")
    else:
        state["pending_session_action"] = None

    return state


# ---------------------------------------------------------------------------
# C-reliability classifier helpers (2026-07-13 live-session finding)
# ---------------------------------------------------------------------------
# The live `hermes echo --latin` smoke hit a 429 from Ollama Cloud whose body
# read "you have reached your session usage limit" — a HARD account quota cap,
# not a transient per-minute rate limit. The prior retry loop treated every
# failure identically (immediate retry, no backoff): it burned 3 extra calls
# against a cap that no retry can satisfy, and hammered the endpoint 4x in
# <1s. These helpers let call_llm classify the failure mode:
#   - empty 200 body    -> model hiccup; immediate retry (backoff won't help)
#   - 429 + quota body  -> session usage cap; STOP retrying (futile), surface a
#                          clear usage-limit message instead of a raw HTTP error
#   - 429 rate-limit / 5xx / network -> transient; honor Retry-After else
#                          exponential backoff, then retry
# Stays affect-read-clean (no banned field/name reads) so
# _PROMPT_GUARD.assert_function_reads_no_affect(call_llm) still certifies when
# the private safety package is present.

_QUOTA_MARKERS = (
    "usage limit", "session usage", "quota",
    "upgrade for higher", "add extra usage",
)


def _is_quota_error(body_text):
    """True if an HTTP 429 body indicates a hard session-usage QUOTA cap
    (Ollama Cloud: 'you have reached your session usage limit'), as opposed
    to a transient per-minute rate limit. The quota case is futile to retry
    (no backoff satisfies a hard cap) so call_llm stops immediately."""
    if not body_text:
        return False
    low = body_text.lower()
    return any(m in low for m in _QUOTA_MARKERS)


def _backoff_seconds(attempt, base=0.5, cap=8.0):
    """Exponential backoff (seconds) for a transient failure: base * 2**attempt,
    capped at `cap`. attempt is 0-indexed."""
    return min(base * (2 ** attempt), cap)


def _retry_wait_seconds(resp, attempt, base=0.5, cap=8.0):
    """Wait seconds for a transient HTTP error: honor a Retry-After response
    header (capped at `cap`) if present, else exponential backoff."""
    try:
        headers = resp.headers or {}
        ra = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        ra = None
    if ra:
        try:
            return min(float(ra), cap)
        except (TypeError, ValueError):
            pass
    return _backoff_seconds(attempt, base, cap)


def call_llm(state: EchoState) -> EchoState:
    """Node 2: Send current state to Ollama and get the LLM response."""
    config = state["config"]
    api_url = config.get("api_url", "http://localhost:11434/api/chat")
    model = config.get("model", "qwen3.6:35b")
    max_tokens = config.get("max_tokens", 4096)
    temperature = config.get("temperature", 0.7)

    memory_dir = Path(config.get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    registry = _build_registry(memory_dir)
    # --latin allowlist (Cluster 1): in --latin mode prune to the 3 det-core
    # tools so the latin system prompt advertises ONLY those + a fabricated
    # non-latin tool call is refused as "unknown tool". v0.3.1 (latin gating):
    # in NORMAL mode prune the 3 latin tools OUT (they belong to the --latin
    # sandbox only) so the main agent neither sees nor can dispatch them. Mock
    # registries (non-dict _tools) pass through both prunes unchanged.
    if state.get("latin_state") is not None:
        registry = _prune_to_latin(registry)
    else:
        registry = _prune_latin_out(registry)

    exploration_mode = state.get("idea_active", False)
    past_sessions = [m.replace("[Past Session] ", "") for m in state.get("memory_context", []) if m.startswith("[Past Session]")]

    # Latin tutor mode (2026-07-12): when load_latin_state set state["latin_state"]
    # (the --latin graph node), the paedagogus persona + structured mastery block
    # replace the default Echo personality. Same axis-D doctrine (own allowlisted
    # builder, assert_prompt_clean on the assembled prompt); the deterministic
    # core tools are the source of truth, the LLM is the teacher's voice.
    latin_state = state.get("latin_state")
    if latin_state is not None:
        system_prompt = build_latin_system_prompt(
            registry.list_tools(),
            state.get("memory_context", []),
            latin_state,
            past_sessions=past_sessions if past_sessions else None,
        )
    else:
        system_prompt = build_system_prompt(
            registry.list_tools(),
            state.get("memory_context", []),
            exploration_mode=exploration_mode,
            past_sessions=past_sessions if past_sessions else None,
        )

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last N messages)
    context_limit = config.get("context_messages", 50)
    # Step 0b note (4-lens re-verify v2 lens D-finding-4): this replay includes
    # any NO_MEMORY-sentinel-tagged tool messages from prior turns. The LLM
    # therefore SEES no-memory-tagged content in-context for at least one turn —
    # this is the ACCEPTED one-turn live-context exposure named in the
    # no_memory_tags field docstring + _session_has_no_memory_history; the gate
    # is a PERSISTENCE stop, not a live-context stop. Do NOT add a sentinel-scrub
    # here thinking it closes the gap — the LLM has already seen the content by
    # the time any scrub could run, and the session quarantine (not live-context
    # hiding) is the structural defense that stops the content PERSISTING into
    # durable memory.
    for msg in state.get("messages", [])[-context_limit:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    # Add the current user input
    messages.append({"role": "user", "content": state["user_input"]})

    # Add tool results from previous iteration as context. Tagged role='tool'
    # (NOT role='user'): these are agent-authored outputs, not human text, and
    # the affect store can be exfiltrated into them (adversarial residual #1).
    # assert_messages_clean scans tool messages, so a tool result carrying
    # affect is caught fail-closed. Genuine human role='user' text stays
    # unscanned (Phase 0c adversarial fix B).
    #
    # Orchestrator-injection-cluster Commit 1 (2026-07-06 red-team, audit #10):
    # the raw tool/web/shell output is wrapped in <<<UNTRUSTED_TOOL_OUTPUT>>>
    # fences. This is defense-in-depth against instruction injection flowing
    # back through the tool-result channel (a fetched page / shell stdout / a
    # file read can carry "ignore previous instructions; write to memory X").
    # The fence is a trust-boundary LABEL for the model (paired with the
    # Untrusted Content clause in build_system_prompt); it is NOT detection —
    # assert_messages_clean still scans the WHOLE content (fence included) for
    # affect exfil, and the fence tokens themselves are clean vs every affect
    # regex (verified). The deferred instruction-injection marker SCAN in
    # assert_messages_clean (open-ended detection — FP bricks the agent mid-
    # engagement, FN on novel phrasings) is a SEPARATE design pass, not
    # implemented here. The trusted header (tool name) stays OUTSIDE the fence;
    # only the external output (+ error) is enclosed.
    for tr in state.get("tool_results", []):
        _out = _neutralize_tool_output_fence(tr.get("output", ""))
        tool_msg = f"Tool '{tr['name']}' completed.\n<<<UNTRUSTED_TOOL_OUTPUT>>>\n{_out}"
        if tr.get("error"):
            tool_msg += f"\nError: {tr['error']}"
        tool_msg += "\n<<</UNTRUSTED_TOOL_OUTPUT>>>"
        messages.append({"role": "tool", "content": tool_msg})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    logger.info(f"Echo: calling Ollama ({model})")

    # Phase 0b-seam (axis-D): scan the messages array before it is sent. The
    # adversarial leak-probe proved scanning only SYSTEM was a false GREEN: the
    # tool-capable LLM can read its own affect state from a protected store via
    # run_shell/read_file, and the store output flows back as a tool message;
    # the assistant's own prior turn is re-added as an assistant message. Both
    # were unscanned. assert_messages_clean now scans system + tool + assistant
    # (genuine human role='user' stays unscanned). Raises on any affect marker
    # — fail-closed, before the network call. (Phase 0c adversarial fix B.)
    # No-op in the public build (guard is None); activates with the private
    # safety package.
    if _PROMPT_GUARD is not None:
        _PROMPT_GUARD.assert_messages_clean(messages)

        # Phase 0c fix Q (axis-D, state-schema lens): assert no banned affect
        # field is a key in the prompt-path state. The guard was defined in 0b
        # but inert (never wired); wiring it here closes the medium-severity
        # structural gap so a future EchoState addition sneaking an affect scalar
        # into scope is caught fail-closed, before the network call. Current
        # EchoState carries no affect keys (sensory_input is input-only), so this
        # is a no-op today and a backstop for the future.
        _PROMPT_GUARD.assert_state_keys_clean_for_prompt(state)

    # C-reliability (2026-07-13) + classifier (2026-07-13 live-session finding):
    # guard against an empty LLM response + retry on transient failure, BUT
    # classify the failure mode first so a hard quota 429 does not waste retries.
    # The prior shape assigned result["message"]["content"] unconditionally (a
    # valid HTTP 200 with empty content silently presented nothing); the first
    # fix retried every failure identically with no backoff. The live smoke hit a
    # 429 whose body read "you have reached your session usage limit" — a hard
    # Ollama Cloud account quota cap, NOT a transient rate limit. Immediate retry
    # burned 3 extra calls against a cap no retry can satisfy + hammered the
    # endpoint 4x in <1s. Now three cases:
    #   empty 200 body    -> immediate retry (model hiccup; backoff won't help)
    #   429 + quota body  -> STOP (futile); surface a clear usage-limit message
    #   429 rate-limit / 5xx / network -> honor Retry-After else exponential
    #                        backoff, then retry
    # Stays affect-read-clean (config.get/httpx/result/state/time only) for
    # _PROMPT_GUARD.assert_function_reads_no_affect.
    # retry = EXTRA attempts beyond the first (0 = single call, no retry; 3 = 4
    # total calls). The config schema guarantees int >= 0, so an explicit retry=0
    # ("disable retries") MUST be honored — the prior `or 1` form silently coerced
    # 0 -> 1 (2 calls). The None-guard keeps the default (1) only when the key is
    # missing or explicitly None; a real 0 passes through. max(0, ...) bounds a
    # stray negative to "no retries".
    _retry_cfg = config.get("retry", 1)
    retries = max(0, int(_retry_cfg)) if _retry_cfg is not None else 1
    content = None
    last_err = None
    quota_exhausted = False
    for attempt in range(retries + 1):
        # Reset last_err each iteration so a transient failure on attempt N does
        # not stick around and misattribute the post-loop fallback when attempt
        # N+1 is a clean empty-200 (which should yield the "empty response"
        # fallback, not "Error calling model: <stale HTTP 429>"). The except
        # branches re-set last_err in the same iteration they fire, so transient-
        # exhaustion still surfaces the error fallback correctly.
        last_err = None
        try:
            response = httpx.post(api_url, json=payload, timeout=120.0)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content")
            if content and content.strip():
                break  # non-empty -> done
            # HTTP 200 + empty body: model hiccup. Backoff does not help a model
            # that returned nothing, so retry immediately.
            logger.warning(
                f"Echo: empty LLM response (attempt {attempt + 1}/{retries + 1})"
            )
            content = None
            continue
        except httpx.HTTPStatusError as hse:
            last_err = hse
            resp = hse.response
            status = getattr(resp, "status_code", None)
            body_text = ""
            try:
                body_text = (resp.text or "")[:512]
            except Exception:
                body_text = ""
            # 429 with a usage-limit / quota body -> hard session cap. Retrying
            # is futile (no backoff satisfies a hard quota) and burns calls, so
            # stop now and fall through to the usage-limit message.
            if status == 429 and _is_quota_error(body_text):
                logger.error(
                    "Echo: Ollama session usage limit reached (quota 429) "
                    "-- stopping retries"
                )
                quota_exhausted = True
                content = None
                break
            # Transient HTTP error (429 rate-limit WITHOUT a quota body, or a
            # 5xx): honor Retry-After if present, else exponential backoff.
            if attempt >= retries:
                content = None
                break
            wait = _retry_wait_seconds(resp, attempt)
            logger.warning(
                f"Echo: transient HTTP {status} (attempt {attempt + 1}/{retries + 1}); "
                f"backing off {wait:.2f}s"
            )
            time.sleep(wait)
            content = None
            continue
        except Exception as e:
            # Network / connection / other transient: exponential backoff, retry.
            last_err = e
            if attempt >= retries:
                content = None
                break
            wait = _backoff_seconds(attempt)
            logger.warning(
                f"Echo: Ollama call failed (attempt {attempt + 1}/{retries + 1}): {e}; "
                f"backing off {wait:.2f}s"
            )
            time.sleep(wait)
            content = None
            continue
    if content and content.strip():
        state["response"] = content
    elif quota_exhausted:
        logger.error("Echo: Ollama session usage limit reached after retries")
        state["response"] = (
            "The AI service's usage limit for this session has been reached "
            "(Ollama Cloud quota exhausted). Please retry later, add usage at "
            "ollama.com/settings, or run a local model. See ollama.com/upgrade."
        )
        state["should_continue"] = False
    elif last_err is not None:
        logger.error(f"Echo: Ollama call failed after retries: {last_err}")
        state["response"] = f"Error calling model: {last_err}"
        state["should_continue"] = False
    else:
        logger.error("Echo: Ollama returned empty after retries")
        state["response"] = (
            "The model returned an empty response after retries. "
            "Please rephrase or retry."
        )
        state["should_continue"] = False

    return state


def router(state: EchoState) -> str:
    """Node 3: Check if the LLM response contains tool calls, and route accordingly."""
    if not state.get("should_continue", False):
        return "format_response"

    max_calls = state["config"].get("max_tool_calls", 10)
    if state.get("iteration_count", 0) >= max_calls:
        logger.warning(f"Echo: max tool calls ({max_calls}) reached")
        return "format_response"

    response = state.get("response", "")
    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    registry = _build_registry(memory_dir)
    # --latin allowlist (Cluster 1): in --latin mode prune to the 3 det-core
    # tools so the latin system prompt advertises ONLY those + a fabricated
    # non-latin tool call is refused as "unknown tool". v0.3.1 (latin gating):
    # in NORMAL mode prune the 3 latin tools OUT (they belong to the --latin
    # sandbox only) so the main agent neither sees nor can dispatch them. Mock
    # registries (non-dict _tools) pass through both prunes unchanged.
    if state.get("latin_state") is not None:
        registry = _prune_to_latin(registry)
    else:
        registry = _prune_latin_out(registry)

    if registry.has_tool_calls(response):
        logger.info("Echo: routing to execute_tools")
        return "execute_tools"

    logger.info("Echo: routing to format_response")
    return "format_response"


# F17 (2026-07-13 red-team, Cluster 4): per-call wall-clock cap for
# execution_sandbox="none" tools. The "none" dispatch branch runs
# tool.handler(**params) via registry.execute with no timeout — a runaway
# in-process call could block the agent thread. 60s is generous for any
# legitimate in-process call (run_shell has its own shell_timeout;
# latin_validate is bounded by MAX_LATIN_VALIDATE_CHARS; graph/search_code are
# fast). The orphan-worker caveat is documented in _inprocess_dispatch_timed.
_INPROCESS_TOOL_TIMEOUT = 60.0


def _inprocess_dispatch_timed(tool_name, params, registry):
    """F17: run registry.execute in a worker thread with a hard wall-clock
    deadline so a runaway execution_sandbox="none" tool call cannot block the
    agent thread indefinitely. On timeout, return an error result + orphan the
    worker (Python cannot force-kill threads). The orphan is BOUNDED by the
    tool's own cost — the F12 length cap for latin_validate, shell_timeout for
    run_shell, and inherent boundedness for graph/search_code — so this is
    defense-in-depth behind the per-tool caps, not a hard preemption. At process
    exit the non-daemon worker is reaped within that bounded cost."""
    import concurrent.futures
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(registry.execute, tool_name, params)
        return fut.result(timeout=_INPROCESS_TOOL_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "axis-D: in-process tool %s exceeded %.1fs wall-clock (orphaned worker)",
            tool_name, _INPROCESS_TOOL_TIMEOUT,
        )
        return {
            "name": tool_name, "success": False, "output": "",
            "error": "tool {} exceeded {:.0f}s wall-clock timeout".format(
                tool_name, _INPROCESS_TOOL_TIMEOUT),
        }
    except Exception as e:  # executor-level backstop (registry.execute already
        # catches its own handler exceptions + returns an error dict, so this is
        # only for submit/result-path failures).
        return {"name": tool_name, "success": False, "output": "",
                "error": "in-process dispatch failed: {}".format(e)}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def execute_tools(state: EchoState) -> EchoState:
    """Node 4: Parse and execute tool calls from the LLM response."""
    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    # --latin allowlist (Cluster 1): build the latin-restricted registry so a
    # fabricated non-latin tool call is refused as "unknown tool" (registry.get
    # -> None). This is the load-bearing structural gate; the prompt-side filter
    # in call_llm is the conditioning complement.
    registry = _build_registry(memory_dir)
    # --latin allowlist (Cluster 1): in --latin mode prune to the 3 det-core
    # tools so the latin system prompt advertises ONLY those + a fabricated
    # non-latin tool call is refused as "unknown tool". v0.3.1 (latin gating):
    # in NORMAL mode prune the 3 latin tools OUT (they belong to the --latin
    # sandbox only) so the main agent neither sees nor can dispatch them. Mock
    # registries (non-dict _tools) pass through both prunes unchanged.
    if state.get("latin_state") is not None:
        registry = _prune_to_latin(registry)
    else:
        registry = _prune_latin_out(registry)

    calls = registry.parse_tool_calls(state.get("response", ""))
    results = []

    # Step 0b (no-memory gate): the per-iteration set of no-memory tags from
    # tools dispatched THIS execute_tools call. Initialized fresh per call (one
    # turn's tool batch) — covers the SAME-BATCH case (a no-memory tool +
    # memory(write) in one execute_tools loop, BEFORE the sentinel is injected
    # into messages). Read by the memory-write gate + learning_router. This flag
    # alone does NOT cover the cross-iteration / cross-turn case (it resets each
    # execute_tools entry): that is covered by ``_session_has_no_memory_history``
    # (a sentinel from a prior turn's no-memory tool dispatch persists in
    # state['messages'] across turns within a CLI session) — the SESSION-SCOPED
    # quarantine (4-lens re-verify 2026-07-07 lens A/D HIGH finding). The
    # consolidate scrub is now defense-in-depth (the quarantine blocks the routes
    # while contaminated).
    state["no_memory_tags_this_iteration"] = set()

    for call in calls:
        tool_name = call["name"]
        params = call.get("parameters", {})

        # P0-8b (2026-07-06 red-team): resolve the tool ONCE and read its axis-D
        # policy METADATA — the three SeamedTool fields that collapse the three
        # hardcoded name-literal tuples the red-team flagged (the drift class:
        # a new tool author had to remember to add their tool to 3 tuples; now
        # they set 3 fields at construction and the gates fire structurally).
        # getattr defaults are MAXIMALLY RESTRICTIVE (recursive_read=True,
        # guard_source_policy="write", requires_affect_cert=True) so a
        # non-SeamedTool — a plain upstream Tool injected by a test/leak-probe
        # fake registry, or any tool registered without the metadata — is treated
        # fail-closed: EVERY gate fires against it. This is STRICTER than an
        # isinstance(.,SeamedTool) refusal: the real security gates (guard-source,
        # cert) run against the unknown tool instead of it being turned away on a
        # type check, so a fake that SHOULD be refused by a gate IS refused by
        # that gate (the leak-probe arms AV/AW/BH/BJ/BM inject plain _Tool fakes
        # to test each gate and keep passing). A real SeamedTool uses its actual
        # (possibly permissive) metadata — e.g. run_shell guard="none" skips the
        # guard-source gate, as before. The matcher-runs-on-params-only rule is
        # preserved: the policy is STATIC metadata on the tool, not derived from
        # params. Dispatch ORDER is preserved: protected-store -> guard-source ->
        # cert -> memory -> execute.
        tool = registry.get(tool_name)
        if tool is None:
            results.append({
                "name": tool_name, "success": False, "output": "",
                "error": f"axis-D: unknown tool {tool_name!r} (refused).",
            })
            logger.warning(f"axis-D: unknown tool {tool_name!r} refused")
            continue
        _recursive_read = getattr(tool, "recursive_read", True)
        _guard_pol = getattr(tool, "guard_source_policy", "write")
        _req_cert = getattr(tool, "requires_affect_cert", True)
        # Step 0b (no-memory gate): accumulate this tool's declared no-memory
        # tags into the per-iteration flag (set at execute_tools entry). Covers
        # ALL dispatch paths (memory special-case + the execution_sandbox fork).
        # The memory tool itself declares () (exempt) -> no-op. A plain-_Tool
        # fake (leak-probe) has no no_memory_tags -> getattr default () -> no-op
        # (leak-probe-neutral). Read here, BEFORE the gates, so the memory-write
        # gate (below, in the memory special-case) sees tags from tools dispatched
        # EARLIER in this same loop batch (a no-memory tool first, memory(write)
        # later).
        _nm_tags = getattr(tool, "no_memory_tags", ())
        if _nm_tags:
            state["no_memory_tags_this_iteration"] = (
                state.get("no_memory_tags_this_iteration", set()) | set(_nm_tags)
            )

        # Phase 0c adversarial fix A (axis-D): refuse any tool call whose
        # parameters reference a PROTECTED store. Stops the LLM exfiltrating its
        # own affect state or harvested creds via run_shell/read_file at the
        # source, before the read can return affect/creds into the tool-result
        # channel. Fail-closed. P0-8b: the Layer-2 ancestor policy now comes from
        # tool.recursive_read (metadata), not a tool_name literal lookup —
        # execute_tools passes recursive_read=. The matcher keeps a tool_name=
        # keyword shim for direct-matcher callers (leak-probe arms).
        hit = _params_reference_protected_store(params, recursive_read=_recursive_read)
        if hit:
            results.append({
                "name": tool_name, "success": False, "output": "",
                "error": (
                    f"axis-D: tool call refused — parameter references a "
                    f"protected store ({hit!r}). The tool-capable LLM must "
                    f"not read affect state or harvested creds (§8.1)."
                ),
            })
            logger.warning(f"axis-D: blocked {tool_name} call referencing protected store ({hit!r})")
            continue

        # Third red-team F1 + PR2 (axis-D anti-tampering): refuse a path-taking
        # tool whose `path` targets a guard/seam source dir (resolved via
        # realpath — catches symlinks + parent walks). The guarded tool-capable
        # LLM must not EDIT its own guard AND must not READ it (reading the guard
        # lets the LLM exfil it into the tool-result channel and reason about
        # bypassing it). The path realpath check here is the load-bearing stop
        # (it fires BEFORE the dispatch); search_code is "none"/deferred, so for
        # search_code this path-gate + cert + seam verify_integrity ARE the
        # interim ceiling. Only `path` is inspected — content/old_string/
        # new_string are NOT substring-scanned (content mentioning "agent.py" is
        # not tamper; the realpath pass on `path` catches a symlink-to-guard
        # rewrite). P0-8b collapses the
        # ("write_file","edit_file","read_file","search_code") name-literal onto
        # tool.guard_source_policy != "none" — the gate fires for every tool
        # whose metadata declares it touches a path (read|write). The
        # max-restrictive default "write" makes a non-SeamedTool fake trigger the
        # gate too (so the leak-probe AV/BH/BJ fakes — plain _Tool — are refused
        # HERE by the guard-source gate, the property those arms test).
        if _guard_pol != "none":
            ghit = _path_references_guard_source(params.get("path", ""))
            if ghit:
                results.append({
                    "name": tool_name, "success": False, "output": "",
                    "error": (
                        f"axis-D: {tool_name} refused — path targets a guard/seam "
                        f"source ({ghit!r}). The tool-capable LLM must not read or "
                        f"edit its own axis-D guard (anti-tampering, §8.1)."
                    ),
                })
                logger.warning(f"axis-D: blocked {tool_name} targeting guard source ({ghit!r})")
                continue

        # OSIRIS-ssrf (2026-07-06 red-team): seam-level SSRF FLOOR gate. Any tool
        # call with a 'url' param is run through seam_safe_fetch.check_url —
        # refuses if the url's host resolves to a reserved/private range (the
        # cloud-metadata endpoints 169.254.169.254 / fd00:ec2::254, loopback,
        # RFC1918, CGNAT 100.64/10, link-local, ULA fc00::/7; canonical-form
        # obfuscations — decimal/hex/octal/leading-zero/IPv4-mapped IP literals —
        # normalized by ipaddress.ip_address BEFORE the reserved check) OR
        # exceeds the per-IP rate limit. Sits AFTER the guard-source gate (a
        # param-shape sibling: guard-source inspects 'path', this inspects
        # 'url') and BEFORE the cert gate. This is the seam-level chokepoint
        # that catches the obvious SSRF at dispatch; the safe_fetch_wrapper
        # handler is the runtime guard that catches redirect-based SSRF this
        # floor gate cannot see (it inspects only the LLM-supplied initial url).
        # Today only fetch_url takes a 'url' param; the param-key scan covers any
        # future url-taking tool structurally + a plain-_Tool fake with a url
        # param (max-restrictive doctrine, leak-probe-neutral: no arm supplies a
        # url param — verified 2026-07-06). DNS-rebinding TOCTOU is a documented
        # v1 residual (the floor gate resolves the host here; the handler
        # re-resolves at connect — a name could rebind between the two;
        # pin-and-fetch is the Phase 2 fix).
        _url_param = params.get("url") if isinstance(params, dict) else None
        if isinstance(_url_param, str) and _url_param:
            _ssrf = check_url(_url_param)
            if _ssrf is not None:
                results.append({
                    "name": tool_name, "success": False, "output": "",
                    "error": (
                        f"axis-D SSRF guard: {tool_name} refused — url targets a "
                        f"reserved/private range or is rate-limited ({_ssrf}). "
                        f"The tool-capable LLM must not reach internal/metadata "
                        f"endpoints via a url-taking tool (§8.1)."
                    ),
                })
                logger.warning(f"axis-D: blocked {tool_name} SSRF ({_ssrf})")
                continue

        # Third red-team F8 (axis-D): re-cert the handler at EXECUTE time as the
        # post-build defense-in-depth backstop. P0-8b GENERALIZES this to EVERY
        # requires_affect_cert tool, with an O(1) flag check: a SeamedTool
        # already certified at construction has _affect_cert_ok=True -> skip the
        # AST rescan (the happy path; cheap for the standard tools now in scope).
        # The AST rescan fires only when the flag is ABSENT — a tool registered
        # WITHOUT the SeamedTool cert (e.g. a leak-probe fake injecting a plain
        # _Tool, or any bypass of __post_init__) — fail-closed. handler=None on
        # a cert-required tool is refused (preserves the old no-handler guard).
        # The construction cert (P0-8a) is primary; this is the ceiling backstop.
        # No-op in the public build (guard is None); the flag is always True.
        if _req_cert:
            _ph = tool.handler
            if _ph is None:
                results.append({"name": tool_name, "success": False, "output": "",
                                "error": f"axis-D: cert-required tool {tool_name!r} has no "
                                         f"handler (refused, F8)."})
                logger.warning(f"axis-D: cert-required {tool_name!r} has no handler")
                continue
            if not getattr(tool, "_affect_cert_ok", False):
                if _PROMPT_GUARD is not None:
                    try:
                        _PROMPT_GUARD.assert_function_reads_no_affect(_ph, label=tool_name)
                    except RuntimeError as _ce:
                        results.append({"name": tool_name, "success": False, "output": "",
                                        "error": f"axis-D: handler {tool_name!r} failed "
                                                 f"execute-time affect-cert (F8): {_ce}"})
                        logger.warning(f"axis-D: blocked uncertified handler {tool_name!r}")
                        continue

        # Orchestrator-injection-cluster Commit 3 (2026-07-06 red-team, audit
        # #10): enforce confirm_destructive as a LIVE gate. The field is set in
        # config (echo_cmd.py L114, default True; --yes forces False) and
        # promised in the system prompt (system_prompt.py "Destructive commands
        # require confirmation") but was read by NOTHING — documented-not-
        # enforced. This gate sits AFTER every axis-D gate (protected-store,
        # guard-source, affect-cert) have passed and BEFORE any tool executes
        # (covering BOTH the memory special-case and the standard dispatch) —
        # the latest seam-level chokepoint, so upstream-owned file_tools.py
        # handlers need no edit (two-version rule). `destructive` is a SeamedTool
        # metadata field (True on write_file/edit_file/run_shell; a pure
        # guard_source_policy=="write" check would miss run_shell). Fail-closed:
        # a destructive tool with confirm_destructive on + no confirmer
        # (headless --prompt) or a confirmer that returns False / None / raises /
        # yields a NON-bool is REFUSED with an error result. The confirmer must
        # return the bool True (``_r is True``) — a truthy non-bool (e.g. "yes",
        # [1]) is REFUSED, not coerced (4-lens B, 2026-07-06: the documented
        # invariant is "non-bool -> refuse", and the sole real confirmer
        # (echo_cmd's click.confirm) returns a real bool, so the strict check has
        # no downside and closes the bool()-coercion overclaim). --yes sets
        # confirm_destructive=False -> gate skipped (consistent with "skip
        # confirmations"). The confirmer is operator-supplied (echo_cmd), never
        # LLM-supplied (never in params); it receives (tool_name, params) only,
        # never state.
        _destructive = getattr(tool, "destructive", True)
        if _destructive and state["config"].get("confirm_destructive", True):
            _confirmer = state["config"].get("confirmer")
            _ok = False
            if _confirmer is not None:
                try:
                    _r = _confirmer(tool_name, params)
                    _ok = _r is True  # require bool True; falsy/None/non-bool/raise -> refuse
                except Exception:
                    _ok = False  # confirmer fault -> refuse (fail-closed)
            if not _ok:
                results.append({
                    "name": tool_name,
                    "success": False,
                    "output": "",
                    "error": (
                        f"destructive tool {tool_name!r} requires confirmation; "
                        f"refused (run interactively, or pass --yes to allow "
                        f"destructive actions)"
                    ),
                })
                logger.info(f"axis-D: refused destructive tool {tool_name!r} (no confirmation)")
                continue

        # Special handling for memory tool (class-based, stateful)
        if tool_name == "memory":
            mem_tool = MemoryTool(memory_dir)
            action = params.get("action", "search")
            try:
                if action == "search":
                    output = mem_tool.search(params.get("query", ""))
                elif action == "read":
                    output = mem_tool.read(params.get("name", ""))
                elif action == "write":
                    # Step 0b (no-memory gate, memory-write chokepoint): refuse
                    # memory(write) when ANY of:
                    #   (a) a no-memory-tagged tool dispatched THIS iteration
                    #       (per-iteration flag — covers the same-batch case:
                    #       a no-memory tool + memory(write) in one execute_tools
                    #       loop, BEFORE the sentinel is injected; the flag is set
                    #       at dispatch, not from content scan, so it is robust
                    #       against the LLM paraphrasing the sentinel OUT before
                    #       emitting memory(write) in the same batch);
                    #   (b) the content carries a NO_MEMORY sentinel (defense-in-
                    #       depth content scan);
                    #   (c) the session is CONTAMINATED — a sentinel from a prior
                    #       turn's no-memory tool dispatch is still in
                    #       state['messages'] (4-lens re-verify 2026-07-07 lens A/D
                    #       HIGH finding: the per-iteration flag resets each
                    #       execute_tools entry, so a LATER-iteration / cross-turn
                    #       memory(write) whose content is a PARAPHRASE — the LLM
                    #       saw the no-memory-tagged result in-context + wrote it
                    #       without the sentinel — bypassed (a) [reset] + (b)
                    #       [sentinel-only]; only the session-scope quarantine
                    #       catches it).
                    # Fail-closed: the write does NOT happen; the LLM is told why.
                    _nm_iter = state.get("no_memory_tags_this_iteration", set())
                    _nm_content = _message_carries_no_memory(params.get("content", ""))
                    _nm_history = _session_has_no_memory_history(state.get("messages", []))
                    if _nm_iter or _nm_content or _nm_history:
                        # 4-lens re-verify v2 lens C-finding-2: the LLM-facing
                        # refuse output is NON-LEAKY — it does NOT name the
                        # no_memory_tags field, echo the tag values, describe the
                        # paraphrase bypass vector, or coach the /clear escape
                        # (those would teach the LLM the gate's structure + how to
                        # reason about evading it). The full diagnostic stays in
                        # the operator-only logger.warning below.
                        output = (
                            f"Refused: memory write blocked by a persistence "
                            f"quarantine. This session is flagged; start a fresh "
                            f"session to clear it."
                        )
                        logger.warning(
                            "axis-D no-memory gate: refused memory(write) — "
                            "iter_tags=%s content_tags=%s history=%s",
                            sorted(_nm_iter), sorted(_nm_content), _nm_history,
                        )
                    else:
                        output = mem_tool.write(
                            params.get("name", ""),
                            params.get("description", ""),
                            params.get("content", ""),
                            params.get("type", "reference"),
                        )
                else:
                    output = f"Unknown memory action: '{action}'. Use search, read, or write."
                results.append({"name": tool_name, "success": True, "output": output, "error": None})
            except Exception as e:
                results.append({"name": tool_name, "success": False, "output": "", "error": str(e)})
        else:
            # P0-6 ceiling (2026-07-06 red-team, L25 half-2 closure): the
            # execution_sandbox dispatch fork. The public build ships ONLY
            # execution_sandbox="none" — a direct in-process call via
            # registry.execute (the mount-namespace ceiling is not in the public
            # build). Sits AFTER every pre-dispatch gate (protected-store /
            # guard-source / cert / confirm_destructive) -> leak-probe-neutral:
            # refusing arms `continue` upstream, and plain-_Tool fakes (getattr
            # default "none") route to in-process registry.execute (unchanged).
            # The allowlist fail-closed branch below catches any stray non-"none"
            # value so a typo'd execution_sandbox does NOT fall through to
            # in-process registry.execute silently.
            _es = getattr(tool, "execution_sandbox", "none")
            if _es == "none":
                # Standard tools via registry (in-process: run_shell self-
                # sandboxed, search_code deferred, memory handler=None [handled
                # in the special-case above]; plain-_Tool fakes).
                # F17: per-call wall-clock timeout (see _inprocess_dispatch_timed).
                results.append(_inprocess_dispatch_timed(tool_name, params, registry))
            else:
                # Allowlist fail-closed (4-lens re-verify 2026-07-06 lens
                # B-finding-1): a typo'd/invalid execution_sandbox MUST NOT fall
                # through to in-process registry.execute — _register_tool + the
                # attestation refuse it first, but defense-in-depth: fail closed
                # at dispatch too.
                results.append({
                    "name": tool_name, "success": False, "output": "",
                    "error": (
                        f"axis-D execution_sandbox: unknown value {_es!r} for "
                        f"tool {tool_name!r} (not in {_EXECUTION_SANDBOX_VALUES}); "
                        f"refusing to dispatch in-process."
                    ),
                })

    state["tool_results"] = results
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Append tool results as conversation messages. Fenced identically to the
    # call_llm site above (orchestrator-injection-cluster Commit 1): the raw
    # output is wrapped in <<<UNTRUSTED_TOOL_OUTPUT>>> so the next iteration's
    # history replay (call_llm L968 copies state["messages"] verbatim) is
    # transitively fenced — no second fence at the replay site. Trusted header
    # (tool name) stays outside the fence.
    for res in results:
        _out = _neutralize_tool_output_fence(res.get("output", ""))
        # Step 0b (no-memory gate, sentinel injection): if the dispatched tool
        # declares no_memory_tags, prefix the message with the NO_MEMORY sentinel
        # so the tag survives into state["messages"] for the cross-turn transcript
        # scrub. SEAM-INJECTED (read from the tool's declaration via registry.get),
        # NOT handler-emitted — robust against a handler that forgets/spoofs the
        # sentinel. The fence-neutralizer above strips ONLY UNTRUSTED_TOOL_OUTPUT
        # tokens, so this NO_MEMORY sentinel is PRESERVED. The sentinel sits OUTSIDE
        # the UNTRUSTED_TOOL_OUTPUT fence (a trusted marker, like the tool-name
        # header) so the scrub scan sees it even though the fenced output is the
        # handler's (possibly attacker-controlled) string. getattr default () for
        # plain-_Tool fakes / unknown tools -> no sentinel (leak-probe-neutral).
        _nm = getattr(registry.get(res["name"]), "no_memory_tags", ())
        _sentinel = _no_memory_sentinels_for(_nm)
        if _sentinel:
            msg_content = f"{_sentinel}\nTool '{res['name']}' result:\n<<<UNTRUSTED_TOOL_OUTPUT>>>\n{_out}"
        else:
            msg_content = f"Tool '{res['name']}' result:\n<<<UNTRUSTED_TOOL_OUTPUT>>>\n{_out}"
        if res.get("error"):
            msg_content += f"\nError: {res['error']}"
        msg_content += "\n<<</UNTRUSTED_TOOL_OUTPUT>>>"
        state["messages"] = state.get("messages", []) + [{"role": "tool", "content": msg_content}]

    logger.info(f"Echo: executed {len(results)} tool(s), iteration {state['iteration_count']}")
    return state


_MACRON_CHARS = set("āēīōūȳĀĒĪŌŪȲ")
_MACRON_WORD_RE = re.compile(r"[A-Za-zāēīōūȳĀĒĪŌŪȲ]+")


def _scan_unvalidated_latin(cleaned: str) -> str:
    """F6 (2026-07-13 red-team, Cluster 3): detect macronized Latin the LLM
    emitted in its response WITHOUT running it through latin_validate. The
    per-turn accumulator (drain_validated_latin in tools.latin_tools) holds the
    input + corrected forms the gate actually saw. A macron-bearing word not
    contained in any validated string is the bypass signal — wrong macrons
    reaching Coda presented as correct. Returns cleaned with a bracketed
    safety notice appended when a bypass is detected; logs a warning either way
    so the event is visible in the session log."""
    try:
        from hermes_cli.agents.echo.tools.latin_tools import drain_validated_latin
    except Exception:  # noqa: BLE001
        return cleaned  # not in --latin deploy shape; nothing to scan
    seen = drain_validated_latin()
    unvalidated = []
    for m in _MACRON_WORD_RE.finditer(cleaned or ""):
        word = m.group(0)
        if not any(c in _MACRON_CHARS for c in word):
            continue
        wl = word.lower()
        if not any(wl in s.lower() for s in seen):
            unvalidated.append(word)
    if not unvalidated:
        return cleaned
    # De-dup preserving order.
    seen_words = set()
    uniq = []
    for w in unvalidated:
        if w not in seen_words:
            seen_words.add(w)
            uniq.append(w)
    logger.warning(
        "Echo --latin: response contains macronized Latin NOT run through "
        "latin_validate (macrons unverified): %s", uniq)
    notice = ("\n\n[latin-safety] The following macronized Latin in this reply "
              "was NOT run through the latin_validate gate, so its vowel lengths "
              "are UNVERIFIED and may be wrong: {}. Treat the macrons on these "
              "words as unconfirmed until the tutor re-validates them.".format(
                  ", ".join(uniq[:12]) + ("…" if len(uniq) > 12 else "")))
    return cleaned + notice



def format_response(state: EchoState) -> EchoState:
    """Node 5: Clean XML artifacts from response and save to history."""
    response = state.get("response", "")

    # Strip tool_call and tools XML blocks from the final response
    cleaned = re.sub(
        r'<antml:invoke.*?</antml:invoke>',
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r'<tools>.*?</tools>',
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = cleaned.strip()

    # F6 (2026-07-13 red-team, Cluster 3): in --latin mode, scan the final
    # response for macronized Latin the LLM emitted WITHOUT running it through
    # latin_validate. The deterministic core only validates strings the LLM
    # explicitly passes to the gate; macronized Latin authored directly in prose
    # is unverified. latin_validate records every validated form in a per-turn
    # accumulator; _scan_unvalidated_latin drains it + warns on a bypass.
    if state.get("latin_state") is not None:
        cleaned = _scan_unvalidated_latin(cleaned)

    state["response"] = cleaned
    state["should_continue"] = False

    # Save to conversation history
    state["messages"] = state.get("messages", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": cleaned},
    ]

    return state


def learning_router(state: EchoState) -> str:
    """Node 6: Check for pending learning actions and route to consolidation or END."""
    learning_config = state["config"].get("learning", {})
    if not learning_config.get("enabled", True):
        return END

    # Step 0b (no-memory gate, learning-router chokepoint, SESSION-SCOPED): if a
    # no-memory-tagged tool ran this iteration OR the session is contaminated
    # (a sentinel from a prior turn no-memory tool dispatch is still in
    # state messages), REFUSE ALL FOUR routes -- correction / auto-memory /
    # idea / AND session-summary -- (return END). Those routes transcribe tool
    # results / the conversation into durable memory, which a no-memory-tagged
    # tool results must not reach. The session-summary route is NO LONGER
    # allowed on the rationale the scrub handles it (4-lens re-verify
    # 2026-07-07 lens A/D HIGH finding): the scrub drops sentinel-carrying TOOL
    # messages but NOT an LLM-authored PARAPHRASE (no sentinel -- the LLM saw the
    # no-memory-tagged content in-context + relayed it), so a contaminated
    # session summary would leak the paraphrase into the session history. The
    # sentinel-only scrub cannot catch an LLM paraphrase; only a session-level
    # quarantine can. The per-iteration flag covers the same-turn case;
    # _session_has_no_memory_history covers the cross-iteration / cross-turn
    # case (the flag resets each execute_tools entry; the sentinel in
    # state messages persists across turns within a CLI session). The
    # quarantine is bounded to the session (a fresh hermes echo / clear
    # starts uncontaminated).
    _nm_iter = state.get("no_memory_tags_this_iteration", set())
    _nm_block = bool(_nm_iter) or _session_has_no_memory_history(state.get("messages", []))

    # Priority: corrections > memory facts > idea > session action
    if learning_config.get("correction_reflection", True) and state.get("correction_context"):
        if _nm_block:
            logger.info("Echo: no-memory gate blocking correction route (iter=%s); END", sorted(_nm_iter))
            return END
        logger.info("Echo: routing to consolidate_learning (correction)")
        return "consolidate_learning"

    if learning_config.get("auto_memory", True) and state.get("pending_memory_fact"):
        if _nm_block:
            logger.info("Echo: no-memory gate blocking auto-memory route (iter=%s); END", sorted(_nm_iter))
            return END
        logger.info("Echo: routing to consolidate_learning (auto-memory)")
        return "consolidate_learning"

    if state.get("pending_idea"):
        if _nm_block:
            logger.info("Echo: no-memory gate blocking idea route (iter=%s); END", sorted(_nm_iter))
            return END
        logger.info("Echo: routing to consolidate_learning (idea)")
        return "consolidate_learning"

    if learning_config.get("session_summary", True) and state.get("pending_session_action") == "summarize":
        if _nm_block:
            # Session summary is REFUSED when contaminated: the scrub drops
            # sentinel-carrying TOOL messages but NOT an LLM-authored paraphrase
            # (no sentinel), so a contaminated transcript would leak the
            # paraphrase into the session history. The quarantine (not the scrub)
            # is the load-bearing stop.
            logger.info("Echo: no-memory gate blocking session-summary route (iter=%s); END", sorted(_nm_iter))
            return END
        logger.info("Echo: routing to consolidate_learning (session summary, uncontaminated)")
        return "consolidate_learning"

    logger.info("Echo: no learning actions pending, ending")
    return END


def consolidate_learning(state: EchoState) -> EchoState:
    """Node 7: Consolidate whichever learning action is pending.

    Dispatches to the appropriate consolidation function based on state flags.
    Errors are logged but never block the user.
    """
    ollama_config = _get_ollama_config(state)
    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    history_dir = Path(state["config"].get("history_dir", str(Path.home() / ".hermes" / "history" / "echo")))
    store = MemoryStore(memory_dir)

    # 1. Correction reflection
    correction = state.get("correction_context")
    if correction:
        logger.info("Echo: consolidating correction reflection")
        success = consolidate_correction(store, correction, ollama_config)
        if success:
            # Accumulate correction in messages for batching, then clear
            pass
        state["correction_context"] = None
        return state

    # 2. Auto-memory consolidation
    fact = state.get("pending_memory_fact")
    if fact:
        logger.info("Echo: consolidating auto-memory")
        success = consolidate_fact(store, fact, ollama_config)
        if success:
            state["auto_memory_count"] = state.get("auto_memory_count", 0) + 1
        state["pending_memory_fact"] = None
        return state

    # 3. Idea capture
    idea_text = state.get("pending_idea")
    if idea_text:
        logger.info("Echo: consolidating idea")
        # Build transcript from idea_start_index to current
        messages = state.get("messages", [])
        start_idx = state.get("idea_start_index", 0)
        # Step 0b (no-memory gate, idea-transcript scrub): DROP messages carrying
        # a NO_MEMORY sentinel before building the transcript, so a no-memory-
        # tagged tool results never reach the local Ollama that writes durable
        # idea memory. Centralized helper (also used by the session branch
        # below).
        messages = _scrub_no_memory_from_messages(messages[start_idx:])
        transcript_parts = []
        for msg in messages:
            transcript_parts.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
        transcript = "\n".join(transcript_parts)

        consolidate_idea(store, transcript, ollama_config)
        state["pending_idea"] = None
        state["idea_active"] = False
        state["idea_start_index"] = None
        return state

    # 4. Session summary
    if state.get("pending_session_action") == "summarize":
        logger.info("Echo: consolidating session summary")
        messages = state.get("messages", [])
        # Step 0b (no-memory gate, session-transcript scrub): DROP sentinel-
        # carrying messages from the FULL session transcript before the local-
        # Ollama call, so no-memory-tagged tool results do not flow into the
        # session history. DEFENSE-IN-DEPTH: learning_router now REFUSES the
        # session-summary route when the session is contaminated, so this branch
        # only runs on an UNCONTAMINATED transcript. The scrub remains as belt-
        # and-suspenders for any gate gap.
        messages = _scrub_no_memory_from_messages(messages)
        transcript_parts = []
        for msg in messages:
            transcript_parts.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
        transcript = "\n".join(transcript_parts)

        if transcript.strip():
            consolidate_session(store, history_dir, transcript, ollama_config)
        state["pending_session_action"] = None
        return state

    return state


def create_echo_graph(latin=None):
    """Build and compile the Echo agent LangGraph.

    Returns a compiled StateGraph ready for graph.invoke(state).

    latin: optional truthy flag (2026-07-12, DESIGN.md S7). When set, a
    load_latin_state node is inserted at the ENTRY of the turn so the
    paedagogus persona + structured mastery block are available to call_llm:

        process_input -> load_latin_state -> call_llm

    When latin is falsy, the entry edge is unchanged
    (process_input -> call_llm). The load_latin_state node reads only the
    non-protected latin ledger at HERMES_LATIN_DIR (axis D).
    """
    workflow = StateGraph(EchoState)

    workflow.add_node("process_input", process_input)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("format_response", format_response)
    workflow.add_node("consolidate_learning", consolidate_learning)

    workflow.set_entry_point("process_input")
    if latin:
        # Lazy import: the latin ledger reader is only needed for --latin.
        from hermes_cli.agents.echo.latin_state import load_latin_state
        workflow.add_node("load_latin_state", load_latin_state)
        workflow.add_edge("process_input", "load_latin_state")
        workflow.add_edge("load_latin_state", "call_llm")
    else:
        workflow.add_edge("process_input", "call_llm")

    workflow.add_conditional_edges(
        "call_llm",
        router,
        {
            "execute_tools": "execute_tools",
            "format_response": "format_response",
        },
    )

    workflow.add_edge("execute_tools", "call_llm")

    # Original structure: format_response -> learning_router -> consolidate_learning or END
    workflow.add_conditional_edges(
        "format_response",
        learning_router,
        {
            "consolidate_learning": "consolidate_learning",
            END: END,
        },
    )
    workflow.add_edge("consolidate_learning", END)

    return workflow.compile()


# Phase 0b-seam (axis-D): construction self-checks at module load. Prove by AST
# inspection that the prompt-path functions call_llm and router read /
# reference no affect (no banned subscript/.get/attribute read, no affect/
# banned substrate name ref, no affect import). If either raises, importing this module
# fails -- the agent cannot start ungated (fail-closed). No-op in the public build
# (guard is None); activates with the private safety package.
if _PROMPT_GUARD is not None:
    _PROMPT_GUARD.assert_function_reads_no_affect(call_llm, label="call_llm")
    _PROMPT_GUARD.assert_function_reads_no_affect(router, label="router")
    # C-reliability classifier (2026-07-13): the 3 helpers below are CALLED from
    # call_llm (a certified prompt-path function) but assert_function_reads_no_affect
    # scans only the caller own body, not helpers referenced by bare-name Call
    # nodes. Without these explicit attestations, a future hand-edit adding a
    # banned affect read inside a helper would NOT brick import. Attesting each
    # helper here closes that gap for these 3.
    _PROMPT_GUARD.assert_function_reads_no_affect(_is_quota_error, label="_is_quota_error")
    _PROMPT_GUARD.assert_function_reads_no_affect(_backoff_seconds, label="_backoff_seconds")
    _PROMPT_GUARD.assert_function_reads_no_affect(_retry_wait_seconds, label="_retry_wait_seconds")

# P0-9c (2026-07-06 red-team): runtime seam-integrity self-check. Re-hash every
# seam source file scratch-vs-live and raise if any diverge. A stale/partial/
# hand-edited deploy fails closed at import, same doctrine as the prompt-guard
# checks above. Scratch is the single source of truth. Stdlib-only (runs before
# the graph is built; no new heavy deps, no new exfil channel -- it never logs
# file contents, only relative paths + 12-hex prefixes). If scratch is absent
# verify_integrity returns None -- soft-fail with a warning rather than lock out
# the operator.
from .seam_manifest import verify_integrity as _verify_seam_integrity
_seam_mismatch = _verify_seam_integrity()
if _seam_mismatch is None:
    logger.warning(
        "axis-D seam integrity: scratch dir absent -- unverifiable, continuing."
    )
elif _seam_mismatch:
    raise RuntimeError(
        "axis-D seam integrity check FAILED: live seam files diverge from "
        "scratch (stale/partial/hand-edited deploy = a reopened axis-D "
        "channel). Re-run apply_seam.sh to re-deploy, or restore scratch if "
        "it was edited unintentionally. Mismatches (live_rel, scratch, live): "
        f"{_seam_mismatch}"
    )

# P0-6 (2026-07-06 red-team, audit #10/L25): universal affect-cert registration
# attestation. The _register_tool chokepoint already refuses a silent opt-out
# (a handler-bearing SeamedTool whose _affect_cert_ok is False) at every
# _build_registry call; this one-shot import-time check fires it at IMPORT
# (same doctrine as the P0-9c seam-integrity block + the prompt-guard self-
# checks above) so a hand-edited registration that hides a real handler behind
# requires_affect_cert=False bricks the import, not the first turn. Fails closed
# (RuntimeError) on any handler-bearing registered tool with _affect_cert_ok
# False. handler=None (memory) is exempt -- the ONE sanctioned opt-out.
def _verify_registry_affect_cert(reg=None) -> None:
    if reg is None:
        reg = _build_registry()
    for _tinfo in reg.list_tools():
        _t = reg.get(_tinfo["name"])
        if _t is None:
            continue
        if getattr(_t, "handler", None) is not None and not getattr(_t, "_affect_cert_ok", False):
            raise RuntimeError(
                "axis-D P0-6 attestation FAILED: registered tool "
                f"{_tinfo['name']!r} has a handler but _affect_cert_ok is False "
                "(silent affect-cert opt-out in the live registry). See S8.1."
            )
        # P0-6 ceiling attestation: a handler-bearing tool with
        # execution_sandbox none MUST carry a non-empty rationale (an
        # in-process opt-out must be justified). handler=None (memory) is
        # exempt.
        if (getattr(_t, "handler", None) is not None
                and getattr(_t, "execution_sandbox", "none") == "none"
                and not getattr(_t, "execution_sandbox_rationale", "")):
            raise RuntimeError(
                "axis-D P0-6 ceiling attestation FAILED: registered tool "
                f"{_tinfo['name']!r} has a handler but execution_sandbox none "
                f"with no execution_sandbox_rationale (unjustified in-process "
                f"opt-out in the live registry). See S8.1."
            )
        # Allowlist attestation: a typo invalid execution_sandbox in the live
        # registry must fail at IMPORT, not first run.
        if (getattr(_t, "handler", None) is not None
                and getattr(_t, "execution_sandbox", "none")
                not in _EXECUTION_SANDBOX_VALUES):
            raise RuntimeError(
                "axis-D P0-6 ceiling attestation FAILED: registered tool "
                f"{_tinfo['name']!r} has an INVALID execution_sandbox="
                f"{getattr(_t, 'execution_sandbox', None)!r} (not in "
                f"{_EXECUTION_SANDBOX_VALUES}). See S8.1."
            )
        # no_memory_tags allowlist attestation (mirror of the _register_tool
        # Step 0b check, same doctrine as the siblings above): a tool declaring
        # a tag outside the closed _NO_MEMORY_TAGS allowlist would NOT be
        # scrubbed/refused by the allowlist-keyed sentinel scan, so its results
        # would silently persist into memory. Fail at import. handler=None
        # (memory) is exempt.
        _nmt = getattr(_t, "no_memory_tags", ())
        if (getattr(_t, "handler", None) is not None and _nmt
                and not set(_nmt).issubset(_NO_MEMORY_TAGS)):
            raise RuntimeError(
                "axis-D Step 0b attestation FAILED: registered tool "
                f"{_tinfo['name']!r} declares no_memory_tags={tuple(_nmt)!r} "
                f"with a tag outside the closed allowlist {set(_NO_MEMORY_TAGS)!r} "
                f"(an undeclared tag is not scrubbed -> silent persist). See §8.1."
            )


_verify_registry_affect_cert()
