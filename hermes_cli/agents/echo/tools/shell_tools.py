"""Shell command execution tool for the Echo agent.

``run_shell`` executes the agent's command as a direct subprocess with a
wall-clock timeout + protected-roots/guard-source path gates. The
mount-namespace ceiling is not in the public build; the floor gates (timeout,
protected-store path gate, guard-source path gate, handler-cert) remain.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


import logging
_log = logging.getLogger(__name__)

_HOME = os.path.expanduser("~")

# Single source of truth for the protected-store PATHS (canonical spellings),
# shared by _PROTECTED_STORE_ROOTS (realpath-level, for the agent.py matcher
# containment). The public build ships with no private cred stores, so
# this tuple is empty -- the protected-store gate is an inert floor, retained
# for extensibility so an operator wiring a private store gets the gate for
# free.
_PROTECTED_STORE_PATHS: tuple = ()

# realpath-level roots for the matcher containment layer (agent.py Layer 2).
# realpath() so a store behind a symlink resolves to its real target -- the
# matcher judges the resolved location, not the spelling.
_PROTECTED_STORE_ROOTS = tuple(os.path.realpath(p) for p in _PROTECTED_STORE_PATHS)

# Guard-source ROOTS (anti-tampering): the dirs whose editing would let
# the tool-capable LLM rewrite its own integrity guard (refusal block, handler-cert).
# Single-sourced HERE (next to _protected_roots) because shell_tools imports
# neither agent.py nor search_tools -- both import _guard_source_roots from
# here, avoiding the agent.py<->search_tools cycle. realpath so a seam behind a
# symlink resolves to its real target. The public build keeps the live seam dir
# + the venv + the latin workspace (a trusted prompt-conditioning source).
_GUARD_SOURCE_ROOTS = tuple(
    os.path.realpath(p)
    for p in (
        os.path.join(_HOME, "hermes-echo", "hermes_cli", "agents", "echo"),  # live seam
        os.path.join(_HOME, "hermes-echo", ".venv"),                        # deployed safety pkg
        # The latin workspace is a
        # TRUSTED prompt-conditioning source. paedagogus.md is interpolated
        # RAW at the top of the --latin SYSTEM prompt (build_latin_system_prompt
        # loads it at build time); ledger.json fields render into the
        # state block that the persona instructs the LLM to open every
        # session from. Treat HERMES_LATIN_DIR as a guard-source root so
        # write_file/edit_file/read_file/search_code refuse to touch
        # paedagogus.md / ledger.json / the data JSONs, closing the in-session
        # LLM-write path to the conditioning surface. The latin deterministic
        # tools (guard_source_policy="none", in-process) bypass this gate and
        # keep reading/writing the ledger directly.
        os.environ.get("HERMES_LATIN_DIR", os.path.join(_HOME, ".hermes", "latin")),
    )
)


def _protected_roots() -> tuple:
    """Resolved realpaths of the protected store roots."""
    return _PROTECTED_STORE_ROOTS


def _guard_source_roots() -> tuple:
    """Resolved realpaths of the guard-source roots (seam + venv + latin)."""
    return _GUARD_SOURCE_ROOTS


def _path_is_contained_in_root(p: str, root: str, allow_ancestor: bool) -> bool:
    """True if resolved path ``p`` is equal to or inside ``root``, or (when
    ``allow_ancestor``) a proper ancestor of ``root``.

      inside-or-equal: commonpath([p, root]) == root   (p is root or under it)
      ancestor:        commonpath([p, root]) == p and p != root  (p is above root)

    This is the SINGLE commonpath-based
    containment primitive, owned by shell_tools (the cycle-free root that already
    owns ``_protected_roots`` / ``_guard_source_roots``). agent.py's
    protected-store matcher + guard-source reference check, and search_tools'
    resolved-in-protected-root check, all call THIS leaf instead of each
    inlining their own ``os.path.commonpath`` (the 3-way duplication that
    motivated this refactor). shell_tools imports neither agent.py nor search_tools, so
    there is no import cycle. The ancestor case closes parent-walk exfil
    (search_code/grep/find on a parent dir that contains the store). A sibling
    dir (e.g. ~/.hermes/memory) is NOT contained -- commonpath is the shared
    parent, equal to neither."""
    try:
        cp = os.path.commonpath([p, root])
    except ValueError:
        # Different drives / mixed absolute-relative -- not contained.
        return False
    if cp == root:
        return True
    if allow_ancestor and cp == p and p != root:
        return True
    return False


def run_shell(command: str, timeout: int = 120, cwd: Optional[str] = None) -> str:
    """Execute a shell command and return the combined output.

    The command runs as a direct subprocess with a wall-clock timeout. The
    protected-roots/guard-source path gates (applied by execute_tools in
    agent.py before this handler is reached) are the floor; the
    mount-namespace ceiling is not in the public build.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds (default 120).
        cwd: Optional working directory for the command.

    Returns:
        Combined stdout, stderr, and exit code of the command.
    """
    working_dir = Path(cwd) if cwd else Path.cwd()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(working_dir),
        )
    except subprocess.TimeoutExpired:
        return f"(command timed out after {timeout}s)"

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"

    return output.strip() or "(no output)"