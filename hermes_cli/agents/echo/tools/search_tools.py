"""Code search tools for the Echo agent.

Containment guard closure: ``search_code`` was
the unsandboxed recursive-read residual. The prior ``allow_ancestor=has_sep`` gate,
applied tool-agnostically in ``agent.py``'s matcher, re-opened the parent-walk
closure for THIS handler: ``grep -r`` (raw subprocess) from ``cwd=home``
rglobs protected store files into the tool-capable LLM, and the matcher's
ancestor rule never fired for ``search_code`` because ``has_sep`` was the wrong
policy for a recursive read (parent-walk IS the attack for search, not a legit
``cd ~``).

A containment pre-check at the top of ``search_code`` closes it: resolve
``base_path`` (``expanduser`` + ``realpath``) and refuse if it is equal-to /
inside / an **ancestor** of any ``_protected_roots()``. This is the load-bearing
closure for the omitted-``path`` / ``path='.'`` / ``'..'`` / ``'~'`` cases the
matcher cannot see (the matcher only sees ``params``, not the handler's ``cwd``
-- omitted ``path`` means ``Path.cwd()`` means ``$HOME`` means ancestor of the
stores, caught HERE not in the matcher). The ``commonpath`` check is imported
from ``shell_tools._path_is_contained_in_root`` (the cycle-free root), not
inlined, avoiding the agent.py<->search_tools import cycle.
"""

import fnmatch
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from hermes_cli.agents.echo.tools.shell_tools import (
    _protected_roots, _guard_source_roots, _path_is_contained_in_root,
)


def _resolved_in_protected_root(resolved: str, roots: tuple,
                                allow_ancestor: bool = True) -> bool:
    """True if ``resolved`` (already realpath'd) is equal-to / inside / (when
    ``allow_ancestor``) a proper ancestor of any protected store root. The
    commonpath containment logic is the SINGLE
    leaf ``shell_tools._path_is_contained_in_root`` (the cycle-free root that
    owns _protected_roots/_guard_source_roots). It was inlined here as the
    third of three commonpath copies -- the drift class this guard prevents.
    shell_tools imports neither agent.py nor search_tools, so importing the
    leaf from it (instead of from agent.py) avoids the agent.py<->search_tools
    import cycle that forced the original inline."""
    return any(
        _path_is_contained_in_root(resolved, root, allow_ancestor)
        for root in roots
    )


def _search_refused_message(resolved: str) -> str:
    """Refusal string for the containment pre-check. Contains 'protected store'
    so leak-probe arm BE can match on it (consistent with the matcher refusal)."""
    return (f"containment guard: search_code refused -- base path resolves to a protected "
            f"store or its ancestor ({resolved!r}). The tool-capable LLM must "
            f"not recursively read a protected store (parent-walk exfil closed).")


def search_code(pattern: str, search_type: str = "grep", path: Optional[str] = None) -> str:
    """Search for files or content in a directory tree.

    Args:
        pattern: The glob pattern (for glob) or regex pattern (for grep).
        search_type: Either 'glob' (filename matching) or 'grep' (content search).
        path: Base directory to search. Defaults to current working directory.

    Returns:
        Newline-separated list of matches, or a "not found" / refusal message.

    A containment pre-check runs BEFORE any filesystem walk. If
    the resolved base path is equal-to / inside / an ancestor of a protected
    store root, the search is refused (parent-walk exfil closed).
    """
    base_path = Path(path).expanduser() if path else Path.cwd()
    _resolved_base = os.path.realpath(str(base_path))
    _roots = _protected_roots()

    # Containment pre-check (load-bearing closure). allow_ancestor=True: a
    # search rooted at $HOME (or /home, or ~) walks INTO the stores, so the base
    # being an ANCESTOR of a store is the attack, not a legit shorthand. This
    # catches omitted-path (cwd=$HOME), '.', '..', '~' -- the cases the matcher
    # cannot see because it only inspects params, not the handler's cwd.
    if _resolved_in_protected_root(_resolved_base, _roots, allow_ancestor=True):
        return _search_refused_message(_resolved_base)

    # Guard-source containment: refuse a search rooted at / inside / an
    # ancestor of a guard-source root (the seam dir, venv, latin workspace).
    # search_code on the guard dir lets the LLM exfil its own guard into the
    # tool-result channel and reason about bypassing it (anti-tampering).
    _groots = _guard_source_roots()
    if _resolved_in_protected_root(_resolved_base, _groots, allow_ancestor=True):
        return (f"containment guard: search_code refused -- base path resolves to a guard/"
                f"seam source or its ancestor ({_resolved_base!r}). The "
                f"tool-capable LLM must not read its own containment guard "
                f"(anti-tampering).")

    if search_type == "glob":
        matches = []
        for f in base_path.rglob("*"):
            if f.is_file() and fnmatch.fnmatch(f.name, pattern):
                # Defense vs a symlinked base whose entries point into a store:
                # skip any match whose realpath is inside a protected root.
                try:
                    _rf = os.path.realpath(str(f))
                except Exception:  # noqa: BLE001
                    continue
                if _resolved_in_protected_root(_rf, _roots, allow_ancestor=False):
                    continue
                matches.append(str(f.relative_to(base_path)))
        return "\n".join(matches) if matches else "No files found."

    elif search_type == "grep":
        # Direct recursive grep (the mount-namespace ceiling is not in the
        # public build; the containment pre-check above is the floor closure).
        # Quote everything so no token is re-globbed by the inner shell.
        try:
            result = subprocess.run(
                [
                    "grep", "-r", "-n",
                    "--include=*",
                    "--",
                    pattern,
                    str(base_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = result.stdout
            if result.returncode != 0 and not out and result.stderr:
                return f"Search error: {result.stderr.strip()[:200]}"
            return out.strip() if out else f"No matches for '{pattern}'."
        except FileNotFoundError:
            return "grep not available. Try 'glob' search_type instead."
        except subprocess.TimeoutExpired:
            return "Search timed out after 30 seconds."

    else:
        return f"Unknown search_type: '{search_type}'. Use 'glob' or 'grep'."


def search_history(query: str, history_dir: str = None, limit: int = 10) -> str:
    """Search past Echo agent session transcripts.

    Greps through ~/.hermes/history/echo/*.jsonl files for matching text.
    Searches both session summaries (first line) and full transcripts.

    Args:
        query: Search term to grep for
        history_dir: Path to history directory (default ~/.hermes/history/echo)
        limit: Max number of matching sessions to return

    Returns:
        Formatted search results
    """
    import json
    from pathlib import Path

    if history_dir is None:
        history_dir = str(Path.home() / ".hermes" / "history" / "echo")

    history_path = Path(history_dir)
    if not history_path.exists():
        return "No session history found. History is created when you /exit a session."

    query_lower = query.lower()
    matches = []

    for jsonl_file in sorted(history_path.glob("*.jsonl"), reverse=True):
        if len(matches) >= limit:
            break

        try:
            content = jsonl_file.read_text(encoding="utf-8")
            lines = content.strip().split("\n")

            # Check first line (summary)
            first_line = lines[0] if lines else ""
            if query_lower in first_line.lower():
                try:
                    summary = json.loads(first_line)
                    matches.append({
                        "date": summary.get("date", jsonl_file.stem),
                        "snippet": summary.get("summary", "")[:200],
                        "topics": summary.get("key_topics", [])[:5],
                    })
                    continue
                except json.JSONDecodeError:
                    pass

            # Check full transcript
            if query_lower in content.lower():
                matches.append({
                    "date": jsonl_file.stem,
                    "snippet": "Match found in transcript",
                    "topics": [],
                })
        except Exception:
            continue

    if not matches:
        return f"No past sessions found matching '{query}'."

    result_lines = ["Past sessions matching your query:"]
    for m in matches:
        topics = ", ".join(m["topics"][:3]) if m["topics"] else "no topics extracted"
        result_lines.append(f"  {m['date']}: {m['snippet'][:150]}")
        if m["topics"]:
            result_lines.append(f"    Topics: {topics}")

    return "\n".join(result_lines)