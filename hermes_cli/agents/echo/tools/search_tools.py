"""Code search tools for the Echo agent."""

import fnmatch
import subprocess
from pathlib import Path
from typing import Optional


def search_code(pattern: str, search_type: str = "grep", path: Optional[str] = None) -> str:
    """Search for files or content in a directory tree.

    Args:
        pattern: The glob pattern (for glob) or regex pattern (for grep).
        search_type: Either 'glob' (filename matching) or 'grep' (content search).
        path: Base directory to search. Defaults to current working directory.

    Returns:
        Newline-separated list of matches, or a "not found" message.
    """
    base_path = Path(path) if path else Path.cwd()

    if search_type == "glob":
        matches = []
        for f in base_path.rglob("*"):
            if f.is_file() and fnmatch.fnmatch(f.name, pattern):
                matches.append(str(f.relative_to(base_path)))
        return "\n".join(matches) if matches else "No files found."

    elif search_type == "grep":
        try:
            result = subprocess.run(
                ["grep", "-r", "-n", "--include=*", pattern, str(base_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip() if result.stdout else f"No matches for '{pattern}'."
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
