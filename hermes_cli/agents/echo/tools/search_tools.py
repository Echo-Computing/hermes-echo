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
