"""File operation tools for the Echo agent.

Provides read_file, write_file, and edit_file — the three core file manipulation
tools the agent uses to interact with the filesystem.
"""

from pathlib import Path
from typing import Optional


def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Read a file from the filesystem, optionally with offset and limit.

    Args:
        path: Absolute path to the file.
        offset: Optional starting line number (0-based).
        limit: Optional maximum number of lines to return.

    Returns:
        The file content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    if offset is not None:
        lines = lines[offset:]
    if limit is not None:
        lines = lines[:limit]

    return "\n".join(lines)


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with new content.

    Args:
        path: Absolute path to the file.
        content: The content to write.

    Returns:
        A confirmation message.
    """
    file_path = Path(path)
    file_path.write_text(content, encoding="utf-8")
    return f"File written: {path}"


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace a string in an existing file.

    By default only the first occurrence of old_string is replaced; pass
    replace_all=True to replace every occurrence.

    Args:
        path: Absolute path to the file.
        old_string: The exact text to replace.
        new_string: The replacement text.
        replace_all: If True, replace every occurrence; if False (default),
            replace only the first.

    Returns:
        A confirmation message.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If old_string is not found in the file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = file_path.read_text(encoding="utf-8")
    if old_string not in content:
        raise ValueError(f"old_string not found in file: {old_string[:80]}...")

    # Tool-call XML delivers parameters as strings, so a bool param arrives as
    # "true"/"false" (a non-empty string is truthy -> a naive `if replace_all:`
    # would treat "false" as True). Coerce robustly. NOTE: str.replace(..., 0)
    # replaces NOTHING (count=0 is a zero-budget cap, not "all"), so the count
    # switch is an if/else, not `0 if replace_all else 1`.
    if isinstance(replace_all, str):
        replace_all = replace_all.strip().lower() in ("true", "1", "yes", "on")
    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
    file_path.write_text(new_content, encoding="utf-8")
    return f"File edited: {path}"
