"""Shell command execution tool for the Echo agent."""

import subprocess
from pathlib import Path
from typing import Optional


def run_shell(command: str, timeout: int = 120, cwd: Optional[str] = None) -> str:
    """Execute a shell command and return the combined output.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds (default 120).
        cwd: Optional working directory for the command.

    Returns:
        Combined stdout, stderr, and exit code of the command.
    """
    working_dir = Path(cwd) if cwd else Path.cwd()

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(working_dir),
    )

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"

    return output.strip() or "(no output)"
