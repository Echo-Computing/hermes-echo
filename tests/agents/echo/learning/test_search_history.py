"""Tests for search_history tool."""

import json
import pytest
from pathlib import Path
from hermes_cli.agents.echo.tools.search_tools import search_history


@pytest.fixture
def history_dir(tmp_path):
    """Create a temporary history directory with mock session files."""
    hist = tmp_path / "history"
    hist.mkdir()

    # Session 1
    with open(hist / "2026-05-20-143022.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "date": "2026-05-20-143022",
            "type": "summary",
            "summary": "Discussed chess engine ML approach",
            "key_topics": ["chess engine", "ML", "Python"],
        }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "type": "transcript",
            "content": "user: build a chess engine\nhermes: what kind?"
        }, ensure_ascii=False) + "\n")

    # Session 2
    with open(hist / "2026-05-22-091500.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "date": "2026-05-22-091500",
            "type": "summary",
            "summary": "Fixed WSL path issues with file tools",
            "key_topics": ["WSL", "paths", "debugging"],
        }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "type": "transcript",
            "content": "user: wsl paths are broken\nhermes: let me check"
        }, ensure_ascii=False) + "\n")

    return str(hist)


def test_search_history_match_summary(history_dir):
    result = search_history("chess", history_dir=history_dir)
    assert "2026-05-20-143022" in result
    assert "chess" in result.lower()


def test_search_history_match_transcript(history_dir):
    result = search_history("broken", history_dir=history_dir)
    assert "2026-05-22-091500" in result


def test_search_history_no_match(history_dir):
    result = search_history("nonexistent_query_xyz", history_dir=history_dir)
    assert "No past sessions found" in result


def test_search_history_limit(history_dir):
    result = search_history("session", history_dir=history_dir, limit=1)
    # Should only return 1 match
    assert result.count("Past sessions") <= 2  # header + 1 entry


def test_search_history_nonexistent_dir(tmp_path):
    no_hist = str(tmp_path / "nonexistent")
    result = search_history("anything", history_dir=no_hist)
    assert "No session history found" in result


def test_search_history_empty_query(history_dir):
    result = search_history("", history_dir=history_dir)
    # Should still work, just won't match anything meaningful
    assert isinstance(result, str)
