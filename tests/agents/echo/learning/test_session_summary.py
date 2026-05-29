"""Tests for session summary module."""

import json
import pytest
from pathlib import Path
from hermes_cli.agents.echo.learning.session_summary import (
    build_summary_prompt,
    parse_summary_response,
    write_session_record,
    extract_learnings,
)
from hermes_cli.agents.echo.memory import MemoryStore


def test_build_summary_prompt_format():
    transcript = "user: hello\nhermes: hi there"
    prompt = build_summary_prompt(transcript)
    assert transcript in prompt
    assert "## Summary" in prompt
    assert "## Key topics" in prompt
    assert "## Decisions made" in prompt
    assert "## Corrections & learnings" in prompt
    assert "## New facts about the user" in prompt
    assert "## Carry-forward" in prompt


def test_parse_summary_response_full():
    response = """## Summary
Built a chess engine prototype.

## Key topics
- chess engine design
- bitboard vs mailbox
- Python performance

## Decisions made
- Use bitboard representation for speed
- Start with UCI protocol

## Corrections & learnings
- Use subprocess instead of os.system

## New facts about the user
- Prefers Python 3.12
- Has RTX 2070 GPU

## Carry-forward
Continue with the chess engine implementation."""

    parsed = parse_summary_response(response)
    assert parsed["summary"] == "Built a chess engine prototype."
    assert "chess engine design" in parsed["key_topics"]
    assert len(parsed["key_topics"]) == 3
    assert len(parsed["decisions"]) == 2
    assert len(parsed["corrections"]) == 1
    assert len(parsed["new_facts"]) == 2
    assert "Continue with" in parsed["carry_forward"]


def test_parse_summary_response_partial():
    response = """## Summary
Just did stuff.

## Carry-forward
Nothing to carry forward."""

    parsed = parse_summary_response(response)
    assert parsed["summary"] == "Just did stuff."
    assert parsed["key_topics"] == []
    assert parsed["decisions"] == []
    assert parsed["carry_forward"] == "Nothing to carry forward."


def test_write_session_record(tmp_path):
    history_dir = tmp_path / "history"
    summary = {
        "summary": ["Did some work"],
        "key_topics": ["testing"],
        "decisions": [],
        "corrections": [],
        "new_facts": [],
        "carry_forward": "Keep going",
    }
    transcript = "user: test\nhermes: ok"

    path = write_session_record(history_dir, transcript, summary)
    assert path.exists()
    assert path.suffix == ".jsonl"

    with open(path) as f:
        first_line = json.loads(f.readline())
        assert first_line["type"] == "summary"
        assert first_line["summary"] == ["Did some work"]
        second_line = json.loads(f.readline())
        assert second_line["type"] == "transcript"
        assert second_line["content"] == transcript


def test_extract_learnings(tmp_path):
    store = MemoryStore(tmp_path)
    summary = {
        "corrections": ["Use subprocess instead of os.system"],
        "new_facts": ["Prefers Python 3.12", "Has RTX 2070 GPU"],
    }

    count = extract_learnings(store, summary)
    assert count == 3  # 1 correction + 2 facts

    # Check feedback/ directory
    feedback_files = list(tmp_path.rglob("*.md"))
    # MEMORY.md should exist
    assert (tmp_path / "MEMORY.md").exists()


def test_extract_learnings_empty(tmp_path):
    store = MemoryStore(tmp_path)
    summary = {
        "corrections": [],
        "new_facts": [],
    }

    count = extract_learnings(store, summary)
    assert count == 0
