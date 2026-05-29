"""Integration tests for learning flow using the full agent graph.

These tests use mock Ollama responses via monkeypatch to avoid requiring
a running Ollama instance. They verify the full graph flow without LLM calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from hermes_cli.agents.echo.agent import create_echo_graph
from hermes_cli.agents.echo.state import EchoState


def make_mock_response(content):
    """Create a mock httpx response object."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"message": {"content": content}}
    return mock


@pytest.fixture
def base_config(tmp_path):
    memory_dir = tmp_path / "memory"
    history_dir = tmp_path / "history"
    return {
        "api_url": "http://localhost:11434/api/chat",
        "model": "qwen3.6:35b",
        "max_tokens": 4096,
        "temperature": 0.7,
        "max_tool_calls": 10,
        "context_messages": 50,
        "shell_timeout": 120,
        "confirm_destructive": False,
        "memory_dir": str(memory_dir),
        "history_dir": str(history_dir),
        "learning": {
            "enabled": True,
            "auto_memory": True,
            "auto_memory_max_per_session": 2,
            "correction_reflection": True,
            "session_summary": True,
            "history_search": True,
            "history_search_limit": 10,
        },
    }


def test_basic_message_flow(base_config):
    """Test that normal messages still work with the new graph (no learning triggers)."""
    graph = create_echo_graph()

    state = EchoState(
        config=base_config,
        messages=[],
        user_input="What is Python?",
    )

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        mock_post.return_value = make_mock_response("Python is a programming language.")

        final_state = graph.invoke(state)

    assert final_state["response"] == "Python is a programming language."
    assert len(final_state["messages"]) == 2  # user + assistant
    assert final_state["messages"][0]["role"] == "user"
    assert final_state["messages"][1]["role"] == "assistant"


def test_correction_detection_in_graph(base_config):
    """Test that correction detection triggers and the graph handles it."""
    graph = create_echo_graph()

    # First turn: assistant says something
    state = EchoState(
        config=base_config,
        messages=[],
        user_input="How do I run a shell command?",
    )

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        mock_post.return_value = make_mock_response("Use os.system to run the command.")

        state = graph.invoke(state)

    assert state["response"] == "Use os.system to run the command."

    # Second turn: user corrects
    state["user_input"] = "no, use subprocess instead"
    state["response"] = None

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        # First call: chat response
        # Second call: correction reflection (via consolidate_learning)
        mock_post.side_effect = [
            make_mock_response("You are right, subprocess is better. I will use that."),
            make_mock_response("""---
name: use-subprocess
description: Prefer subprocess over os.system
metadata:
  type: feedback
---

Prefer subprocess.run over os.system.
**Why:** More secure and modern.
**How to apply:** Use subprocess.run for shell commands."""),
        ]

        final_state = graph.invoke(state)

    assert "subprocess" in final_state["response"].lower()
    # Verify correction context was processed (state flag cleared)
    assert final_state.get("correction_context") is None


def test_idea_flow(base_config):
    """Test /idea exploration mode in the graph."""
    graph = create_echo_graph()

    state = EchoState(
        config=base_config,
        messages=[],
        user_input="/idea build a chess engine",
        idea_active=True,
        idea_start_index=0,
        pending_idea=None,
    )

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        mock_post.return_value = make_mock_response(
            "What kind of chess engine? ML experiment, competitive, or something else?"
        )

        state = graph.invoke(state)

    assert "chess engine" in state["response"].lower()
    assert state.get("idea_active") is True


def test_idea_save_flow(base_config, tmp_path):
    """Test /idea save consolidation."""
    graph = create_echo_graph()

    # Simulate after a conversation about the idea
    state = EchoState(
        config=base_config,
        messages=[
            {"role": "user", "content": "/idea build a chess engine"},
            {"role": "assistant", "content": "What kind? ML or competitive?"},
            {"role": "user", "content": "ML experiment, Python, long-term"},
            {"role": "assistant", "content": "Good. Bitboard or mailbox?"},
            {"role": "user", "content": "/idea save"},
        ],
        user_input="/idea save",
        idea_active=True,
        idea_start_index=0,
        pending_idea="build a chess engine",
    )

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        # First call: chat response to /idea save
        # Second call: idea consolidation
        mock_post.side_effect = [
            make_mock_response("Saved as chess-ml-engine. I have captured everything."),
            make_mock_response("""---
name: chess-ml-engine
description: From-scratch Python chess engine for ML training
metadata:
  type: project
---

**Goal:** Build a chess engine from scratch for ML training.
**Context:** Long-term project, Python, from-scratch.
**Constraints:** RTX 2070 GPU, Python 3.12.
**Approaches explored:** Bitboard vs mailbox.
**Decisions made:** Bitboard representation.
**Open questions:** Target strength, opening book."""),
        ]

        final_state = graph.invoke(state)

    assert "Saved" in final_state["response"]
    assert final_state.get("idea_active") is False
    assert final_state.get("pending_idea") is None
    assert final_state.get("idea_start_index") is None

    # Verify memory was written
    project_dir = tmp_path / "memory" / "projects"
    assert project_dir.exists()
    md_files = list(project_dir.glob("*.md"))
    assert len(md_files) >= 1


def test_exit_session_summary(base_config, tmp_path):
    """Test /exit session summary flow."""
    graph = create_echo_graph()

    state = EchoState(
        config=base_config,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Help me with Python"},
            {"role": "assistant", "content": "Sure, what do you need?"},
        ],
        user_input="/exit",
        pending_session_action="summarize",
    )

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        # First call: chat response (goodbye)
        # Second call: session summary
        mock_post.side_effect = [
            make_mock_response("Session saved. Goodbye."),
            make_mock_response("""## Summary
Chatted about Python.

## Key topics
- Python
- help

## Decisions made
- None

## Corrections & learnings
- None

## New facts about the user
- None

## Carry-forward
Nothing specific."""),
        ]

        final_state = graph.invoke(state)

    assert "Goodbye" in final_state["response"]

    # Verify JSONL was written
    history_dir = tmp_path / "history"
    jsonl_files = list(history_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    with open(jsonl_files[0]) as f:
        first_line = json.loads(f.readline())
        assert first_line["type"] == "summary"


def test_learning_disabled(base_config):
    """Test that learning can be disabled via config."""
    base_config["learning"]["enabled"] = False
    graph = create_echo_graph()

    state = EchoState(
        config=base_config,
        messages=[],
        user_input="no, use subprocess instead",
        correction_context={"user_msg": "no, use subprocess instead", "prior_response": "Use os.system"},
    )

    with patch('hermes_cli.agents.echo.agent.httpx.post') as mock_post:
        # Only one call: the chat response (no consolidation)
        mock_post.return_value = make_mock_response("OK, I will use subprocess.")

        final_state = graph.invoke(state)

    assert "subprocess" in final_state["response"].lower()
    # Correction context should persist since consolidation was skipped
    # (the graph still detected it but did not route to consolidation)
