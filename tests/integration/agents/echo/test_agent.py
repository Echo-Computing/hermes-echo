"""Integration tests for the Echo agent LangGraph — smoke tests only."""

import pytest
from pathlib import Path

from hermes_cli.agents.echo.agent import create_echo_graph
from hermes_cli.agents.echo.state import EchoState


class TestEchoGraph:
    """Smoke tests that verify the graph compiles and basic routing works."""

    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = create_echo_graph()
        assert graph is not None

    def test_graph_with_simple_state(self):
        """Graph should process a simple state (may fail if Ollama not running)."""
        graph = create_echo_graph()
        state = EchoState(
            user_input="Hello",
            config={
                "api_url": "http://localhost:11434/api/chat",
                "model": "qwen3.6:35b",
                "max_tool_calls": 3,
                "context_messages": 50,
                "memory_dir": "/tmp/test_echo_graph",
                "max_tokens": 256,
                "temperature": 0.7,
            },
            messages=[],
        )

        result = graph.invoke(state)

        # Graph should complete (will contain error if Ollama not running, but should finish)
        assert "response" in result or result.get("should_continue") is False
