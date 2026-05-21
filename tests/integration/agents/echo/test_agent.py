""""Integration smoke tests for the Echo agent."""

import pytest

from hermes_cli.agents.echo.agent import create_echo_graph
from hermes_cli.agents.echo.state import EchoState


@pytest.mark.integration
def test_echo_graph_compiles():
    """Verify the LangGraph compiles without errors."""
    graph = create_echo_graph()
    assert graph is not None


@pytest.mark.integration
def test_echo_graph_handles_missing_ollama():
    """Graph should complete even when Ollama is not reachable."""
    graph = create_echo_graph()

    state = EchoState(
        user_input="Hello",
        config={
            "api_url": "http://localhost:11434/api/chat",
            "model": "qwen3.6:35b",
            "max_tool_calls": 3,
            "context_messages": 50,
            "memory_dir": "/tmp/test_echo_smoke",
            "max_tokens": 64,
            "temperature": 0.7,
        },
        messages=[],
    )

    result = graph.invoke(state)
    assert "response" in result
    assert result.get("should_continue") is False

