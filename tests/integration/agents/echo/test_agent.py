"""Integration tests for the Echo agent LangGraph."""

import pytest

from hermes_cli.agents.echo.agent import create_echo_graph
from hermes_cli.agents.echo.state import EchoState


@pytest.mark.integration
def test_echo_graph_builds():
    """Smoke test: the graph compiles without errors."""
    graph = create_echo_graph()
    assert graph is not None


@pytest.mark.integration
def test_echo_graph_routing_no_tool_call():
    """Graph should route to format_response when there are no tool calls."""
    graph = create_echo_graph()

    state = EchoState(
        user_input="Hello!",
        config={
            "api_url": "http://localhost:11434/api/chat",
            "model": "qwen3:8b",
            "max_tool_calls": 3,
            "context_messages": 50,
            "memory_dir": "/tmp/test_echo_memory",
            "max_tokens": 256,
            "temperature": 0.7,
        },
        messages=[],
    )

    # Note: This requires Ollama to be running with qwen3:8b pulled.
    # Without Ollama, the call_llm node will fail gracefully and set should_continue=False.
    result = graph.invoke(state)

    # The graph should complete (response set, should_continue False)
    assert "response" in result
    assert result.get("should_continue") is False


@pytest.mark.integration
def test_echo_graph_state_fields_preserved():
    """Fields should be preserved through the graph execution."""
    graph = create_echo_graph()

    state = EchoState(
        user_input="What is 2+2?",
        config={
            "api_url": "http://localhost:11434/api/chat",
            "model": "qwen3:8b",
            "max_tool_calls": 3,
            "context_messages": 50,
            "memory_dir": "/tmp/test_echo_memory",
            "max_tokens": 256,
            "temperature": 0.7,
        },
        messages=[],
        workspace="/tmp",
    )

    result = graph.invoke(state)

    # User input should be preserved
    assert result.get("user_input") == "What is 2+2?"
    # Workspace should be preserved
    assert result.get("workspace") == "/tmp"
    # Should have completed
    assert result.get("should_continue") is False
