"""Tests for the tool registry and XML parser."""

import pytest
from hermes_cli.agents.echo.tools.registry import ToolRegistry, Tool


def test_parse_valid_tool_call():
    reg = ToolRegistry()
    text = """Some text
<tool_call>
  <name>read_file</name>
  <parameters>
    <path>/home/test.py</path>
  </parameters>
</tool_call>
More text"""
    calls = reg.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["parameters"]["path"] == "/home/test.py"


def test_parse_multiple_tool_calls():
    reg = ToolRegistry()
    text = """<tool_call>
  <name>run_shell</name>
  <parameters>
    <command>ls -la</command>
  </parameters>
</tool_call>
<tool_call>
  <name>read_file</name>
  <parameters>
    <path>/etc/hosts</path>
  </parameters>
</tool_call>"""
    calls = reg.parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "run_shell"
    assert calls[1]["name"] == "read_file"


def test_parse_no_tool_calls():
    reg = ToolRegistry()
    text = "Just a normal response without any tools."
    assert reg.has_tool_calls(text) is False
    assert reg.parse_tool_calls(text) == []


def test_parse_malformed_tool_call():
    reg = ToolRegistry()
    text = "<tool_call><name>broken</name><parameters><path>"  # missing closing tags
    calls = reg.parse_tool_calls(text)
    assert len(calls) == 0


def test_execute_missing_tool():
    reg = ToolRegistry()
    result = reg.execute("nonexistent", {})
    assert result["success"] is False
    assert "not found" in result["error"]


def test_execute_successful():
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="Echo back",
        parameters=[{"name": "message", "type": "string", "required": True, "description": "Message"}],
        handler=lambda message: f"Echo: {message}",
    ))
    result = reg.execute("echo", {"message": "hello"})
    assert result["success"] is True
    assert result["output"] == "Echo: hello"


def test_execute_handler_raises():
    reg = ToolRegistry()
    def failing_handler():
        raise ValueError("boom")
    reg.register(Tool(
        name="failer",
        description="Always fails",
        parameters=[],
        handler=failing_handler,
    ))
    result = reg.execute("failer", {})
    assert result["success"] is False
    assert "boom" in result["error"]


def test_register_and_list_tools():
    reg = ToolRegistry()
    reg.register(Tool(
        name="tool1",
        description="First tool",
        parameters=[],
        handler=lambda: None,
    ))
    reg.register(Tool(
        name="tool2",
        description="Second tool",
        parameters=[{"name": "x", "type": "int", "required": True, "description": "X param"}],
        handler=lambda x: x,
    ))
    tools = reg.list_tools()
    assert len(tools) == 2
    names = [t["name"] for t in tools]
    assert "tool1" in names
    assert "tool2" in names
