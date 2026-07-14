"""Tests for the Echo agent system prompt builder."""

from hermes_cli.agents.echo.system_prompt import build_system_prompt


def test_build_system_prompt_with_tools_and_memory():
    tools = [
        {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": [
                {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
                {"name": "offset", "type": "int", "required": False, "description": "Start line number"},
            ],
        },
        {
            "name": "run_shell",
            "description": "Execute a shell command",
            "parameters": [
                {"name": "command", "type": "string", "required": True, "description": "Shell command to execute"},
            ],
        },
    ]
    memory = ["GPU: RTX 2070 with 8GB VRAM", "User: test_user"]

    prompt = build_system_prompt(tools, memory)

    # Should contain identity
    assert "Hermes" in prompt
    assert "Echo Agent" in prompt
    assert "test_user" in prompt

    # Should contain tool definitions
    assert "read_file" in prompt
    assert "run_shell" in prompt
    assert "<tools>" in prompt
    assert "</tools>" in prompt
    assert '<tool name="read_file">' in prompt
    assert '<tool name="run_shell">' in prompt

    # Should contain tool call instructions
    assert "<tool_call>" in prompt

    # Should contain memory
    assert "Loaded Memory" in prompt
    assert "RTX 2070" in prompt

    # Should contain behavior rules
    assert "Behavior Rules" in prompt

    # Should contain constraints
    assert "Constraints" in prompt

    # Should be wrapped in <system> tags
    assert prompt.startswith("<system>")


def test_build_system_prompt_no_memory():
    prompt = build_system_prompt([], [])

    # No memory section when empty
    assert "Loaded Memory" not in prompt

    # Still has tools section even with no tools
    assert "<tools>" in prompt

    # Still has identity
    assert "Hermes" in prompt


def test_build_system_prompt_tool_with_no_parameters():
    tools = [
        {
            "name": "get_time",
            "description": "Get current time",
            "parameters": [],
        }
    ]
    prompt = build_system_prompt(tools, [])
    assert "get_time" in prompt
    assert "<parameters>" in prompt
