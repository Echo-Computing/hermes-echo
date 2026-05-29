"""Tests for correction micro-reflection."""

import pytest
from pathlib import Path
from hermes_cli.agents.echo.learning.reflector import (
    build_reflection_prompt,
    parse_reflection_response,
)


def test_build_reflection_prompt_format():
    context = {
        "user_msg": "no, use subprocess instead of os.system",
        "prior_response": "I'll use os.system to run that command",
    }
    prompt = build_reflection_prompt(context)

    assert "no, use subprocess instead of os.system" in prompt
    assert "I'll use os.system to run that command" in prompt
    assert "---" in prompt
    assert "name:" in prompt
    assert "type: feedback" in prompt
    assert "**Why:**" in prompt
    assert "**How to apply:**" in prompt


def test_parse_reflection_response_valid():
    response = """---
name: use-subprocess-instead
description: Prefer subprocess over os.system for shell commands
metadata:
  type: feedback
---

Prefer subprocess.run over os.system for executing shell commands.
**Why:** os.system is deprecated and less secure.
**How to apply:** Use subprocess.run with a list of arguments for any shell command execution."""

    parsed = parse_reflection_response(response)
    assert parsed["name"] == "use-subprocess-instead"
    assert parsed["description"] == "Prefer subprocess over os.system for shell commands"
    assert parsed["type"] == "feedback"
    assert "subprocess.run" in parsed["content"]
    assert "**Why:**" in parsed["content"]


def test_parse_reflection_response_no_frontmatter():
    response = "Just a plain rule: use subprocess instead of os.system."
    parsed = parse_reflection_response(response)
    assert parsed["name"] == "correction"
    assert parsed["type"] == "feedback"
    assert "use subprocess" in parsed["content"]
