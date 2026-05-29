"""Tests for auto-memory consolidation."""

import pytest
from pathlib import Path
from hermes_cli.agents.echo.learning.auto_memory import (
    build_auto_memory_prompt,
    parse_auto_memory_response,
)


def test_build_auto_memory_prompt_format():
    prompt = build_auto_memory_prompt("I prefer using async/await over raw callbacks")
    assert "I prefer using async/await over raw callbacks" in prompt
    assert "---" in prompt
    assert "name:" in prompt
    assert "type:" in prompt
    assert "**Why:**" in prompt
    assert "user | feedback | project | reference" in prompt


def test_parse_auto_memory_response_user_type():
    response = """---
name: async-await-preference
description: Prefer async/await over raw callbacks for async code
metadata:
  type: user
---

Always use async/await for asynchronous operations instead of raw callbacks.
**Why:** async/await is more readable and easier to debug.
**How to apply:** Use async/await in all new async code."""

    parsed = parse_auto_memory_response(response)
    assert parsed["name"] == "async-await-preference"
    assert parsed["description"] == "Prefer async/await over raw callbacks for async code"
    assert parsed["type"] == "user"
    assert "async/await" in parsed["content"]


def test_parse_auto_memory_response_project_type():
    response = """---
name: project-architecture
description: Project uses microservices architecture
metadata:
  type: project
---

The project follows a microservices architecture with event-driven communication.
**Why:** Scalability and team autonomy.
**How to apply:** New features should be built as separate services."""

    parsed = parse_auto_memory_response(response)
    assert parsed["name"] == "project-architecture"
    assert parsed["type"] == "project"


def test_parse_auto_memory_response_no_frontmatter():
    response = "The user prefers Python 3.12 for all projects."
    parsed = parse_auto_memory_response(response)
    assert parsed["name"] == "memory"
    assert parsed["type"] == "reference"
    assert "Python 3.12" in parsed["content"]
