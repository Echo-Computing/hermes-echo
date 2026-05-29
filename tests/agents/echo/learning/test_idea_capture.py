"""Tests for idea capture module."""

import pytest
from hermes_cli.agents.echo.learning.idea_capture import (
    build_exploration_prompt,
    build_idea_save_prompt,
    parse_idea_response,
)


def test_build_exploration_prompt_content():
    prompt = build_exploration_prompt()
    assert "exploring a project idea" in prompt
    assert "Ask clarifying questions" in prompt
    assert "trade-offs" in prompt
    assert "Don't propose solutions" in prompt


def test_build_idea_save_prompt_format():
    transcript = "user: build a chess engine\nhermes: What kind of chess engine?"
    prompt = build_idea_save_prompt(transcript)

    assert "Extract a structured project concept" in prompt
    assert transcript in prompt
    assert "---" in prompt
    assert "name:" in prompt
    assert "type: project" in prompt
    assert "**Goal:**" in prompt
    assert "**Context:**" in prompt
    assert "**Constraints:**" in prompt
    assert "**Approaches explored:**" in prompt
    assert "**Decisions made:**" in prompt
    assert "**Open questions:**" in prompt


def test_parse_idea_response_valid():
    response = """---
name: chess-ml-engine
description: From-scratch Python chess engine for ML training
metadata:
  type: project
---

**Goal:** Build a chess engine from scratch for ML training purposes.
**Context:** Long-term project to understand chess engine internals.
**Constraints:** Python, from-scratch, long-term timeline.
**Approaches explored:** Bitboard vs mailbox representation, UCI protocol vs GUI.
**Decisions made:** Bitboard representation, UCI protocol first.
**Open questions:** Target playing strength, opening book strategy."""

    parsed = parse_idea_response(response)
    assert parsed["name"] == "chess-ml-engine"
    assert parsed["description"] == "From-scratch Python chess engine for ML training"
    assert parsed["type"] == "project"
    assert "**Goal:**" in parsed["content"]
    assert "Bitboard" in parsed["content"]


def test_parse_idea_response_no_frontmatter():
    response = "Just a raw project idea without proper formatting."
    parsed = parse_idea_response(response)
    assert parsed["name"] == "project-idea"
    assert parsed["type"] == "project"
    assert parsed["content"] == response
