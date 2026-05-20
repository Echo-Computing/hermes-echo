"""Tests for the Echo agent memory tool."""

import pytest
from pathlib import Path
from hermes_cli.agents.echo.tools.memory_tools import MemoryTool


def test_memory_tool_search(tmp_path):
    tool = MemoryTool(tmp_path)
    tool.write("gpu", "GPU info", "RTX 2070 8GB", "knowledge")

    result = tool.search("RTX")
    assert "gpu" in result
    assert "GPU info" in result


def test_memory_tool_search_no_match(tmp_path):
    tool = MemoryTool(tmp_path)
    result = tool.search("nonexistent_xyz")
    assert "No matching" in result


def test_memory_tool_read(tmp_path):
    tool = MemoryTool(tmp_path)
    tool.write("test-entry", "Test", "Some content here", "project")

    result = tool.read("test-entry")
    assert "Some content here" in result


def test_memory_tool_read_not_found(tmp_path):
    tool = MemoryTool(tmp_path)
    result = tool.read("nonexistent")
    assert "not found" in result


def test_memory_tool_write(tmp_path):
    tool = MemoryTool(tmp_path)
    result = tool.write("new-mem", "New memory", "Some content", "feedback")
    assert "saved" in result
    # Verify it was actually written
    content = tool.read("new-mem")
    assert "Some content" in content
