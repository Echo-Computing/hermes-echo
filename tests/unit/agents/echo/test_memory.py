import pytest
from pathlib import Path
from hermes_cli.agents.echo.memory import MemoryStore


def test_memory_write_and_read(tmp_path):
    store = MemoryStore(tmp_path)
    store.write("test-mem", "Test description", "This is test content.", "project")

    result = store.read("test-mem")
    assert result is not None
    assert "test-mem" in result
    assert "Test description" in result
    assert "This is test content." in result


def test_memory_search(tmp_path):
    store = MemoryStore(tmp_path)
    store.write("gpu-info", "GPU specs", "RTX 2070 with 8GB VRAM", "knowledge")
    store.write("project-foo", "Foo project", "Something else entirely", "project")

    results = store.search("VRAM")
    assert len(results) == 1
    assert results[0]["name"] == "gpu-info"


def test_memory_index_created(tmp_path):
    store = MemoryStore(tmp_path)
    store.write("entry-1", "First entry", "Content A", "feedback")

    index = tmp_path / "MEMORY.md"
    assert index.exists()
    index_text = index.read_text()
    assert "entry-1" in index_text
    assert "First entry" in index_text


def test_memory_read_nonexistent(tmp_path):
    store = MemoryStore(tmp_path)
    result = store.read("nonexistent")
    assert result is None


def test_memory_search_no_match(tmp_path):
    store = MemoryStore(tmp_path)
    store.write("test", "A test", "Content", "project")
    results = store.search("zzz_nonexistent_zzz")
    assert len(results) == 0


def test_memory_overwrite(tmp_path):
    store = MemoryStore(tmp_path)
    store.write("test", "First desc", "First content", "project")
    store.write("test", "Updated desc", "Updated content", "feedback")

    result = store.read("test")
    assert "Updated desc" in result
    assert "Updated content" in result
