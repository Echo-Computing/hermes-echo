"""Tests for Echo agent file operation tools."""

import pytest
from pathlib import Path
from hermes_cli.agents.echo.tools.file_tools import read_file, write_file, edit_file


def test_read_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3")
    assert read_file(str(f)) == "line1\nline2\nline3"


def test_read_file_with_offset(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("a\nb\nc\nd\ne")
    assert read_file(str(f), offset=2) == "c\nd\ne"


def test_read_file_with_offset_and_limit(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("a\nb\nc\nd\ne")
    assert read_file(str(f), offset=1, limit=2) == "b\nc"


def test_write_file(tmp_path):
    f = tmp_path / "new.txt"
    result = write_file(str(f), "hello world")
    assert f.read_text() == "hello world"
    assert "written" in result


def test_write_file_overwrite(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("old content")
    write_file(str(f), "new content")
    assert f.read_text() == "new content"


def test_edit_file(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello world")
    result = edit_file(str(f), "hello", "goodbye")
    assert f.read_text() == "goodbye world"
    assert "edited" in result


def test_edit_file_first_occurrence_only(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("foo bar foo")
    edit_file(str(f), "foo", "baz")
    assert f.read_text() == "baz bar foo"


def test_read_missing_file():
    with pytest.raises(FileNotFoundError):
        read_file("/nonexistent/path/file.txt")


def test_edit_missing_file():
    with pytest.raises(FileNotFoundError):
        edit_file("/nonexistent/path/file.txt", "old", "new")


def test_edit_old_string_not_found(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello world")
    with pytest.raises(ValueError, match="old_string not found"):
        edit_file(str(f), "goodbye", "replacement")
