"""Tests for the Echo agent feedback loop module."""

import pytest
from pathlib import Path
from hermes_cli.agents.echo.feedback import FeedbackLoop


def test_capture_avoid_pattern(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("Don't use tabs for indentation", "OK")
    assert result is not None
    assert "avoid_pattern" in result


def test_capture_avoid_ever(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("Don't ever use tabs", "OK")
    assert result is not None
    assert "avoid_pattern" in result


def test_capture_preference(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("I prefer using spaces over tabs", "OK")
    assert result is not None
    assert "preference" in result


def test_capture_no_feedback(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("What's in this file?", "It contains...")
    assert result is None


def test_capture_duplicate(tmp_path):
    fb = FeedbackLoop(tmp_path)
    fb.capture("Don't use tabs", "OK")
    result = fb.capture("Don't use tabs", "OK")
    assert result is None  # Duplicate


def test_summarize_patterns(tmp_path):
    fb = FeedbackLoop(tmp_path)
    fb.capture("Don't use tabs", "OK")
    fb.capture("I prefer spaces", "OK")
    fb.capture("Use spaces instead of tabs", "OK")
    fb.capture("Don't use spaces in config", "OK")
    fb.capture("I avoid tabs", "OK")

    patterns = fb.summarize_patterns()
    assert "avoid_pattern" in patterns


def test_read_all_feedback(tmp_path):
    fb = FeedbackLoop(tmp_path)
    fb.capture("Don't use thing one", "OK")
    fb.capture("Prefer thing two", "OK")

    feedbacks = fb.read_all_feedback()
    assert len(feedbacks) == 2


def test_capture_from_now_on(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("From now on I prefer Python", "OK")
    assert result is not None
    assert "from_now_on" in result


def test_capture_use_instead(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("Use spaces instead of tabs", "OK")
    assert result is not None
    assert "preference" in result


def test_capture_note(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("Note that I'm a developer, not a designer", "OK")
    assert result is not None
    assert "note" in result


def test_capture_remember(tmp_path):
    fb = FeedbackLoop(tmp_path)
    result = fb.capture("Remember that I am a developer", "OK")
    assert result is not None
    assert "remember" in result


def test_summarize_not_enough_feedback(tmp_path):
    fb = FeedbackLoop(tmp_path)
    fb.capture("Feedback one", "OK")
    fb.capture("Feedback two", "OK")

    patterns = fb.summarize_patterns()
    assert "not enough feedback" in patterns.lower()


def test_feedback_stored_in_memory(tmp_path):
    fb = FeedbackLoop(tmp_path)
    fb.capture("Don't use tabs", "OK")

    # Verify the file was actually written
    md_files = list(tmp_path.rglob("*.md"))
    feedback_files = [f for f in md_files if f.stem.startswith("feedback-")]
    assert len(feedback_files) >= 1

    content = feedback_files[0].read_text()
    assert "Trigger:" in content
    assert "avoid_pattern" in content
