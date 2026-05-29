"""Tests for correction, fact, and command detection."""

import pytest
from hermes_cli.agents.echo.learning.detector import (
    detect_correction,
    detect_fact,
    detect_command,
)


class TestDetectCorrection:
    """Test correction pattern detection."""

    def test_directive_no(self):
        assert detect_correction("no, use subprocess instead") is True
        assert detect_correction("no, try pathlib") is True
        assert detect_correction("no, do this way") is True
        assert detect_correction("no, go with the flat structure") is True
        assert detect_correction("no, we should use async") is True

    def test_negative_dont(self):
        assert detect_correction("don't use tabs for indentation") is True
        assert detect_correction("do not mock the database") is True
        assert detect_correction("don't use spaces here") is True

    def test_instead(self):
        assert detect_correction("use pathlib instead of os.path") is True
        assert detect_correction("instead of that, try this") is True

    def test_actually(self):
        assert detect_correction("actually, the port is 8080") is True
        assert detect_correction("it's actually configured differently") is True

    def test_prefer(self):
        assert detect_correction("I prefer flat directories") is True
        assert detect_correction("I would prefer using async") is True

    def test_always_never(self):
        assert detect_correction("always use Python 3.12") is True
        assert detect_correction("never use global variables") is True

    def test_no_correction_normal_message(self):
        assert detect_correction("What does this file do?") is False
        assert detect_correction("Can you help me with something?") is False
        assert detect_correction("The weather is nice today") is False

    def test_no_correction_code_question(self):
        assert detect_correction("how do I read a file in Python") is False


class TestDetectFact:
    """Test fact detection tiers."""

    def test_explicit_remember(self):
        tier, text = detect_fact("remember that I use Python 3.12")
        assert tier == "explicit"
        assert text == "remember that I use Python 3.12"

    def test_explicit_save_this(self):
        tier, text = detect_fact("save this for later")
        assert tier == "explicit"
        assert text == "save this for later"

    def test_explicit_dont_forget(self):
        tier, text = detect_fact("don't forget the GPU only has 8GB VRAM")
        assert tier == "explicit"
        assert text == "don't forget the GPU only has 8GB VRAM"

    def test_explicit_REMEMBER_uppercase(self):
        tier, text = detect_fact("REMEMBER: this is important")
        assert tier == "explicit"
        assert text == "REMEMBER: this is important"

    def test_strong_prefer(self):
        tier, text = detect_fact("I prefer using spaces over tabs")
        assert tier == "strong"
        assert text == "I prefer using spaces over tabs"

    def test_strong_setup(self):
        tier, text = detect_fact("my setup uses a 4K monitor")
        assert tier == "strong"
        assert text == "my setup uses a 4K monitor"

    def test_strong_lets_go_with(self):
        tier, text = detect_fact("let's go with the async approach")
        assert tier == "strong"
        assert text == "let's go with the async approach"

    def test_strong_plan_is(self):
        tier, text = detect_fact("the plan is to deploy on Friday")
        assert tier == "strong"
        assert text == "the plan is to deploy on Friday"

    def test_weak_signal_ignored(self):
        tier, text = detect_fact("this project uses React for the frontend")
        assert tier is None
        assert text == ""

    def test_weak_signal_tool_choice_ignored(self):
        tier, text = detect_fact("I'll use pytest for testing")
        assert tier is None
        assert text == ""

    def test_no_fact_normal_message(self):
        tier, text = detect_fact("What's the weather?")
        assert tier is None
        assert text == ""


class TestDetectCommand:
    """Test command detection."""

    def test_idea_command(self):
        cmd = detect_command("/idea build a chess engine")
        assert cmd["command"] == "idea"
        assert cmd["arg"] == "build a chess engine"

    def test_idea_command_multiline_arg(self):
        cmd = detect_command("/idea a honeypot that mimics an AI inference server")
        assert cmd["command"] == "idea"
        assert "honeypot" in cmd["arg"]

    def test_idea_save(self):
        cmd = detect_command("/idea save")
        assert cmd["command"] == "idea_save"
        assert cmd["arg"] is None

    def test_idea_save_with_whitespace(self):
        cmd = detect_command("/idea save  ")
        assert cmd["command"] == "idea_save"

    def test_exit_command(self):
        cmd = detect_command("/exit")
        assert cmd["command"] == "exit"
        assert cmd["arg"] is None

    def test_exit_with_whitespace(self):
        cmd = detect_command("/exit   ")
        assert cmd["command"] == "exit"

    def test_no_command_normal_message(self):
        cmd = detect_command("hey how are you")
        assert cmd["command"] is None
        assert cmd["arg"] is None

    def test_no_command_partial_match(self):
        cmd = detect_command("idea about something")  # no leading /
        assert cmd["command"] is None

    def test_no_command_idea_without_arg(self):
        cmd = detect_command("/idea")  # needs an argument
        assert cmd["command"] is None
