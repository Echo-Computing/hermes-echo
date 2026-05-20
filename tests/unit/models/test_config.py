import pytest
from pathlib import Path
from hermes_cli.models.config import EchoConfig, HermesConfig


def test_echo_config_defaults():
    config = EchoConfig()
    assert config.model == "qwen3.6:35b"
    assert config.max_tool_calls == 10
    assert config.context_messages == 50
    assert config.shell_timeout == 120
    assert config.confirm_destructive is True
    assert config.auto_memory is True
    assert config.memory_dir == Path.home() / ".hermes" / "memory"


def test_hermes_config_has_echo():
    config = HermesConfig()
    assert config.echo.model == "qwen3.6:35b"
