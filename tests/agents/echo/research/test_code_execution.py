"""Unit tests for code execution sandbox restrictions."""

import pytest
import tempfile
import os
from hermes_cli.agents.echo.research.nodes.code_execution import (
    _execute_sandboxed,
    SANDBOX_TIMEOUT,
    ALLOWED_IMPORTS,
    BLOCKED_MODULES,
)


class TestSandboxExecution:
    """Test the sandboxed code execution function."""

    def test_simple_print(self):
        """Simple print statement should execute successfully."""
        result = _execute_sandboxed("print('hello world')", 1)
        assert result.success
        assert "hello world" in result.stdout
        assert result.exit_code == 0

    def test_basic_math(self):
        """Basic math operations should work."""
        code = """
result = sum(range(1, 101))
print("Sum 1-100:", result)
"""
        result = _execute_sandboxed(code, 1)
        assert result.success
        assert "5050" in result.stdout

    def test_allowed_import_works(self):
        """Allowed modules should be importable."""
        code = """
import math
import json
print("sqrt(16) =", math.sqrt(16))
data = json.dumps({"key": "value"})
print("JSON:", data)
"""
        result = _execute_sandboxed(code, 1)
        assert result.success
        assert "4.0" in result.stdout
        assert '"key"' in result.stdout

    def test_blocked_shutil_import(self):
        """Attempting to import shutil should fail."""
        code = """
import shutil
shutil.rmtree('/tmp')
"""
        result = _execute_sandboxed(code, 1)
        assert not result.success
        assert "ImportError" in result.stderr or "blocked" in result.stderr.lower()

    def test_blocked_subprocess_import(self):
        """Attempting to import subprocess should fail."""
        code = """
import subprocess
subprocess.run(['ls'])
"""
        result = _execute_sandboxed(code, 1)
        assert not result.success

    def test_blocked_socket_import(self):
        """Attempting to import socket should fail."""
        code = """
import socket
s = socket.socket()
"""
        result = _execute_sandboxed(code, 1)
        assert not result.success

    def test_statistics_module(self):
        """Statistics module should be importable."""
        code = """
import statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Mean:", statistics.mean(data))
print("Stdev:", statistics.stdev(data))
"""
        result = _execute_sandboxed(code, 1)
        assert result.success
        assert "5.5" in result.stdout

    def test_timeout_handling(self):
        """Very long execution should time out."""
        code = """
import time
time.sleep(60)
print("This should not print")
"""
        # Override timeout for test
        import hermes_cli.agents.echo.research.nodes.code_execution as ce
        old_timeout = ce.SANDBOX_TIMEOUT
        ce.SANDBOX_TIMEOUT = 2
        try:
            result = _execute_sandboxed(code, 1)
            assert not result.success
            assert "timed out" in result.stderr.lower()
        finally:
            ce.SANDBOX_TIMEOUT = old_timeout

    def test_syntax_error_captured(self):
        """Python syntax errors should be captured in stderr."""
        code = "this is not valid python {{{{{"
        result = _execute_sandboxed(code, 1)
        assert not result.success
        assert "SyntaxError" in result.stderr

    def test_runtime_error_captured(self):
        """Runtime errors should be captured."""
        code = """
x = 1 / 0
print(x)
"""
        result = _execute_sandboxed(code, 1)
        assert not result.success
        assert "ZeroDivisionError" in result.stderr

    def test_stdout_captured(self):
        """Stdout should be captured and returned."""
        code = """
for i in range(5):
    print("Line", i + 1)
"""
        result = _execute_sandboxed(code, 1)
        assert result.success
        assert result.stdout.count("Line") == 5


class TestSandboxConfiguration:
    """Test sandbox configuration values."""

    def test_allowed_imports_not_empty(self):
        """Allowed imports should contain common data science modules."""
        assert "math" in ALLOWED_IMPORTS
        assert "statistics" in ALLOWED_IMPORTS
        assert "json" in ALLOWED_IMPORTS

    def test_blocked_modules_contain_dangerous(self):
        """Blocked modules should contain dangerous ones."""
        assert "subprocess" in BLOCKED_MODULES
        assert "socket" in BLOCKED_MODULES
        assert "ctypes" in BLOCKED_MODULES
        assert "shutil" in BLOCKED_MODULES

    def test_sandbox_timeout_reasonable(self):
        """Timeout should be between 5 and 120 seconds."""
        assert 5 <= SANDBOX_TIMEOUT <= 120
