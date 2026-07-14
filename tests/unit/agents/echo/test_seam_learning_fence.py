"""v0.3.1 — assert_messages_clean fence wired into the learning-loop Ollama calls.

The 4 learning modules (reflector / idea_capture / session_summary / auto_memory)
each send a constructed single-user-message prompt to the local Ollama for
consolidation. v0.3.1 wires assert_messages_clean before each httpx.post so an
affect marker in the constructed prompt is caught fail-closed before the network
call. In the public build the guard is None (the anima safety package is absent)
so the wiring is a no-op; these tests monkeypatch the module's _PROMPT_GUARD to a
fake to pin the wiring WITHOUT needing the private anima package.

Hermetic: no real Ollama call is made (httpx.post is monkeypatched).
"""

import pytest
from hermes_cli.agents.echo.memory import MemoryStore
from hermes_cli.agents.echo.learning import reflector, auto_memory, idea_capture, session_summary


class _RecordingGuard:
    """Fake prompt guard: records that assert_messages_clean ran + optionally raises."""
    def __init__(self, raise_marker=False):
        self.called = False
        self.captured = None
        self._raise = raise_marker

    def assert_messages_clean(self, messages):
        self.called = True
        self.captured = messages
        if self._raise:
            raise RuntimeError("affect marker detected — fence fail-closed")


def _raise_no_ollama(*a, **k):
    raise RuntimeError("no ollama in test")


def _should_not_reach(*a, **k):
    pytest.fail("httpx.post reached despite the guard raising fail-closed before it")


def test_consolidate_correction_runs_fence_and_catches_raise(monkeypatch, tmp_path):
    """The fence runs BEFORE the Ollama call; a guard raise is caught by the
    surrounding try -> consolidate_correction returns False (best-effort skip,
    never blocks the user). The httpx.post is never reached."""
    guard = _RecordingGuard(raise_marker=True)
    monkeypatch.setattr(reflector, "_PROMPT_GUARD", guard)
    monkeypatch.setattr(reflector.httpx, "post", _should_not_reach)
    store = MemoryStore(tmp_path)
    ok = reflector.consolidate_correction(
        store, {"user_msg": "use spaces", "prior_response": "used tabs"},
        {"api_url": "http://localhost:11434/api/chat", "model": "x"})
    assert ok is False
    assert guard.called is True
    assert isinstance(guard.captured, list) and guard.captured


def test_consolidate_idea_runs_fence_and_catches_raise(monkeypatch, tmp_path):
    """Fence-wiring pin for idea_capture.consolidate_idea — the guard runs before
    httpx.post; a raise is caught -> returns False (best-effort skip), the network
    call is never reached. Confirms the fence is wired for all 4 learning modules,
    not just reflector + auto_memory."""
    guard = _RecordingGuard(raise_marker=True)
    monkeypatch.setattr(idea_capture, "_PROMPT_GUARD", guard)
    monkeypatch.setattr(idea_capture.httpx, "post", _should_not_reach)
    store = MemoryStore(tmp_path)
    ok = idea_capture.consolidate_idea(
        store, "maybe ship the latin data bundle so --latin is not a shell",
        {"api_url": "http://localhost:11434/api/chat", "model": "x"})
    assert ok is False
    assert guard.called is True
    assert isinstance(guard.captured, list) and guard.captured


def test_consolidate_session_runs_fence_and_catches_raise(monkeypatch, tmp_path):
    """Fence-wiring pin for session_summary.consolidate_session — same wiring as
    the other three modules. Needs a history_dir (write_session_record target);
    the guard raise fires before httpx.post so the history write is never
    reached either."""
    guard = _RecordingGuard(raise_marker=True)
    monkeypatch.setattr(session_summary, "_PROMPT_GUARD", guard)
    monkeypatch.setattr(session_summary.httpx, "post", _should_not_reach)
    store = MemoryStore(tmp_path)
    history_dir = tmp_path / "hist"
    history_dir.mkdir()
    ok = session_summary.consolidate_session(
        store, history_dir, "user asked about the latin data bundle; agent summarized",
        {"api_url": "http://localhost:11434/api/chat", "model": "x"})
    assert ok is False
    assert guard.called is True
    assert isinstance(guard.captured, list) and guard.captured, "fence did not receive the messages"


def test_consolidate_correction_fence_noop_when_guard_none(monkeypatch, tmp_path):
    """Public build: _PROMPT_GUARD is None -> no fence call (no AttributeError on
    the None guard) + the best-effort try/except still wraps the Ollama call."""
    monkeypatch.setattr(reflector, "_PROMPT_GUARD", None)
    monkeypatch.setattr(reflector.httpx, "post", _raise_no_ollama)
    store = MemoryStore(tmp_path)
    ok = reflector.consolidate_correction(
        store, {"user_msg": "use spaces", "prior_response": "used tabs"},
        {"api_url": "http://localhost:11434/api/chat", "model": "x"})
    assert ok is False  # httpx raised -> caught -> best-effort False


def test_consolidate_fact_runs_fence_and_catches_raise(monkeypatch, tmp_path):
    """Same wiring pin for auto_memory.consolidate_fact — confirms the pattern is
    applied consistently across the learning modules, not just reflector."""
    guard = _RecordingGuard(raise_marker=True)
    monkeypatch.setattr(auto_memory, "_PROMPT_GUARD", guard)
    monkeypatch.setattr(auto_memory.httpx, "post", _should_not_reach)
    store = MemoryStore(tmp_path)
    # consolidate_fact dedups via store.search first; force no existing entry so
    # it reaches the prompt-build + guard path.
    monkeypatch.setattr(store, "search", lambda _q: [])
    ok = auto_memory.consolidate_fact(
        store, "I prefer Python 3.12 for the unit tests",
        {"api_url": "http://localhost:11434/api/chat", "model": "x"})
    assert ok is False
    assert guard.called is True
    assert isinstance(guard.captured, list) and guard.captured