"""Latin tutor state-loader graph node (seam, --latin mode only).

Reads the latin ledger at HERMES_LATIN_DIR/ledger.json (default
~/.hermes/latin/ledger.json) — or cold-starts a default ledger on first
run — builds a structured mastery-state block, and sets state['latin_state'] so
build_latin_system_prompt can render it into the latin system prompt. Runs
BEFORE call_llm: in --latin mode the graph entry edge is rewired
process_input -> load_latin_state -> call_llm (see create_echo_graph).

This node reads ONLY the non-protected latin ledger at HERMES_LATIN_DIR
(a sibling of the protected stores, NOT a protected store itself). It imports
no private safety package. It is a graph node (same tier as
process_input/format_response) — not a SeamedTool handler, so it is not
AST-affect-attested, but verify_integrity covers it (seam file) and the
latin_state value it produces flows into call_llm which IS attested (reads no
banned affect field; assert_state_keys_clean_for_prompt permits the
'latin_state' key). The ledger is owned by the latin tools (latin_srs/
latin_validate write; this node only reads).

DESIGN.md §7.1 (state load graph node) + §8.1 (state-aware, AI-driven).
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_log = logging.getLogger(__name__)

_DEFAULT_LATIN_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "latin")

# F9 (2026-07-13 red-team, Cluster 2 structural stop): id-list fields rendered
# into the <latin_state> block (weak_spots, paradigm_only_flags) and skill keys
# are attacker-controlled ledger content. A poisoned ledger could otherwise
# inject fence tokens (<<<UNTRUSTED_MEMORY>>>), angle brackets, or tool-call
# backticks into the SYSTEM prompt. This charset admits real construction ids
# ("ablative_absolute", "subjunctive", "decl_I", "ablative absolute" with a
# space) but blocks <, >, backtick, {, }, [, ], |, & — the structural
# injection vectors. Non-matching items are DROPPED (with a logged warning so
# Coda notices the ledger got dirty) rather than rendered.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_:\- ]+$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """F11 crash-safe mastery cast: a non-numeric 'mastery' in a corrupt ledger
    must not raise ValueError out of the graph node (which would brick --latin).
    Log + fall back to default instead."""
    try:
        return float(value)
    except (TypeError, ValueError):
        _log.warning("latin_state: non-numeric mastery %r -> default %.1f", value, default)
        return default


def _sanitize_id_list(items: Any, field_name: str) -> list:
    """F9: keep only items matching _SAFE_ID_RE; drop the rest with a warning."""
    if not isinstance(items, list):
        return []
    kept = []
    for it in items:
        if isinstance(it, str) and _SAFE_ID_RE.match(it):
            kept.append(it)
        else:
            _log.warning("latin_state: dropped %s entry %r (failed safe-id schema)", field_name, it)
    return kept


def _sanitize_skills(skills: Any) -> Dict[str, Dict[str, float]]:
    """F9 + F11: keep only safe-keyed skills with a crash-safe mastery float."""
    if not isinstance(skills, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for sid, s in skills.items():
        if not (isinstance(sid, str) and _SAFE_ID_RE.match(sid)):
            _log.warning("latin_state: dropped skill key %r (failed safe-id schema)", sid)
            continue
        mastery = _safe_float((s or {}).get("mastery", 0.0)) if isinstance(s, dict) else 0.0
        out[sid] = {"mastery": mastery}
    return out


def _latin_dir() -> Path:
    """HERMES_LATIN_DIR env (default ~/.hermes/latin). Sibling of the protected
    stores, NOT a protected store itself (the in-process latin tools write here
    directly)."""
    return Path(os.environ.get("HERMES_LATIN_DIR", _DEFAULT_LATIN_DIR))


def _ledger_path() -> Path:
    return _latin_dir() / "ledger.json"


def _default_ledger() -> Dict[str, Any]:
    """Cold-start ledger (DESIGN.md ledger_schema.md, version 1)."""
    return {
        "version": 1,
        "profile": {
            "current_ginn_ch": 1,
            "current_fr_ch": 1,
            "stage": 1,
            "week": 1,
            "vocab_count": 0,
        },
        "skills": {},
        "cards": {},
        "error_patterns": [],
        # subjunctive: Ginn LIV-LX front-loads the paradigm before FR 27+ supplies
        # in-context reading (DESIGN.md §4 bidirectional ordering).
        "paradigm_only_flags": ["subjunctive"],
        "sessions": [],
    }


# Transient marker key (F11): _read_ledger tags a fallback ledger so
# load_latin_state can surface corruption in the rendered block. Stripped before
# any write (the latin tools re-read the file fresh, so this never persists).
_CORRUPT_MARKER = "__ledger_corrupt__"


def _default_ledger_with_flag() -> Dict[str, Any]:
    ledger = _default_ledger()
    ledger[_CORRUPT_MARKER] = True
    return ledger


def _atomic_write_ledger(ledger: Dict[str, Any]) -> None:
    p = _ledger_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _read_ledger() -> Dict[str, Any]:
    """Read the ledger; cold-start + persist a default if absent. Falls back to
    an in-memory default if the ledger is corrupt or the FS is unwritable.

    F11 (2026-07-13 red-team, Cluster 2): corruption is NON-SILENT. The prior
    bare ``except Exception: return _default_ledger()`` swallowed a corrupt
    ledger and substituted a fresh default with no signal — Coda would lose all
    mastery/SRS progress and never know. Now a corrupt ledger logs a WARNING
    (so it shows in the session) before the in-memory fallback; the caller
    (load_latin_state) also tags latin_state with ``ledger_corrupt=True`` so
    the rendered block + the LLM can surface it. Defaulting still happens so
    --latin boots instead of bricking the graph node."""
    p = _ledger_path()
    if not p.exists():
        ledger = _default_ledger()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_ledger(ledger)
        except Exception as e:  # noqa: BLE001
            _log.warning("latin_state: could not persist cold-start ledger (%r); "
                         "running in-memory (writes will retry)", e)
        return ledger
    raw = p.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        _log.warning("latin_state: ledger.json is corrupt (%r) — falling back to a "
                     "fresh default. Back up + inspect %s; mastery/SRS progress in "
                     "the corrupt file is not loaded.", e, p)
        return _default_ledger_with_flag()
    if not isinstance(parsed, dict):
        _log.warning("latin_state: ledger.json top-level is %s, not an object — "
                     "falling back to a fresh default.", type(parsed).__name__)
        return _default_ledger_with_flag()
    return parsed


def _is_due(card: Dict[str, Any], now: datetime) -> bool:
    due = (card.get("fsrs") or {}).get("due")
    if not due:
        return False
    try:
        return datetime.fromisoformat(due) <= now
    except Exception:
        return False


def load_latin_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node: read the latin ledger, build the structured latin_state
    block, set state['latin_state']. Passthrough for translate_permitted so the
    latin builder + tutor tools see the per-turn /translate flag."""
    ledger = _read_ledger()
    ledger_corrupt = bool(ledger.pop(_CORRUPT_MARKER, False))
    profile = ledger.get("profile", {})
    if not isinstance(profile, dict):
        _log.warning("latin_state: ledger 'profile' is %s, not an object — using defaults.",
                     type(profile).__name__)
        profile = {}
    sessions = ledger.get("sessions", [])
    last_session = sessions[-1] if isinstance(sessions, list) and sessions else {}
    if not isinstance(last_session, dict):
        last_session = {}
    cards = ledger.get("cards", {})
    if not isinstance(cards, dict):
        _log.warning("latin_state: ledger 'cards' is %s, not an object — SRS due count = 0.",
                     type(cards).__name__)
        cards = {}
    now = datetime.now()
    due_count = sum(1 for c in cards.values() if isinstance(c, dict) and _is_due(c, now))
    # F9 structural stop: schema-validate id-list fields + skill keys before they
    # render into the SYSTEM prompt; F11: crash-safe mastery cast.
    weak_spots = _sanitize_id_list(last_session.get("weak_spots", []), "weak_spots")
    paradigm_only_flags = _sanitize_id_list(ledger.get("paradigm_only_flags", []),
                                            "paradigm_only_flags")
    skills_snapshot = _sanitize_skills(ledger.get("skills", {}))
    # Profile ints: wrong type -> None (rendered as None) rather than crash.
    def _safe_int(v):
        return v if isinstance(v, int) else None
    latin_state = {
        "current_ginn_ch": _safe_int(profile.get("current_ginn_ch")),
        "current_fr_ch": _safe_int(profile.get("current_fr_ch")),
        "stage": profile.get("stage", 1) if isinstance(profile.get("stage", 1), int) else 1,
        "week": profile.get("week", 1) if isinstance(profile.get("week", 1), int) else 1,
        "vocab_count": _safe_int(profile.get("vocab_count")) or 0,
        "srs_due_count": due_count,
        # weak_spots are skill-id / construction references (structured, not free
        # prose) so the rendered block cannot trip assert_prompt_clean.
        "weak_spots": weak_spots,
        "paradigm_only_flags": paradigm_only_flags,
        "translate_permitted": bool(state.get("translate_permitted", False)),
        "skills_snapshot": skills_snapshot,
        # F11: surface corruption to the LLM + Coda rather than silently loading
        # a fresh default. The persona instructs the tutor to tell Coda.
        "ledger_corrupt": ledger_corrupt,
    }
    state["latin_state"] = latin_state
    return state