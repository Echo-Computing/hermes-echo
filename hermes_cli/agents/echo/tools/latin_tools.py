"""Latin deterministic-core CertifiedTool handlers (seam).

Correctness-critical operations that NEVER trust the LLM:
  - latin_validate : graded gate — LatinCy parse + macron lexicon + proper-noun
                     allowlist. Attempts deterministic macron correction on a
                     parse-able string before flagging. Returns accept/warn/reject.
  - latin_srs      : FSRS-6 review scheduling + ledger update. The scheduler is
                     the sole authority on when a card is due.
  - latin_paradigm : static finite declension/conjugation tables citing A&G.

Handlers read only LLM-supplied params + the non-protected latin data
files (macron_lexicon.json, proper_nouns.json, paradigm_tables.json) + the
non-protected latin ledger at HERMES_LATIN_DIR (a sibling of the protected
stores, NOT a protected store itself). No private safety package import, no
banned protected-state field read, no protected-store path string constant. spaCy +
fsrs are lazy-imported on first use (the ~500MB la_core_web_lg loads on the
first latin_validate call, NOT at module import — keeps Hermes import fast
and keeps the model out of the venv path for non-latin graphs). Each handler
is module-level so inspect.getsource resolves for the CertifiedTool handler-cert
(scan_function_for_handler at registration + the import-time
_verify_registry_handler_cert attestation).

Registered in agent.py _build_registry as DIRECT CertifiedTool(...,
execution_sandbox="none", requires_handler_cert=True) — NOT a ToolPlugin (a
mount-namespace ceiling is not in the public build).
The ledger I/O helpers here are the canonical writer; latin_state.py imports
the read helper for its pre-LLM graph node.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default latin workspace when HERMES_LATIN_DIR is unset. A sibling of the
# other hermes stores (~/.hermes/memory, ~/.hermes/history, ~/.hermes/graphs).
# shell_tools._GUARD_SOURCE_ROOTS uses the SAME default expression so the latin
# workspace is a guard-source root even without HERMES_LATIN_DIR set (the two
# defaults must agree or the conditioning surface is unprotected).
_DEFAULT_LATIN_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "latin")

# Macron folding: combine the precomposed macron vowels (lowercase AND uppercase)
# + the combining macron (U+0304) back to their bare ASCII vowel so an
# unmacronized LLM string can be matched against a stored macronized form (and
# vice-versa). Both cases are mapped so an all-caps / title-case macronized token
# (e.g. 'RŌMA', 'RŌma') folds correctly — accent-insensitive matching must not
# depend on case (Phase 5 4-lens re-verify, detcore finding).
_MACRON_MAP = str.maketrans({
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ȳ": "y",
    "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U", "Ȳ": "Y",
})
_COMBINING_MACRON = "̄"

# DoS bounds:
# MAX_LATIN_VALIDATE_CHARS — application-level cap on the latin_string passed to
# latin_validate BEFORE the O(n) LatinCy spaCy parse. spaCy's own max_length
# (1,000,000) is the sole bound without this, and a single ~500KB call blocks
# the in-process agent thread for ~73s (no per-tool-call timeout on the
# execution_sandbox="none" path). 8000 chars is ~a page of Latin — far beyond any
# real tutor passage — so a legitimate validate never hits it; a huge/pasted
# string is rejected instantly + deterministically instead of stalling. (F12.)
MAX_LATIN_VALIDATE_CHARS = 8000
# MAX_ERROR_PATTERNS — cap on the ledger's error_patterns list. _record_error_pattern
# appends a new entry for every DISTINCT form string with no bound, so a long
# session (or a prompt-injected loop) bloats the ledger + makes every subsequent
# latin_validate/latin_srs read+write O(n). LFU eviction (drop lowest count, tie-
# break oldest last_seen) keeps the most-frequent / most-recent patterns the
# tutor actually surfaces. (F16.)
MAX_ERROR_PATTERNS = 500

# F13: the vowel set used to make the reject
# path meaningful. LatinCy is a statistical tagger that assigns a lemma+POS to
# EVERY token (including gibberish like 'asdf'), so the prior `not any_lemma`
# reject branch was effectively dead code for any non-empty string. With a
# 42-lemma starter lexicon, most REAL Latin words are out-of-lexicon too, so an
# aggressive "reject if no in-lexicon token" heuristic would brick legitimate
# beginner sentences (e.g. 'Mater femina est'). The safe discriminator: every
# Latin word contains at least one Latin vowel — reject when NO token carries a
# Latin vowel (pure punctuation / numbers / symbols / non-Latin script), which
# the old gate returned `warn` for. Grammar correctness (agreement/case/tense)
# is the LLM teacher's role BY DESIGN — this gate is a parse-recovery + macron-
# correction + vocab-recognition gate, NOT a grammatical correctness gate (the
# docstring + paedagogus.md are honest about this).
_LATIN_VOWELS = set("aeiouyāēīōūȳAEIOUYĀĒĪŌŪȲ")

# F6: per-turn accumulator of the Latin strings
# the LLM actually ran through latin_validate (input + macron-corrected output
# forms). format_response drains this to warn when the LLM emitted macronized
# Latin in its response WITHOUT going through the gate — the bypass that would
# let wrong macrons reach the user presented as correct. Cleared each format_response
# drain, so it is scoped to one turn (the CLI processes one turn at a time).
# Module-level (not state-threaded) because CertifiedTool handlers receive only
# params, not EchoState, and threading it through execute_tools would touch the
# attested dispatch path. This is a WARNING net only; the deterministic core
# remains the source of truth for validated strings.
_VALIDATED_THIS_TURN: set = set()


def _record_validated(forms) -> None:
    """Record the input + corrected forms of a latin_validate call (accept/warn)."""
    if isinstance(forms, str):
        if forms.strip():
            _VALIDATED_THIS_TURN.add(forms.strip())
    elif isinstance(forms, (list, tuple)):
        for f in forms:
            if isinstance(f, str) and f.strip():
                _VALIDATED_THIS_TURN.add(f.strip())


def drain_validated_latin() -> set:
    """Return + clear the per-turn validated-Latin accumulator (called by
    format_response at end of turn)."""
    seen = set(_VALIDATED_THIS_TURN)
    _VALIDATED_THIS_TURN.clear()
    return seen


def _strip_macrons(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.translate(_MACRON_MAP).replace(_COMBINING_MACRON, "")


def _latin_dir() -> Path:
    return Path(os.environ.get("HERMES_LATIN_DIR", _DEFAULT_LATIN_DIR))


def bundled_latin_data_dir() -> Path:
    """The read-only in-package latin data dir (v0.3.1: a public data subset
    ships in-tree at hermes_cli/agents/echo/latin_data/ so `hermes echo --latin`
    is usable out of the box — paradigm_tables.json + proper_nouns.json +
    macron_lexicon.json + the public paedagogus.md). This dir is the FLOOR for
    the three read-only data files when HERMES_LATIN_DIR is unset. The WRITABLE
    ledger home stays _latin_dir() (env or ~/.hermes/latin) so the shell_tools
    guard-source-root agreement + the hermetic-test {}-on-missing contract are
    unchanged (a bundled dir must never receive ledger writes)."""
    return Path(__file__).resolve().parent.parent / "latin_data"


def _ledger_path() -> Path:
    return _latin_dir() / "ledger.json"


def _default_ledger() -> Dict[str, Any]:
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
        "paradigm_only_flags": ["subjunctive"],
        "sessions": [],
    }


def _atomic_write_ledger(ledger: Dict[str, Any]) -> None:
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _read_ledger() -> Dict[str, Any]:
    p = _ledger_path()
    if not p.exists():
        ledger = _default_ledger()
        try:
            _atomic_write_ledger(ledger)
        except Exception:
            pass
        return ledger
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _default_ledger()


# --- lazy-loaded model + data caches (NOT module-level heavy state) ---
_NLP = None
_LEXICON = None
_PROPER_NOUNS = None
_PARADIGM_TABLES = None


def _get_nlp():
    """Lazy-load la_core_web_lg on first latin_validate call."""
    global _NLP
    if _NLP is None:
        import spacy  # lazy: keeps module import light + venv clean for non-latin graphs
        _NLP = spacy.load("la_core_web_lg")
    return _NLP


def _get_lexicon() -> Dict[str, Any]:
    global _LEXICON
    if _LEXICON is None:
        # Case-fold the keys (drop the `_schema` metadata key) so the .lower()
        # lookup matches regardless of how the JSON is keyed. The macron_lexicon
        # convention is lowercase keys, but case-folding makes the lookup robust
        # to a Capitalized entry + to LatinCy returning a sentence-initial cap.
        raw = _load_json("macron_lexicon.json")
        _LEXICON = {str(k).lower(): v for k, v in raw.items()
                    if not str(k).startswith("_")}
    return _LEXICON


def _get_proper_nouns() -> Dict[str, Any]:
    global _PROPER_NOUNS
    if _PROPER_NOUNS is None:
        # Case-fold the keys (drop the `_schema` metadata key). proper_nouns.json
        # stores Capitalized keys ('Caesar','Roma',...) matching how LatinCy
        # returns PROPN lemmas; the lookup is `proper.get(lemma.lower())`, so
        # case-folding here is what makes a Capitalized JSON key match a
        # lowercased query (Phase 5 4-lens re-verify, detcore HIGH finding —
        # without this the proper-noun allowlist never matched the real data +
        # macron correction was silently disabled for every macron-bearing
        # proper noun: Rōma, Trōia, Cicero, ...).
        raw = _load_json("proper_nouns.json")
        _PROPER_NOUNS = {str(k).lower(): v for k, v in raw.items()
                         if not str(k).startswith("_")}
    return _PROPER_NOUNS


def _get_paradigm_tables() -> Dict[str, Any]:
    global _PARADIGM_TABLES
    if _PARADIGM_TABLES is None:
        _PARADIGM_TABLES = _load_json("paradigm_tables.json")
    return _PARADIGM_TABLES


def _load_json(name: str) -> Dict[str, Any]:
    # v0.3.1: a public read-only data subset ships bundled (paradigm_tables,
    # proper_nouns, macron_lexicon) so the tutor is usable without
    # HERMES_LATIN_DIR. When HERMES_LATIN_DIR is SET (power user / live dir /
    # hermetic tests) respect it EXACTLY — a missing file is {} (preserves the
    # test contract + the user's explicit dir choice; no silent bundled
    # shadowing of a dir the user pointed at). Only when it is UNSET do we fall
    # back to the user dir ~/.hermes/latin, then the bundled read-only data.
    env_dir = os.environ.get("HERMES_LATIN_DIR")
    if env_dir:
        try:
            return json.loads((Path(env_dir) / name).read_text(encoding="utf-8"))
        except Exception:
            return {}
    for d in (_latin_dir(), bundled_latin_data_dir()):
        try:
            return json.loads((d / name).read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _record_error_pattern(ledger: Dict[str, Any], pattern: str) -> None:
    """Accumulate an error pattern (gentle-mode focused form)."""
    now = datetime.now().isoformat()
    patterns = ledger.setdefault("error_patterns", [])
    for ep in patterns:
        if ep.get("pattern") == pattern:
            ep["count"] = int(ep.get("count", 0)) + 1
            ep["last_seen"] = now
            return
    patterns.append({"pattern": pattern, "count": 1, "last_seen": now})
    # F16: bound the list. Evict the single
    # least-frequent entry (lowest count, tie-break oldest last_seen) when the
    # cap is exceeded, so the ledger cannot grow without bound via distinct form
    # strings. Preserves the most-frequent / most-recent patterns the tutor
    # surfaces next session.
    if len(patterns) > MAX_ERROR_PATTERNS:
        _evict_idx = 0
        for _i in range(1, len(patterns)):
            _pi, _ei = patterns[_i], patterns[_evict_idx]
            if (int(_pi.get("count", 0)), _pi.get("last_seen", "")) < (
                    int(_ei.get("count", 0)), _ei.get("last_seen", "")):
                _evict_idx = _i
        patterns.pop(_evict_idx)


def _macronize_token(token_text: str, lemma: str) -> Tuple[str, str]:
    """Return (macronized_form, source) for a token given its lemma. Tries the
    lexicon citation + principal parts (accent-insensitive match). Returns
    (token_text unchanged, "none") if no macronized form is known."""
    lex = _get_lexicon()
    entry = lex.get(lemma)
    if not entry:
        return token_text, "unknown_vocab"
    candidates = []
    cit = entry.get("citation")
    if cit:
        candidates.append(cit)
    for pp in entry.get("principal_parts", []):
        if pp:
            candidates.append(pp)
    bare_tok = _strip_macrons(token_text).lower()
    for cand in candidates:
        if _strip_macrons(cand).lower() == bare_tok:
            return cand, "lexicon"
    return token_text, "macron_unknown"


def latin_validate(latin_string: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Graded gate. Parse the Latin string with LatinCy,
    recover lemmas, attempt deterministic macron correction from the curated
    lexicon, flag-and-warn on proper nouns + unknown macronization, reject on a
    true parse failure (no lemma) OR on non-Latin input (no Latin vowel in any
    token) OR on an oversized/empty string (DoS / empty guards).

    HONEST SCOPE (F13): this is a parse-RECOVERY + macron-
    CORRECTION + vocab-RECOGNITION gate, NOT a grammatical correctness gate. It
    cannot detect agreement / case / tense / voice errors — LatinCy recovers a
    lemma for almost any token (including gibberish), so wrong-but-parseable
    Latin returns warn/accept. Grammar correctness is the LLM teacher's role BY
    DESIGN (paedagogus.md); the deterministic core owns macrons, paradigms, and
    scheduling — not grammar judgment. The macron oracle covers only the starter
    lexicon's citation + principal-parts forms; inflected running prose and
    out-of-lexicon words return macron_unknown/unknown_vocab (warn, not reject).

    Records error patterns on reject/warn. Returns {verdict, lemmas,
    macron_corrections, macronized_text, proper_nouns, unknown_vocab,
    diagnostics}."""
    if not isinstance(latin_string, str) or not latin_string.strip():
        return {"verdict": "reject", "lemmas": [], "macron_corrections": [],
                "macronized_text": "", "proper_nouns": [], "unknown_vocab": [],
                "diagnostics": ["empty input"]}
    # F12: bound the O(n) LatinCy parse. Reject
    # instantly on an oversized string instead of blocking the in-process agent
    # thread for tens of seconds. 8000 chars is far beyond any real tutor passage.
    if len(latin_string) > MAX_LATIN_VALIDATE_CHARS:
        return {"verdict": "reject", "lemmas": [], "macron_corrections": [],
                "macronized_text": latin_string, "proper_nouns": [],
                "unknown_vocab": [],
                "diagnostics": [
                    "input too long: {} chars exceeds {} limit".format(
                        len(latin_string), MAX_LATIN_VALIDATE_CHARS)]}
    try:
        nlp = _get_nlp()
        doc = nlp(latin_string)
    except Exception as exc:
        return {"verdict": "reject", "lemmas": [], "macron_corrections": [],
                "macronized_text": latin_string, "proper_nouns": [],
                "unknown_vocab": [], "diagnostics": ["parse error: {}".format(exc)]}

    proper = _get_proper_nouns()
    lemmas: List[Dict[str, str]] = []
    corrections: List[Dict[str, str]] = []
    proper_found: List[str] = []
    unknown_vocab: List[str] = []
    macron_unknown: List[str] = []
    out_tokens: List[str] = []

    any_lemma = False
    has_latin_vowel = False  # F13: make the reject path meaningful (see _LATIN_VOWELS)
    for tok in doc:
        lemma = (tok.lemma_ or "").strip()
        text = tok.text
        if lemma and lemma.lower() not in {"", "-pron-"}:
            any_lemma = True
        if not has_latin_vowel and any(c in _LATIN_VOWELS for c in text):
            has_latin_vowel = True
        lemmas.append({"text": text, "lemma": lemma, "pos": tok.pos_, "morph": tok.morph_.key if hasattr(tok, "morph_") else str(tok.morph)})
        # proper-noun allowlist: flag-and-warn, apply known citation macrons
        pentry = proper.get(lemma.lower()) if lemma else None
        pentry_cit = proper.get(text.lower())
        if pentry or pentry_cit:
            cit = (pentry or pentry_cit).get("citation", text)
            if _strip_macrons(cit).lower() == _strip_macrons(text).lower() and cit != text:
                corrections.append({"original": text, "corrected": cit})
                out_tokens.append(cit)
            else:
                out_tokens.append(text)
            proper_found.append(text)
            continue
        # lexicon macron correction
        macronized, source = _macronize_token(text, lemma.lower() if lemma else "")
        if source == "lexicon" and macronized != text:
            corrections.append({"original": text, "corrected": macronized})
            out_tokens.append(macronized)
        elif source == "lexicon":
            out_tokens.append(macronized)
        elif source == "unknown_vocab":
            unknown_vocab.append(text)
            out_tokens.append(text)
        else:  # macron_unknown — lemma recovered, inflection not in starter lexicon
            macron_unknown.append(text)
            out_tokens.append(text)

    macronized_text = " ".join(out_tokens)

    # Verdict (graded gate).
    diagnostics: List[str] = []
    # F13: reject on no-lemma (true parse failure) OR no Latin vowel in any
    # token (pure punctuation/numbers/symbols/non-Latin script — LatinCy assigns
    # these lemmas, so the no-lemma branch alone is dead code; the vowel check is
    # the safe discriminator that doesn't brick legitimate out-of-lexicon Latin).
    if not any_lemma or not has_latin_vowel:
        verdict = "reject"
        if not any_lemma:
            diagnostics.append("no lemma recovered — true parse failure")
        if not has_latin_vowel:
            diagnostics.append("no Latin-script vowel in any token — not Latin")
    elif unknown_vocab or macron_unknown or proper_found:
        verdict = "warn"
        if proper_found:
            diagnostics.append("proper nouns flagged: {}".format(", ".join(proper_found)))
        if macron_unknown:
            diagnostics.append("macron unknown for inflected forms (lemma recovered, flagging not rejecting): {}".format(", ".join(macron_unknown)))
        if unknown_vocab:
            diagnostics.append("vocab outside starter lexicon: {}".format(", ".join(unknown_vocab)))
    else:
        verdict = "accept"

    if verdict in ("reject", "warn"):
        try:
            ledger = _read_ledger()
            # Form-granular error patterns (DESIGN §8.6 gentle-mode focused
            # form): record each distinct form BY CATEGORY so the tutor can
            # surface the single most-frequent weak form next session, not a
            # coarse verdict-level string (Phase 5 4-lens re-verify, detcore
            # finding — the prior coarse "validate:warn:..." string could not
            # drive focused-form selection). Reject still records a verdict-
            # level marker (no forms to enumerate on a true parse failure).
            for form in macron_unknown:
                _record_error_pattern(ledger, "macron_unknown:{}".format(form))
            for form in unknown_vocab:
                _record_error_pattern(ledger, "unknown_vocab:{}".format(form))
            for form in proper_found:
                _record_error_pattern(ledger, "proper_noun:{}".format(form))
            if verdict == "reject":
                _record_error_pattern(ledger,
                    "validate:reject:lemmas_recovered={}".format(any_lemma))
            _atomic_write_ledger(ledger)
        except Exception:
            pass  # ledger write is best-effort; the verdict still returns

    # F6: record the input + corrected forms so format_response can detect
    # macronized Latin the LLM emitted WITHOUT going through the gate.
    if verdict in ("accept", "warn"):
        _record_validated(latin_string)
        _record_validated(macronized_text)
        _record_validated([c.get("to") for c in corrections if isinstance(c, dict)])

    return {
        "verdict": verdict,
        "lemmas": lemmas,
        "macron_corrections": corrections,
        "macronized_text": macronized_text,
        "proper_nouns": proper_found,
        "unknown_vocab": unknown_vocab,
        "diagnostics": diagnostics,
    }


def latin_srs(card_id: str, rating: str, front: Optional[str] = None,
              back: Optional[str] = None) -> Dict[str, Any]:
    """FSRS-6 review scheduling + ledger update. If the card is
    new and front+back are supplied, create it; otherwise review the existing
    card. The FSRS scheduler is the SOLE authority on the next due date — the
    LLM never sets the schedule. rating is one of again/hard/good/easy."""
    from fsrs import Scheduler, Card, Rating, State  # lazy

    rating_map = {
        "again": Rating.Again, "hard": Rating.Hard,
        "good": Rating.Good, "easy": Rating.Easy,
    }
    r = rating_map.get(str(rating).strip().lower())
    if r is None:
        return {"success": False, "error": "invalid rating {!r}; use again/hard/good/easy".format(rating)}

    ledger = _read_ledger()
    cards = ledger.setdefault("cards", {})
    card_data = cards.get(card_id)

    if card_data is None:
        if not front or not back:
            return {"success": False, "error":
                    "card {!r} not found; supply front + back to create it".format(card_id)}
        card = Card()
        reps = 0
        lapses = 0
        stored_front = front
        stored_back = back
    else:
        fsrs_state = card_data.get("fsrs", {}) or {}
        card = Card()
        try:
            card.state = State(int(fsrs_state.get("state", 1)))
        except Exception:
            card.state = State.Learning
        card.step = fsrs_state.get("step")  # None once in Review state
        card.stability = fsrs_state.get("stability")
        card.difficulty = fsrs_state.get("difficulty")
        try:
            card.due = datetime.fromisoformat(fsrs_state["due"]) if fsrs_state.get("due") else card.due
        except Exception:
            pass
        try:
            card.last_review = datetime.fromisoformat(fsrs_state["last_review"]) if fsrs_state.get("last_review") else None
        except Exception:
            pass
        reps = int(card_data.get("reps", 0) or 0)
        lapses = int(card_data.get("lapses", 0) or 0)
        stored_front = front if front else card_data.get("front", "")
        stored_back = back if back else card_data.get("back", "")

    scheduler = Scheduler()
    new_card, _log = scheduler.review_card(card, r)
    reps += 1
    if r == Rating.Again:
        lapses += 1

    cards[card_id] = {
        "front": stored_front,
        "back": stored_back,
        "fsrs": {
            "state": int(new_card.state),
            # fsrs sets step=None once the card leaves the Learning state
            # (State.Review); preserve None rather than int(None).
            "step": int(new_card.step) if new_card.step is not None else None,
            "stability": new_card.stability,
            "difficulty": new_card.difficulty,
            "due": new_card.due.isoformat() if new_card.due else None,
            "last_review": new_card.last_review.isoformat() if new_card.last_review else None,
        },
        "reps": reps,
        "lapses": lapses,
    }
    try:
        _atomic_write_ledger(ledger)
    except Exception as exc:
        return {"success": False, "error": "ledger write failed: {}".format(exc)}

    return {
        "success": True,
        "card_id": card_id,
        "due": cards[card_id]["fsrs"]["due"],
        "reps": reps,
        "lapses": lapses,
        "state": cards[card_id]["fsrs"]["state"],
    }


def latin_paradigm(kind: str, args: Optional[str] = None) -> Dict[str, Any]:
    """Static finite declension/conjugation tables. Pure
    lookup — the LLM never generates a paradigm. kind = a table id matching a
    key in paradigm_tables.json (e.g. 'declension:I', 'declension:II_m',
    'conjugation:1_present_active', 'conjugation:sum_present_active'), or
    'list' to enumerate the available tables. Returns the table cells + the A&G
    section citation; for a gap (table not yet curated) returns the available
    list + an A&G pointer so the tutor can cite it."""
    tables = _get_paradigm_tables()
    if not kind or kind == "list":
        decls = sorted((tables.get("declensions") or {}).keys())
        conjs = sorted((tables.get("conjugations") or {}).keys())
        return {"available": {"declensions": ["declension:" + k for k in decls],
                              "conjugations": ["conjugation:" + k for k in conjs]},
                "note": "starter scope; gaps cite A&G. Expansion is a later phase."}
    # kind = "<section>:<id>"
    section, _, tid = kind.partition(":")
    block = (tables.get(section + "s") or tables.get(section) or {})
    entry = block.get(tid)
    if not entry:
        return {"available": "use latin_paradigm kind='list' for the table ids",
                "ag_reference": "see Allen & Greenough New Latin Grammar (1903)",
                "note": "table {!r} not yet curated".format(kind)}
    return {"kind": kind, "table": entry,
            "ag_reference": entry.get("ag_section", "A&G"),
            "note": "static lookup — deterministic, not LLM-generated"}