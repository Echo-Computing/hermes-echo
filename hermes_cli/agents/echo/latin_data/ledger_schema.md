# Ledger schema — `HERMES_LATIN_DIR/ledger.json`

The ledger is the tutor's **only persistent state**. Owned by the deterministic-core tools (`latin_srs` writes after every review; `latin_validate` writes error patterns on reject/warn; `load_latin_state` reads pre-LLM). It is NOT Hermes memory (memory entries are untrusted data the model ignores; the ledger must not live there).

Cold-started by `load_latin_state` on first run if absent. Atomic writes (tmp + rename). The in-process `execution_sandbox="none"` tool writes the ledger directly to `HERMES_LATIN_DIR/ledger.json` (default `~/.hermes/latin/ledger.json`).

## Schema (version 1)

```json
{
  "version": 1,
  "profile": {
    "current_ginn_ch": <int|null>,
    "current_fr_ch": <int|null>,
    "stage": <1|2>,
    "week": <int>,
    "vocab_count": <int>
  },
  "skills": {
    "<skill_id>": {
      "mastery": <0.0-1.0>,
      "last_practiced": "<iso8601>"
    }
  },
  "cards": {
    "<card_id>": {
      "front": "<str>",
      "back": "<str>",
      "fsrs": {
        "stability": <float>,
        "difficulty": <float>,
        "last_review": "<iso8601>",
        "due": "<iso8601>",
        "reps": <int>,
        "lapses": <int>
      }
    }
  },
  "error_patterns": [
    {"pattern": "<str>", "count": <int>, "last_seen": "<iso8601>"}
  ],
  "paradigm_only_flags": ["<construction Ginn teaches before FR catches up>"],
  "sessions": [
    {"date": "<iso8601>", "ginn_ch": <int>, "fr_ch": <int>, "weak_spots": ["<str>"]}
  ]
}
```

## Skill IDs (per-skill mastery)

Curriculum-aligned, mirroring the Year-1/Year-2 milestones:

- Declensions: `decl_I`, `decl_II`, `decl_III`, `decl_IV`, `decl_V`
- Conjugations (active+passive, all tenses): `conj_1`, `conj_2`, `conj_3`, `conj_4`, `conj_3_io`
- Irregular verbs: `irreg_sum`, `irreg_fero`, `irreg_volo`, `irreg_eo`, `irreg_fio`
- Pronouns: `pronouns_personal`, `pronouns_relative`, `pronouns_demonstrative`
- Voice/mood: `subjunctive_present`, `subjunctive_imperfect`, `subjunctive_perfect`, `subjunctive_pluperfect`, `passive_periphrastic`, `gerundive`
- Constructions: `ablative_absolute`, `participles`, `gerund`, `indirect_discourse`, `purpose_clauses`, `result_clauses`, `cum_clauses`, `indirect_questions`, `conditionals`, `comparison`
- Production: `composition_en_to_la`, `spoken_latin`

## Paradigm-only flags (bidirectional ordering)

A construction is flagged here when Ginn front-loads its paradigm before FR supplies in-context reading. The tutor teaches the paradigm (strict drills) but records the gap; the flag clears when FR catches up.

Starter flag: `subjunctive` (Ginn LIV–LX teaches the paradigm; in-context FR reading is FR 27+, Year 2).

## FSRS-6 fields

`stability` + `difficulty` are FSRS-6 per-card memory state (the `fsrs` lib updates them on every review). `due` is the scheduler's next-review timestamp — the tutor surfaces cards where `due <= now`. `reps` = total reviews, `lapses` = times lapsed to relearning. The tutor NEVER sets these from the LLM — `latin_srs` is the only writer.

## Error patterns

Accumulated by `latin_validate` on `reject`/`warn` verdicts. `{pattern, count, last_seen}` — the tutor surfaces recurring patterns to focus the next session's strict-drill block (gentle-mode focused-form logic).