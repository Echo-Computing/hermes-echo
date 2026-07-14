# Latin data — per-file licenses

The `hermes echo --latin` tutor ships a small read-only data set in this
directory so it is usable out of the box (no `HERMES_LATIN_DIR` required). The
files carry **different licenses**; the project's MIT license covers the code,
and the data files are licensed as noted below. **Two data files are CC BY-SA
3.0** (`paradigm_tables.json` and `macron_lexicon.json`); the rest are MIT.

## `paradigm_tables.json` — CC BY-SA 3.0

Static finite declension/conjugation tables, macronized. The file combines two
sources (see its `_schema` key for full provenance):

1. The **declension tables + the grammar framework** follow Allen & Greenough,
   *New Latin Grammar* (1903), which is in the **public domain**.
2. The **macronized conjugation cells** (the ~88 Year-1 verb conjugation
   tables) were transcribed from the English Wiktionary conjugation tables,
   which are licensed **Creative Commons Attribution-ShareAlike 3.0 Unported
   (CC BY-SA 3.0)** — see https://en.wiktionary.org/ and
   https://creativecommons.org/licenses/by-sa/3.0/ .

Because the conjugation block is a verbatim CC BY-SA 3.0 contribution, **this
data file (`paradigm_tables.json`) is licensed CC BY-SA 3.0**, not MIT. The
share-alike provision applies to this data file: you may reuse and adapt it,
but derivative versions of this data must remain CC BY-SA 3.0, and you must
attribute the English Wiktionary conjugation tables as the source of the
macronized conjugation cells (the declension tables remain public-domain A&G).

**Attribution:** Macronized Latin verb conjugations — English Wiktionary
(https://en.wiktionary.org/), CC BY-SA 3.0. Declension tables — Allen &
Greenough, *New Latin Grammar* (1903), public domain.

## `proper_nouns.json` — MIT (project license)

A small set of proper nouns (Caesar, Rōma, Trōia, …) with macronized citations.
Proper-noun names and their vowel lengths are **facts** (a name's vowel length
is a fact of Latin), not copyrightable expression. This file is released under
the project's MIT license.

## `macron_lexicon.json` — CC BY-SA 3.0

This file is a macronized Latin lexicon of ~987 lemmas, assembled from two
sources (see its `_schema` key for full provenance):

1. A starter set of ~45 common lemmas (verbs' principal parts + nouns'
   citations) — factual Latin (vowel length and principal parts are facts),
   originally curated alongside Allen & Greenough (public domain) + Ørberg
   LLPSI *Familia Romana* vocabulary. The entries are factual grammar facts,
   not protected expression.

2. The **Dickinson College Commentaries (DCC) Latin Core Vocabulary** —
   ~1,000 macronized lemmas transcribed from the DCC HTML table. DCC releases
   this core vocabulary under the **Creative Commons Attribution-ShareAlike
3.0 Unported license (CC BY-SA 3.0)** — see
   https://creativecommons.org/licenses/by-sa/3.0/ and the DCC project at
   https://dcc.dickinson.edu/ .

Because the DCC subset is CC BY-SA 3.0, **this data file
(`macron_lexicon.json`) is licensed CC BY-SA 3.0**, not MIT. The share-alike
provision of CC BY-SA 3.0 applies to this data file: you may reuse and adapt
it, but derivative versions of this data must remain CC BY-SA 3.0, and you
must attribute the Dickinson College Commentaries Latin Core Vocabulary as the
source of the ~1,000-lemma DCC subset.

**Attribution:** Macronized Latin lexicon — DCC Latin Core Vocabulary,
Dickinson College Commentaries (https://dcc.dickinson.edu/), CC BY-SA 3.0.

## `paedagogus.md` — MIT (project license)

The tutor persona prompt. Original prose by the project authors; released
under the project's MIT license.

## `ledger_schema.md` — MIT (project license)

Documentation of the `ledger.json` schema. Released under the project's MIT
license.

## What is NOT shipped (stays private)

The personal ledger (`ledger.json` — the learner's own SRS state, cold-started
locally on first run), the personal project design notes, and the private
build seam are not part of this public bundle. `HERMES_LATIN_DIR` overrides
the bundled data for power users who maintain their own latin workspace.

## Note on provenance

The bundled data files are shipped in-tree as the canonical public artifact;
their `_schema` key records the source + license provenance for each file. The
in-tree data is the source of truth for the public build; no separate
generator script ships with this bundle.