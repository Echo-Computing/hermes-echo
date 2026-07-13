# hermes-echo

> **hermes-echo** — an extended fork of [yasutoshi-lab/Hermes](https://github.com/yasutoshi-lab/Hermes), maintained by [Echo-Computing](https://github.com/Echo-Computing).
>
> MIT license; upstream copyright retained. The original Hermes is a local-LLM research agent built on Ollama; this fork keeps all of it and adds an interactive tool-using chat agent, a Latin tutor, and a collaborative research mode that reuses the upstream research graph.
>
> A Japanese translation of the upstream README lives at [README_JA.md](./README_JA.md) (pending update for the fork's additions).

## 1. What hermes-echo is

Upstream **Hermes** is a locally executable CLI research agent: it runs web searches through SearxNG, validates findings with an LLM loop, and writes cited Markdown reports — all on a local LLM via Ollama, with no external API billing.

**hermes-echo** keeps that research pipeline intact and layers two new surfaces on top of the same local-LLM core:

- an interactive **Echo agent** — a LangGraph chat agent that can read/write/edit files, run shell commands, search code and history, write to memory, search the web, and fetch URLs (SSRF-guarded), all from a chat loop; and
- a **Latin tutor** (`hermes echo --latin`) — a deterministic spaced-repetition paedagogus that uses a morphological parser and static paradigm tables as the source of truth, with the LLM only rendering and declining.

The fork also adds a **collaborative research mode** (`hermes echo --research`) that drives the upstream research graph from the chat agent's tool surface.

## 2. The Echo agent and its tools

`hermes echo` opens an interactive chat session. The agent can call any of these tools:

| Tool | What it does | Guard |
|------|--------------|-------|
| `read_file` | Read a file from disk | path gate |
| `write_file` | Write a file to disk | protected-roots / guard-source path gate |
| `edit_file` | Apply an exact-string edit to a file | protected-roots / guard-source path gate |
| `run_shell` | Run a shell command via `subprocess.run` | F17 wall-clock timeout + protected-roots/guard-source path gate |
| `search_code` | Grep the working tree (ripgrep-backed) | path gate |
| `search_history` | Search prior session history | read-only |
| memory write | Persist a tagged memory entry | tag scrub / no-memory sentinel |
| `search_web` | Web search (SearxNG) | upstream provider |
| `fetch_url` | Fetch a URL and convert to markdown | **SSRF `check_url` floor** |
| `graph` | Query / mutate a tree-sitter code graph | **destructive-tool confirmer + `UNTRUSTED_TOOL_OUTPUT` fence** |

Every tool result that comes back into the agent context is wrapped by the `UNTRUSTED_TOOL_OUTPUT` / `UNTRUSTED_MEMORY` injection fences, so tool output cannot inject new instructions into the prompt.

## 3. The learning loop

The Echo agent captures four kinds of learning across sessions:

- **correction** — when the user corrects the agent, the correction is reflected into a memory entry so the same mistake is not repeated;
- **fact** — durable facts the user asks the agent to remember;
- **idea** — half-formed ideas (`/idea`, persisted with `/idea save`) that can be recalled later without committing them as facts; and
- **session** — end-of-session summaries written to memory.

These are written through the memory tool and reloaded on the next `hermes echo` start, so the agent's behaviour improves incrementally without retraining the underlying model.

## 4. The Latin tutor (`hermes echo --latin`)

The Latin tutor is a **deterministic core** with the LLM confined to rendering and declining:

- **FSRS-6 spaced repetition** schedules each paradigm card;
- a **LatinCy parse gate** (`la_core_web_lg`) validates that the learner's answer parses as the expected morphology before it is scored — the LLM never decides whether a form is correct;
- a **macron lexicon** is the source of truth for vowel length;
- **static paradigm tables** (declension / conjugation / case-use) are the source of truth for the forms themselves — the LLM only renders the prompt and declines a form the tables say exists; and
- a `/translate` escape hatch drops into free translation when the learner wants out of the drill.

Set `HERMES_LATIN_DIR` to point the tutor at a directory of paradigm and lexicon data.

## 5. Collaborative research (`hermes echo --research`)

```
hermes echo --research "your research question" --rounds 4 --debates 3
```

Research mode drives the upstream Hermes research graph — query generation, parallel SearxNG search, content analysis, draft aggregation, and the validator loop — from the Echo agent's tool surface. `--rounds` controls the validation loop depth; `--debates` controls how many adversarial review passes run before the final report.

## 6. Why it's better than upstream

Concrete, non-marketing deltas over `yasutoshi-lab/Hermes`:

- **interactive tool-using chat agent** — upstream is batch (`hermes run`); the fork adds a stateful chat loop with file/shell/search/memory/web/graph tools;
- **SSRF guard** on `fetch_url` (`check_url` floor blocks private-network targets);
- **injection fences** (`UNTRUSTED_TOOL_OUTPUT` / `UNTRUSTED_MEMORY`) wrapping every tool result that re-enters the prompt;
- **Latin tutor** with a deterministic FSRS-6 + parse-gate + paradigm-table core;
- **graph tool** — a tree-sitter DSL for querying and mutating a code graph, behind a destructive-tool confirmer;
- **learning loop** — correction / fact / idea / session capture that persists across sessions;
- **tool certificate + seam integrity attestation** — every registered tool carries a certificate, and seamed tools are integrity-checked at import time, so a swapped tool module is detected rather than silently loaded;
- **generalized tool certificate** — the cert mechanism is not special-cased to one tool family.

The upstream research pipeline, SearxNG integration, Langfuse tracing, and config/task/history commands are unchanged.

## 7. Install

```bash
git clone https://github.com/Echo-Computing/hermes-echo.git
cd hermes-echo
pip install -e .
```

Requirements:

- **Python 3.10+**
- **Ollama** running on `localhost:11434` with a model pulled (e.g. `ollama pull ...`)
- (optional) `pip install -e ".[graph]"` for the tree-sitter code-graph tool (`tree-sitter`, `tree-sitter-python`, `networkx`)
- (optional) `python -m spacy download la_core_web_lg` for the Latin tutor's parse gate

## 8. Usage

```bash
# interactive tool-using chat agent
hermes echo

# collaborative multi-agent research mode
hermes echo --research "your research question" --rounds 4 --debates 3

# Latin tutor (deterministic spaced-repetition drill)
hermes echo --latin
```

In-session commands inside `hermes echo`:

| Command | Action |
|---------|--------|
| `/help` | list commands |
| `/clear` | clear the conversation context |
| `/model` | show / switch the active Ollama model |
| `/idea` | capture a half-formed idea |
| `/idea save` | persist the current idea to memory |
| `/translate` | (Latin tutor) escape hatch into free translation |

The upstream batch commands (`hermes init`, `hermes task`, `hermes run`, `hermes log`, `hermes history`) are unchanged; see `doc/command/`.

## 9. Security posture (honest)

This is an honest statement of what ships and what does not.

**Floor gates that ship in this build:**

- **SSRF `check_url`** — `fetch_url` rejects private-network and link-local targets before any request is made;
- **destructive-tool confirmer** — graph mutations and other destructive tool calls require confirmation;
- **injection fences** — `UNTRUSTED_TOOL_OUTPUT` and `UNTRUSTED_MEMORY` wrap every tool result that re-enters the prompt, so tool output cannot inject instructions;
- **path gates** — protected-roots and guard-source markers stop file/shell tools from touching paths outside the workspace;
- **seam certificate + integrity attestation** — seamed tools are integrity-checked at import time; a swapped module is detected, not silently loaded;
- **tool certificate** — every registered tool carries a certificate;
- **no-memory scrub** — the `NO_MEMORY` sentinel and tag-scrub machinery prevent tagged tool output from being written to long-term memory (the scrub machinery ships dormant and is wired where the memory tool writes).

**Ceilings NOT in the public build (these stay in a private fork):**

- a **mount-namespace sandbox ceiling** — the floor gates above ship; the sandbox ceiling that would confine tool execution in an isolated mount namespace does not. A future release may reintroduce the sandbox ceiling via an installable profile.
- a **dark-web OSINT transport** — not shipped;
- the **red-team tool stubs** — not shipped;
- the **affect substrate** (`--continuous` / `--lesion`) — not shipped;
- **extension auto-discovery** — the extension directory is excluded from the public build, so tool plugins are not auto-loaded.

This is an honest protection downgrade relative to the private fork: the floor gates are real and active; the execution sandbox ceiling is not.

## 10. Banned-token release guard

The fork ships a release guard that fails any push whose diff reintroduces private-surface tokens:

- `.githooks/pre-push` — a local pre-push hook;
- `.github/workflows/banned-tokens.yml` — a CI workflow that scans on push and pull request.

Enable the local hook with:

```bash
git config core.hooksPath .githooks
```

## 11. What's usable at v0.3.0

### v0.3.0 — first public release

**NEW:**

- Latin tutor (`hermes echo --latin`) — deterministic FSRS-6 spaced repetition + LatinCy parse gate + macron lexicon + static paradigm tables as source of truth; the LLM renders and declines only.
- graph tool — tree-sitter DSL for querying and mutating a code graph, behind a destructive-tool confirmer and an `UNTRUSTED_TOOL_OUTPUT` fence.
- SSRF-safe `fetch_url` — `check_url` floor blocks private-network targets.
- injection fences — `UNTRUSTED_TOOL_OUTPUT` and `UNTRUSTED_MEMORY` wrap every tool result that re-enters the prompt.
- destructive-tool confirmer — graph mutations and other destructive calls require confirmation.
- no-memory scrub machinery (dormant) — `NO_MEMORY` sentinel + tag scrub wired where the memory tool writes.
- SeamedTool certificate + seam integrity attestation — swapped seamed modules are detected at import time.
- generalized tool certificate — the cert mechanism is not special-cased to one tool family.
- learning-loop refinements — correction / fact / idea / session capture.
- banned-token release guard — `.githooks/pre-push` + `.github/workflows/banned-tokens.yml`.

**USABLE AT v0.3.0:**

- interactive `hermes echo` + all tools (read/write/edit, run_shell, search_code, search_history, memory, search_web, fetch_url, graph);
- `hermes echo --research` with `--rounds` / `--debates`;
- `hermes echo --latin` Latin tutor;
- graph tool;
- SSRF-safe fetch;
- injection fences + destructive confirmer + path gates + seam cert + integrity attestation;
- learning loop.

**NOT IN THIS RELEASE (private fork):**

- the affect substrate (`--continuous` / `--lesion`);
- the mount-namespace sandbox ceiling — the floor gates ship, the ceiling does not; a future release may reintroduce it via an installable profile;
- the red-team tool stubs;
- the dark-web OSINT transport;
- ToolPlugin extension auto-discovery.

## 12. License and attribution

MIT License. Upstream copyright belongs to the [yasutoshi-lab/Hermes](https://github.com/yasutoshi-lab/Hermes) authors and is retained; fork additions are (c) Echo-Computing, released under the same MIT terms.

## 13. Acknowledgements

- The [yasutoshi-lab/Hermes](https://github.com/yasutoshi-lab/Hermes) team — the local-LLM research agent this fork builds on, including the SearxNG integration, the LangGraph research pipeline, and the validation loop.
- The Ollama project for the local inference runtime.
- The LatinCy / spaCy `la_core_web_lg` model authors, and the FSRS-6 spaced-repetition algorithm authors, used by the Latin tutor's deterministic core.
- The tree-sitter project, used by the graph tool.