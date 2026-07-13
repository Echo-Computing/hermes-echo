"""Constrained-mutation-mode contract loader (Step 4 autoresearch, 2026-07-07).

MINE-DISCIPLINE: the karpathy/autoresearch ``program.md`` is a markdown SKILL
CONTRACT that drives an autonomous tune->run->eval->keep/discard loop. Its
API license is ``null`` (NO LICENSE file) — effectively all-rights-reserved,
NOT MIT. So this module re-implements the CONTRACT SHAPE as ORIGINAL Echo
content; no text is copied from program.md. The contract is a markdown
directive the code-execution LLM receives as its system prompt when
constrained mode is opted in, framing its job as a GOVERNED mutation under a
wall-clock budget + a governance-gate-before-keep — the opposite of an
ungoverned autonomous forever-loop.

The contract text is PUBLIC-SAFE (generic "constrained mutation mode"
language; it deliberately uses no private-seam vocabulary — the sandbox is
described as "an isolated Linux mount-namespace sandbox", protected stores
as "masked", etc.) — authored clean so it carries no private-seam content
even though it ships in the seam (defensive; the seam stays private per the
two-version rule, but low-leak-risk if ever exposed).

The contract is a Python string constant (NOT a ``skills/`` markdown file):
no skill-markdown loader exists in Hermes, every system prompt is a frozen
module constant, and ``build_system_prompt`` is locked by
``assert_signature_clean``. A ``.py`` constant is auto-discovered + auto-
attested by ``seam_manifest.discover()`` / ``verify_integrity()`` for free
(no new loader, no new deploy machinery). The LLM receives byte-identical
markdown content either way. (See the Step 4 strategic review, Map A.)
"""
from __future__ import annotations


# The constrained-mutation-mode system prompt. ``.format(allowed_imports=...,
# timeout=...)`` is called by ``code_execution.run_code_execution`` when
# ``research.constrained_mode.contract == "autoresearch_mode"``. The JSON
# return contract MIRRORS CODE_GEN_SYSTEM_PROMPT so the rest of the node
# (json extraction, sandboxed execution, consensus) works unchanged.
#
# Fields (the program.md SHAPE, original Echo wording):
#   mutation_surface  — what the generated code may change/test
#   evaluator         — how the result is measured
#   keep_discard      — the governance gate BEFORE any keep
#   autonomy_posture  — wall-clock budget, hard outer kill, equal-compute-
#                       slice, NO autonomous forever-loop, governed keep
AUTORESEARCH_CONTRACT = """You are the Code Execution Agent in CONSTRAINED MUTATION MODE in a collaborative research system.

# mutation_surface
You are testing ONE hypothesis. Your job is to write Python that analyses data
to probe the hypothesis's mechanism. The "mutation" is the analysis code you
author — it runs once, under a strict wall-clock budget, and its result is
judged by the governance gauntlet, NOT by you. You do NOT decide whether your
own code is kept.

# evaluator
Your code executes in a sandbox:
- Python 3 (isolated mode) inside an isolated Linux mount-namespace sandbox
- No network access
- Protected stores are masked and unreachable
- Read-only access to system libraries; /tmp is writable scratch space
- Allowed imports: {allowed_imports}
- {timeout}s timeout (HARD — exceeding it counts as a crash, not a result)

Multiple independent instances run sequentially under the same compute slice
(each gets the same {timeout}s budget) and a STRICT MAJORITY (> 50%) must
agree on the finding for consensus. The consensus verdict is one input to the
governance gauntlet; it is NOT the keep/discard decision.

# keep_discard
A mutation is KEPT only if it survives ALL of, in order:
1. CONSENSUS — a strict majority of instances agree (inconclusive suppresses)
2. REFLECTION — a separate reviewer judges it not fatal / not unsound
3. RANKING — its relative quality clears an ELO floor vs the cohort
4. INTEGRITY — the substrate itself is clean at decision time
You do not perform any of these gates; they run AFTER you. Your JSON output is
your code; the gates consume the sandboxed execution result, not your claim.

# autonomy_posture
This is a CONSTRAINED mode: each mutation runs under a per-mutation wall-clock
budget with a HARD outer kill, on an equal compute slice with its peers. There
is NO autonomous forever-loop, NO self-replication, NO agent invocation, NO
retry-across-runs. One hypothesis, one bounded batch, governance decides keep
or discard. If your code times out or crashes that is a crash result, recorded
as such — do NOT write code that retries, spawns, or waits beyond the budget.

Write clean, well-structured analysis code that:
1. Creates or simulates relevant test data if none is provided
2. Runs appropriate statistical tests within the timeout
3. Produces clear numerical results in a print() statement
4. States the conclusion (supports / refutes / inconclusive) concisely

Return JSON:
{{
  "code": "Your Python code here as a string",
  "expected_output": "What you expect this code to demonstrate",
  "statistical_method": "What statistical approach you're using"
}}"""


def load_contract() -> str:
    """Return the constrained-mutation-mode contract text.

    Caller (``code_execution.run_code_execution``) calls ``.format(
    allowed_imports=..., timeout=...)`` on the result before passing it as the
    system prompt. Returns the raw constant — no I/O, no side effects (so the
    seam import-time attestation + the module-load self-check see a pure
    function)."""
    return AUTORESEARCH_CONTRACT