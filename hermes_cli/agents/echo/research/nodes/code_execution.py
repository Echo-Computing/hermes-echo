"""Code Execution agent (Finch-style) -- writes and executes Python to analyze data.

Launches multiple independent instances (run SEQUENTIALLY, one after another,
under the same per-instance compute slice) with consensus voting.
Adapted from Robin: 3 instances (not 8) for local execution.

SEAM OVERRIDE of the upstream hermes_cli/agents/echo/research/nodes/
code_execution.py. The upstream version ran LLM-authored Python via
``subprocess.run(["python3","-I", script_path])`` with NO FS masks, NO net
isolation, and an import blacklist that OMITS ``os`` -- so ``import os;
os.system(...)`` could read protected stores, exfiltrate over the network, and
``open(...,'w')`` the guard source files. This override runs the code via
``python3 -I -`` (isolated mode, stdin pipe) with an expanded import blacklist
(BLOCKED_MODULES) prepended as a guard hook. The mount-namespace ceiling is not
in the public build; the import guard + isolated mode are the floor. The
CODE_GEN_SYSTEM_PROMPT is corrected: the upstream falsely told the LLM there
was no network/FS access while nothing enforced that.
"""

import subprocess
import tempfile
import os
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.agents.echo.research.models import CodeExecutionResult, ConsensusResult
from hermes_cli.tools.ollama_client import OllamaClient
# Step 4 autoresearch: constrained-mutation-mode contract. Imported here (not
# at the top of the module) ONLY because it is a seam-sibling; load_contract()
# is a pure constant return, no I/O, so this import adds no exfil surface.
from hermes_cli.agents.echo.research.contract_loader import load_contract


# Sandbox restrictions
SANDBOX_TIMEOUT = 30  # seconds
ALLOWED_IMPORTS = {
    "numpy", "scipy", "pandas", "matplotlib",
    "json", "csv", "math", "statistics",
    "collections", "itertools", "pathlib", "re",
    "typing", "dataclasses", "functools", "operator",
    "time", "random",
}
# Defense-in-depth import blacklist. The upstream omitted the process /
# async-spawn modules, so ``import subprocess`` (and ``asyncio.subprocess``,
# ``multiprocessing``, ``pty``) let sandboxed code spawn helper processes.
# The import guard hook is the floor boundary in the public build (the
# mount-namespace ceiling is not shipped); it must NOT be relied on as the
# sole boundary in a deployment that wires private stores.
#
# IMPORTANT: the import hook intercepts TRANSITIVE imports, so we CANNOT block
# modules that numpy/pandas/scipy/matplotlib import during their own init.
# ``os``, ``signal``, ``fcntl`` and ``builtins`` are all imported
# transitively by the allowed libs (e.g. numpy/__init__.py does ``import os``)
# -- blocking them would break ``import numpy`` itself. They are deliberately
# NOT listed. Only process-spawn modules that the allowed libs do NOT pull in
# transitively are blocked here.
BLOCKED_MODULES = [
    "subprocess", "socket", "requests", "ctypes", "shutil",
    "pty", "multiprocessing", "asyncio",
]

CODE_GEN_SYSTEM_PROMPT = """You are the Code Execution Agent (Finch) in a scientific research system (inspired by the Robin multi-agent system from Nature).

Your job is to write Python code that analyzes data to test a hypothesis.
The code will be executed in a sandbox with:
- Python 3 (isolated mode, -I flag) with an import guard
- No network libraries (socket/requests blocked by import guard)
- No subprocess spawning (subprocess/multiprocessing/pty blocked)
- Allowed imports: {allowed_imports}
- {timeout}s timeout

Write clean, well-structured analysis code that:
1. Creates or simulates relevant test data if none is provided
2. Runs appropriate statistical tests
3. Produces clear numerical results
4. States the conclusion clearly in a print() statement

Return JSON:
{{
  "code": "Your Python code here as a string",
  "expected_output": "What you expect this code to demonstrate",
  "statistical_method": "What statistical approach you're using"
}}"""


async def run_code_execution(state: ResearchState) -> ResearchState:
    """Code Execution node: run Finch-style sequential multi-instance analysis on testable hypotheses."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})
    research_config = config.get("research", {})

    num_instances = research_config.get("parallel_instances", 3)

    # Step 4 autoresearch: constrained-mutation mode (opt-in, default off).
    # When ``constrained_mode.contract == "autoresearch_mode"`` the system
    # prompt is swapped for the governed-mutation contract; per-mutation
    # wall-clock budget + a hard outer kill bound the batch; equal-compute-
    # slice = the same timeout= value per sequential instance call. The keep/
    # discard DECISION is NOT made here -- it is the governance gauntlet's
    # composite (reflection+consensus+ranking+integrity) at format_report. The
    # ``proposal`` dict emitted below is INFORMATIONAL and overwritten by the
    # gauntlet (Goodhart defense: no scalar metric decides keep).
    constrained = research_config.get("constrained_mode", {}) or {}
    per_mutation_budget = int(constrained.get("per_mutation_budget", SANDBOX_TIMEOUT))
    # The outer_kill default is scaled to the configured budget so a legitimate
    # large per_mutation_budget cannot silently trigger premature outer-kill of
    # later instances (instances run SEQUENTIALLY, so the worst-case batch time
    # is num_instances * per_mutation_budget; +30s slack for LLM + overhead).
    # An explicit outer_kill in config still wins. Default floor of 10*
    # SANDBOX_TIMEOUT preserves the unconstrained-path behavior (huge, never
    # reached) when the budget is the module default.
    _default_outer_kill = max(10 * SANDBOX_TIMEOUT, num_instances * per_mutation_budget + 30)
    outer_kill = float(constrained.get("outer_kill", _default_outer_kill))
    constrained_on = bool(constrained) and constrained.get("contract") == "autoresearch_mode"

    hypotheses = state.get("hypotheses", [])
    # Focus on top-ranked hypotheses (by ELO) for code analysis
    alive = sorted(
        [h for h in hypotheses if h.get("status") in ("alive", "refined")],
        key=lambda h: h.get("elo_rating", 1500),
        reverse=True,
    )

    if not alive:
        logger.info("CodeExecution: no hypotheses to test", extra={"category": "RESEARCH"})
        return state

    # Take the top 1-2 hypotheses for code analysis (expensive operation)
    top_hypotheses = alive[:2]

    logger.info(
        "CodeExecution: analyzing {} hypotheses with {} instances each".format(
            len(top_hypotheses), num_instances
        ),
        extra={"category": "RESEARCH"},
    )

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 2),
        temperature=0.4,
    )

    all_results = state.get("code_execution_results", [])

    try:
        for hyp in top_hypotheses:
            # --- Generate code from each instance (sequential, equal compute slice) ---
            instances = []
            # Step 4: constrained mode swaps the system prompt for the
            # governed-mutation contract (still produces the SAME JSON return
            # shape, so the rest of the node is unchanged). The timeout
            # placeholder reflects the per-mutation budget, not the module
            # default -- so the LLM knows the real hard bound it is under.
            if constrained_on:
                system_prompt = load_contract().format(
                    allowed_imports=", ".join(sorted(ALLOWED_IMPORTS)),
                    timeout=per_mutation_budget,
                )
            else:
                system_prompt = CODE_GEN_SYSTEM_PROMPT.format(
                    allowed_imports=", ".join(sorted(ALLOWED_IMPORTS)),
                    timeout=SANDBOX_TIMEOUT,
                )

            for i in range(num_instances):
                prompt = (
                    "Write Python code to test this hypothesis:\n\n"
                    "**Hypothesis:** {title}\n"
                    "**Description:** {desc}\n"
                    "**Mechanism:** {mechanism}\n\n"
                    "Generate analysis code (instance {instance_num}/{total}). "
                    "Approach the problem independently -- use your own statistical methods.".format(
                        title=hyp.get("title", ""),
                        desc=hyp.get("description", ""),
                        mechanism=hyp.get("mechanism", "Not specified"),
                        instance_num=i + 1,
                        total=num_instances,
                    )
                )

                try:
                    response = await client.chat(prompt, system_prompt, temperature=0.4)

                    import json as _json
                    import re as _re

                    # Strip markdown code blocks first
                    clean_response = response
                    md_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, _re.DOTALL)
                    if md_match:
                        clean_response = md_match.group(1).strip()

                    json_match = _re.search(r"\{.*\}", clean_response, _re.DOTALL)

                    if json_match:
                        data = _json.loads(json_match.group())
                        code = data.get("code", "")
                    else:
                        # Fallback: extract code block
                        code_match = _re.search(r"```python\n(.*?)```", response, _re.DOTALL)
                        code = code_match.group(1) if code_match else response

                    instances.append({
                        "instance_id": i + 1,
                        "code": code,
                        "expected_output": data.get("expected_output", "") if json_match else "",
                    })

                except Exception as e:
                    logger.warning(
                        "Code gen instance {} failed: {}".format(i + 1, e),
                        extra={"category": "RESEARCH"},
                    )
                    instances.append({
                        "instance_id": i + 1,
                        "code": "# Generation failed: {}".format(e),
                        "expected_output": "",
                    })

            # --- Execute each instance in sandbox ---
            # Step 4: constrained mode imposes a hard OUTER kill across the
            # whole instance batch (a wall-clock budget per hypothesis, not
            # just per instance). Equal-compute-slice = the same per-instance
            # timeout= value; the outer kill bounds the SUM. A batch that
            # blows the outer kill stops; remaining instances are recorded as
            # timed-out (crash) -- NOT run. The default outer_kill is scaled to
            # the configured per_mutation_budget (max(10*SANDBOX_TIMEOUT,
            # num_instances*budget+30)) so a large budget does not prematurely
            # kill later instances; an explicit outer_kill in config wins.
            # NOTE: outer_kill is a per-batch START gate (checked at loop top
            # only), NOT a hard mid-instance wall-clock kill.
            execution_results = []
            deadline = time.monotonic() + outer_kill
            for inst in instances:
                if time.monotonic() > deadline:
                    # outer kill: mark remaining instances as timed-out crash
                    # results so the gauntlet sees them as crash (not silent
                    # skip). The keep/discard decision still runs downstream.
                    execution_results.append(CodeExecutionResult(
                        instance_id=inst["instance_id"],
                        code=inst["code"],
                        stdout="",
                        stderr="outer kill: batch exceeded {}s wall-clock budget".format(outer_kill),
                        exit_code=-1,
                        success=False,
                        findings="",
                    ))
                    continue
                result = _execute_sandboxed(
                    inst["code"], inst["instance_id"],
                    timeout=per_mutation_budget if constrained_on else SANDBOX_TIMEOUT,
                )
                execution_results.append(result)

            # --- Analyze results with LLM to extract findings ---
            for result in execution_results:
                if result.success and result.stdout.strip():
                    findings_prompt = (
                        "Analyze this code execution output in the context of the hypothesis:\n\n"
                        "**Hypothesis:** {title}\n"
                        "**Execution stdout:**\n{stdout}\n"
                        "**Execution stderr:**\n{stderr}\n\n"
                        "In 1-2 sentences, what does this result tell us about the hypothesis? "
                        "Does it support, refute, or is it inconclusive?".format(
                            title=hyp.get("title", ""),
                            stdout=result.stdout[:2000],
                            stderr=result.stderr[:500],
                        )
                    )
                    try:
                        findings = await client.chat(findings_prompt, temperature=0.3)
                        result.findings = findings.strip()
                    except Exception:
                        result.findings = result.stdout[:200] if result.stdout else "No output"

            # --- Consensus ---
            consensus = _compute_consensus(execution_results)

            all_results.append({
                "hypothesis_id": hyp.get("id", ""),
                "hypothesis_title": hyp.get("title", ""),
                "instances": [
                    {
                        "instance_id": r.instance_id,
                        "success": r.success,
                        "stdout": r.stdout[:500],
                        "stderr": r.stderr[:300],
                        "findings": r.findings,
                    }
                    for r in execution_results
                ],
                "consensus_reached": consensus.consensus_reached,
                "agreeing_instances": consensus.agreeing_instances,
                "total_instances": consensus.total_instances,
                "majority_finding": consensus.majority_finding,
                "verdict": consensus.verdict,
                # Step 4: INFORMATIONAL keep/discard PROPOSAL. This is NOT the
                # keep/discard decision -- the governance gauntlet
                # (reflection+consensus+ranking+integrity at format_report)
                # makes that decision and OVERWRITES this proposal. Recorded
                # here so the provenance ledger's governance_verdict can cite
                # what the code-execution layer proposed vs what the gauntlet
                # decided (Goodhart observability: the metric the LLM cites is
                # not the metric that decides keep).
                "proposal": {
                    "action": "keep" if consensus.consensus_reached else "discard",
                    "budget_used": per_mutation_budget if constrained_on else SANDBOX_TIMEOUT,
                    "constrained_mode": constrained_on,
                    "reason": "consensus {} ({} of {} instances)".format(
                        consensus.verdict,
                        consensus.agreeing_instances,
                        consensus.total_instances,
                    ),
                },
            })

            logger.info(
                "CodeExecution: hypothesis '{}' -- consensus: {} ({}/{})".format(
                    hyp.get("title", "")[:50],
                    consensus.verdict,
                    consensus.agreeing_instances,
                    consensus.total_instances,
                ),
                extra={"category": "RESEARCH"},
            )

        state["code_execution_results"] = all_results

    except Exception as e:
        logger.error("CodeExecution error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "code_execution", "error": str(e)})

    finally:
        await client.close()

    return state


def _execute_sandboxed(code: str, instance_id: int, timeout: int = SANDBOX_TIMEOUT) -> CodeExecutionResult:
    """Execute Python code in a guarded subprocess.

    Runs the LLM-authored code via ``python3 -I -`` (isolated mode, piped over
    stdin so no temp script file is written to the host FS). A defense-in-depth
    import blacklist (BLOCKED_MODULES) is prepended as a guard hook; the import
    guard is the floor boundary in the public build (the mount-namespace
    ceiling is not shipped). ``python3 -I`` (isolated mode) ignores
    PYTHONPATH/user-site so system-installed numpy/pandas/scipy are importable
    from the standard interpreter path. The import guard intercepts
    transitive imports, blocking process-spawn + network modules that the
    allowed libs do NOT pull in transitively.
    """

    blocked_list = sorted(BLOCKED_MODULES)

    # Build guard using repr() to avoid format-string collisions
    guard_lines = [
        "# Sandbox import guard (defense-in-depth blacklist; the import hook",
        "# is the floor boundary in the public build).",
        "import builtins",
        "_original_import = builtins.__import__",
        "",
        "def _sandbox_import(name, *args, **kwargs):",
        "    blocked = " + repr(blocked_list),
        "    top_level = name.split('.')[0]",
        "    if top_level in blocked:",
        "        raise ImportError(" + repr("Module '{}' is blocked in sandbox") + ".format(repr(top_level)))",
        "    return _original_import(name, *args, **kwargs)",
        "",
        "builtins.__import__ = _sandbox_import",
        "",
        "# Execute the analysis code",
        "",
    ]
    guard = "\n".join(guard_lines)

    # Pipe the guard + LLM-authored code to python3 -I - over stdin so no temp
    # script file is written to the host FS. python3 -I (isolated mode) ignores
    # PYTHONPATH/user-site; system-installed numpy/pandas/scipy are importable
    # from the standard interpreter path.
    stdin_payload = guard + "\n" + code

    try:
        result = subprocess.run(
            ["python3", "-I", "-"],
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return CodeExecutionResult(
            instance_id=instance_id,
            code=code,
            stdout=result.stdout[:5000],
            stderr=result.stderr[:2000],
            exit_code=result.returncode,
            success=result.returncode == 0,
            findings="",
        )
    except subprocess.TimeoutExpired:
        return CodeExecutionResult(
            instance_id=instance_id,
            code=code,
            stdout="",
            stderr="Execution timed out after {} seconds".format(timeout),
            exit_code=-1,
            success=False,
            findings="",
        )
    except Exception as e:
        return CodeExecutionResult(
            instance_id=instance_id,
            code=code,
            stdout="",
            stderr=str(e),
            exit_code=-1,
            success=False,
            findings="",
        )


def _compute_consensus(results: List[CodeExecutionResult]) -> ConsensusResult:
    """Compute consensus across code execution instances.

    A strict majority (> 50%, i.e. threshold = total // 2 + 1) must agree for
    consensus to be reached.
    """
    total = len(results)
    successful = [r for r in results if r.success]

    if not successful:
        return ConsensusResult(
            total_instances=total,
            agreeing_instances=0,
            consensus_reached=False,
            majority_finding=None,
            all_findings=[],
            verdict="inconclusive",
        )

    findings = [r.findings for r in successful if r.findings]

    if not findings:
        return ConsensusResult(
            total_instances=total,
            agreeing_instances=0,
            consensus_reached=False,
            majority_finding=None,
            all_findings=findings,
            verdict="inconclusive",
        )

    # Simple majority: group similar findings by keyword overlap
    # In production, you'd use embedding similarity; here we use word overlap
    groups = _group_similar_findings(findings)
    largest_group = max(groups, key=len)
    agreeing = len(largest_group)

    # Consensus requires true majority: > 50% (e.g., 2 of 3, 2 of 2, 3 of 4)
    threshold = total // 2 + 1
    consensus_reached = agreeing >= threshold

    verdict = "accepted" if consensus_reached else "inconclusive"

    return ConsensusResult(
        total_instances=total,
        agreeing_instances=agreeing,
        consensus_reached=consensus_reached,
        majority_finding=largest_group[0] if largest_group else None,
        all_findings=findings,
        verdict=verdict,
    )


def _group_similar_findings(findings: List[str]) -> List[List[str]]:
    """Group findings by word overlap similarity.

    Simple Jaccard-like grouping: two findings are similar if they share
    significant word overlap.
    """
    if len(findings) <= 1:
        return [findings] if findings else []

    # Tokenize
    tokenized = []
    for f in findings:
        tokens = set(f.lower().split())
        tokenized.append(tokens)

    # Group by overlap
    groups = []
    used = set()

    for i in range(len(findings)):
        if i in used:
            continue
        group = [findings[i]]
        used.add(i)
        for j in range(i + 1, len(findings)):
            if j in used:
                continue
            # Jaccard similarity
            intersection = tokenized[i] & tokenized[j]
            union = tokenized[i] | tokenized[j]
            if union and len(intersection) / len(union) > 0.3:
                group.append(findings[j])
                used.add(j)
        groups.append(group)

    return groups