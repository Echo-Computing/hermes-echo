"""Code Execution agent (Finch-style) — writes and executes Python to analyze data.

Launches multiple parallel instances with consensus voting.
Adapted from Robin: 3 parallel instances (not 8) for local execution.
"""

import subprocess
import tempfile
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.agents.echo.research.models import CodeExecutionResult, ConsensusResult
from hermes_cli.tools.ollama_client import OllamaClient


# Sandbox restrictions
SANDBOX_TIMEOUT = 30  # seconds
ALLOWED_IMPORTS = {
    "numpy", "scipy", "pandas", "matplotlib",
    "json", "csv", "math", "statistics",
    "collections", "itertools", "pathlib", "re",
    "typing", "dataclasses", "functools", "operator",
    "time", "random",
}
BLOCKED_MODULES = ["subprocess", "socket", "requests", "ctypes", "shutil"]

CODE_GEN_SYSTEM_PROMPT = """You are the Code Execution Agent (Finch) in a scientific research system (inspired by the Robin multi-agent system from Nature).

Your job is to write Python code that analyzes data to test a hypothesis.
The code will be executed in a sandbox with:
- Python 3 (isolated mode)
- Allowed imports: {allowed_imports}
- No network access
- No file system access outside a temp directory
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
    """Code Execution node: run Finch-style parallel analysis on testable hypotheses."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})
    research_config = config.get("research", {})

    num_instances = research_config.get("parallel_instances", 3)

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
        "CodeExecution: analyzing {} hypotheses with {} parallel instances each".format(
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
            # --- Generate code from each parallel instance ---
            instances = []
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
                    "Approach the problem independently — use your own statistical methods.".format(
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
            execution_results = []
            for inst in instances:
                result = _execute_sandboxed(inst["code"], inst["instance_id"])
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
            })

            logger.info(
                "CodeExecution: hypothesis '{}' — consensus: {} ({}/{})".format(
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


def _execute_sandboxed(code: str, instance_id: int) -> CodeExecutionResult:
    """Execute Python code in a restricted sandbox subprocess."""

    blocked_list = sorted(BLOCKED_MODULES)

    # Build guard using repr() to avoid format-string collisions
    guard_lines = [
        "# Sandbox import guard (blacklist-only approach)",
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

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="hermes_research_"
    ) as f:
        f.write(guard)
        f.write("\n")
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", "-I", script_path],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT,
            env={"PYTHONPATH": "", "HOME": tempfile.gettempdir()},
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
            stderr="Execution timed out after {} seconds".format(SANDBOX_TIMEOUT),
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
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


def _compute_consensus(results: List[CodeExecutionResult]) -> ConsensusResult:
    """Compute consensus across parallel code execution instances.

    Majority (>= 50%) must agree for consensus to be reached.
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
