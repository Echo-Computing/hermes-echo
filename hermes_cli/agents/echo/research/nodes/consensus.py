"""Consensus node — final check on code execution results.

No LLM calls — pure logic comparing results from parallel code execution instances.
"""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState


async def run_consensus(state: ResearchState) -> ResearchState:
    """Consensus node: review all code execution results and update hypothesis confidence.

    This is a lightweight post-processing step that:
    1. Reviews consensus results from code execution
    2. Flags hypotheses with strong experimental support
    3. Updates ELO ratings based on experimental evidence
    """

    results = state.get("code_execution_results", [])
    hypotheses = state.get("hypotheses", [])

    if not results:
        logger.info("Consensus: no execution results to process", extra={"category": "RESEARCH"})
        return state

    logger.info(
        "Consensus: processing {} execution result sets".format(len(results)),
        extra={"category": "RESEARCH"},
    )

    for result in results:
        hyp_id = result.get("hypothesis_id", "")
        verdict = result.get("verdict", "inconclusive")
        consensus_reached = result.get("consensus_reached", False)
        majority_finding = result.get("majority_finding", "")

        # Find the hypothesis
        for h in hypotheses:
            if h.get("id") == hyp_id:
                # Boost ELO for experimentally supported hypotheses
                if consensus_reached and verdict == "accepted":
                    h["elo_rating"] = h.get("elo_rating", 1500) + 25
                    logger.info(
                        "Consensus: {} — ELO boosted for experimental support".format(
                            h.get("title", "")[:50]
                        ),
                        extra={"category": "RESEARCH"},
                    )
                elif not consensus_reached:
                    # Slight penalty for inconclusive results
                    h["elo_rating"] = h.get("elo_rating", 1500) - 10

                # Attach consensus finding to hypothesis
                if majority_finding:
                    critiques = h.get("critiques", [])
                    critiques.append({
                        "severity": "info",
                        "critique": "Code analysis consensus: {}".format(majority_finding),
                        "verdict": verdict,
                    })
                    h["critiques"] = critiques
                break

    # Log summary
    accepted = sum(1 for r in results if r.get("verdict") == "accepted")
    inconclusive = sum(1 for r in results if r.get("verdict") == "inconclusive")
    logger.info(
        "Consensus: {} accepted, {} inconclusive".format(accepted, inconclusive),
        extra={"category": "RESEARCH"},
    )

    return state
