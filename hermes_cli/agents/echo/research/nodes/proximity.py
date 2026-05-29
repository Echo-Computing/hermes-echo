"""Proximity agent — detects duplicate/similar hypotheses using LLM-based semantic comparison."""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.tools.ollama_client import OllamaClient


PROXIMITY_SYSTEM_PROMPT = """You are the Proximity Agent in a multi-agent scientific research system (inspired by Google's Co-Scientist).

Your job is to compare pairs of hypotheses and determine if they are essentially the same idea expressed differently.

For each pair, evaluate:
1. Are they proposing the same underlying mechanism?
2. Would they be tested the same way?
3. Are the differences just surface-level wording?

Return JSON:
{
  "comparisons": [
    {
      "hypothesis_a": "<title or ID of first>",
      "hypothesis_b": "<title or ID of second>",
      "are_duplicates": true/false,
      "similarity": 0.0-1.0,
      "reasoning": "Why they are or aren't duplicates",
      "better_version": "a|b|neither"  — which is the stronger formulation, if duplicates
    }
  ],
  "groups": [
    ["hypothesis_title_1", "hypothesis_title_2"],  — groups of duplicates
  ]
}"""


async def run_proximity(state: ResearchState) -> ResearchState:
    """Proximity node: detect and group duplicate hypotheses."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})

    hypotheses = state.get("hypotheses", [])
    alive = [h for h in hypotheses if h.get("status") == "alive"]

    if len(alive) < 2:
        logger.info("Proximity: not enough hypotheses to compare", extra={"category": "RESEARCH"})
        return state

    logger.info(
        "Proximity: comparing {} hypotheses for duplicates".format(len(alive)),
        extra={"category": "RESEARCH"},
    )

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 3),
        temperature=0.2,
    )

    try:
        # Build prompt comparing all alive hypotheses
        hyp_list = "\n\n".join(
            "### Hypothesis {idx}\n"
            "**ID:** {id}\n"
            "**Title:** {title}\n"
            "**Description:** {desc}\n"
            "**Mechanism:** {mechanism}".format(
                idx=i + 1,
                id=h.get("id", ""),
                title=h.get("title", ""),
                desc=h.get("description", ""),
                mechanism=h.get("mechanism", ""),
            )
            for i, h in enumerate(alive)
        )

        prompt = (
            "Compare the following hypotheses and identify which are essentially "
            "the same idea expressed differently:\n\n"
            "{hypotheses}".format(hypotheses=hyp_list)
        )

        response = await client.chat(prompt, PROXIMITY_SYSTEM_PROMPT, temperature=0.2)

        import json
        import re
        json_match = re.search(r"\{.*\}", response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                groups = data.get("groups", [])

                # Process groups: mark duplicates, keep the best version
                for group in groups:
                    if len(group) < 2:
                        continue

                    # Find the hypotheses in this group
                    group_hyps = []
                    for title in group:
                        idx = _find_by_title_or_id(title, hypotheses)
                        if idx is not None:
                            group_hyps.append(idx)

                    if len(group_hyps) < 2:
                        continue

                    # Keep the one with highest ELO, mark others as merged
                    best_idx = max(group_hyps, key=lambda i: hypotheses[i].get("elo_rating", 1500))
                    best_hyp = hypotheses[best_idx]

                    for idx in group_hyps:
                        if idx != best_idx:
                            hyp = hypotheses[idx]
                            hyp["status"] = "merged"
                            # Track merge lineage
                            merged_from = best_hyp.get("merged_from", [])
                            merged_from.append(hyp.get("id", ""))
                            best_hyp["merged_from"] = merged_from
                            logger.info(
                                "Proximity: merged '{}' into '{}'".format(
                                    hyp.get("title", "")[:50],
                                    best_hyp.get("title", "")[:50],
                                ),
                                extra={"category": "RESEARCH"},
                            )

                logger.info(
                    "Proximity: processed {} duplicate groups".format(len(groups)),
                    extra={"category": "RESEARCH"},
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse proximity JSON: {}".format(e), extra={"category": "RESEARCH"})

    except Exception as e:
        logger.error("Proximity error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "proximity", "error": str(e)})

    finally:
        await client.close()

    return state


def _find_by_title_or_id(title_or_id: str, hypotheses: list) -> int:
    """Find a hypothesis by exact ID match or title substring."""
    for i, h in enumerate(hypotheses):
        if h.get("id") == title_or_id:
            return i
    for i, h in enumerate(hypotheses):
        if title_or_id[:30] in h.get("title", ""):
            return i
    return None
