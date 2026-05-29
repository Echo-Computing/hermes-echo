"""Evolution agent — refines hypotheses based on critiques and combines complementary ones."""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.agents.echo.research.models import Hypothesis
from hermes_cli.tools.ollama_client import OllamaClient


EVOLUTION_SYSTEM_PROMPT = """You are the Evolution Agent in a multi-agent scientific research system (inspired by Google's Co-Scientist).

Your job is to take hypotheses that survived reflection (with their critiques) and:
1. Refine each hypothesis to address specific critiques
2. Identify hypotheses that complement each other and can be combined
3. Bridge logical gaps identified during reflection
4. Strengthen mechanistic explanations

For each surviving hypothesis, produce a refined version that:
- Addresses every specific critique
- Fills identified logic gaps
- Strengthens the mechanistic explanation
- Is more specific and testable

Also identify 0-2 pairs of hypotheses that could be combined into a single, stronger hypothesis.

Return JSON:
{
  "refinements": [
    {
      "original_hypothesis": "<title or ID>",
      "refined_title": "Refined hypothesis title",
      "refined_description": "Improved description addressing all critiques",
      "refined_mechanism": "Stronger mechanistic explanation",
      "changes_made": ["what was improved", ...]
    }
  ],
  "combinations": [
    {
      "hypothesis_a": "<title or ID>",
      "hypothesis_b": "<title or ID>",
      "combined_title": "Title for the combined hypothesis",
      "combined_description": "Synthesis of both ideas",
      "combined_mechanism": "How the combined mechanisms work together",
      "synergy": "Why these two are stronger together than apart"
    }
  ]
}"""


async def run_evolution(state: ResearchState) -> ResearchState:
    """Evolution node: refine hypotheses and combine complementary ones."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})

    hypotheses = state.get("hypotheses", [])
    alive = [h for h in hypotheses if h.get("status") == "alive"]

    if not alive:
        logger.info("Evolution: no hypotheses to evolve", extra={"category": "RESEARCH"})
        return state

    logger.info(
        "Evolution: refining {} hypotheses".format(len(alive)),
        extra={"category": "RESEARCH"},
    )

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 3),
        temperature=0.5,
    )

    try:
        # Build prompt with hypotheses and their critiques
        hyp_parts = []
        for i, h in enumerate(alive):
            part = "### Hypothesis {idx}\n**ID:** {id}\n**Title:** {title}\n**Description:** {desc}\n**Mechanism:** {mechanism}\n".format(
                idx=i + 1,
                id=h.get("id", ""),
                title=h.get("title", ""),
                desc=h.get("description", ""),
                mechanism=h.get("mechanism", ""),
            )
            critiques = h.get("critiques", [])
            if critiques:
                part += "**Critiques to address:**\n"
                for c in critiques:
                    part += "- [{sev}] {crit}\n".format(
                        sev=c.get("severity", "minor"),
                        crit=c.get("critique", ""),
                    )
                    for gap in c.get("logic_gaps", []):
                        part += "  - Logic gap: {}\n".format(gap)
            hyp_parts.append(part)

        prompt = (
            "Original Research Goal: {goal}\n\n"
            "Refine these hypotheses and combine complementary ones:\n\n"
            "{hypotheses}".format(
                goal=state["research_goal"],
                hypotheses="\n\n".join(hyp_parts),
            )
        )

        response = await client.chat(prompt, EVOLUTION_SYSTEM_PROMPT, temperature=0.5)

        import json
        import re

        # Strip markdown code blocks first
        clean_response = response
        md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if md_match:
            clean_response = md_match.group(1).strip()

        json_match = re.search(r"\{.*\}", clean_response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                round_num = state.get("current_round", 1)

                # Process refinements
                refinements = data.get("refinements", [])
                title_to_refinement = {}
                for ref in refinements:
                    orig_title = ref.get("original_hypothesis", "")
                    title_to_refinement[orig_title] = ref

                for h in hypotheses:
                    if h.get("status") != "alive":
                        continue
                    # Check if this hypothesis has a refinement
                    matched = False
                    for orig_title, ref in title_to_refinement.items():
                        if _titles_match(orig_title, h.get("title", "")) or orig_title == h.get("id", ""):
                            h["title"] = ref.get("refined_title", h["title"])
                            h["description"] = ref.get("refined_description", h["description"])
                            h["mechanism"] = ref.get("refined_mechanism", h["mechanism"])
                            h["status"] = "refined"
                            history = h.get("refinement_history", [])
                            history.append("Round {}: {}".format(
                                round_num,
                                "; ".join(ref.get("changes_made", ["refined"]))
                            ))
                            h["refinement_history"] = history
                            matched = True
                            break

                # Process combinations
                combinations = data.get("combinations", [])
                for combo in combinations:
                    combined = Hypothesis(
                        title=combo.get("combined_title", ""),
                        description=combo.get("combined_description", ""),
                        mechanism=combo.get("combined_mechanism", ""),
                        elo_rating=1600.0,  # Slight boost for being a synthesis
                        round_created=round_num,
                        status="alive",
                    )

                    # Find and mark the source hypotheses
                    ids_to_merge = []
                    for key in ("hypothesis_a", "hypothesis_b"):
                        target = combo.get(key, "")
                        for h in hypotheses:
                            if h.get("status") in ("alive", "refined"):
                                if _titles_match(target, h.get("title", "")) or target == h.get("id", ""):
                                    ids_to_merge.append(h.get("id", ""))
                                    h["status"] = "merged"
                                    break

                    combined.merged_from = ids_to_merge
                    hypotheses.append(combined.to_dict())
                    logger.info(
                        "Evolution: combined into '{}'".format(combined.title[:60]),
                        extra={"category": "RESEARCH"},
                    )

                # Count results
                alive_count = sum(1 for h in hypotheses if h.get("status") in ("alive", "refined"))
                merged_count = sum(1 for h in hypotheses if h.get("status") == "merged")
                logger.info(
                    "Evolution: {} alive/refined, {} merged into others, {} new combinations".format(
                        alive_count, merged_count, len(combinations)
                    ),
                    extra={"category": "RESEARCH"},
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse evolution JSON: {}".format(e), extra={"category": "RESEARCH"})

    except Exception as e:
        logger.error("Evolution error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "evolution", "error": str(e)})

    finally:
        await client.close()

    return state


def _titles_match(a: str, b: str) -> bool:
    """Check if two titles likely refer to the same hypothesis."""
    if not a or not b:
        return False
    a_clean = a.lower().strip()
    b_clean = b.lower().strip()
    if a_clean == b_clean:
        return True
    if len(a_clean) > 15 and a_clean[:30] in b_clean:
        return True
    if len(b_clean) > 15 and b_clean[:30] in a_clean:
        return True
    return False
