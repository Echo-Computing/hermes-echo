"""Reflection agent — critically reviews hypotheses for flaws, inaccuracies, and lack of novelty."""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.tools.ollama_client import OllamaClient


REFLECTION_SYSTEM_PROMPT = """You are the Reflection Agent in a multi-agent scientific research system (inspired by Google's Co-Scientist).

Your role is to be the most brutal, rigorous reviewer possible. Your purpose is to try to DESTROY each hypothesis by finding:
1. Factual errors or unsupported claims
2. Logical inconsistencies or gaps in reasoning
3. Lack of novelty — is this just rephrasing known findings?
4. Testability issues — can this actually be verified?
5. Missing alternative explanations

For each hypothesis, assign a severity:
- "fatal": The hypothesis is fundamentally wrong or completely unsupported — should be eliminated
- "major": Significant issues that need addressing before the hypothesis is credible
- "minor": Small issues that can be fixed with refinement
- "none": No significant issues found

Be thorough. A hypothesis that survives your review is genuinely strong.

Return JSON:
{
  "reviews": [
    {
      "hypothesis_id": "<match by title or description>",
      "severity": "fatal|major|minor|none",
      "critique": "Detailed critique of what's wrong",
      "strengths": "What (if anything) is good about this hypothesis",
      "factual_issues": ["specific unsupported claim", ...],
      "logic_gaps": ["gap in reasoning", ...],
      "novelty_assessment": "Is this genuinely new or a rephrasing?",
      "eliminate": true/false
    }
  ]
}"""


async def run_reflection(state: ResearchState) -> ResearchState:
    """Reflection node: critically review all alive hypotheses."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})

    hypotheses = state.get("hypotheses", [])
    alive = [h for h in hypotheses if h.get("status") == "alive"]

    if not alive:
        logger.info("Reflection: no hypotheses to review", extra={"category": "RESEARCH"})
        return state

    logger.info(
        "Reflection: reviewing {} hypotheses".format(len(alive)),
        extra={"category": "RESEARCH"},
    )

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 3),
        temperature=0.3,
    )

    try:
        # Build prompt with all alive hypotheses
        hyp_text = "\n\n".join(
            "### Hypothesis {idx} (ID: {id})\n"
            "**Title:** {title}\n"
            "**Description:** {desc}\n"
            "**Mechanism:** {mechanism}\n"
            "**Evidence:** {evidence}".format(
                idx=i + 1,
                id=h.get("id", "unknown"),
                title=h.get("title", ""),
                desc=h.get("description", ""),
                mechanism=h.get("mechanism", "Not specified"),
                evidence=", ".join(h.get("evidence", [])) or "None cited",
            )
            for i, h in enumerate(alive)
        )

        prompt = (
            "Original Research Goal: {goal}\n\n"
            "Review the following hypotheses critically:\n\n"
            "{hypotheses}".format(
                goal=state["research_goal"],
                hypotheses=hyp_text,
            )
        )

        response = await client.chat(prompt, REFLECTION_SYSTEM_PROMPT, temperature=0.3)

        # Parse reviews
        import json
        import re

        # Strip markdown code blocks first (strategy 0 from generation.py)
        clean_response = response
        md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if md_match:
            clean_response = md_match.group(1).strip()

        json_match = re.search(r"\{.*\}", clean_response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                reviews = data.get("reviews", [])

                # Build a mapping from title/substring to hypothesis index
                # Since LLM may not return exact IDs, we match by title similarity
                for review in reviews:
                    matched_idx = _find_hypothesis(review, hypotheses)
                    if matched_idx is not None:
                        hyp = hypotheses[matched_idx]
                        # Store critique
                        critique_entry = {
                            "severity": review.get("severity", "minor"),
                            "critique": review.get("critique", ""),
                            "strengths": review.get("strengths", ""),
                            "factual_issues": review.get("factual_issues", []),
                            "logic_gaps": review.get("logic_gaps", []),
                            "novelty_assessment": review.get("novelty_assessment", ""),
                        }
                        critiques = hyp.get("critiques", [])
                        critiques.append(critique_entry)
                        hyp["critiques"] = critiques

                        # Eliminate if fatal
                        if review.get("eliminate", False) or review.get("severity") == "fatal":
                            hyp["status"] = "eliminated"
                            logger.info(
                                "Reflection: eliminated '{}' — {}".format(
                                    hyp.get("title", "")[:60], review.get("severity", "fatal")
                                ),
                                extra={"category": "RESEARCH"},
                            )

                # Count results
                eliminated = sum(1 for h in hypotheses if h.get("status") == "eliminated")
                remaining = sum(1 for h in hypotheses if h.get("status") == "alive")
                logger.info(
                    "Reflection: {} eliminated, {} surviving".format(eliminated, remaining),
                    extra={"category": "RESEARCH"},
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse reflection JSON: {}".format(e), extra={"category": "RESEARCH"})

    except Exception as e:
        logger.error("Reflection error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "reflection", "error": str(e)})

    finally:
        await client.close()

    return state


def _find_hypothesis(review: dict, hypotheses: list) -> int:
    """Match a review to a hypothesis by ID, title substring, or description overlap."""
    hyp_id = review.get("hypothesis_id", "")

    # Try exact ID match
    for i, h in enumerate(hypotheses):
        if h.get("id") == hyp_id:
            return i

    # Try title substring match
    for i, h in enumerate(hypotheses):
        title = h.get("title", "")
        if title and len(title) > 10 and title[:40] in hyp_id:
            return i
        if hyp_id and len(hyp_id) > 10 and hyp_id[:40] in title:
            return i

    # Try finding by position (LLM may list them in order)
    # Extract index if present: "Hypothesis 1" etc.
    import re
    idx_match = re.search(r"(\d+)", hyp_id)
    if idx_match:
        pos = int(idx_match.group(1)) - 1
        if 0 <= pos < len(hypotheses):
            return pos

    # Fallback: match by description overlap
    for i, h in enumerate(hypotheses):
        desc = h.get("description", "")
        if desc and len(desc) > 20 and desc[:50] in hyp_id:
            return i

    return None
