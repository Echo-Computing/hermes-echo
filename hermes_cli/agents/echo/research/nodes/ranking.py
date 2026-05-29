"""Ranking agent — ELO tournament system for hypothesis evaluation.

Pairs hypotheses in head-to-head debates judged by an AI. Uses ELO rating
math (adapted from chess) to rank ideas by novelty, plausibility, and impact.
"""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.agents.echo.research.models import (
    DebateResult,
    Hypothesis,
    elo_expected,
    elo_update,
    select_debate_pairs,
    ELO_K_FACTOR,
)
from hermes_cli.tools.ollama_client import OllamaClient


JUDGE_SYSTEM_PROMPT = """You are a scientific judge in an automated hypothesis tournament (inspired by Google's Co-Scientist ranking system).

Two hypotheses are competing. Evaluate both based on:

1. **Novelty** — Is this genuinely new, or a rephrasing of known ideas?
2. **Plausibility** — Is the mechanism scientifically sound?
3. **Testability** — Can this be empirically verified?
4. **Impact** — If true, how significant would this discovery be?

Be objective and rigorous. Do not favor longer descriptions over concise ones.

Return JSON:
{
  "winner": "A" or "B" or "tie",
  "score_a": 0.0-1.0,
  "score_b": 0.0-1.0,
  "reasoning": "Brief explanation of the decision",
  "strengths_a": "What A does well",
  "strengths_b": "What B does well",
  "weakness_a": "Where A falls short",
  "weakness_b": "Where B falls short"
}"""


async def run_ranking(state: ResearchState) -> ResearchState:
    """Ranking node: run ELO tournament on all alive hypotheses."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})
    research_config = config.get("research", {})

    max_debates = research_config.get("debates_per_round", 10)

    hypotheses = [h for h in state.get("hypotheses", []) if h.get("status") in ("alive", "refined")]

    if len(hypotheses) < 2:
        logger.info("Ranking: not enough hypotheses for tournament", extra={"category": "RESEARCH"})
        return state

    # Convert to Hypothesis objects for ELO math
    hyp_objects = [Hypothesis.from_dict(h) for h in hypotheses]

    # Select debate pairs by ELO proximity
    pairs = select_debate_pairs(hyp_objects, max_debates)

    if not pairs:
        logger.info("Ranking: no valid debate pairs", extra={"category": "RESEARCH"})
        return state

    logger.info(
        "Ranking: starting tournament — {} debates among {} hypotheses".format(
            len(pairs), len(hyp_objects)
        ),
        extra={"category": "RESEARCH"},
    )

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 3),
        temperature=0.2,
    )

    tournament_results = state.get("tournament_results", [])
    id_to_hyp = {h.id: h for h in hyp_objects}

    try:
        for debate_num, (hyp_a, hyp_b) in enumerate(pairs):
            logger.info(
                "Ranking: debate {}/{} — {} vs {}".format(
                    debate_num + 1, len(pairs),
                    hyp_a.title[:50], hyp_b.title[:50],
                ),
                extra={"category": "RESEARCH"},
            )

            prompt = (
                "Debate {idx}/{total}\n\n"
                "### Hypothesis A (ELO: {elo_a:.0f})\n"
                "**Title:** {title_a}\n"
                "**Description:** {desc_a}\n"
                "**Mechanism:** {mech_a}\n\n"
                "### Hypothesis B (ELO: {elo_b:.0f})\n"
                "**Title:** {title_b}\n"
                "**Description:** {desc_b}\n"
                "**Mechanism:** {mech_b}\n\n"
                "Judge these two hypotheses and declare a winner."
            ).format(
                idx=debate_num + 1,
                total=len(pairs),
                elo_a=hyp_a.elo_rating,
                title_a=hyp_a.title,
                desc_a=hyp_a.description,
                mech_a=hyp_a.mechanism or "Not specified",
                elo_b=hyp_b.elo_rating,
                title_b=hyp_b.title,
                desc_b=hyp_b.description,
                mech_b=hyp_b.mechanism or "Not specified",
            )

            try:
                response = await client.chat(prompt, JUDGE_SYSTEM_PROMPT, temperature=0.2)

                import json
                import re
                json_match = re.search(r"\{.*\}", response, re.DOTALL)

                if json_match:
                    try:
                        verdict = json.loads(json_match.group())
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                    winner = verdict.get("winner", "tie")

                    # Calculate ELO updates
                    expected_a = elo_expected(hyp_a.elo_rating, hyp_b.elo_rating)
                    expected_b = 1.0 - expected_a

                    if winner == "A":
                        actual_a, actual_b = 1.0, 0.0
                    elif winner == "B":
                        actual_a, actual_b = 0.0, 1.0
                    else:  # tie
                        actual_a, actual_b = 0.5, 0.5

                    elo_a_before = hyp_a.elo_rating
                    elo_b_before = hyp_b.elo_rating

                    hyp_a.elo_rating = elo_update(hyp_a.elo_rating, expected_a, actual_a, ELO_K_FACTOR)
                    hyp_b.elo_rating = elo_update(hyp_b.elo_rating, expected_b, actual_b, ELO_K_FACTOR)

                    result = DebateResult(
                        hypothesis_a_id=hyp_a.id,
                        hypothesis_b_id=hyp_b.id,
                        winner=winner,
                        reasoning=verdict.get("reasoning", ""),
                        elo_a_before=elo_a_before,
                        elo_b_before=elo_b_before,
                        elo_a_after=hyp_a.elo_rating,
                        elo_b_after=hyp_b.elo_rating,
                    )

                    tournament_results.append({
                        "a_id": result.hypothesis_a_id,
                        "b_id": result.hypothesis_b_id,
                        "winner": result.winner,
                        "reasoning": result.reasoning,
                        "elo_a_before": result.elo_a_before,
                        "elo_b_before": result.elo_b_before,
                        "elo_a_after": result.elo_a_after,
                        "elo_b_after": result.elo_b_after,
                    })

            except Exception as e:
                logger.warning(
                    "Debate {}/{} failed: {}".format(debate_num + 1, len(pairs), e),
                    extra={"category": "RESEARCH"},
                )
                continue

        # --- Post-tournament: update hypotheses in state ---
        # Update ELO ratings
        for hyp_obj in hyp_objects:
            for h in hypotheses:
                if h.get("id") == hyp_obj.id:
                    h["elo_rating"] = hyp_obj.elo_rating
                    break

        # Sort by ELO and eliminate bottom 25% if they dropped
        sorted_hyps = sorted(hyp_objects, key=lambda h: h.elo_rating, reverse=True)
        cutoff_idx = max(int(len(sorted_hyps) * 0.75), 1)

        for hyp_obj in sorted_hyps[cutoff_idx:]:
            # Only eliminate if they participated in debates and their ELO dropped
            debates_participated = sum(
                1 for r in tournament_results
                if r.get("a_id") == hyp_obj.id or r.get("b_id") == hyp_obj.id
            )
            if debates_participated > 0 and hyp_obj.elo_rating < 1450:
                for h in hypotheses:
                    if h.get("id") == hyp_obj.id:
                        h["status"] = "eliminated"
                        break

        state["tournament_results"] = tournament_results

        # Log leaderboard
        leaderboard = sorted(
            [h for h in hypotheses if h.get("status") in ("alive", "refined")],
            key=lambda h: h.get("elo_rating", 1500),
            reverse=True,
        )
        if leaderboard:
            logger.info(
                "Ranking: tournament complete — top: '{}' ({:.0f} ELO)".format(
                    leaderboard[0].get("title", "")[:60],
                    leaderboard[0].get("elo_rating", 1500),
                ),
                extra={"category": "RESEARCH"},
            )

    except Exception as e:
        logger.error("Ranking error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "ranking", "error": str(e)})

    finally:
        await client.close()

    return state
