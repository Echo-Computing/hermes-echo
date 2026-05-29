"""Supervisor agent — decomposes research goals into sub-questions and controls rounds."""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.tools.ollama_client import OllamaClient


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent in a multi-agent scientific research system (inspired by Google's Co-Scientist).

Your role is purely administrative:
1. Decompose the research goal into 3-5 specific, testable sub-questions
2. Track what has been learned so far
3. Identify what still needs investigation

You do NOT generate hypotheses yourself — that is the Generation Agent's job.

For each sub-question, think about:
- What specific aspect of the problem does it address?
- What kind of evidence would answer it?
- Is it narrow enough to be investigable?

Return your analysis as JSON:
{
  "sub_questions": ["question 1", "question 2", ...],
  "rationale": "Why these sub-questions were chosen",
  "knowledge_gaps": ["what we still don't know", ...]
}"""


SUPERVISOR_REFINE_PROMPT = """You are the Supervisor Agent. Review progress after a research round and plan the next.

Current round: {current_round}/{max_rounds}
Original goal: {goal}

Surviving hypotheses from last round:
{hypotheses_summary}

Tournament results:
{tournament_summary}

What was learned:
{learnings}

Determine:
1. Should we continue to another round? (consider: are hypotheses converging? are there unexplored angles?)
2. If continuing: refine sub-questions based on what we've learned
3. If stopping: summarize why

Return JSON:
{{
  "continue": true/false,
  "reasoning": "Why continue or stop",
  "refined_sub_questions": ["updated question 1", ...] or [],
  "new_focus_areas": ["specific angles to explore next", ...]
}}"""


async def run_supervisor(state: ResearchState) -> ResearchState:
    """Supervisor node: decompose goal or refine for next round."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})
    research_config = config.get("research", {})

    max_rounds = research_config.get("max_rounds", 3)

    if "max_rounds" not in state or not state.get("max_rounds"):
        state["max_rounds"] = max_rounds

    round_num = state.get("current_round", 0)

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 3),
        temperature=0.3,
    )

    try:
        if round_num == 0:
            # First invocation — decompose the goal
            logger.info("Supervisor: decomposing research goal", extra={"category": "RESEARCH"})

            response = await client.chat(
                state["research_goal"],
                SUPERVISOR_SYSTEM_PROMPT,
                temperature=0.3,
            )

            import json
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group())
                    state["sub_questions"] = plan.get("sub_questions", [state["research_goal"]])
                except (json.JSONDecodeError, KeyError, ValueError):
                    state["sub_questions"] = [state["research_goal"]]
            else:
                state["sub_questions"] = [state["research_goal"]]

            state["current_round"] = 1
            logger.info(
                "Supervisor: decomposed into {} sub-questions".format(len(state["sub_questions"])),
                extra={"category": "RESEARCH"},
            )

        else:
            # Refinement round — incorporate learnings
            logger.info("Supervisor: planning round {}".format(round_num + 1), extra={"category": "RESEARCH"})

            # Summarize hypotheses for the supervisor
            hypotheses = [h for h in state.get("hypotheses", []) if h.get("status") == "alive"]
            hyp_summary = "\n".join(
                "- [{elo:.0f} ELO] {title}: {desc}".format(
                    elo=h.get("elo_rating", 1500),
                    title=h.get("title", ""),
                    desc=h.get("description", "")[:200],
                )
                for h in hypotheses[:10]
            ) or "(no surviving hypotheses)"

            # Summarize tournament
            tourney = state.get("tournament_results", [])
            tourney_summary = "{} debates held".format(len(tourney)) if tourney else "(no tournament data)"

            # Extract learnings from code execution
            learnings_parts = []
            for result in state.get("code_execution_results", []):
                if result.get("consensus_reached"):
                    learnings_parts.append(
                        "Code analysis consensus: {}".format(result.get("majority_finding", ""))
                    )
            learnings = "\n".join(learnings_parts) or "(no code analysis data)"

            prompt = SUPERVISOR_REFINE_PROMPT.format(
                current_round=round_num,
                max_rounds=state["max_rounds"],
                goal=state["research_goal"],
                hypotheses_summary=hyp_summary,
                tournament_summary=tourney_summary,
                learnings=learnings,
            )

            response = await client.chat(prompt, temperature=0.3)

            import json
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    decision = json.loads(json_match.group())
                    if decision.get("continue", True):
                        state["current_round"] = round_num + 1
                        if decision.get("refined_sub_questions"):
                            state["sub_questions"] = decision["refined_sub_questions"]
                    else:
                        state["research_complete"] = True
                        state["current_round"] = round_num  # Don't increment — we're done
                        logger.info(
                            "Supervisor: research complete — {}".format(decision.get("reasoning", "")),
                            extra={"category": "RESEARCH"},
                        )
                except (json.JSONDecodeError, KeyError, ValueError):
                    if round_num < state.get("max_rounds", 20):
                        state["current_round"] = round_num + 1
                    else:
                        state["research_complete"] = True
            else:
                # Supervisor couldn't decide — default to continuing if under safety limit
                if round_num < state.get("max_rounds", 20):
                    state["current_round"] = round_num + 1
                else:
                    state["research_complete"] = True
                    logger.info("Supervisor: safety limit reached after {} rounds".format(round_num),
                                extra={"category": "RESEARCH"})

    except Exception as e:
        logger.error("Supervisor error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "supervisor", "error": str(e)})
        # Fall through — let next round proceed with existing sub_questions
        if round_num == 0:
            state["sub_questions"] = [state["research_goal"]]
            state["current_round"] = 1
        elif round_num < state.get("max_rounds", 3):
            state["current_round"] = round_num + 1
        else:
            state["current_round"] = state.get("max_rounds", 3) + 1

    finally:
        await client.close()

    return state
