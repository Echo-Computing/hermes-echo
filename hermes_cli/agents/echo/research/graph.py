"""Collaborative multi-agent research graph.

Implements the Co-Scientist/Robin architecture as a LangGraph StateGraph:

    supervisor -> generation -> reflection -> proximity -> evolution -> ranking
                                                                          |
                                                                          v
                                                                 code_execution
                                                                          |
                                                                          v
                                                                   consensus
                                                                          |
                                              ┌───────────────────────────┘
                                              | (continue)    | (done)
                                              v               v
                                          supervisor      format_report
                                              |               |
                                              ...             END
"""

from langgraph.graph import StateGraph, END
from loguru import logger

from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.agents.echo.research.nodes import (
    run_supervisor,
    run_generation,
    run_reflection,
    run_proximity,
    run_evolution,
    run_ranking,
    run_code_execution,
    run_consensus,
)
from hermes_cli.agents.echo.research.nodes.format_report import format_report


def supervisor_router(state: ResearchState) -> str:
    """Determine whether to continue the research loop or finalize."""
    current = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 3)

    if current == 0:
        # First invocation — go to generation
        return "generation"

    if max_rounds > 0 and current > max_rounds:
        # Done — format final report
        logger.info("Graph: research complete after {} rounds".format(max_rounds), extra={"category": "RESEARCH"})
        return "format_report"

    # Continue the loop for the next round
    logger.info("Graph: continuing to round {}".format(current), extra={"category": "RESEARCH"})
    return "generation"


def should_continue_research(state: ResearchState) -> str:
    """After consensus, decide: loop back to supervisor or end."""
    current = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 20)

    # Supervisor explicitly said we're done
    if state.get("research_complete", False):
        logger.info("Graph: supervisor concluded research", extra={"category": "RESEARCH"})
        return "format_report"

    # Safety limit
    if current > max_rounds:
        logger.info("Graph: safety limit reached after {} rounds".format(max_rounds), extra={"category": "RESEARCH"})
        return "format_report"

    # Check if we have enough hypotheses
    alive = [h for h in state.get("hypotheses", []) if h.get("status") in ("alive", "refined")]
    if not alive:
        logger.info("Graph: no surviving hypotheses, ending", extra={"category": "RESEARCH"})
        return "format_report"

    return "supervisor"


def create_research_graph():
    """Build and compile the collaborative research LangGraph.

    Returns a compiled StateGraph ready for graph.ainvoke(state).
    """
    workflow = StateGraph(ResearchState)

    # Register all nodes
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("generation", run_generation)
    workflow.add_node("reflection", run_reflection)
    workflow.add_node("proximity", run_proximity)
    workflow.add_node("evolution", run_evolution)
    workflow.add_node("ranking", run_ranking)
    workflow.add_node("code_execution", run_code_execution)
    workflow.add_node("consensus", run_consensus)
    workflow.add_node("format_report", format_report)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routes to generation or format_report
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "generation": "generation",
            "format_report": "format_report",
        },
    )

    # Linear chain: generation -> reflection -> proximity -> evolution -> ranking
    workflow.add_edge("generation", "reflection")
    workflow.add_edge("reflection", "proximity")
    workflow.add_edge("proximity", "evolution")
    workflow.add_edge("evolution", "ranking")
    workflow.add_edge("ranking", "code_execution")
    workflow.add_edge("code_execution", "consensus")

    # After consensus: loop back to supervisor or finalize
    workflow.add_conditional_edges(
        "consensus",
        should_continue_research,
        {
            "supervisor": "supervisor",
            "format_report": "format_report",
        },
    )

    # Format report ends the workflow
    workflow.add_edge("format_report", END)

    return workflow.compile()
