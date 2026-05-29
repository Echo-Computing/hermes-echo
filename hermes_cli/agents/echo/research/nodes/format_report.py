"""Format report node — produces the final structured research report."""

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState


async def format_report(state: ResearchState) -> ResearchState:
    """Format the final research report from all accumulated findings."""

    logger.info("FormatReport: assembling final research report", extra={"category": "RESEARCH"})

    hypotheses = state.get("hypotheses", [])

    # Group hypotheses by status
    alive = sorted(
        [h for h in hypotheses if h.get("status") in ("alive", "refined")],
        key=lambda h: h.get("elo_rating", 1500),
        reverse=True,
    )
    eliminated = [h for h in hypotheses if h.get("status") == "eliminated"]
    merged = [h for h in hypotheses if h.get("status") == "merged"]

    # Build leaderboard
    leaderboard = []
    for rank, h in enumerate(alive[:10], 1):
        leaderboard.append({
            "rank": rank,
            "title": h.get("title", ""),
            "description": h.get("description", ""),
            "mechanism": h.get("mechanism", ""),
            "elo_rating": h.get("elo_rating", 1500),
            "evidence": h.get("evidence", []),
            "critiques": h.get("critiques", []),
            "round_created": h.get("round_created", 0),
        })

    # Tournament summary
    tournament = state.get("tournament_results", [])
    debates_total = len(tournament)
    a_wins = sum(1 for t in tournament if t.get("winner") == "A")
    b_wins = sum(1 for t in tournament if t.get("winner") == "B")
    ties = sum(1 for t in tournament if t.get("winner") == "tie")

    # Code execution summary
    code_results = state.get("code_execution_results", [])
    experiments_with_consensus = sum(1 for r in code_results if r.get("consensus_reached"))
    accepted_findings = sum(1 for r in code_results if r.get("verdict") == "accepted")

    final_report = {
        "title": "Research Report: {}".format(state.get("research_goal", ""))[:100],
        "rounds_completed": state.get("current_round", 0),
        "total_hypotheses": len(hypotheses),
        "surviving_hypotheses": len(alive),
        "eliminated_hypotheses": len(eliminated),
        "merged_hypotheses": len(merged),
        "leaderboard": leaderboard,
        "tournament_summary": {
            "total_debates": debates_total,
            "wins_a": a_wins,
            "wins_b": b_wins,
            "ties": ties,
            "top_hypothesis": alive[0].get("title", "") if alive else "None",
            "top_elo": alive[0].get("elo_rating", 1500) if alive else 0,
        },
        "experimental_validation": {
            "experiments_run": len(code_results),
            "experiments_with_consensus": experiments_with_consensus,
            "accepted_findings": accepted_findings,
        },
        "methodology": {
            "architecture": "Multi-agent collaborative research (Co-Scientist + Robin inspired)",
            "agents": [
                "Supervisor — goal decomposition and round control",
                "Generation — literature search and hypothesis generation",
                "Reflection — critical review and fact-checking",
                "Proximity — duplicate detection",
                "Evolution — refinement and combination",
                "Ranking — ELO tournament with AI judging",
                "Code Execution — Finch-style sandboxed analysis",
                "Consensus — majority-vote verification",
            ],
        },
        "sub_questions": state.get("sub_questions", []),
    }

    state["final_report"] = final_report

    # Log summary
    logger.info(
        "FormatReport: report complete — top hypothesis: '{}' ({:.0f} ELO)".format(
            final_report["tournament_summary"]["top_hypothesis"][:60],
            final_report["tournament_summary"]["top_elo"],
        ),
        extra={"category": "RESEARCH"},
    )

    return state
