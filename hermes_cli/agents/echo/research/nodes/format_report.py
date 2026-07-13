"""Format report node — produces the final structured research report.

SEAM OVERRIDE (Step 4 autoresearch, 2026-07-07): extends the upstream
format_report to ALSO append one results.tsv provenance row per hypothesis
that reached this terminal node, so the governance gauntlet's keep/discard
decision is durable + auditable. The upstream built ``final_report`` and
returned; this override adds the TSV write loop AFTER ``final_report`` is
assembled + BEFORE return, calling ``research.provenance.write_outcomes_for_run``.

GOODHART INVARIANT (load-bearing): this node is the ONLY writer of TSV rows,
and it runs AFTER the whole gauntlet (reflection -> consensus -> ranking +
seam-integrity). The ``status`` column is the gauntlet's composite outcome,
NOT a scalar metric — see ``research/provenance.py``. No earlier node writes
rows; no metric alone decides keep/discard.

The seam-integrity signal (``verify_integrity()``) is read here so a kept
hypothesis is NEVER recorded as ``keep`` when the substrate itself is
compromised or unverifiable. ``verify_integrity()`` returns ``[]`` (clean),
a non-empty list (a breach), or ``None`` (soft-fail — scratch absent). BOTH
a non-empty list AND ``None`` force ``discard`` for every row: an
unverifiable substrate must not silently keep (``build_outcome`` treats any
truthy ``seam_integrity`` as a breach; ``None`` is coerced to the
``["unverifiable"]`` marker so it is truthy).
"""
import hashlib
import time

from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
# Step 4: provenance ledger writer (seam-owned, auto-attested).
from hermes_cli.agents.echo.research.provenance import (
    write_outcomes_for_run,
    default_tsv_path,
)


def _run_id(state: ResearchState) -> str:
    """Derive a stable run id for the TSV ``run_id`` column. Combines a UTC
    timestamp (unique per run — format_report runs once per run) with a short
    hash of the research goal (human-grep-able)."""
    goal = str(state.get("research_goal", ""))[:120]
    goal_hash = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:6]
    return "{}-{}".format(time.strftime("%Y%m%d-%H%M%S", time.gmtime()), goal_hash)


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

    # --- Step 4 autoresearch: append results.tsv provenance rows ---
    # The ONLY writer of TSV rows; runs after the whole gauntlet. The status
    # column is the gauntlet composite (reflection+consensus+ranking+integrity),
    # NEVER a scalar metric. A seam breach (non-empty verify_integrity) forces
    # discard for every row (build_outcome enforces it). Failures here MUST
    # NOT block the final report (the report is the user-facing artifact); a
    # provenance write error is logged + swallowed.
    try:
        # Late import: seam_manifest is seam-owned; reading verify_integrity
        # here records the substrate state at the keep decision point.
        from hermes_cli.agents.echo.seam_manifest import verify_integrity
        # verify_integrity() returns:
        #   []      -> clean (no breach)
        #   [..]    -> non-empty: a seam breach (paths that drifted)
        #   None    -> soft-fail: the scratch dir is absent/unverifiable
        #              (e.g. ANIMA_SEAM_SCRATCH unset / F: unplugged)
        # A non-empty list AND None are BOTH unsafe to keep over: an
        # unverifiable substrate must NOT silently record `keep` rows (that
        # is the exact fail-open the `or []` coercion below once created —
        # `None or []` -> `[]` reads as clean). So coerce the None RETURN to
        # a non-empty marker, fail-closed: build_outcome treats any truthy
        # seam_integrity as a breach -> discard for every row.
        try:
            _si = verify_integrity()
        except Exception:
            # A RAISED exception is also an unverifiable substrate — coerce
            # to the same marker (the except branch is the second route).
            _si = None
        seam_integrity = _si if _si is not None else ["unverifiable"]
        run_id = _run_id(state)
        written = write_outcomes_for_run(
            hypotheses=hypotheses,
            code_results=code_results,
            seam_integrity=seam_integrity,
            run_id=run_id,
            config=state.get("config", {}),
            resource_per_hyp=0.0,  # Phase A: 0.0; PATH 1 measures actual wall-clock
            # elapsed around the instance batch (time.monotonic delta), NOT the
            # per_mutation_budget cap — the `resource` column documents
            # wall-clock seconds CONSUMED, so PATH 1 must record consumption.
        )
        logger.info(
            "FormatReport: wrote {} provenance row(s) to {}".format(
                written, default_tsv_path(state.get("config", {})),
            ),
            extra={"category": "RESEARCH"},
        )
    except Exception as _prov_err:  # noqa: BLE001
        # Provenance is operator-state, never user-facing; a write failure
        # (e.g. read-only FS, no home dir) is logged + swallowed so the
        # research report still returns. The gauntlet decision still
        # happened in-state; only the durable ledger write was lost.
        logger.warning(
            "FormatReport: provenance write failed ({}); report unaffected".format(_prov_err),
            extra={"category": "RESEARCH"},
        )

    # Log summary
    logger.info(
        "FormatReport: report complete — top hypothesis: '{}' ({:.0f} ELO)".format(
            final_report["tournament_summary"]["top_hypothesis"][:60],
            final_report["tournament_summary"]["top_elo"],
        ),
        extra={"category": "RESEARCH"},
    )

    return state