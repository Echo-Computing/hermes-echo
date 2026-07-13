"""Research-run provenance ledger (Step 4 autoresearch, 2026-07-07).

A small append-only TSV record of research-run OUTCOMES — one row per
hypothesis that reached the terminal ``format_report`` node — so the
governance gauntlet's keep/discard decision is OBSERVABLE and AUDITABLE
after the fact, not silently lost in the final report dict.

WHY A TSV (not JSON/memory-markdown):
  ``consolidate_learning`` (agent.py) writes per-turn memory markdown for
  RETRIEVAL; ``session_summary`` writes per-session JSONL. NEITHER is an
  append-only provenance ledger of research-run outcomes. A TSV is the
  minimal shape for "one line per kept/discarded hypothesis, grep-able,
  diff-able across runs" — and it is the schema karpathy/autoresearch's
  results.tsv mined as a DESIGN (MINE-DISCIPLINE: no code copied; the
  schema is re-implemented as original Echo content).

GOODHART DEFENSE (load-bearing):
  The ``status`` column (keep|discard|crash) is NOT computed from a scalar
  metric the LLM cites. It is the COMPOSITE of three independent governance
  nodes + a seam-integrity check, composed in ``compose_governance_verdict``
  and written ONLY at ``format_report`` (the terminal node, after the whole
  gauntlet has run). The ``code_execution`` node's ``proposal`` dict is
  informational and is OVERWRITTEN by the gauntlet outcome — see
  ``code_execution.py`` + ``format_report.py``. No node writes a keep row
  directly from a metric.

LOCATION / GIT:
  The ledger lives at ``~/.hermes/learning/results.tsv`` (configurable via
  ``config['research']['learning_dir']``) — OUTSIDE the repo, so it is
  git-clean by default (no .gitignore entry needed, no seam-manifest entry).
  It is operator state, not source. NEVER copy it into the repo.

This module is a seam-owned ``.py`` file (auto-discovered + auto-attested by
``seam_manifest.discover()`` / ``verify_integrity()``); it is public-safe
(generic "research provenance" language — it deliberately uses no
private-seam vocabulary, only the research-graph's own terms).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# The column order of the TSV ledger. Stable across versions (append-only
# contract: new columns go at the END; never reorder/rename). The header row
# written to a fresh ledger is exactly "\t".join(TSV_COLUMNS).
TSV_COLUMNS: Tuple[str, ...] = (
    "run_id",          # the research run id (echo_cmd per-invocation)
    "timestamp",       # ISO-ish UTC string of row write
    "commit",          # the hypothesis id (the "mutation" identifier)
    "metric",          # the consensus verdict / majority finding on the run
    "resource",        # wall-clock seconds consumed by the run (or 0.0)
    "status",          # keep | discard | crash  (the gauntlet outcome)
    "governance_verdict",  # composed: reflection + consensus + ranking + integrity
)


@dataclass
class ResearchOutcome:
    """One row in the results.tsv provenance ledger.

    A research run produces N hypotheses; the governance gauntlet (reflection
    -> consensus -> ranking + seam-integrity) decides each one's status. This
    dataclass is the durable record of that decision. ``status`` is the
    gauntlet outcome, NOT a scalar metric — see module docstring (Goodhart).
    """
    run_id: str
    timestamp: str
    commit: str            # hypothesis id
    metric: str            # consensus verdict / majority finding
    resource: float        # wall-clock seconds (0.0 if unmeasured)
    status: str            # "keep" | "discard" | "crash"
    governance_verdict: str  # composed string of the 4 governance signals

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "commit": self.commit,
            "metric": self.metric,
            "resource": self.resource,
            "status": self.status,
            "governance_verdict": self.governance_verdict,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchOutcome":
        return cls(
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", ""),
            commit=d.get("commit", ""),
            metric=d.get("metric", ""),
            resource=float(d.get("resource", 0.0)),
            status=d.get("status", "discard"),
            governance_verdict=d.get("governance_verdict", ""),
        )

    def to_tsv_row(self) -> str:
        """Tab-joined row, column order == TSV_COLUMNS. No newlines inside
        fields (tabs/newlines in governance_verdict are replaced with spaces
        so the row stays one physical line)."""
        def _clean(v) -> str:
            return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")
        return "\t".join(_clean(getattr(self, c)) for c in TSV_COLUMNS)


def default_tsv_path(config: Optional[dict] = None) -> Path:
    """Resolve the ledger path. Default ``~/.hermes/learning/results.tsv``;
    override via ``config['research']['learning_dir']`` (mirrors the
    memory_dir / history_dir pattern in agent.py). The path is OUTSIDE the
    repo by construction (under the operator's home, not the clone)."""
    research_cfg = (config or {}).get("research", {}) if config else {}
    if isinstance(research_cfg, dict):
        learning_dir = research_cfg.get("learning_dir")
    else:
        # a config object with attributes (defensive; not the hot path)
        learning_dir = getattr(research_cfg, "learning_dir", None)
    base = Path(learning_dir) if learning_dir else Path.home() / ".hermes" / "learning"
    return base / "results.tsv"


def _now_timestamp() -> str:
    # Use time.strftime over gmtime to avoid Date.now-style helpers being
    # unavailable in some sandboxes; this runs in the echo process, not in
    # the research sandbox, so stdlib time is fine.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compose_governance_verdict(
    hypothesis: dict,
    code_result: Optional[dict],
    seam_integrity: Optional[list],
) -> str:
    """Compose the 4 governance signals into one verdict string for the TSV.

    This is the Goodhart load-bearing composition: the ``status`` decision is
    NOT a scalar metric — it is the composite of
      (1) reflection: hypothesis status (alive/refined/eliminated/merged)
      (2) consensus: the code_execution verdict (accepted/inconclusive/...)
      (3) ranking:   the ELO rating the ranking node assigned
      (4) integrity: verify_integrity() — [] = clean, ["unverifiable"] =
          absent/unverifiable substrate (None coerced by format_report),
          any other non-empty list = a seam breach
    The composition is recorded for audit; the keep/discard DECISION derived
    from it lives in ``build_outcome`` (single source of truth for status).
    """
    status = hypothesis.get("status", "alive")
    elo = hypothesis.get("elo_rating", 1500)
    if code_result is not None:
        consensus = code_result.get("verdict", "n/a")
        consensus_reached = code_result.get("consensus_reached", False)
    else:
        consensus = "no-code"
        consensus_reached = False
    if not seam_integrity:
        integ = "clean"
    elif seam_integrity == ["unverifiable"]:
        # None return from verify_integrity (absent scratch) coerced to this
        # marker by format_report — distinct from a real breach so the ledger
        # is honest about WHY the row was discarded.
        integ = "unverifiable"
    else:
        integ = "BREACH:{}".format(len(seam_integrity))
    return "reflection={}|consensus={}|elo={}|integrity={}".format(
        status, ("accepted" if consensus_reached else consensus), elo, integ
    )


def build_outcome(
    hypothesis: dict,
    code_result: Optional[dict],
    seam_integrity: Optional[list],
    run_id: str,
    resource: float = 0.0,
) -> ResearchOutcome:
    """Build a ResearchOutcome from a post-gauntlet hypothesis.

    STATUS RULE (the gauntlet outcome — NOT a scalar metric):
      keep    if status in (alive, refined) AND integrity clean AND the
              consensus gate passed (code, when it ran, reached consensus)
      discard if status == eliminated (reflection fatal or ranking ELO floor)
      discard if a seam breach / unverifiable substrate (integrity not clean)
      discard if code ran, all instances succeeded, but consensus was
              INCONCLUSIVE (no majority) — contract gate #1: "inconclusive
              suppresses". The consensus NODE only applies a -10 ELO penalty
              and never sets status='eliminated', so without this branch an
              alive hypothesis with inconclusive consensus would silently keep
              — contradicting the 4-gate claim. (reflection runs BEFORE
              code_execution in the graph, so it cannot consume the consensus
              result; the discard must be enforced here.)
      crash   if any code_execution instance failed (non-zero exit / timeout)
              — detected via the code_result having no consensus_reached and a
              non-empty stderr-ish signal. Conservative: if the hypothesis was
              eliminated by reflection/ranking, that wins over crash.

    The metric column records the consensus verdict (informational); the
    governance_verdict column records all 4 signals.
    """
    governance = compose_governance_verdict(hypothesis, code_result, seam_integrity)
    hyp_status = hypothesis.get("status", "alive")
    integ_breached = bool(seam_integrity)

    # crash detection: code ran but no instance succeeded (consensus not
    # reached AND there were code instances that failed)
    crashed = False
    if code_result is not None:
        instances = code_result.get("instances", [])
        if instances and not code_result.get("consensus_reached", False):
            crashed = any(not inst.get("success", False) for inst in instances)

    # consensus gate: code ran, instances succeeded, but no majority. The
    # contract (gate #1) says inconclusive suppresses — discard, NOT keep.
    # Only fires when code actually ran (code_result is not None) and none
    # of the instances crashed (a crash is recorded as 'crash' below, not
    # silently demoted to a consensus-discard).
    consensus_inconclusive = (
        code_result is not None
        and not crashed
        and not code_result.get("consensus_reached", False)
    )

    if crashed and hyp_status != "eliminated":
        status = "crash"
    elif hyp_status == "eliminated":
        status = "discard"
    elif integ_breached:
        # a seam breach / unverifiable substrate is fail-closed — never keep
        # into a compromised substrate
        status = "discard"
    elif consensus_inconclusive:
        # contract gate #1: inconclusive consensus suppresses (discard)
        status = "discard"
    elif hyp_status in ("alive", "refined"):
        status = "keep"
    else:
        # merged / unknown — treat as discard (conservative; merged hypotheses
        # are superseded by their combination)
        status = "discard"

    metric = ""
    if code_result is not None:
        metric = code_result.get("majority_finding") or code_result.get("verdict", "")

    return ResearchOutcome(
        run_id=run_id,
        timestamp=_now_timestamp(),
        commit=hypothesis.get("id", ""),
        metric=metric,
        resource=float(resource),
        status=status,
        governance_verdict=governance,
    )


def append_results_tsv(path: Path, outcome: ResearchOutcome) -> None:
    """Append one outcome row to the TSV ledger; create the file + header if
    new. Mkdir parents. Writes the header row (TSV_COLUMNS) to a fresh file so
    the ledger is self-describing. Atomicity is best-effort (open-append +
    flush); this is operator-state, not transactional — a partial row from a
    crash is readable as a short line and never corrupts prior rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    # Open for append (create if missing). Use a simple write — the ledger is
    # operator state under ~/.hermes, not a hot loop.
    with open(path, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write("\t".join(TSV_COLUMNS) + "\n")
        fh.write(outcome.to_tsv_row() + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # not all backing filesystems support fsync (drvfs etc.)


def write_outcomes_for_run(
    hypotheses: List[dict],
    code_results: List[dict],
    seam_integrity: Optional[list],
    run_id: str,
    config: Optional[dict] = None,
    resource_per_hyp: float = 0.0,
) -> int:
    """Convenience: write one TSV row per hypothesis for a completed run.

    Matches each hypothesis to its code_execution result by ``hypothesis_id``
    (code_results entries carry ``hypothesis_id``). Returns the number of
    rows written. Used by ``format_report`` (the terminal node, the ONLY
    caller — keeps the Goodhart invariant: rows are written after the whole
    gauntlet).
    """
    # index code results by hypothesis id for O(1) lookup
    by_id = {}
    for cr in code_results or []:
        hid = cr.get("hypothesis_id", "")
        if hid:
            by_id[hid] = cr

    path = default_tsv_path(config)
    written = 0
    for hyp in hypotheses or []:
        cr = by_id.get(hyp.get("id", ""))
        outcome = build_outcome(
            hyp, cr, seam_integrity, run_id, resource=resource_per_hyp,
        )
        append_results_tsv(path, outcome)
        written += 1
    return written