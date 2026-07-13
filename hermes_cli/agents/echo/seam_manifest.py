"""P0-9 (2026-07-06 red-team) — single source of truth for the axis-D seam.

Two consumers share this module so the seam file list CANNOT drift between them
(the echo_cmd.py-was-copied-but-not-attested gap is the precedent this closes):

  1. apply_seam.sh (deploy-time): imports discover() to auto-generate the cp +
     backup + chmod + byte-identical attestation lists from a glob of the
     scratch dir. No hard-coded file manifest to rot.
  2. agent.py (runtime, module-load self-check): calls verify_integrity() to
     re-hash every seam file scratch-vs-live and raise if any diverge — the
     runtime counterpart to apply_seam.sh's deploy-time attestation, so a
     stale/partial/hand-edited deploy (e.g. a reverted P0-1/P0-3 fix that
     reopens an axis-D exfil channel) fails closed at import instead of
     silently running an un-sandboxed code_execution node.

Threat model: ACCIDENTAL DRIFT / STALE DEPLOY, NOT a tampering attacker (the
path guards in agent.py already refuse write_file/edit_file on guard sources).
The seam scratch dir (a deploy-time source-of-truth path, absent in the
public build -> verify_integrity soft-fails) is the single source of truth — verify_integrity()
re-reads it at runtime, so there is NO stored manifest that can drift
independently of the files it describes. The file LIST itself is re-derived
from scratch, so a stale live copy of THIS module cannot hide a newly-added
seam file.

Stdlib-only (hashlib, pathlib, os, sys): runs at agent.py module load BEFORE
the graph is built and must not add heavy deps or a new exfil channel. NEVER
logs or returns file contents — only relative paths + 12-hex hash prefixes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple

# The seam scratch dir (source of truth). Override via env for testability.
DEFAULT_SCRATCH = os.path.join(os.path.expanduser("~"), ".hermes", "seam-scratch")

# Subdirs under scratch that are NOT seam source (excluded from discover()).
# seam-tests/ holds seam-coupled tests (copied + run by a separate mechanism).
_NON_SEAM_PARTS = {"__pycache__", "seam-tests"}

# Legacy scratch-top-level files whose live path does NOT mirror scratch <rel>
# under agents/echo/. New seam files SHOULD follow the mirror convention
# (scratch <rel> -> agents/echo/<rel>) and need NO entry here.
#   echo_cmd.py  lives in commands/, not agents/echo/.
# Each legacy exception MUST be listed here — omitting one silently drops that
# seam on a fresh restore (the v2 breach class the apply_seam.sh header warns
# of). Verified against the live tree 2026-07-06.
_OVERRIDES = {
    "echo_cmd.py": "commands/echo_cmd.py",
}


def _is_helper(name: str) -> bool:
    """Excluded one-off scratch helpers. Rule = single underscore + lowercase
    second char: KEEPS __init__.py / dunder modules (double underscore) and
    excludes _run_probe.py, _verify_*.py, etc. A plain '_*' would WRONGLY drop
    a package's __init__.py."""
    return name.startswith("_") and len(name) > 1 and name[1].islower()


def _live_rel(scratch_rel: str) -> str:
    """Map a scratch-relative path to its live-clone-relative path (relative to
    hermes_cli/). Default mirrors under agents/echo/; legacy exceptions in
    _OVERRIDES."""
    return _OVERRIDES.get(scratch_rel, "agents/echo/" + scratch_rel)


def discover(scratch_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """Auto-discover the seam source file set as (scratch_rel, live_rel) pairs.

    Glob scratch_dir for *.py, excluding __pycache__, the seam-tests/ subtree,
    and _<lower> helper scripts. Returns a sorted, deterministic list. Raises
    if scratch is absent or contains no seam .py files — a deploy with no
    source of truth must fail closed, not silently deploy nothing and pass a
    vacuous attestation (the false-green-stale-deploy vector P0-9 closes)."""
    scratch = Path(scratch_dir or os.environ.get("ANIMA_SEAM_SCRATCH", DEFAULT_SCRATCH))
    if not scratch.is_dir():
        raise FileNotFoundError(
            f"seam scratch dir absent: {scratch}. Cannot discover seam file "
            f"set — refusing to deploy/attest an empty seam (fail-closed)."
        )
    pairs: List[Tuple[str, str]] = []
    for p in sorted(scratch.rglob("*.py")):
        if any(part in _NON_SEAM_PARTS for part in p.parts):
            continue
        if _is_helper(p.name):
            continue
        scratch_rel = p.relative_to(scratch).as_posix()
        pairs.append((scratch_rel, _live_rel(scratch_rel)))
    if not pairs:
        raise RuntimeError(
            f"seam scratch dir {scratch} contains no seam .py files — refusing "
            f"to deploy an empty seam (fail-closed)."
        )
    return pairs


def _hash(path: Path) -> str:
    """sha256 of CRLF-normalized UTF-8 text, 12-hex prefix. MUST match
    apply_seam.sh's attestation hash byte-for-byte (same normalization) so a
    deploy that passes apply_seam.sh also passes the runtime check — no
    false-positive lockout deadlock."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def verify_integrity(
    scratch_dir: Optional[str] = None,
    *,
    live_root: Optional[str] = None,
) -> Optional[List[Tuple[str, str, str]]]:
    """Re-hash every seam source file scratch-vs-live. Returns:
      - None  if scratch is absent (unverifiable — F: unplugged; agent.py's
              anima import would already have failed in that case, so this is
              defensive). Caller SOFT-FAILS (warn), does NOT raise — avoid
              locking the operator out of the agent on a portable deploy.
      - []    if every live seam file is byte-identical (CRLF-normalized) to
              its scratch source (clean deploy).
      - [(live_rel, scratch_hash, live_hash_or_'MISSING'), ...]  for each
              divergent file (stale/partial/hand-edited deploy = axis-D breach).
              Caller HARD-FAILS (raise) — same doctrine as the prompt-guard
              module-load checks in agent.py.

    The file LIST is re-derived from scratch (discover()), so a stale live
    copy of this module cannot hide a newly-added seam file. Never logs or
    returns file contents — only relative paths + 12-hex prefixes (no new
    exfil channel through the log sink)."""
    scratch = Path(scratch_dir or os.environ.get("ANIMA_SEAM_SCRATCH", DEFAULT_SCRATCH))
    if not scratch.is_dir():
        return None  # unverifiable (env), NOT a mismatch — caller soft-fails
    # LIVE_ROOT = the hermes_cli/ dir. This module lives at
    # hermes_cli/agents/echo/seam_manifest.py, so three parents up is hermes_cli/.
    # (Only valid when imported from the LIVE location; apply_seam.sh imports
    # this module from scratch for discover() only and never calls this.)
    # `live_root` is a testability hook (tests pass a temp dir); the runtime
    # caller in agent.py leaves it None -> derived from __file__.
    live_root_path = Path(live_root) if live_root else Path(__file__).resolve().parent.parent.parent
    mismatches: List[Tuple[str, str, str]] = []
    for scratch_rel, live_rel in discover(str(scratch)):
        s = scratch / scratch_rel
        d = live_root_path / live_rel
        if not d.exists():
            mismatches.append((live_rel, _hash(s), "MISSING"))
            continue
        sh = _hash(s)
        dh = _hash(d)
        if sh != dh:
            mismatches.append((live_rel, sh, dh))
    return mismatches