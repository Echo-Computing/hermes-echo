"""Hermes CLI - ローカルLLM情報収集エージェント"""

# Single source of truth for the version is pyproject.toml ([project].version,
# consumed by hatchling into the installed dist metadata). Read it back from
# importlib.metadata so __version__ never drifts from the installed wheel again
# (v0.3.1 fix: this was a hardcoded "0.1.0" stale since v0.2.0). Fallback covers
# running from a source tree that has not been pip-installed.
try:
    from importlib.metadata import version as _dist_version, PackageNotFoundError

    try:
        __version__ = _dist_version("hermes-cli")
    except PackageNotFoundError:
        __version__ = "0.3.1"
except ImportError:  # pragma: no cover - py<3.8 not supported anyway
    __version__ = "0.3.1"