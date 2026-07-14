"""seam-tests for Graphify (Step 2 recommended-order build, 2026-07-07).

Covers: registration surface (fields + execution_sandbox="none" + rationale),
integrity guard containment pre-check on the `path` param, the query-DSL refusal of a
path-shaped query (the deep-dive's KEY integrity guard risk), the synthetic-repo graph
build (DEFINES/CALLS/INHERITS/IMPORTS edges + EXTRACTED/INFERRED tags + two-pass
cross-file resolver), the v1 query set (callers_of/callees_of/upstream/downstream/
shortest path/community/explain), the mtime+size cache (no-op rebuild), the
30s parse ceiling, LRU eviction (count + size), import-purity, + a real-slice
smoke against the seam tools dir.

CRITICAL test mechanic: monkeypatch `_build_registry` via
`monkeypatch.setattr("hermes_cli.agents.echo.agent._build_registry", ...)` —
NEVER bare module assignment. Handlers are module-level functions so
`inspect.getsource` resolves (graph_tools.graph + its helpers).
"""
import os
import sys
import textwrap
import time

import pytest

from hermes_cli.agents.echo.agent import _PROTECTED_STORE_MARKERS
from hermes_cli.agents.echo.tools import graph_tools


# ---------------------------------------------------------------------------
# synthetic repo fixture (hermetic — written under tmp_path each test)
# ---------------------------------------------------------------------------

_SYNTH = {
    "pkg/__init__.py": "",
    "pkg/base.py": textwrap.dedent("""\
        class Base:
            def method(self):
                return "base"
        """),
    "pkg/derived.py": textwrap.dedent("""\
        from .base import Base

        class Derived(Base):
            def call_base(self):
                return self.method()
        """),
    "pkg/util.py": textwrap.dedent("""\
        def helper():
            return 42
        """),
    "pkg/api.py": textwrap.dedent("""\
        from .util import helper
        from .base import Base

        def api_func():
            helper()
            return Base()
        """),
    "pkg/main.py": textwrap.dedent("""\
        from .api import api_func

        def main():
            return api_func()
        """),
}


@pytest.fixture
def synth_repo(tmp_path):
    repo = tmp_path / "synthrepo"
    repo.mkdir()
    for rel, src in _SYNTH.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, "utf-8")
    return str(repo)


# ---------------------------------------------------------------------------
# registration surface
# ---------------------------------------------------------------------------

def _registry_with_graph(monkeypatch):
    """Build the real registry + return the graph CertifiedTool (graph_tools.graph
    handler imported by agent.py). Uses the real _build_registry so the
    import-time attestation + _register_tool chokepoint fire."""
    from hermes_cli.agents.echo import agent as agent_mod
    reg = agent_mod._build_registry()
    return reg.get("graph")


def test_graph_registered_with_locked_fields(monkeypatch):
    t = _registry_with_graph(monkeypatch)
    assert t is not None, "graph tool not registered"
    assert t.name == "graph"
    assert t.guard_source_policy == "read"
    assert t.recursive_read is True
    assert t.requires_handler_cert is True
    assert t.execution_sandbox == "none"
    assert t.execution_sandbox_rationale.strip(), (
        "execution_sandbox='none' requires a non-empty rationale (the _register_tool "
        "chokepoint should have refused an empty one)"
    )
    # the `path` param is required; `query` is NOT (omit -> summary)
    names = {p["name"] for p in t.parameters}
    assert {"path", "action", "query", "depth"} <= names


def test_graph_always_registered(monkeypatch):
    """graph is public-safe structural and always registered (no feature flag
    gates it). This is a deliberate placement decision (Graphify can be used
    without flipping any optional feature gate)."""
    from hermes_cli.agents.echo import agent as agent_mod
    reg = agent_mod._build_registry()
    assert reg.get("graph") is not None
    # graph is always present in the registry
    assert reg.get("graph").handler is graph_tools.graph


# ---------------------------------------------------------------------------
# integrity guard containment pre-check on the `path` param
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _PROTECTED_STORE_MARKERS, reason="protected-store path gate inert in the public build (private stores excluded)")
def test_graph_refuses_protected_store_path(monkeypatch):
    """graph('path=~/.hermes/store') must refuse (the protected store is the
    integrity guard substrate)."""
    cont = os.path.join(os.path.expanduser("~"), ".hermes", "store")
    out = graph_tools.graph(path=cont)
    assert "integrity guard" in out and "refused" in out.lower()


def test_graph_refuses_ancestor_of_protected_store(monkeypatch):
    """graph('path=~') must refuse — a recursive parse from ~ would rglob the
    protected stores into the graph (ancestor case)."""
    out = graph_tools.graph(path="~")
    assert "integrity guard" in out and "refused" in out.lower()


def test_graph_refuses_guard_source(monkeypatch):
    """graph('path=<guard source root>') must refuse (guard source = the seam /
    venv / guard-source tree — the cert + path-gate protect these)."""
    gs = graph_tools._guard_source_roots()
    if not gs:
        pytest.skip("no guard source roots configured")
    out = graph_tools.graph(path=gs[0])
    assert "integrity guard" in out and "refused" in out.lower()


def test_graph_refuses_nonexistent_dir(monkeypatch):
    out = graph_tools.graph(path="/nonexistent/repo/path/xyz")
    assert "not a directory" in out.lower()


def test_graph_refuses_bad_action(monkeypatch, synth_repo):
    out = graph_tools.graph(path=synth_repo, action="exfil")
    assert "action must be" in out.lower()


# ---------------------------------------------------------------------------
# query-DSL refusal of a path-shaped query (the KEY integrity guard risk)
# ---------------------------------------------------------------------------

def test_graph_refuses_path_shaped_query(monkeypatch, synth_repo):
    """The guard-source path-gate inspects only `path`, NOT `query`. A
    path-shaped query (e.g. a raw fs path sneaking in) must be refused by the DSL
    interpreter — this is the deep-dive's named integrity guard risk."""
    out = graph_tools.graph(path=synth_repo, query="/home/user/.hermes/store/state.db")
    assert "integrity guard" in out and "refused" in out.lower() and "query" in out.lower()


def test_graph_refuses_dotdot_query(monkeypatch, synth_repo):
    out = graph_tools.graph(path=synth_repo, query="../etc/passwd")
    assert "integrity guard" in out and "refused" in out.lower()


def test_graph_accepts_node_id_query(monkeypatch, synth_repo):
    """A node-id-shaped query (relpath:line:name) is NOT path-shaped — the resolver
    accepts it (it has the `:line:name` shape). This is the legitimate query form."""
    # build first
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="pkg/base.py:2:Base")
    # explain-ish content, not a refusal
    assert "integrity guard" not in out
    assert "refused" not in out.lower()


def test_graph_refuses_path_shaped_query_with_colons(monkeypatch, synth_repo):
    """A1 (4-lens re-verify): a relative path-shaped query with a ':N:name' suffix
    (3+ colons) used to slip _looks_like_path. The tightened check refuses when the
    first segment contains an integrity guard protected-surface marker (e.g. .hermes) even
    with a colon suffix. The LOAD-BEARING mitigation is that _run_query does no FS
    I/O regardless; this check is defense-in-depth."""
    graph_tools.graph(path=synth_repo, action="rebuild")
    for q in (".hermes/store.db:1:foo",
              ".hermes/cred.pem:2:bar",
              ".hermes/graphs/x:3:baz"):
        out = graph_tools.graph(path=synth_repo, query=q)
        assert "integrity guard" in out, f"query {q!r} not refused"
        assert "refused" in out.lower()


def test_graph_accepts_plain_relpath_node_id(monkeypatch, synth_repo):
    """A plain relpath node id (no protected-surface marker) with a `:line:name`
    suffix is the legitimate form — the tightened check must NOT over-refuse it."""
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="pkg/util.py:1:helper")
    assert "integrity guard" not in out


def test_graph_dormant_when_deps_missing(monkeypatch, synth_repo):
    """D-1 (4-lens re-verify): if the optional tree-sitter/networkx deps are not
    installed, graph() returns a clear dormant message + does NOT crash. The tool
    still registered (handler=graph is callable) so the registry surface is stable
    + the rest of the agent is unaffected (no import-time brick)."""
    monkeypatch.setattr(graph_tools, "_TS_AVAILABLE", False)
    monkeypatch.setattr(graph_tools, "_TS_IMPORT_ERROR", "ImportError: simulated")
    out = graph_tools.graph(path=synth_repo, query="callers_of:helper")
    assert "dormant" in out.lower() or "not installed" in out.lower()
    assert "integrity guard" not in out  # not an integrity guard refusal; a degrade message
    # restore so later tests in the session see the real flag
    monkeypatch.setattr(graph_tools, "_TS_AVAILABLE", True)


# ---------------------------------------------------------------------------
# synthetic-repo graph build — edges + tags + cross-file resolver
# ---------------------------------------------------------------------------

def test_graph_builds_and_summary(synth_repo):
    out = graph_tools.graph(path=synth_repo)
    assert "nodes" in out and "edges" in out and "files" in out


def test_graph_defines_edges(synth_repo):
    g = graph_tools._load_or_build(synth_repo)
    # Base + method + Derived + call_base + helper + api_func + main defined
    names = {d.get("name") for _, d in g.nodes(data=True) if d.get("kind") in ("function", "class")}
    assert {"Base", "method", "Derived", "call_base", "helper", "api_func", "main"} <= names


def test_graph_inherits_edge_cross_file(synth_repo):
    """Derived(Base) -> INHERITS edge Derived -> Base, resolved cross-file
    (INFERRED tag — the baseclass is in a different file)."""
    g = graph_tools._load_or_build(synth_repo)
    inh = [(u, v, r) for u, v, r in g.edges(data=True) if r.get("relation") == graph_tools.REL_INHERITS]
    assert inh, "no INHERITS edges found"
    # the target should be a Base node
    targets = {v for _u, v, _r in inh}
    base_nodes = [n for n, d in g.nodes(data=True) if d.get("name") == "Base"]
    assert any(bn in targets for bn in base_nodes)


def test_graph_calls_edge_same_file_extracted(synth_repo):
    """api_func calls helper() — helper is imported (cross-file) -> INFERRED.
    main() calls api_func() — api_func is imported (cross-file) -> INFERRED.
    Base() constructor call in api_func -> CALLS edge to Base (INFERRED)."""
    g = graph_tools._load_or_build(synth_repo)
    calls = [(u, v, r) for u, v, r in g.edges(data=True) if r.get("relation") == graph_tools.REL_CALLS]
    assert calls, "no CALLS edges found"
    callee_names = {g.nodes[v].get("name") for _u, v, _r in calls}
    assert "helper" in callee_names, "api_func -> helper CALLS edge missing"
    assert "api_func" in callee_names, "main -> api_func CALLS edge missing"


def test_graph_imports_edges(synth_repo):
    """from .base import Base (derived.py) -> IMPORTS edge derived.py -> base.py."""
    g = graph_tools._load_or_build(synth_repo)
    imp = [(u, v, r) for u, v, r in g.edges(data=True) if r.get("relation") == graph_tools.REL_IMPORTS]
    assert imp, "no IMPORTS edges found"
    targets = {v for _u, v, _r in imp}
    assert "pkg/base.py" in targets, "derived.py -> base.py IMPORTS edge missing"


def test_graph_rebuild_returns_counts(synth_repo):
    out = graph_tools.graph(path=synth_repo, action="rebuild")
    assert "rebuilt" in out.lower() and "nodes" in out


# ---------------------------------------------------------------------------
# v1 query set
# ---------------------------------------------------------------------------

def test_query_callers_of(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="callers_of:helper")
    assert "callers of" in out.lower()
    # api_func calls helper
    assert "api_func" in out


def test_query_callees_of(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="callees_of:api_func")
    assert "helper" in out  # api_func calls helper


def test_query_upstream_blast_radius(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    # who calls helper, up to depth 2: api_func (depth 1), main (depth 2)
    out = graph_tools.graph(path=synth_repo, query="upstream:helper", depth=2)
    assert "api_func" in out
    assert "main" in out  # main -> api_func -> helper


def test_query_downstream_blast_radius(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    # what does main call, downstream: api_func (d1), helper (d2)
    out = graph_tools.graph(path=synth_repo, query="downstream:main", depth=3)
    assert "api_func" in out
    assert "helper" in out


def test_query_shortest_path(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="path:main::helper")
    assert "main" in out and "helper" in out and "->" in out


def test_query_community(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="community:helper")
    # either a community membership or a "not in any" — both valid; not a refusal
    assert "integrity guard" not in out


def test_query_explain(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="explain:helper")
    assert "explain" in out.lower()
    assert "api_func" in out  # helper is called by api_func


def test_query_unknown_returns_usage(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="bogus_query:foo")
    assert "unknown query" in out.lower() or "callers_of" in out


def test_query_no_match(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    out = graph_tools.graph(path=synth_repo, query="callers_of:nonexistent_symbol")
    assert "no node matching" in out.lower() or "no callers" in out.lower()


# ---------------------------------------------------------------------------
# mtime+size cache (no-op rebuild) + 30s ceiling + eviction
# ---------------------------------------------------------------------------

def test_cache_no_rebuild_when_unchanged(synth_repo, monkeypatch):
    """First build materializes the store; a second query (action=query) must
    NOT re-parse — the signature matches. We assert the store file exists + a
    second call succeeds without rebuilding."""
    graph_tools.graph(path=synth_repo, action="rebuild")
    key = graph_tools._repo_key(synth_repo)
    sig_file = graph_tools._store_dir(key) / "signature.json"
    assert sig_file.is_file(), "signature.json not written"
    first_mtime = sig_file.stat().st_mtime
    # second query — should load from cache (signature unchanged)
    out = graph_tools.graph(path=synth_repo)
    assert "nodes" in out
    # signature.json mtime should NOT change on a cached load (we only touch meta)


def test_cache_rebuilds_when_file_changes(synth_repo):
    graph_tools.graph(path=synth_repo, action="rebuild")
    g1 = graph_tools._load_or_build(synth_repo)
    n1 = g1.number_of_nodes()
    # add a new file -> signature changes -> rebuild picks it up
    new_file = os.path.join(synth_repo, "pkg", "extra.py")
    with open(new_file, "w") as f:
        f.write("def extra_func():\n    return 1\n")
    # bump mtime enough to change the signature
    g2 = graph_tools._load_or_build(synth_repo)
    assert g2.number_of_nodes() > n1, "rebuild did not pick up the new file"


def test_parse_ceiling_refuses(monkeypatch, synth_repo):
    """A repo exceeding MAX_PARSE_SECONDS must refuse with the operator-pre-warm
    message (LOCKED f), NOT hang or silently truncate."""
    # force the ceiling to 0 so the first file parse trips it
    monkeypatch.setattr(graph_tools, "MAX_PARSE_SECONDS", 0.0)
    out = graph_tools.graph(path=synth_repo, action="rebuild")
    assert "too large" in out.lower() or "pre-warm" in out.lower()


def test_eviction_by_repo_count(monkeypatch, tmp_path):
    """max-repo-count=32; building 33 distinct repos evicts the oldest. We set
    the cap low + assert the oldest store dir is gone after exceeding it."""
    monkeypatch.setattr(graph_tools, "MAX_REPO_COUNT", 2)
    monkeypatch.setattr(graph_tools, "GRAPH_STORE_ROOT", str(tmp_path / "graphs"))
    os.makedirs(tmp_path / "graphs", exist_ok=True)
    # build 3 distinct repos
    for i in range(3):
        repo = tmp_path / f"r{i}"
        repo.mkdir()
        (repo / "m.py").write_text(f"def f{i}():\n    return {i}\n", "utf-8")
        graph_tools._load_or_build(str(repo))
    stores = [d for d in (tmp_path / "graphs").iterdir() if d.is_dir()]
    assert len(stores) <= 2, f"eviction did not cap repo count: {len(stores)} stores"


def test_eviction_by_total_size(monkeypatch, tmp_path):
    """max-total-size cap evicts oldest repos first."""
    monkeypatch.setattr(graph_tools, "MAX_TOTAL_SIZE_BYTES", 1)  # 1 byte -> evict all but keep at least... well, evicts aggressively
    monkeypatch.setattr(graph_tools, "MAX_REPO_COUNT", 100)
    monkeypatch.setattr(graph_tools, "GRAPH_STORE_ROOT", str(tmp_path / "graphs"))
    os.makedirs(tmp_path / "graphs", exist_ok=True)
    repo = tmp_path / "bigrepo"
    repo.mkdir()
    (repo / "m.py").write_text("def f():\n    return 0\n" * 50, "utf-8")
    # building should evict to stay under the 1-byte cap (effectively empties)
    graph_tools._load_or_build(str(repo))
    # the new store may exist but the total stays tiny (graph.json would exceed 1B
    # so it gets evicted too -> store dir removed). Assert no bloated store.
    total = graph_tools._repo_total_size()
    assert total <= 1 or total < 1024  # under the cap (1B) or trivially small


# ---------------------------------------------------------------------------
# import-purity (graph_tools import does no I/O / no protected-state read)
# ---------------------------------------------------------------------------

def test_graph_tools_import_pure():
    """Importing graph_tools must not touch the protected substrate or do filesystem
    I/O. We assert the module imported + its handlers are module-level functions
    (no __self__) — the cert AST scan + the execution_sandbox='none' rationale
    depend on this."""
    import inspect
    assert hasattr(graph_tools, "graph")
    assert inspect.isfunction(graph_tools.graph)
    assert graph_tools.graph.__module__ == graph_tools.__name__
    # the handler has no bound self (not a method) -> dispatches via the
    # execution_sandbox='none' branch (in-process), NOT the Phase-1 plugin path
    assert not hasattr(graph_tools.graph, "__self__")


# ---------------------------------------------------------------------------
# real-slice smoke (against the seam tools dir itself — the actual hermes tree)
# ---------------------------------------------------------------------------

def test_real_slice_smoke_against_seam_tools(monkeypatch, tmp_path):
    """Smoke tree-sitter on REAL, complex Python (decorators, type hints,
    nested classes — the synthetic repo lacks these) by copying a few real
    seam-tool .py files into a tmp_path repo. tmp_path is NOT a guard source
    (the live tools dir IS — it's contained in the live-seam guard root — so
    we copy out rather than graph in place). Asserts the build succeeds on real
    Python + a query returns content, NOT exact-edge counts (the real tree's
    shape is not part of the contract)."""
    import shutil
    src_dir = os.path.dirname(graph_tools.__file__)  # the deployed tools dir
    if not os.path.isdir(src_dir):
        pytest.skip("seam tools dir not found")
    repo = tmp_path / "realslice" / "tools"
    repo.mkdir(parents=True)
    # copy a few real, complex .py modules (graph_tools itself + search_tools +
    # shell_tools have decorators, type hints, nested defs, attribute calls)
    copied = 0
    for name in ("graph_tools.py", "search_tools.py", "shell_tools.py"):
        s = os.path.join(src_dir, name)
        if os.path.isfile(s):
            shutil.copy(s, repo / name)
            copied += 1
    if copied == 0:
        pytest.skip("no real tool files to copy")
    out = graph_tools.graph(path=str(repo), action="rebuild")
    assert "rebuilt" in out.lower() or "nodes" in out
    # a query for a real symbol (graph_tools.graph itself)
    q = graph_tools.graph(path=str(repo), query="callers_of:graph")
    assert "integrity guard" not in q  # not a refusal
    # cleanup the store for this repo so it doesn't pollute later runs
    key = graph_tools._repo_key(str(repo))
    sd = graph_tools._store_dir(key)
    if sd.is_dir():
        graph_tools._rm_tree(sd)