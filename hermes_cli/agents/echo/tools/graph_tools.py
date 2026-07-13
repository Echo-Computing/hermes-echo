"""Code-structure graph tool for the Echo agent (Graphify — Step 2 of the
recommended-order addon build, 2026-07-07).

DECISION-LOCKED (see [[hermes_echo_addon_eval_2026-07-06]] — all 6 pre-build
confirmations LOCKED):

- BUILD-LEANER from tree-sitter + networkx, NOT vendored. The vendored graphify
  ``extract.py`` is a ~17k-line monolith + graspologic (Leiden, Py3.13-incompatible);
  networkx ships ``louvain_communities`` + ``greedy_modularity_communities``
  built-in, so Leiden/graspologic is unnecessary. ~200-500 LOC of our own code.
- CORE SEAM TOOL, NOT a ToolPlugin. A core seam tool (module-level function,
  no ``__self__``) registers in ``_build_registry`` + dispatches in-process via
  the ``execution_sandbox="none"`` branch. Registered as a SIBLING of
  ``search_code``.
- ``execution_sandbox="none"`` rationale: in-process tree-sitter parse is a pure
  Python function call; reads only the LLM-supplied ``path`` (guard-source-gated)
  + writes only the non-protected graph store; ``verify_integrity`` covers handler
  source; no subprocess, no net, no cred. A mount-namespace ceiling is not in the
  public build; in-process writes to the graph store persist freely.
- ``query`` param is a DSL / node-id, NEVER a raw fs path. The guard-source path-
  gate inspects ONLY the ``path`` param; a path-shaped ``query`` would NOT be
  gate-checked (the deep-dive's KEY axis-D risk). The LOAD-BEARING mitigation is
  that ``_run_query`` does NO filesystem I/O — it resolves ONLY against already-
  built graph nodes (a path-shaped query cannot reach the FS even if it slips the
  ``_looks_like_path`` defense-in-depth check, because the resolver matches node
  ids / symbol names, never opens a path). ``_looks_like_path`` is DEFENSE-IN-
  DEPTH (a best-effort refusal of obvious path-shaped queries); it is NOT the
  primary containment. The ``path`` param is the sole FS entry point + is the
  only gate-checked surface.
- Graph store at ``~/.hermes/graphs/<repo-sha8>/`` — SIBLING of the protected
  stores, NOT in ``_PROTECTED_STORE_PATHS`` (in-process writes freely).
- v1 query set: ``callers_of``, ``callees_of``, ``upstream``/``downstream`` blast-
  radius (depth-limited), ``path`` (shortest call path), ``community`` (label),
  ``explain`` (node summary). NOT import-chain/SCC/betweenness in v1.
- Python-only in v1 (intra-file AND cross-file edges); other tree-sitter-supported
  languages get NO edges in v1 (``_discover_py_files`` rglobs only ``.py``). v2
  adds intra-file edges for other langs. Echo's seam is all .py.
- STATIC / STRUCTURAL snapshot: a CALLS edge reflects a static call-site, NOT
  runtime dispatch — the path may never execute. Blast-radius / shortest-path
  results are STATIC REACHABILITY, not runtime reachability. Named so callers do
  not read them as execution traces.
- Eviction: max-total-size 256 MB + max-repo-count 32, LRU on access time, atomic
  per-``<repo-sha8>/`` dir.
- ``action=rebuild`` LLM-triggered IS allowed, guarded by (1) mtime+size signature
  cache (rebuild only when stale), (2) a 30s max-parse ceiling (refuse w/ an
  "operator pre-warm via CLI" message if exceeded). The ceiling is REACTIVE /
  per-file: it fires AFTER each ``_parse_file`` returns, not as a hard wall-clock
  interrupt mid-parse — a single pathological oversized ``.py`` file can run past
  30s before the check fires (a v1 residual; a hard interrupt would need wrapping
  ``_parse_file`` in a timeout). This bounds the PARSE cost of a rebuild, NOT the
  in-process dispatch wall-clock (there is no per-tool-call timeout on the
  ``execution_sandbox="none"`` path — the 30s ceiling is the only time bound).

Two-version rule: this is the PRIVATE seam source (the seam source graph_tools.py is copied to its live location at
``~/hermes-echo/hermes_cli/agents/echo/tools/graph_tools.py`` via apply_seam.sh,
NEVER pushed to the public Echo-Computing/hermes-echo tree). The PUBLIC sanitized
fork (``codegraph-echo``) strips the seam-internal refs (``_protected_roots`` /
``_guard_source_roots`` -> a generic blocklist; drop the ``verify_integrity`` hook;
rephrase private substrate refs -> "protected paths"/"integrity check") + is a
SEPARATE two-version deliverable, deferred until the private seam is GREEN + the
step-7 adversarial re-verify upholds ``execution_sandbox="none"`` (so a re-verify
overturn does not waste the public-fork work). Graph vocab (graph_tools.py,
graph_query/graph_path/graph_explain, graph.json, codegraph-echo) carries NONE of
the banned hygiene tokens natively; hazards are INHERITANCE only + are caught by
the public-hygiene grep gate.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

# tree-sitter + networkx are OPTIONAL deps (graph extra; see pyproject.toml
# [project.optional-dependencies] graph). A fresh `uv sync` without the extra,
# CI, the public codegraph-echo fork, or /usr/bin/python3 (no tree_sitter) must
# NOT brick the ENTIRE echo agent at import. Wrap the imports: on ImportError,
# set _TS_AVAILABLE=False + graph() returns a clear dormant message. The tool
# still registers (handler=graph is a module-level callable) so the registry
# surface is stable, but it no-ops without the deps. (D-1, 4-lens re-verify.)
try:
    import networkx as nx
    from tree_sitter import Language, Parser
    import tree_sitter_python as _tspython
    _PY_LANGUAGE = Language(_tspython.language())
    _PY_PARSER = Parser(_PY_LANGUAGE)
    _TS_AVAILABLE = True
    _TS_IMPORT_ERROR = ""
except ImportError as _e:  # tree-sitter / tree-sitter-python / networkx not installed
    _TS_AVAILABLE = False
    _TS_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    _PY_PARSER = None
    nx = None  # module-level refs (graph() short-circuits before any nx use)

from hermes_cli.agents.echo.tools.shell_tools import (
    _protected_roots, _guard_source_roots, _path_is_contained_in_root,
)


# ---------------------------------------------------------------------------
# config (module-level constants; NOT LLM-supplied — seam config, not tool params)
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser("~")
GRAPH_STORE_ROOT = os.path.join(_HOME, ".hermes", "graphs")
MAX_TOTAL_SIZE_BYTES = 256 * 1024 * 1024   # 256 MB across all repos (LOCKED e)
MAX_REPO_COUNT = 32                        # secondary cap (LOCKED e)
MAX_PARSE_SECONDS = 30.0                   # rebuild ceiling (LOCKED f)
DEFAULT_DEPTH = 2

# The closed relation enum (we own the code; no free-form relations).
REL_DEFINES = "DEFINES"
REL_CALLS = "CALLS"
REL_IMPORTS = "IMPORTS"
REL_INHERITS = "INHERITS"
_RELATIONS = (REL_DEFINES, REL_CALLS, REL_IMPORTS, REL_INHERITS)

# Edge provenance tags: EXTRACTED = the target is parsed from this repo's source;
# INFERRED = the target is resolved by the cross-file resolver (name match).
TAG_EXTRACTED = "EXTRACTED"
TAG_INFERRED = "INFERRED"

# File extensions tree-sitter-python can parse. v1 is Python-only (intra-file +
# cross-file); other tree-sitter langs get no edges in v1 (v2 adds intra-file).
_PY_SUFFIXES = (".py",)

# _PY_LANGUAGE / _PY_PARSER are constructed in the try/except import block above
# (D-1 graceful degrade). _PY_PARSER is None when deps are missing.


# ---------------------------------------------------------------------------
# containment pre-check (mirrors search_code: the `path` param IS guard-source-
# gated, but the handler must re-fire for the omitted-path / '.' / '~' / '..'
# cases the matcher cannot see — the matcher sees params, not the handler cwd)
# ---------------------------------------------------------------------------

def _resolved(path: Optional[str]) -> str:
    """expanduser + realpath the given path (or cwd if None/empty). Mirrors
    search_code._resolved."""
    p = os.path.realpath(os.path.expanduser(path or "") or ".")
    return p


def _refuse_path(resolved: str) -> str:
    """Refusal string for a path that resolves to a protected store / guard
    source / ancestor. Names 'protected store' so a leak-probe arm can match on
    it (consistent with search_code._search_refused_message)."""
    return (f"axis-D: graph refused — base path resolves to a protected store "
            f"or guard source or its ancestor ({resolved!r}). The tool-capable "
            f"LLM must not graph the axis-D substrate. Use a workspace path.")


def _containment_check(path: Optional[str]) -> Optional[str]:
    """Return a refusal string if ``path`` resolves to a protected store / guard
    source / ancestor (the axis-D containment pre-check). Returns None if OK.

    ``allow_ancestor=True`` so a base ABOVE a store (e.g. ``~`` / ``/home``)
    is refused — a recursive parse from there would rglob the stores into the
    graph. The matcher's path-gate inspects only the ``path`` param; the omitted
    path / ``.`` / ``~`` cases reach the handler cwd + are caught HERE."""
    resolved = _resolved(path)
    for root in _protected_roots():
        if _path_is_contained_in_root(resolved, root, allow_ancestor=True):
            return _refuse_path(resolved)
    for root in _guard_source_roots():
        if _path_is_contained_in_root(resolved, root, allow_ancestor=True):
            return _refuse_path(resolved)
    return None


# ---------------------------------------------------------------------------
# repo key + graph store layout
# ---------------------------------------------------------------------------

def _repo_key(repo_root: str) -> str:
    """Stable 8-hex key for a repo root (sha1 of the realpath, first 8 hex). Two
    different realpath strings never collide in practice; the key is the graph
    store dir name. NOT a content hash (a content hash would force a re-key on
    every edit -> no cache; the mtime+size signature handles staleness)."""
    return hashlib.sha1(os.path.realpath(repo_root).encode("utf-8")).hexdigest()[:8]


def _store_dir(repo_key: str) -> Path:
    return Path(GRAPH_STORE_ROOT) / repo_key


def _repo_total_size() -> int:
    """Sum of graph.json sizes across all repo store dirs (the eviction metric)."""
    root = Path(GRAPH_STORE_ROOT)
    if not root.is_dir():
        return 0
    total = 0
    for d in root.iterdir():
        if d.is_dir():
            gj = d / "graph.json"
            if gj.is_file():
                try:
                    total += gj.stat().st_size
                except OSError:
                    pass
    return total


def _evict_lru() -> None:
    """Enforce MAX_TOTAL_SIZE_BYTES + MAX_REPO_COUNT. Eviction is atomic per-
    repo (remove the whole ``<repo-sha8>/`` dir, never a single file). LRU on
    access time (meta.json mtime updated on every load)."""
    root = Path(GRAPH_STORE_ROOT)
    if not root.is_dir():
        return
    repos = []
    for d in root.iterdir():
        if d.is_dir():
            meta = d / "meta.json"
            try:
                mtime = meta.stat().st_mtime if meta.is_file() else d.stat().st_mtime
                gj = d / "graph.json"
                size = gj.stat().st_size if gj.is_file() else 0
            except OSError:
                continue
            repos.append((mtime, size, d))
    # Evict by count first (oldest access), then by size.
    repos.sort(key=lambda t: t[0])  # oldest first
    while len(repos) > MAX_REPO_COUNT:
        _mt, _sz, d = repos.pop(0)
        _rm_tree(d)
    # Now enforce size cap (oldest first).
    total = sum(sz for _mt, sz, _d in repos)
    i = 0
    while total > MAX_TOTAL_SIZE_BYTES and i < len(repos):
        _mt, sz, d = repos[i]
        _rm_tree(d)
        total -= sz
        i += 1


def _rm_tree(p: Path) -> None:
    try:
        for child in p.iterdir():
            if child.is_dir():
                _rm_tree(child)
            else:
                child.unlink()
        p.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# staleness signature (mtime + size per file — rebuild only when stale)
# ---------------------------------------------------------------------------

def _signature(repo_root: str, py_files) -> dict:
    """{relative_path: [mtime, size]} for every .py file under repo_root. Cheap
    (one stat per file); the rebuild gate compares this to the stored
    signature.json — a no-op rebuild costs N stats, not a re-parse."""
    sig = {}
    root = Path(repo_root)
    for f in py_files:
        try:
            st = f.stat()
        except OSError:
            continue
        rel = str(f.relative_to(root)).replace(os.sep, "/")
        sig[rel] = [st.st_mtime, st.st_size]
    return sig


def _is_stale(repo_key: str, current_sig: dict) -> bool:
    """True if the stored signature differs from current (missing store, missing
    file, changed mtime/size, or extra file)."""
    sd = _store_dir(repo_key)
    sj = sd / "signature.json"
    if not sj.is_file():
        return True
    try:
        stored = json.loads(sj.read_text("utf-8"))
    except (OSError, ValueError):
        return True
    if set(stored.keys()) != set(current_sig.keys()):
        return True
    for k, v in current_sig.items():
        if stored.get(k) != v:
            return True
    return False


# ---------------------------------------------------------------------------
# tree-sitter parse → per-file symbols + raw references
# ---------------------------------------------------------------------------

def _node_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _qualified_name(node, src: bytes) -> str:
    """Best-effort name of a function/class node (handles decorators, async)."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier"):
            return _node_text(src, child)
    return ""


def _parse_file(path: Path) -> dict:
    """Parse one .py file with tree-sitter; return
    ``{defines:[...], calls:[...], imports:[...], inherits:[...]}`` where each
    entry is a dict of raw, file-local reference data (NOT yet cross-resolved).

    - defines: {name, kind (function|class), line, parent (enclosing class or "")}
    - calls:   {name, line, caller (the enclosing function qualified name or "")}
    - imports: {stmt, line, kind (from|import), module, names (list), level (rel)}
    - inherits:{name, line, subclass (the enclosing class qualified name)}
    """
    try:
        src = path.read_bytes()
    except OSError:
        return {"defines": [], "calls": [], "imports": [], "inherits": []}
    tree = _PY_PARSER.parse(src)
    root = tree.root_node

    defines = []
    calls = []
    imports = []
    inherits = []

    def _walk(node, enclosing_class=""):
        # class / function definitions
        if node.type == "class_definition":
            name = _qualified_name(node, src)
            qn = name
            # bases (inheritance)
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            inherits.append({
                                "name": _node_text(src, arg),
                                "line": arg.start_point[0] + 1,
                                "subclass": qn,
                            })
            for child in node.children:
                if child.type == "block":
                    _walk(child, enclosing_class=qn)
            defines.append({"name": name, "kind": "class",
                            "line": node.start_point[0] + 1, "parent": ""})
            return
        if node.type == "function_definition":
            name = _qualified_name(node, src)
            qn = f"{enclosing_class}.{name}" if enclosing_class else name
            defines.append({"name": name, "kind": "function",
                            "line": node.start_point[0] + 1,
                            "parent": enclosing_class})
            # collect calls inside this function body (the caller context)
            for child in node.children:
                if child.type == "block":
                    _collect_calls(child, caller=qn)
            return
        # import statements (module-level + nested; we collect all)
        if node.type == "import_statement":
            names, module, level = _parse_import(node, src, is_from=False)
            imports.append({"stmt": _node_text(src, node),
                            "line": node.start_point[0] + 1,
                            "kind": "import", "module": module,
                            "names": names, "level": level})
            return
        if node.type == "import_from_statement":
            names, module, level = _parse_import(node, src, is_from=True)
            imports.append({"stmt": _node_text(src, node),
                            "line": node.start_point[0] + 1,
                            "kind": "from", "module": module,
                            "names": names, "level": level})
            return
        for child in node.children:
            _walk(child, enclosing_class)

    def _collect_calls(node, caller):
        if node.type == "call":
            fn = node.children[0] if node.children else None
            if fn is not None:
                # only bare-name + attribute calls (foo() / self.foo() / obj.foo())
                if fn.type == "identifier":
                    calls.append({"name": _node_text(src, fn),
                                  "line": node.start_point[0] + 1,
                                  "caller": caller})
                elif fn.type == "attribute":
                    # obj.method -> record the attribute name (method resolution
                    # is v2; we record the tail so callers_of:method finds it)
                    attr = fn.children[-1] if fn.children else None
                    if attr is not None and attr.type == "identifier":
                        calls.append({"name": _node_text(src, attr),
                                      "line": node.start_point[0] + 1,
                                      "caller": caller})
            return
        for child in node.children:
            _collect_calls(child, caller)

    _walk(root)
    return {"defines": defines, "calls": calls, "imports": imports,
            "inherits": inherits}


def _parse_import(node, src: bytes, is_from: bool):
    """Return (names, module, level) for an import statement. ``level`` is the
    relative-import dot count (0 = absolute). ``names`` is the list of bound
    local names (for ``import x`` -> [x]; ``from x import a, b`` -> [a, b];
    ``from . import y`` -> [y]).

    tree-sitter-python shape (verified 0.25):
      ``from .base import Base``  -> import_from_statement( relative_import(
        import_prefix '.', dotted_name 'base' ), dotted_name 'Base' )
      ``from pkg import api_func`` -> import_from_statement( dotted_name 'pkg',
        dotted_name 'api_func' )
      ``import x.y.z``            -> import_statement( dotted_name 'x.y.z' )
    So the FIRST module-specifier child is ``relative_import`` (relative) OR
    ``dotted_name`` (absolute module); the REST are imported names
    (``dotted_name`` / ``identifier`` / ``aliased_import`` / ``wildcard_import``).
    A relative_import's level = dots in its ``import_prefix`` children; its
    relative module = its ``dotted_name`` child (may be absent for ``from .``)."""
    names = []
    module = ""
    level = 0
    module_set = False
    for child in node.children:
        t = child.type
        if not module_set and t == "relative_import":
            for sub in child.children:
                if sub.type == "import_prefix":
                    level += _node_text(src, sub).count(".")
                elif sub.type == "dotted_name":
                    module = _node_text(src, sub)
            module_set = True
            continue
        if not module_set and t == "dotted_name":
            module = _node_text(src, child)
            module_set = True
            continue
        # past the module specifier — these are imported names
        if t == "dotted_name":
            names.append(_node_text(src, child))
        elif t == "aliased_import":
            # import x as y -> bind y; from x import a as b -> bind b (the alias)
            ids = [s for s in child.children if s.type == "identifier"]
            if ids:
                names.append(_node_text(src, ids[-1]))
        elif t == "identifier":
            names.append(_node_text(src, child))
        elif t == "wildcard_import":
            names.append("*")
    # for plain `import x.y.z`, bind the head only (Python binds `x`).
    if not is_from and module and not names:
        names = [module.split(".")[0]]
    return names, module, level


# ---------------------------------------------------------------------------
# two-pass cross-file resolver (the only non-trivial piece — Python-only in v1)
# ---------------------------------------------------------------------------

def _build_index(all_files: dict) -> dict:
    """Pass 1: collect a name -> [node_id] index of every DEFINES across the repo
    + a per-file import map (local_name -> module path or qualified symbol).

    ``all_files`` = {relpath: parsed_dict}.
    Returns (defines_index, per_file_imports) where:
      defines_index = {name: [node_id, ...]}  (name = unqualified symbol name)
      per_file_imports = {relpath: {local_name: target_module_dotted}}
    """
    defines_index = {}
    for rel, parsed in all_files.items():
        for d in parsed["defines"]:
            nm = d["name"]
            if not nm:
                continue
            defines_index.setdefault(nm, []).append(_node_id(rel, d))
    per_file_imports = {}
    for rel, parsed in all_files.items():
        imap = {}
        for imp in parsed["imports"]:
            if imp["kind"] == "from" and imp["level"] == 0:
                # from X import a, b -> bind a, b to module X
                for nm in imp["names"]:
                    if nm and nm != "*":
                        imap[nm] = imp["module"]
            elif imp["kind"] == "import":
                # import x [as y] -> bind the local alias to the dotted module
                for nm in imp["names"]:
                    if nm:
                        imap[nm] = imp["module"] or nm
            # relative imports (level>0): resolve against the package path; v1
            # records the module name as-is (the resolver matches by name; full
            # relative resolution is a v1 best-effort, v2 hardens it).
            elif imp["kind"] == "from" and imp["level"] > 0:
                pkg = rel.rsplit("/", 1)[0] if "/" in rel else ""
                for _ in range(imp["level"] - 1):
                    pkg = pkg.rsplit("/", 1)[0] if "/" in pkg else ""
                base = (pkg + "." + imp["module"]) if imp["module"] else pkg
                for nm in imp["names"]:
                    if nm and nm != "*":
                        imap[nm] = base
        per_file_imports[rel] = imap
    return defines_index, per_file_imports


def _node_id(rel: str, d: dict) -> str:
    """Stable node id: relpath:line:name (unique within a repo)."""
    parent = d.get("parent", "")
    qn = f"{parent}.{d['name']}" if parent else d["name"]
    return f"{rel}:{d['line']}:{qn}"


def _resolve_call(name: str, rel: str, defines_index: dict,
                  per_file_imports: dict):
    """Resolve a bare-name call ``name`` in file ``rel`` to a list of node_ids +
    a tag (EXTRACTED if same-file def, INFERRED if cross-file / import-bound).

    Order: (1) same-file def named ``name``; (2) import map binds ``name`` to a
    module whose exported symbol named ``name`` exists in the repo; (3) any repo
    def named ``name``. Unresolved -> ([], INFERRED) (no edge)."""
    # (1) same-file def
    same_file = [nid for nid in defines_index.get(name, [])
                 if nid.startswith(rel + ":")]
    if same_file:
        return same_file, TAG_EXTRACTED
    # (2) import-bound: the name was imported from a module; if a repo def of
    # that name exists, it's the target (v1: assume the imported name == the def
    # name; full module->file mapping is v2). Tag INFERRED.
    imap = per_file_imports.get(rel, {})
    if name in imap and defines_index.get(name):
        return defines_index[name][:], TAG_INFERRED
    # (3) any repo def named name (ambiguous -> all, INFERRED)
    if defines_index.get(name):
        return defines_index[name][:], TAG_INFERRED
    return [], TAG_INFERRED


# ---------------------------------------------------------------------------
# graph build (nx.DiGraph) + store load/save
# ---------------------------------------------------------------------------

def _discover_py_files(repo_root: str) -> list:
    """All .py files under repo_root (recursive), sorted for determinism.
    Skips common venv / cache / hidden dirs."""
    out = []
    root = Path(repo_root)
    skip = {"__pycache__", ".venv", ".venv312", ".venv313", "venv", "node_modules",
            ".git", ".pytest_cache", ".mypy_cache", ".tox", "build", "dist"}
    for p in sorted(root.rglob("*.py")):
        parts = set(p.relative_to(root).parts)
        if parts & skip:
            continue
        if any(part.startswith(".") and part not in (".", "..") for part in parts):
            continue
        out.append(p)
    return out


def _build_nx_graph(repo_root: str, all_files: dict) -> nx.DiGraph:
    """Pass 2: build the nx.DiGraph from the per-file parsed data + the
    cross-file resolver. Nodes carry {name, kind, file, line}; edges carry
    {relation, tag}. DEFINES edges go module->symbol (so the file is a node too)."""
    g = nx.DiGraph()
    defines_index, per_file_imports = _build_index(all_files)

    # file nodes (kind=module)
    for rel in all_files:
        g.add_node(rel, name=rel, kind="module", file=rel, line=0)

    # symbol nodes + DEFINES edges
    for rel, parsed in all_files.items():
        for d in parsed["defines"]:
            nid = _node_id(rel, d)
            g.add_node(nid, name=d["name"], kind=d["kind"], file=rel,
                       line=d["line"])
            g.add_edge(rel, nid, relation=REL_DEFINES, tag=TAG_EXTRACTED)

    # CALLS edges (caller fn -> callee def, cross-resolved)
    for rel, parsed in all_files.items():
        for c in parsed["calls"]:
            caller = c["caller"]
            caller_nid = _find_def_node(g, rel, caller)
            if caller_nid is None:
                continue
            targets, tag = _resolve_call(c["name"], rel, defines_index,
                                         per_file_imports)
            for tgt in targets:
                if tgt in g and tgt != caller_nid:
                    g.add_edge(caller_nid, tgt, relation=REL_CALLS, tag=tag)

    # INHERITS edges (subclass -> baseclass def, cross-resolved)
    for rel, parsed in all_files.items():
        for h in parsed["inherits"]:
            sub_nid = _find_def_node(g, rel, h["subclass"])
            if sub_nid is None:
                continue
            targets, tag = _resolve_call(h["name"], rel, defines_index,
                                         per_file_imports)
            for tgt in targets:
                if tgt in g and tgt != sub_nid:
                    g.add_edge(sub_nid, tgt, relation=REL_INHERITS, tag=tag)

    # IMPORTS edges (file -> file, by best-effort module->path mapping). v1:
    # map a dotted module to a relpath (pkg.mod -> pkg/mod.py or pkg/mod/__init__.py)
    # and add a file->file IMPORTS edge if the target file is in the graph.
    for rel, parsed in all_files.items():
        seen = set()
        for imp in parsed["imports"]:
            mod = imp["module"] or ""
            if not mod or imp["level"] > 0:
                # relative import: resolve against package path
                if imp["level"] > 0:
                    pkg = rel.rsplit("/", 1)[0] if "/" in rel else ""
                    for _ in range(imp["level"] - 1):
                        pkg = pkg.rsplit("/", 1)[0] if "/" in pkg else ""
                    base = (pkg + "." + mod) if mod else pkg
                    mod = base
                if not mod:
                    continue
            tgt_rel = _module_to_relpath(mod)
            if tgt_rel and tgt_rel != rel and tgt_rel in g and tgt_rel not in seen:
                seen.add(tgt_rel)
                g.add_edge(rel, tgt_rel, relation=REL_IMPORTS, tag=TAG_INFERRED)

    return g


def _find_def_node(g: nx.DiGraph, rel: str, qualified_name: str) -> Optional[str]:
    """Find the graph node for a symbol ``qualified_name`` defined in ``rel``."""
    for nid, data in g.nodes(data=True):
        if data.get("file") == rel and data.get("kind") in ("function", "class"):
            # match the qualified name: rel:line:parent.name or rel:line:name
            if nid.endswith(":" + qualified_name):
                return nid
    # fall back to the file's first def whose name tail matches
    tail = qualified_name.split(".")[-1]
    for nid, data in g.nodes(data=True):
        if data.get("file") == rel and data.get("name") == tail:
            return nid
    return None


def _module_to_relpath(module: str) -> Optional[str]:
    """Best-effort map a dotted module name to a repo-relative path
    (``pkg.mod`` -> ``pkg/mod.py`` or ``pkg/mod/__init__.py``). The caller checks
    membership in the graph; this just produces candidates. v1 returns the .py
    form (the __init__ form is a v1 simplification — both checked by the caller)."""
    if not module:
        return None
    return module.replace(".", "/") + ".py"


def _graph_to_json(g: nx.DiGraph) -> dict:
    """Serialize nx.DiGraph to a JSON-safe dict (node_link_data with str attrs)."""
    nodes = []
    for nid, data in g.nodes(data=True):
        nodes.append({"id": nid, **{k: str(v) for k, v in data.items()}})
    edges = []
    for u, v, data in g.edges(data=True):
        edges.append({"source": u, "target": v,
                      **{k: str(v2) for k, v2 in data.items()}})
    return {"nodes": nodes, "edges": edges}


def _graph_from_json(data: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in data.get("nodes", []):
        nid = n.pop("id")
        g.add_node(nid, **n)
    for e in data.get("edges", []):
        u = e.pop("source")
        v = e.pop("target")
        g.add_edge(u, v, **e)
    return g


def _save_graph(repo_key: str, g: nx.DiGraph, sig: dict, repo_root: str) -> None:
    sd = _store_dir(repo_key)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "graph.json").write_text(json.dumps(_graph_to_json(g)), "utf-8")
    (sd / "signature.json").write_text(json.dumps(sig), "utf-8")
    (sd / "meta.json").write_text(json.dumps({
        "repo_root": os.path.realpath(repo_root),
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
        "built_at": time.time(),
    }), "utf-8")


def _load_graph(repo_key: str) -> Optional[nx.DiGraph]:
    gj = _store_dir(repo_key) / "graph.json"
    if not gj.is_file():
        return None
    try:
        return _graph_from_json(json.loads(gj.read_text("utf-8")))
    except (OSError, ValueError):
        return None


def _touch_meta(repo_key: str) -> None:
    """Update meta.json mtime (the LRU access-time signal) on every load."""
    mp = _store_dir(repo_key) / "meta.json"
    try:
        if mp.is_file():
            os.utime(mp, None)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# load-or-build (lazy + mtime cache + 30s ceiling + eviction)
# ---------------------------------------------------------------------------

def _load_or_build(repo_root: str, force: bool = False) -> nx.DiGraph:
    key = _repo_key(repo_root)
    py_files = _discover_py_files(repo_root)
    sig = _signature(repo_root, py_files)
    stale = force or _is_stale(key, sig)
    if not stale:
        g = _load_graph(key)
        if g is not None:
            _touch_meta(key)
            return g
    # rebuild — enforce the 30s parse ceiling (LOCKED f)
    t0 = time.monotonic()
    all_files = {}
    for f in py_files:
        rel = str(f.relative_to(repo_root)).replace(os.sep, "/")
        all_files[rel] = _parse_file(f)
        if time.monotonic() - t0 > MAX_PARSE_SECONDS:
            raise _ParseTooLarge(repo_root)
    g = _build_nx_graph(repo_root, all_files)
    _save_graph(key, g, sig, repo_root)
    _evict_lru()  # after saving — enforce caps on the FULL set incl. the new store
    # (the just-saved store has the newest mtime -> survives LRU; a single repo
    # bigger than MAX_TOTAL_SIZE_BYTES evicts itself -> always-rebuild degenerate,
    # acceptable for an absurdly low cap, never hit at the real 256MB)
    return g


class _ParseTooLarge(Exception):
    """Raised when a rebuild would exceed MAX_PARSE_SECONDS. The handler turns
    this into the operator-pre-warm refusal (LOCKED f)."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root


# ---------------------------------------------------------------------------
# query DSL (callers_of / callees_of / upstream / downstream / path / community /
# explain). The query is NEVER a raw fs path — a path-shaped query is refused.
# ---------------------------------------------------------------------------

_QUERY_PREFIXES = ("callers_of:", "callees_of:", "upstream:", "downstream:",
                   "path:", "community:", "explain:")

# a node-id-shaped query: relpath:line:qualified_name (heuristic — contains a
# path separator + a colon + a .py-ish leading segment). Used to REFUSE a raw fs
# path sneaking in via the query param (the guard-source gate inspects only
# `path`, not `query`). DEFENSE-IN-DEPTH ONLY: the load-bearing mitigation is
# that _run_query does NO filesystem I/O (it resolves only against already-built
# graph nodes), so even a path-shaped query that slips this check cannot reach
# the FS. This check just refuses the obvious cases for a cleaner refusal msg.
def _looks_like_path(q: str) -> bool:
    s = q.strip()
    if not s:
        return False
    # explicit query DSL — not a path
    if any(s.startswith(p) for p in _QUERY_PREFIXES):
        return False
    # a bare node id the resolver produced: rel/path.py:line:name — allow if it
    # has the `:line:name` shape. A raw fs path like /home/x/foo or ../bar has NO
    # second colon; refuse it.
    if s.startswith(("/", "~")) or s.startswith(".."):
        return True
    # if it looks like a windows/posix path with a dot-suffix + no `:N:name`, treat
    # as path-shaped — refuse.
    parts = s.split(":")
    first = parts[0]
    # a path-ish first segment: contains a separator, a backslash, or ends .py —
    # refuse even with a `:N:name` suffix (e.g. a protected-store path or a
    # private source path are path-shaped, not node ids a caller would
    # legitimately pass). A genuine node id's first segment is a relpath
    # (pkg/mod.py) — but a relpath containing a protected-store marker is still
    # path-shaped from the caller's intent; refuse it.
    if "/" in first or "\\" in first or first.endswith(".py"):
        # allow ONLY if first is a plain relpath with no protected marker
        # (a legit node id like 'pkg/util.py:12:helper'). A protected-store
        # marker in the first segment => refuse regardless of suffix.
        _MARKERS = (".hermes",)
        if any(m in first for m in _MARKERS):
            return True
        # else it's a plain relpath node id — allow
        return False
    if len(parts) < 3:
        # could still be a bare symbol name (allowed) — only refuse if path-ish
        if "/" in s or "\\" in s or s.endswith(".py"):
            return True
        return False
    return False


def _resolve_node_id(g: nx.DiGraph, q: str) -> Optional[str]:
    """Resolve a query target to a node id. ``q`` after the DSL prefix may be a
    node id (rel:line:name) or a bare symbol name (first match)."""
    if q in g:
        return q
    # bare name -> first matching symbol node
    for nid, data in g.nodes(data=True):
        if data.get("kind") in ("function", "class") and data.get("name") == q:
            return nid
    # tail match on qualified name
    for nid in g.nodes():
        if nid.endswith(":" + q):
            return nid
    return None


def _describe_node(g: nx.DiGraph, nid: str) -> str:
    d = g.nodes[nid]
    return f"{d.get('name','?')} ({d.get('kind','?')}) at {d.get('file','?')}:{d.get('line','?')}"


def _run_query(g: nx.DiGraph, query: str, depth: int) -> str:
    """Interpret the DSL. Returns a human-readable string result."""
    q = (query or "").strip()
    if not q:
        # default: a small summary of the graph
        return (f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
                f"across {sum(1 for _,d in g.nodes(data=True) if d.get('kind')=='module')} files. "
                f"Use a query like 'callers_of:foo', 'callees_of:bar', "
                f"'upstream:baz', 'downstream:baz', 'path:foo::bar', "
                f"'community:foo', 'explain:foo'.")

    if _looks_like_path(q):
        return ("axis-D: graph query refused — the query looks like a filesystem "
                "path. The `query` param is a DSL / node-id, not a path; use the "
                "`path` param for the repo root.")

    depth = max(1, min(int(depth or DEFAULT_DEPTH), 10))

    if q.startswith("callers_of:"):
        tgt = _resolve_node_id(g, q[len("callers_of:"):])
        if tgt is None:
            return f"no node matching {q[len('callers_of:'):]}!r"
        callers = [u for u, _, r in g.in_edges(tgt, data=True)
                   if r.get("relation") == REL_CALLS]
        if not callers:
            return f"no callers of {_describe_node(g, tgt)} (it is a root)."
        return "callers of " + _describe_node(g, tgt) + ":\n" + "\n".join(
            f"  - {_describe_node(g, c)}" for c in sorted(set(callers)))

    if q.startswith("callees_of:"):
        tgt = _resolve_node_id(g, q[len("callees_of:"):])
        if tgt is None:
            return f"no node matching {q[len('callees_of:'):]}!r"
        callees = [v for _, v, r in g.out_edges(tgt, data=True)
                   if r.get("relation") == REL_CALLS]
        if not callees:
            return f"{_describe_node(g, tgt)} calls nothing (it is a leaf)."
        return "callees of " + _describe_node(g, tgt) + ":\n" + "\n".join(
            f"  - {_describe_node(g, c)}" for c in sorted(set(callees)))

    if q.startswith("upstream:") or q.startswith("downstream:"):
        is_up = q.startswith("upstream:")
        tgt = _resolve_node_id(g, q[len("upstream:") if is_up else len("downstream:"):])
        if tgt is None:
            return f"no node matching {q.split(':', 1)[1]}!r"
        # walk CALLS edges only (the call graph), reverse for upstream
        visited = set()
        frontier = [tgt]
        rels = {REL_CALLS}
        for _step in range(depth):
            nxt = []
            for n in frontier:
                edges = g.in_edges(n, data=True) if is_up else g.out_edges(n, data=True)
                for u, v, r in (edges if is_up else edges):
                    other = u if is_up else v
                    if r.get("relation") in rels and other not in visited and other != tgt:
                        visited.add(other)
                        nxt.append(other)
            if not nxt:
                break
            frontier = nxt
        if not visited:
            return f"no {'upstream' if is_up else 'downstream'} callers within depth {depth} of {_describe_node(g, tgt)}."
        label = "upstream (callers)" if is_up else "downstream (callees)"
        return f"{label} of {_describe_node(g, tgt)} (depth {depth}):\n" + "\n".join(
            f"  - {_describe_node(g, n)}" for n in sorted(visited))

    if q.startswith("path:"):
        # path:a::b -> shortest CALLS path from a to b
        rest = q[len("path:"):]
        if "::" not in rest:
            return "path query needs the form path:src::dst (e.g. path:foo::bar)."
        a, b = rest.split("::", 1)
        na = _resolve_node_id(g, a.strip())
        nb = _resolve_node_id(g, b.strip())
        if na is None or nb is None:
            return f"no node matching {a!r} or {b!r}."
        callg = nx.DiGraph()
        for u, v, r in g.edges(data=True):
            if r.get("relation") == REL_CALLS:
                callg.add_edge(u, v)
        try:
            p = nx.shortest_path(callg, na, nb)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return f"no call path {a!r} -> {b!r}."
        return "shortest call path " + a + " -> " + b + ":\n" + " -> ".join(
            _describe_node(g, n).split(" (")[0] for n in p)

    if q.startswith("community:"):
        tgt = _resolve_node_id(g, q[len("community:"):])
        # build the undirected call graph + run greedy_modularity. louvain_communities
        # is also available (nx.algorithms.community) but is RANDOMIZED (needs a seed
        # for stable output); greedy_modularity_communities is DETERMINISTIC, so v1
        # uses it for reproducible community labels. v2 may expose louvain behind an
        # explicit seed param.
        ug = nx.Graph()
        for u, v, r in g.edges(data=True):
            if r.get("relation") == REL_CALLS:
                ug.add_edge(u, v)
        try:
            comms = nx.algorithms.community.greedy_modularity_communities(ug)
        except Exception:
            comms = []
        # find which community tgt is in
        if tgt is None:
            return (f"community detection: {len(comms)} communities found. "
                    f"Use community:<node> to find a node's community.")
        for i, comm in enumerate(comms):
            if tgt in comm:
                members = sorted(comm)
                return (f"community #{i} of {_describe_node(g, tgt)} "
                        f"({len(members)} members):\n" + "\n".join(
                            f"  - {_describe_node(g, m)}" for m in members[:50]))
        return f"{_describe_node(g, tgt)} is not in any detected community."

    if q.startswith("explain:"):
        tgt = _resolve_node_id(g, q[len("explain:"):])
        if tgt is None:
            return f"no node matching {q[len('explain:'):]}!r"
        callers = [u for u, _, r in g.in_edges(tgt, data=True) if r.get("relation") == REL_CALLS]
        callees = [v for _, v, r in g.out_edges(tgt, data=True) if r.get("relation") == REL_CALLS]
        defs = [u for u, _, r in g.in_edges(tgt, data=True) if r.get("relation") == REL_DEFINES]
        inh = [v for _, v, r in g.out_edges(tgt, data=True) if r.get("relation") == REL_INHERITS]
        lines = [f"explain {_describe_node(g, tgt)}:"]
        if defs:
            lines.append(f"  defined in: {_describe_node(g, defs[0]).split(' at ')[-1]}")
        if callers:
            lines.append(f"  called by ({len(callers)}): " + ", ".join(sorted({g.nodes[c]['name'] for c in callers})))
        else:
            lines.append("  called by: (none — root)")
        if callees:
            lines.append(f"  calls ({len(callees)}): " + ", ".join(sorted({g.nodes[c]['name'] for c in callees})))
        else:
            lines.append("  calls: (none — leaf)")
        if inh:
            lines.append(f"  inherits from: " + ", ".join(sorted({g.nodes[c]['name'] for c in inh})))
        return "\n".join(lines)

    return (f"unknown query {q!r}. Use one of: callers_of:, callees_of:, "
            f"upstream:, downstream:, path:a::b, community:, explain: — or omit for a summary.")


# ---------------------------------------------------------------------------
# the handler (the registered tool — module-level function, no __self__)
# ---------------------------------------------------------------------------

def graph(path: Optional[str] = None, action: str = "query",
          query: Optional[str] = None, depth: int = DEFAULT_DEPTH) -> str:
    """Build + query a code-structure graph for a Python repo.

    params (per the registered SeamedTool schema):
      path   — repo root to graph (required; guard-source-gated; the containment
               pre-check refuses a protected store / guard source / ancestor).
      action — query | rebuild | explain (default query). ``rebuild`` forces a
               re-parse (guarded by the mtime+size cache + a 30s ceiling).
      query  — a DSL / node-id, NEVER a raw fs path. Forms:
               callers_of:<name>, callees_of:<name>, upstream:<name>,
               downstream:<name>, path:<src>::<dst>, community:<name>,
               explain:<name>. Omit for a graph summary.
      depth  — blast-radius depth for upstream/downstream (default 2, cap 10).

    Returns a human-readable string. The graph is cached at
    ``~/.hermes/graphs/<repo-sha8>/`` (sibling of the protected stores; not
    masked — in-process writes freely). Eviction: 256 MB total / 32 repos LRU.
    """
    # 0. graceful degrade: if the optional graph deps (tree-sitter / networkx) are
    # not installed, the tool is DORMANT — refuse with a clear message rather than
    # crashing. The tool still registers (handler=graph is callable) so the
    # registry surface is stable across envs (D-1, 4-lens re-verify).
    if not _TS_AVAILABLE:
        return ("graph refused — optional code-graph deps not installed "
                f"({_TS_IMPORT_ERROR}). Install `hermes-cli[graph]` (tree-sitter, "
                "tree-sitter-python, networkx) to enable this tool. The tool is "
                "dormant in this env; the rest of the agent is unaffected.")

    # 1. axis-D containment pre-check (the `path` param entry point).
    refusal = _containment_check(path)
    if refusal is not None:
        return refusal

    repo_root = _resolved(path)
    if not os.path.isdir(repo_root):
        return f"graph refused — path is not a directory: {repo_root!r}"

    action = (action or "query").strip().lower()
    if action not in ("query", "rebuild", "explain"):
        return (f"graph refused — action must be query|rebuild|explain, got "
                f"{action!r}.")

    # 2. load-or-build (lazy + cache; rebuild only on stale or action=rebuild).
    force = (action == "rebuild")
    try:
        g = _load_or_build(repo_root, force=force)
    except _ParseTooLarge:
        return (f"graph refused — repo too large for an inline rebuild "
                f"({MAX_PARSE_SECONDS:.0f}s ceiling). Operator pre-warm via CLI: "
                f"run `python -c \"import hermes_cli.agents.echo.tools.graph_tools "
                f"as g; g._load_or_build({repo_root!r}, force=True)\"` or reduce "
                f"scope. The mtime cache will serve subsequent queries.")

    if action == "rebuild":
        return (f"graph rebuilt: {g.number_of_nodes()} nodes, "
                f"{g.number_of_edges()} edges for {repo_root!r}. "
                f"Run a query next.")

    # 3. query (action == query OR explain — explain is a query DSL form too).
    return _run_query(g, query, depth)