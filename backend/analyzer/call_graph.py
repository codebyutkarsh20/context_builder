"""
call_graph.py — Build a directed call/import/containment graph from ParsedFile data.

Graph node IDs follow the convention:
  - file      : "<rel_path>"
  - function  : "<rel_path>::<func_name>"
  - method    : "<rel_path>::<class_name>::<method_name>"
  - class     : "<rel_path>::<class_name>"

Edge types:
  CONTAINS  — structural containment (file→class, file→function, class→method)
  IMPORTS   — file A imports from file B (resolved best-effort)
  CALLS     — function/method body references another known callable
  INHERITS  — class B inherits from class A
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_id(rel_path: str) -> str:
    return rel_path


def _func_id(rel_path: str, func_name: str) -> str:
    return f"{rel_path}::{func_name}"


def _class_id(rel_path: str, class_name: str) -> str:
    return f"{rel_path}::{class_name}"


def _method_id(rel_path: str, class_name: str, method_name: str) -> str:
    return f"{rel_path}::{class_name}::{method_name}"


def _resolve_import_module(
    module: str,
    is_from: bool,
    importer_path: str,
    known_paths: set[str],
) -> Optional[str]:
    """
    Try to map an import module string to a relative file path present in known_paths.

    Handles:
    - Absolute imports: "a.b.c" → "a/b/c.py" or "a/b/c/__init__.py"
    - Relative imports: ".sibling" or "..parent.sibling"
    """
    if not module:
        return None

    # --- relative import (starts with one or more dots) ---
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        remainder = module.lstrip(".")
        # Compute base directory of importer
        importer_dir = str(PurePosixPath(importer_path).parent)
        # Go up 'dots - 1' levels
        base = PurePosixPath(importer_dir)
        for _ in range(dots - 1):
            base = base.parent
        if remainder:
            candidate_base = base / remainder.replace(".", "/")
        else:
            candidate_base = base

        for suffix in (".py", "/__init__.py"):
            candidate = str(candidate_base) + suffix
            # Normalise leading "./"
            candidate = candidate.lstrip("./")
            if candidate in known_paths:
                return candidate
        return None

    # --- absolute import ---
    module_as_path = module.replace(".", "/")
    for suffix in (".py", "/__init__.py"):
        candidate = module_as_path + suffix
        if candidate in known_paths:
            return candidate
    return None


# ---------------------------------------------------------------------------
# CALLS-edge inference: very lightweight — no full dataflow
# ---------------------------------------------------------------------------

# Builtins and very common names to skip for CALLS matching (too noisy)
_SKIP_CALL_NAMES = {
    # Python builtins
    "len", "print", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "type", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "super", "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "repr", "id", "hash", "abs", "max", "min", "sum", "all", "any", "next", "iter",
    "open", "input", "round", "format", "vars", "dir", "callable", "property",
    "staticmethod", "classmethod", "object", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "AttributeError", "RuntimeError", "NotImplementedError",
    "StopIteration", "OSError", "IOError", "FileNotFoundError",
    # Common noise names that create false cross-file matches
    "get", "set", "update", "delete", "create", "save", "add", "remove",
    "execute", "run", "init", "setup", "close", "read", "write",
}


def _scan_body_for_calls(
    source_code: Optional[str],
    func_name_to_ids: dict[str, list[str]],
    caller_file: str = "",
    file_imports: dict[str, str] | None = None,
) -> list[str]:
    """
    Scan raw source text for function call patterns that match known callable names.
    Returns a list of target node IDs.

    Uses receiver-qualified matching to avoid name collisions:
    - `self.method()` → match methods in the same class only
    - `variable.method()` → match methods in the class the variable was imported as
    - `ClassName(...)` → match class constructors
    - `bare_function()` → match functions in the same file first, then others

    The file_imports dict maps alias/variable names to their resolved file path,
    e.g. {"family_service": "app/service/family_service.py"}.
    """
    if not source_code:
        return []
    import re

    file_imports = file_imports or {}

    # Strip comments and string literals to reduce false positives
    cleaned = re.sub(r"#[^\n]*", "", source_code)
    cleaned = re.sub(r'"""[\s\S]*?"""', "", cleaned)
    cleaned = re.sub(r"'''[\s\S]*?'''", "", cleaned)
    cleaned = re.sub(r'"[^"\n]*"', "", cleaned)
    cleaned = re.sub(r"'[^'\n]*'", "", cleaned)

    targets: list[str] = []
    seen: set[str] = set()

    # 1. Match attribute-style calls: receiver.method(...)
    #    This handles self.method(), service.method(), ClassName.method(), etc.
    attr_calls = re.findall(r"\b(\w+)\.(\w+)\s*\(", cleaned)
    for receiver, method_name in attr_calls:
        if method_name in _SKIP_CALL_NAMES:
            continue
        if method_name not in func_name_to_ids:
            continue
        key = f"{receiver}.{method_name}"
        if key in seen:
            continue
        seen.add(key)

        candidates = func_name_to_ids[method_name]

        if receiver in ("self", "cls"):
            # self.method() → prefer methods in the same file
            for cid in candidates:
                if cid.startswith(caller_file + "::"):
                    targets.append(cid)
        elif receiver in file_imports:
            # Known import: e.g. family_service.create_family → match in that file
            target_file = file_imports[receiver]
            for cid in candidates:
                if cid.startswith(target_file + "::"):
                    targets.append(cid)
        else:
            # Unknown receiver — try to match as a class name
            # e.g. FamilyService.create_family or family_crud.create_family
            matched = False
            for cid in candidates:
                # Match "ClassName::method" pattern in node ID
                parts = cid.split("::")
                if len(parts) >= 2:
                    class_or_file = parts[-2] if len(parts) >= 3 else parts[0]
                    # Check if receiver looks like it could be this class
                    # e.g., receiver="family_crud" matches class "FamilyCRUD" (snake → PascalCase)
                    receiver_lower = receiver.lower().replace("_", "")
                    class_lower = class_or_file.lower().replace("_", "").split("/")[-1].split(".")[0]
                    if receiver_lower == class_lower:
                        targets.append(cid)
                        matched = True
            # If no class matched, don't add anything (avoids phantom edges)

    # 2. Match bare function calls: function_name(...)
    #    Only match if there's exactly 1 candidate, or if candidate is in the same file
    bare_calls = re.findall(r"(?<!\.)\b([A-Za-z_]\w*)\s*\(", cleaned)
    for ident in bare_calls:
        if ident in _SKIP_CALL_NAMES:
            continue
        if ident not in func_name_to_ids:
            continue
        if ident in seen:
            continue
        seen.add(ident)

        candidates = func_name_to_ids[ident]

        # Prefer same-file match
        same_file = [c for c in candidates if c.startswith(caller_file + "::")]
        if same_file:
            targets.extend(same_file)
        elif len(candidates) == 1:
            # Unique name across entire codebase — safe to add
            targets.append(candidates[0])
        # If multiple candidates and none in same file, skip (ambiguous)

    return targets


# ---------------------------------------------------------------------------
# CallGraphBuilder
# ---------------------------------------------------------------------------

class CallGraphBuilder:
    """
    Build a call / import / containment graph from a list of ParsedFile dicts
    (as produced by CodeParser.parse_all()).

    Usage::

        from backend.analyzer.code_parser import CodeParser
        from backend.analyzer.call_graph import CallGraphBuilder

        parsed = CodeParser(repo_path).parse_all()
        graph_data = CallGraphBuilder(parsed).build()
    """

    TOP_HOTSPOTS = 20

    def __init__(self, parsed: list[dict]) -> None:
        self._parsed = parsed
        # Map rel_path → ParsedFile for quick lookup
        self._file_map: dict[str, dict] = {pf["path"]: pf for pf in parsed}
        self._known_paths: set[str] = set(self._file_map.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> dict:
        G = nx.DiGraph()

        # Phase 1: add all nodes
        self._add_nodes(G)

        # Phase 2: containment edges
        self._add_contains_edges(G)

        # Phase 3: import edges
        self._add_import_edges(G)

        # Phase 4: inheritance edges
        self._add_inherits_edges(G)

        # Phase 5: call edges (best-effort, name-based, skipped for large repos)
        total_funcs = sum(
            len(pf.get("functions", [])) + sum(len(c.get("methods", [])) for c in pf.get("classes", []))
            for pf in self._parsed
        )
        if total_funcs <= 2000:
            self._add_call_edges(G)
        else:
            logger.info("Skipping CALLS phase (>2000 callables) for performance")

        # Phase 6: external call detection
        self._detect_external_calls(G)

        # Phase 7: PageRank
        try:
            pr: dict[str, float] = nx.pagerank(G, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge; using uniform scores")
            pr = {n: 1.0 / max(len(G), 1) for n in G.nodes}

        # Attach scores to node attributes
        for node_id, score in pr.items():
            G.nodes[node_id]["pagerank"] = score

        nodes = self._serialize_nodes(G, pr)
        edges = self._serialize_edges(G)
        hotspots = self._compute_hotspots(nodes)

        return {
            "nodes": nodes,
            "edges": edges,
            "hotspots": hotspots,
        }

    # ------------------------------------------------------------------
    # Phase 1: nodes
    # ------------------------------------------------------------------

    def _add_nodes(self, G: nx.DiGraph) -> None:
        for pf in self._parsed:
            rel = pf["path"]

            # File node
            G.add_node(
                _file_id(rel),
                label=rel,
                type="file",
                file=rel,
            )

            # Class nodes
            for cls in pf.get("classes", []):
                cid = _class_id(rel, cls["name"])
                G.add_node(cid, label=cls["name"], type="class", file=rel,
                           line_start=cls.get("line_start", 0), line_end=cls.get("line_end", 0))

                # Method nodes
                for method in cls.get("methods", []):
                    mid = _method_id(rel, cls["name"], method["name"])
                    G.add_node(mid, label=method["name"], type="function", file=rel,
                               line_start=method.get("line_start", 0), line_end=method.get("line_end", 0))

            # Top-level function nodes
            for fn in pf.get("functions", []):
                fid = _func_id(rel, fn["name"])
                G.add_node(fid, label=fn["name"], type="function", file=rel,
                           line_start=fn.get("line_start", 0), line_end=fn.get("line_end", 0))

    # ------------------------------------------------------------------
    # Phase 2: CONTAINS edges
    # ------------------------------------------------------------------

    def _add_contains_edges(self, G: nx.DiGraph) -> None:
        for pf in self._parsed:
            rel = pf["path"]
            fnode = _file_id(rel)

            for cls in pf.get("classes", []):
                cid = _class_id(rel, cls["name"])
                G.add_edge(fnode, cid, type="CONTAINS")
                for method in cls.get("methods", []):
                    mid = _method_id(rel, cls["name"], method["name"])
                    G.add_edge(cid, mid, type="CONTAINS")

            for fn in pf.get("functions", []):
                fid = _func_id(rel, fn["name"])
                G.add_edge(fnode, fid, type="CONTAINS")

    # ------------------------------------------------------------------
    # Phase 3: IMPORTS edges
    # ------------------------------------------------------------------

    def _add_import_edges(self, G: nx.DiGraph) -> None:
        for pf in self._parsed:
            rel = pf["path"]
            src_fnode = _file_id(rel)

            for imp in pf.get("imports", []):
                target_path = _resolve_import_module(
                    imp.get("module", ""),
                    imp.get("is_from", False),
                    rel,
                    self._known_paths,
                )
                if target_path and target_path != rel:
                    tgt_fnode = _file_id(target_path)
                    if G.has_node(tgt_fnode):
                        G.add_edge(src_fnode, tgt_fnode, type="IMPORTS")

    # ------------------------------------------------------------------
    # Phase 4: INHERITS edges
    # ------------------------------------------------------------------

    def _add_inherits_edges(self, G: nx.DiGraph) -> None:
        # Build a lookup: class name → list of class node IDs (could be in many files)
        class_name_to_ids: dict[str, list[str]] = {}
        for pf in self._parsed:
            rel = pf["path"]
            for cls in pf.get("classes", []):
                cid = _class_id(rel, cls["name"])
                class_name_to_ids.setdefault(cls["name"], []).append(cid)

        for pf in self._parsed:
            rel = pf["path"]
            for cls in pf.get("classes", []):
                child_cid = _class_id(rel, cls["name"])
                for base in cls.get("bases", []):
                    # base might be "Base", "module.Base", etc. — use last segment
                    base_name = base.split(".")[-1]
                    for parent_cid in class_name_to_ids.get(base_name, []):
                        if parent_cid != child_cid:
                            G.add_edge(child_cid, parent_cid, type="INHERITS")

    # ------------------------------------------------------------------
    # Phase 5: CALLS edges
    # ------------------------------------------------------------------

    def _add_call_edges(self, G: nx.DiGraph) -> None:
        """
        Best-effort: re-parse function body source text to find identifier
        references to known callables.

        Uses receiver-qualified matching to avoid phantom edges from name collisions.
        E.g., `service.create_family()` only links to the service's method, not
        to every class that has a `create_family` method.
        """
        # Build callable name → node IDs index (functions + methods)
        func_name_to_ids: dict[str, list[str]] = {}
        for pf in self._parsed:
            rel = pf["path"]
            for fn in pf.get("functions", []):
                fid = _func_id(rel, fn["name"])
                func_name_to_ids.setdefault(fn["name"], []).append(fid)
            for cls in pf.get("classes", []):
                for method in cls.get("methods", []):
                    mid = _method_id(rel, cls["name"], method["name"])
                    func_name_to_ids.setdefault(method["name"], []).append(mid)

        # For each file, load lines, build import map, then scan each function/method body
        for pf in self._parsed:
            rel = pf["path"]
            abs_path = pf.get("abs_path", "")
            try:
                lines = _read_lines(abs_path)
            except OSError:
                continue

            # Build per-file import alias → resolved file path mapping
            # e.g., "from app.service.family_service import FamilyService"
            #   → {"FamilyService": "app/service/family_service.py"}
            # or "family_service = FamilyService()" style instantiation
            file_imports: dict[str, str] = {}
            for imp in pf.get("imports", []):
                mod = imp.get("module", "")
                resolved = _resolve_import_module(mod, imp.get("is_from", False), rel, self._known_paths)
                if resolved:
                    # For "from X import Y", the imported name is the alias
                    alias = imp.get("alias")
                    names = imp.get("names", [])
                    if alias:
                        file_imports[alias] = resolved
                    elif names:
                        for n in names:
                            name = n.get("name", "") if isinstance(n, dict) else str(n)
                            if name:
                                file_imports[name] = resolved
                    elif not imp.get("is_from") and mod:
                        short = mod.split(".")[-1]
                        file_imports[short] = resolved

            # Top-level functions
            for fn in pf.get("functions", []):
                if "line_start" not in fn or "line_end" not in fn:
                    continue
                caller_id = _func_id(rel, fn["name"])
                body_text = _slice_lines(lines, fn["line_start"], fn["line_end"])
                for target_id in _scan_body_for_calls(body_text, func_name_to_ids, rel, file_imports):
                    if target_id != caller_id and not G.has_edge(caller_id, target_id):
                        G.add_edge(caller_id, target_id, type="CALLS")

            # Methods
            for cls in pf.get("classes", []):
                for method in cls.get("methods", []):
                    if "line_start" not in method or "line_end" not in method:
                        continue
                    caller_id = _method_id(rel, cls["name"], method["name"])
                    body_text = _slice_lines(lines, method["line_start"], method["line_end"])
                    for target_id in _scan_body_for_calls(body_text, func_name_to_ids, rel, file_imports):
                        if target_id != caller_id and not G.has_edge(caller_id, target_id):
                            G.add_edge(caller_id, target_id, type="CALLS")

    # ------------------------------------------------------------------
    # Phase 6: External call detection
    # ------------------------------------------------------------------

    _ATTR_CALL_RE = __import__("re").compile(r"\b(\w+(?:\.\w+)+)\s*\(")

    def _detect_external_calls(self, G: nx.DiGraph) -> None:
        """Annotate function/method nodes with external_calls (calls to non-repo libraries)."""
        # Build set of all known internal callable names
        internal_names: set[str] = set()
        for pf in self._parsed:
            for fn in pf.get("functions", []):
                internal_names.add(fn["name"])
            for cls in pf.get("classes", []):
                internal_names.add(cls["name"])
                for method in cls.get("methods", []):
                    internal_names.add(method["name"])

        # Build import alias → module mapping
        import_aliases: dict[str, str] = {}  # per-file: {alias: module}
        for pf in self._parsed:
            for imp in pf.get("imports", []):
                mod = imp.get("module", "")
                alias = imp.get("alias")
                if alias:
                    import_aliases[alias] = mod
                elif not imp.get("is_from") and mod:
                    # import foo → foo is available
                    short = mod.split(".")[-1]
                    import_aliases[short] = mod

        for pf in self._parsed:
            rel = pf["path"]
            abs_path = pf.get("abs_path", "")
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except (OSError, FileNotFoundError):
                continue

            all_funcs = [
                (_func_id(rel, fn["name"]), fn)
                for fn in pf.get("functions", [])
            ]
            for cls in pf.get("classes", []):
                for method in cls.get("methods", []):
                    all_funcs.append(
                        (_method_id(rel, cls["name"], method["name"]), method)
                    )

            for node_id, fn in all_funcs:
                if "line_start" not in fn or "line_end" not in fn:
                    continue
                body = "".join(lines[max(0, fn["line_start"] - 1):fn["line_end"]])
                ext_calls: set[str] = set()

                # Find attribute-style calls: module.func()
                for match in self._ATTR_CALL_RE.finditer(body):
                    dotted = match.group(1)
                    parts = dotted.split(".")
                    base = parts[0]
                    # If base is an import alias or known stdlib/library, it's external
                    if base in import_aliases or (base not in internal_names and base not in ("self", "cls", "super")):
                        ext_calls.add(dotted)

                if ext_calls and node_id in G:
                    G.nodes[node_id]["external_calls"] = sorted(ext_calls)[:20]  # cap

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_nodes(G: nx.DiGraph, pr: dict[str, float]) -> list[dict]:
        nodes: list[dict] = []
        for node_id, attrs in G.nodes(data=True):
            node = {
                "id": node_id,
                "label": attrs.get("label", node_id),
                "type": attrs.get("type", "file"),
                "file": attrs.get("file", ""),
                "pagerank": pr.get(node_id, 0.0),
            }
            # Include line numbers for functions/classes (enables precise localization)
            if attrs.get("line_start"):
                node["line_start"] = attrs["line_start"]
            if attrs.get("line_end"):
                node["line_end"] = attrs["line_end"]
            if attrs.get("external_calls"):
                node["external_calls"] = attrs["external_calls"]
            if attrs.get("reads_from"):
                node["reads_from"] = attrs["reads_from"]
            if attrs.get("writes_to"):
                node["writes_to"] = attrs["writes_to"]
            nodes.append(node)
        return nodes

    @staticmethod
    def _serialize_edges(G: nx.DiGraph) -> list[dict]:
        edges: list[dict] = []
        for src, dst, attrs in G.edges(data=True):
            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "type": attrs.get("type", "CONTAINS"),
                }
            )
        return edges

    def _compute_hotspots(self, nodes: list[dict]) -> list[dict]:
        sorted_nodes = sorted(nodes, key=lambda n: n["pagerank"], reverse=True)
        return [
            {
                "id": n["id"],
                "label": n["label"],
                "pagerank": n["pagerank"],
                "type": n["type"],
            }
            for n in sorted_nodes[: self.TOP_HOTSPOTS]
        ]


# ---------------------------------------------------------------------------
# File I/O utilities
# ---------------------------------------------------------------------------

from functools import lru_cache

@lru_cache(maxsize=512)
def _read_lines_cached(abs_path: str) -> tuple[str, ...]:
    with open(abs_path, encoding="utf-8", errors="replace") as fh:
        return tuple(fh.readlines())


def _read_lines(abs_path: str) -> list[str]:
    return list(_read_lines_cached(abs_path))


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    """Return lines [start, end] (1-indexed, inclusive) joined as a string."""
    # Clamp to available lines
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "".join(lines[s:e])
