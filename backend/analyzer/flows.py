"""Execution flow detection, tracing, criticality scoring, and dead-code analysis.

Works on the in-memory ``graph_data`` dict produced by ``call_graph.py``
(keys: ``nodes``, ``edges``, ``hotspots``).  No external DB required.

Ported from *code-review-graph* and adapted to our graph.json schema.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECURITY_KEYWORDS: frozenset[str] = frozenset({
    "auth", "login", "password", "token", "session", "crypt", "secret",
    "credential", "permission", "sql", "query", "execute", "connect",
    "socket", "request", "http", "sanitize", "validate", "encrypt",
    "decrypt", "hash", "sign", "verify", "admin", "privilege",
})

# Decorator patterns that indicate a function is a framework entry point.
_FRAMEWORK_DECORATOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"app\.(get|post|put|delete|patch|route|websocket)", re.I),
    re.compile(r"router\.(get|post|put|delete|patch|route)", re.I),
    re.compile(r"blueprint\.(route|before_request|after_request)", re.I),
    re.compile(r"click\.(command|group)", re.I),
    re.compile(r"celery\.(task|shared_task)", re.I),
    re.compile(r"api_view", re.I),
    re.compile(r"@(Get|Post|Put|Delete|Patch|RequestMapping)", re.I),
]

# Name patterns for conventional entry points.
_ENTRY_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^main$"),
    re.compile(r"^__main__$"),
    re.compile(r"^test_"),
    re.compile(r"^Test[A-Z]"),
    re.compile(r"^on_"),
    re.compile(r"^handle_"),
    re.compile(r"^process_"),
]


# ---------------------------------------------------------------------------
# Internal index helpers
# ---------------------------------------------------------------------------


def _build_indices(graph_data: dict) -> tuple[
    dict[str, dict],           # nodes_by_id
    dict[str, list[str]],      # calls_forward: source_id -> [target_id, ...]
    set[str],                  # called_ids: all targets of CALLS edges
    dict[str, list[str]],      # tested_by_targets: node_id -> [test_node_id, ...]
    dict[str, list[str]],      # calls_reverse: target_id -> [source_id, ...]
]:
    """Build lookup indices from graph_data for efficient traversal."""
    nodes_by_id: dict[str, dict] = {}
    for node in graph_data.get("nodes", []):
        nodes_by_id[node["id"]] = node

    calls_forward: dict[str, list[str]] = defaultdict(list)
    calls_reverse: dict[str, list[str]] = defaultdict(list)
    called_ids: set[str] = set()
    tested_by_targets: dict[str, list[str]] = defaultdict(list)

    for edge in graph_data.get("edges", []):
        etype = edge.get("type", "")
        if etype == "CALLS":
            calls_forward[edge["source"]].append(edge["target"])
            calls_reverse[edge["target"]].append(edge["source"])
            called_ids.add(edge["target"])
        elif etype == "TESTED_BY":
            tested_by_targets[edge["target"]].append(edge["source"])

    return nodes_by_id, calls_forward, called_ids, tested_by_targets, calls_reverse


# ---------------------------------------------------------------------------
# Entry-point detection
# ---------------------------------------------------------------------------


def _has_framework_decorator(node: dict) -> bool:
    """Return True if *node* has a decorator matching a framework pattern."""
    decorators = node.get("decorators")
    if not decorators:
        return False
    if isinstance(decorators, str):
        decorators = [decorators]
    for dec in decorators:
        for pat in _FRAMEWORK_DECORATOR_PATTERNS:
            if pat.search(dec):
                return True
    return False


def _matches_entry_name(node: dict) -> bool:
    """Return True if *node*'s label matches a conventional entry-point pattern."""
    name = node.get("label", "")
    # For methods like "Class::method", check the method part.
    if "::" in name:
        name = name.rsplit("::", 1)[-1]
    for pat in _ENTRY_NAME_PATTERNS:
        if pat.search(name):
            return True
    return False


def detect_entry_points(
    graph_data: dict,
    called_ids: set[str] | None = None,
) -> list[dict]:
    """Find functions/methods that are entry points in the graph.

    An entry point is a function/method node that:
    1. Has a framework decorator (``@app.get``, ``@router.post``, etc.), OR
    2. Matches a conventional name pattern (``main``, ``test_*``, etc.), OR
    3. Has no incoming CALLS edges (true root).

    Parameters
    ----------
    graph_data:
        The graph dict.
    called_ids:
        Pre-built set of node IDs that are targets of CALLS edges.
        If ``None``, built internally via ``_build_indices``.
    """
    if called_ids is None:
        _, _, called_ids, _, _ = _build_indices(graph_data)

    entry_points: list[dict] = []
    for node in graph_data.get("nodes", []):
        ntype = node.get("type", "")
        if ntype not in ("function", "method"):
            continue

        is_entry = False

        # Framework decorator match (highest signal).
        if _has_framework_decorator(node):
            is_entry = True

        # Conventional name match.
        if _matches_entry_name(node):
            is_entry = True

        # True root: nobody calls this function.
        if node["id"] not in called_ids:
            is_entry = True

        if is_entry:
            entry_points.append(node)

    return entry_points


# ---------------------------------------------------------------------------
# Flow tracing (BFS through CALLS edges)
# ---------------------------------------------------------------------------


def _trace_single_flow(
    node: dict,
    nodes_by_id: dict[str, dict],
    calls_forward: dict[str, list[str]],
    max_depth: int = 15,
) -> Optional[dict]:
    """Trace a single execution flow from *node* via forward BFS.

    Returns a flow dict or ``None`` if the flow is trivial (single-node).
    """
    path_ids: list[str] = []
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    queue.append((node["id"], 0))
    visited.add(node["id"])
    path_ids.append(node["id"])

    actual_depth = 0

    while queue:
        current_id, depth = queue.popleft()
        if depth > actual_depth:
            actual_depth = depth
        if depth >= max_depth:
            continue

        for target_id in calls_forward.get(current_id, []):
            if target_id in visited:
                continue
            if target_id not in nodes_by_id:
                continue
            visited.add(target_id)
            path_ids.append(target_id)
            queue.append((target_id, depth + 1))

    # Skip trivial single-node flows.
    if len(path_ids) < 2:
        return None

    files = list({
        nodes_by_id[nid].get("file", "")
        for nid in path_ids
        if nid in nodes_by_id
    })
    files = [f for f in files if f]  # drop empty

    label = node.get("label", node["id"])
    if "::" in label:
        label = label.rsplit("::", 1)[-1]

    return {
        "name": label,
        "entry_point": node["id"],
        "entry_point_file": node.get("file", ""),
        "path": path_ids,
        "depth": actual_depth,
        "node_count": len(path_ids),
        "file_count": len(files),
        "files": files,
        "criticality": 0.0,  # computed below
    }


def trace_flows(
    graph_data: dict,
    max_depth: int = 15,
    *,
    _indices: tuple | None = None,
) -> list[dict]:
    """Trace execution flows from every entry point via forward BFS.

    Returns a list of flow dicts sorted by criticality descending.
    Pre-built indices can be passed via ``_indices`` to avoid redundant work.
    """
    if _indices is not None:
        nodes_by_id, calls_forward, called_ids, tested_by_targets, _ = _indices
    else:
        nodes_by_id, calls_forward, called_ids, tested_by_targets, _ = _build_indices(
            graph_data
        )
    entry_points = detect_entry_points(graph_data, called_ids=called_ids)

    flows: list[dict] = []
    for ep in entry_points:
        flow = _trace_single_flow(ep, nodes_by_id, calls_forward, max_depth)
        if flow is not None:
            flow["criticality"] = compute_criticality(
                flow, nodes_by_id, calls_forward, tested_by_targets
            )
            flows.append(flow)

    flows.sort(key=lambda f: f["criticality"], reverse=True)
    return flows


# ---------------------------------------------------------------------------
# Criticality scoring
# ---------------------------------------------------------------------------


def compute_criticality(
    flow: dict,
    nodes_by_id: dict[str, dict],
    calls_forward: dict[str, list[str]],
    tested_by_targets: dict[str, list[str]],
) -> float:
    """Score a flow from 0.0 to 1.0 based on multiple weighted factors.

    Weights:
      - File spread:          0.30
      - External calls:       0.20
      - Security sensitivity: 0.25
      - Test coverage gap:    0.15
      - Depth:                0.10
    """
    path_ids: list[str] = flow.get("path", [])
    if not path_ids:
        return 0.0

    nodes = [nodes_by_id[nid] for nid in path_ids if nid in nodes_by_id]
    if not nodes:
        return 0.0

    # --- File spread (0.0 - 1.0) ---
    file_count = len({n.get("file", "") for n in nodes} - {""})
    file_spread = min((file_count - 1) / 4.0, 1.0) if file_count > 1 else 0.0

    # --- External calls (0.0 - 1.0) ---
    # Targets of CALLS that don't resolve to known nodes.
    external_count = 0
    for nid in path_ids:
        for target in calls_forward.get(nid, []):
            if target not in nodes_by_id:
                external_count += 1
    external_score = min(external_count / 5.0, 1.0)

    # --- Security sensitivity (0.0 - 1.0) ---
    security_hits = 0
    for n in nodes:
        name_lower = n.get("label", "").lower()
        id_lower = n["id"].lower()
        for kw in SECURITY_KEYWORDS:
            if kw in name_lower or kw in id_lower:
                security_hits += 1
                break
    security_score = min(security_hits / max(len(nodes), 1), 1.0)

    # --- Test coverage gap (0.0 - 1.0) ---
    tested_count = sum(1 for nid in path_ids if tested_by_targets.get(nid))
    coverage = tested_count / max(len(nodes), 1)
    test_gap = 1.0 - coverage

    # --- Depth (0.0 - 1.0) ---
    depth_score = min(flow.get("depth", 0) / 10.0, 1.0)

    # --- Weighted sum ---
    criticality = (
        file_spread * 0.30
        + external_score * 0.20
        + security_score * 0.25
        + test_gap * 0.15
        + depth_score * 0.10
    )
    return round(min(max(criticality, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Dead-code detection
# ---------------------------------------------------------------------------


def find_dead_code(
    graph_data: dict,
    *,
    _indices: tuple | None = None,
) -> list[dict]:
    """Find unreachable functions: no callers, no tests, not entry points, not dunder.

    Returns a list of dicts with ``id``, ``label``, ``file``, ``type`` keys.
    Pre-built indices can be passed via ``_indices`` to avoid redundant work.
    """
    if _indices is not None:
        nodes_by_id, calls_forward, called_ids, tested_by_targets, _ = _indices
    else:
        nodes_by_id, calls_forward, called_ids, tested_by_targets, _ = _build_indices(
            graph_data
        )
    entry_point_ids = {ep["id"] for ep in detect_entry_points(graph_data, called_ids=called_ids)}

    dead: list[dict] = []
    for node in graph_data.get("nodes", []):
        ntype = node.get("type", "")
        if ntype not in ("function", "method"):
            continue

        nid = node["id"]
        name = node.get("label", "")
        if "::" in name:
            name = name.rsplit("::", 1)[-1]

        # Skip dunder methods.
        if name.startswith("__") and name.endswith("__"):
            continue

        # Skip test functions (they are callers, not callees).
        if node.get("is_test"):
            continue

        # Has callers?
        if nid in called_ids:
            continue

        # Has tests?
        if tested_by_targets.get(nid):
            continue

        # Is entry point?
        if nid in entry_point_ids:
            continue

        dead.append({
            "id": nid,
            "label": node.get("label", nid),
            "file": node.get("file", ""),
            "type": ntype,
        })

    return dead


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_flows(
    graph_data: dict,
    communities: list[dict] | None = None,
    max_depth: int = 15,
) -> dict:
    """Build execution flows, dead code list, and summary stats.

    Parameters
    ----------
    graph_data:
        The dict from ``graph.json`` (keys: ``nodes``, ``edges``).
    communities:
        Optional Leiden communities list (unused for now, reserved for
        community-crossing risk scoring in Phase 2).
    max_depth:
        Maximum BFS depth for flow tracing.

    Returns
    -------
    dict with keys:
      - ``flows``: list of flow dicts sorted by criticality
      - ``dead_code``: list of dead-code node dicts
      - ``entry_point_count``: total entry points detected
      - ``flow_count``: number of non-trivial flows
      - ``dead_code_count``: number of dead-code nodes
    """
    indices = _build_indices(graph_data)
    _, _, called_ids, _, _ = indices
    entry_points = detect_entry_points(graph_data, called_ids=called_ids)
    flows = trace_flows(graph_data, max_depth=max_depth, _indices=indices)
    dead_code = find_dead_code(graph_data, _indices=indices)

    logger.info(
        "Flows: %d entry points, %d flows traced, %d dead-code nodes",
        len(entry_points),
        len(flows),
        len(dead_code),
    )

    return {
        "flows": flows,
        "dead_code": dead_code,
        "entry_point_count": len(entry_points),
        "flow_count": len(flows),
        "dead_code_count": len(dead_code),
    }
