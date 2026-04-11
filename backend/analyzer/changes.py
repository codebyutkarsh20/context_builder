"""Change impact analysis — maps git diffs to affected graph nodes and flows.

Works on the in-memory ``graph_data`` dict from ``graph.json`` and the
``flows_data`` dict from ``flows.json``.  No external DB required.

Ported from *code-review-graph* and adapted to our schema.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .flows import SECURITY_KEYWORDS, _build_indices

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = int(os.environ.get("CHANGE_GIT_TIMEOUT", "30"))
_SAFE_GIT_REF = re.compile(r"^[A-Za-z0-9_.~^/@{}\-]+$")


# ---------------------------------------------------------------------------
# 1. Parse git diff into file → line-range pairs
# ---------------------------------------------------------------------------


def parse_git_diff(
    repo_path: str | Path,
    base_ref: str = "HEAD~1",
) -> dict[str, list[tuple[int, int]]]:
    """Run ``git diff --unified=0`` and extract changed line ranges per file.

    Returns a mapping of relative file paths to ``(start, end)`` tuples.
    Empty dict on error.
    """
    if not _SAFE_GIT_REF.match(base_ref):
        logger.warning("Invalid git ref rejected: %s", base_ref)
        return {}
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", base_ref, "--"],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=_GIT_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning("git diff failed (rc=%d): %s", result.returncode, result.stderr[:200])
            return {}
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git diff error: %s", exc)
        return {}

    return _parse_unified_diff(result.stdout)


def _parse_unified_diff(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse unified diff into file → line-range mappings."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    file_pattern = re.compile(r"^\+\+\+ b/(.+)$")
    hunk_pattern = re.compile(r"^@@ .+? \+(\d+)(?:,(\d+))? @@")

    for line in diff_text.splitlines():
        file_match = file_pattern.match(line)
        if file_match:
            current_file = file_match.group(1)
            continue

        hunk_match = hunk_pattern.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
            end = start if count == 0 else start + count - 1
            ranges.setdefault(current_file, []).append((start, end))

    return ranges


# ---------------------------------------------------------------------------
# 2. Map changed line ranges to graph nodes
# ---------------------------------------------------------------------------


def map_changes_to_nodes(
    graph_data: dict,
    changed_ranges: dict[str, list[tuple[int, int]]],
) -> list[dict]:
    """Find graph nodes whose line ranges overlap the changed lines.

    Returns deduplicated list of overlapping node dicts.
    """
    # Index nodes by file path for fast lookup.
    nodes_by_file: dict[str, list[dict]] = defaultdict(list)
    for node in graph_data.get("nodes", []):
        f = node.get("file", "")
        if f:
            nodes_by_file[f].append(node)

    seen: set[str] = set()
    result: list[dict] = []

    for file_path, ranges in changed_ranges.items():
        # Try exact match first, then suffix match.
        candidates = nodes_by_file.get(file_path, [])
        if not candidates:
            for stored_path, nodes in nodes_by_file.items():
                if stored_path.endswith(file_path) or file_path.endswith(stored_path):
                    candidates.extend(nodes)

        for node in candidates:
            if node["id"] in seen:
                continue
            ls = node.get("line_start", 0)
            le = node.get("line_end", 0)
            if not ls or not le:
                continue
            for start, end in ranges:
                if ls <= end and le >= start:
                    result.append(node)
                    seen.add(node["id"])
                    break

    return result


# ---------------------------------------------------------------------------
# 3. Risk scoring per changed node
# ---------------------------------------------------------------------------


def compute_change_risk(
    node: dict,
    nodes_by_id: dict[str, dict],
    calls_forward: dict[str, list[str]],
    called_ids: set[str],
    tested_by_targets: dict[str, list[str]],
    flows: list[dict] | None = None,
    communities: list[dict] | None = None,
    calls_reverse: dict[str, list[str]] | None = None,
) -> float:
    """Compute a risk score (0.0 – 1.0) for a single changed node.

    Weights:
      - Flow participation:   0.25  (how many flows include this node)
      - Test coverage gap:    0.20  (no tests = high risk)
      - Caller count:         0.20  (many callers = wider blast radius)
      - Security sensitivity: 0.15  (security keyword match)
      - Community crossing:   0.20  (callers from different communities)
    """
    nid = node["id"]
    score = 0.0

    # --- Flow participation (cap 0.25) ---
    if flows:
        flow_count = sum(1 for f in flows if nid in f.get("path", []))
        score += min(flow_count * 0.05, 0.25)

    # --- Test coverage gap ---
    has_test = bool(tested_by_targets.get(nid))
    score += 0.05 if has_test else 0.20

    # --- Caller count (cap 0.20) ---
    callers = calls_reverse.get(nid, []) if calls_reverse else []
    caller_count = len(callers)
    score += min(caller_count / 10.0, 0.20)

    # --- Security sensitivity ---
    name_lower = node.get("label", "").lower()
    id_lower = nid.lower()
    if any(kw in name_lower or kw in id_lower for kw in SECURITY_KEYWORDS):
        score += 0.15

    # --- Community crossing (cap 0.20) ---
    if communities and callers:
        node_community = _find_community(nid, communities)
        cross = 0
        for src in callers:
            src_community = _find_community(src, communities)
            if src_community and node_community and src_community != node_community:
                cross += 1
        score += min(cross * 0.05, 0.20)

    return round(min(max(score, 0.0), 1.0), 4)


def _find_community(node_id: str, communities: list[dict]) -> str | None:
    """Find which community a node belongs to."""
    for comm in communities:
        if node_id in comm.get("members", []):
            return comm.get("name", "")
    return None


# ---------------------------------------------------------------------------
# 4. Affected flows detection
# ---------------------------------------------------------------------------


def find_affected_flows(
    changed_node_ids: set[str],
    flows: list[dict],
) -> list[dict]:
    """Find flows that include any of the changed nodes."""
    affected: list[dict] = []
    for flow in flows:
        path_set = set(flow.get("path", []))
        if path_set & changed_node_ids:
            affected.append(flow)
    affected.sort(key=lambda f: f.get("criticality", 0), reverse=True)
    return affected


# ---------------------------------------------------------------------------
# 5. Top-level change detection
# ---------------------------------------------------------------------------


def detect_changes(
    repo_path: str | Path,
    graph_data: dict,
    flows_data: dict | None = None,
    communities: list[dict] | None = None,
    base_ref: str = "HEAD~1",
) -> dict[str, Any]:
    """Full change impact analysis: diff → parse → map → score.

    Parameters
    ----------
    repo_path:
        Path to the git repo root.
    graph_data:
        The dict from ``graph.json``.
    flows_data:
        The dict from ``flows.json`` (optional; improves risk scoring).
    communities:
        Leiden communities list (optional; enables community-crossing risk).
    base_ref:
        Git ref to diff against.

    Returns
    -------
    dict with keys:
      - ``changed_files``: list of changed file paths
      - ``changed_nodes``: list of dicts with risk scores
      - ``affected_flows``: list of affected flow dicts
      - ``test_gaps``: list of untested changed nodes
      - ``risk_summary``: overall risk score and summary text
    """
    # Parse diff.
    diff_ranges = parse_git_diff(repo_path, base_ref)
    changed_files = list(diff_ranges.keys())

    # Map to nodes.
    if diff_ranges:
        changed_nodes = map_changes_to_nodes(graph_data, diff_ranges)
    else:
        # Fallback: all nodes in changed files.
        changed_nodes = [
            n for n in graph_data.get("nodes", [])
            if n.get("file", "") in changed_files
        ]

    # Filter to functions/methods for risk scoring.
    changed_funcs = [
        n for n in changed_nodes
        if n.get("type", "") in ("function", "method", "class")
    ]

    # Build indices once.
    nodes_by_id, calls_forward, called_ids, tested_by_targets, calls_reverse = _build_indices(
        graph_data
    )
    flows = (flows_data or {}).get("flows", [])

    # Score each changed function.
    scored_nodes: list[dict[str, Any]] = []
    for node in changed_funcs:
        risk = compute_change_risk(
            node, nodes_by_id, calls_forward, called_ids,
            tested_by_targets, flows, communities, calls_reverse,
        )
        scored_nodes.append({
            "id": node["id"],
            "label": node.get("label", node["id"]),
            "file": node.get("file", ""),
            "type": node.get("type", ""),
            "risk_score": risk,
        })

    # Overall risk: max of individual scores.
    overall_risk = max((n["risk_score"] for n in scored_nodes), default=0.0)

    # Affected flows.
    changed_node_ids = {n["id"] for n in changed_nodes}
    affected_flows = find_affected_flows(changed_node_ids, flows)

    # Test gaps: changed functions without TESTED_BY edges.
    test_gaps: list[dict] = []
    for node in changed_funcs:
        if node.get("is_test"):
            continue
        if not tested_by_targets.get(node["id"]):
            test_gaps.append({
                "id": node["id"],
                "label": node.get("label", node["id"]),
                "file": node.get("file", ""),
            })

    # Summary.
    summary_parts = [
        f"Analyzed {len(changed_files)} changed file(s):",
        f"  {len(changed_funcs)} changed function(s)",
        f"  {len(affected_flows)} affected flow(s)",
        f"  {len(test_gaps)} test gap(s)",
        f"  Overall risk: {overall_risk:.2f}",
    ]

    return {
        "changed_files": changed_files,
        "changed_nodes": scored_nodes,
        "affected_flows": affected_flows,
        "test_gaps": test_gaps,
        "risk_summary": {
            "overall_risk": overall_risk,
            "summary": "\n".join(summary_parts),
        },
    }
