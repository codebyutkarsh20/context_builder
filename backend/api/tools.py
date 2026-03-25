"""
tools.py — MCP tool endpoints for the AI Deploy Agent.

Three tool endpoints that agents can call:
1. get_function_context — full context for a function (description, callers, callees, rules)
2. get_blast_radius — downstream impact analysis for modified files
3. search_by_concept — semantic search via ChromaDB embeddings
"""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.utils import validate_repo_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


# ---------------------------------------------------------------------------
# Shared data loaders
# ---------------------------------------------------------------------------

def _load_graph(repo: str) -> dict:
    p = _DATA_DIR / repo / "graph.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _load_enriched(repo: str) -> dict:
    p = _DATA_DIR / repo / "enriched_nodes.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _load_rules(repo: str) -> list[dict]:
    p = _DATA_DIR / repo / "business_rules.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def _node_short_name(node_id: str) -> str:
    return node_id.split("::")[-1] if "::" in node_id else node_id


def _node_file(node_id: str) -> str:
    return node_id.split("::")[0] if "::" in node_id else node_id


# ---------------------------------------------------------------------------
# 1. get_function_context
# ---------------------------------------------------------------------------

@router.get("/function-context/{function_id:path}")
def get_function_context(
    function_id: str,
    repo: str = Query(..., description="Repository name"),
):
    """Return full context for a function: description, callers, callees, rules, decisions."""
    t0 = time.monotonic()
    repo = validate_repo_name(repo)

    graph = _load_graph(repo)
    enriched = _load_enriched(repo)

    if not graph and not enriched:
        raise HTTPException(status_code=404, detail=f"Repository '{repo}' not found or not indexed")

    # Find the function in enriched nodes (exact match)
    node = enriched.get(function_id)
    graph_node = None
    for n in graph.get("nodes", []):
        if n.get("id") == function_id:
            graph_node = n
            break

    if not node and not graph_node:
        raise HTTPException(status_code=404, detail=f"Function '{function_id}' not found")

    # Merge data from both sources
    file_path = (node or {}).get("file") or (graph_node or {}).get("file") or _node_file(function_id)
    description = ""
    if node:
        description = node.get("llm_summary") or node.get("summary") or node.get("docstring") or ""
    if not description and graph_node:
        description = graph_node.get("summary") or graph_node.get("docstring") or ""

    params = (node or {}).get("params", [])
    if isinstance(params, list):
        params = [p if isinstance(p, str) else p.get("name", str(p)) for p in params]

    # Callers and callees from edges
    edges = graph.get("edges", [])
    callers = []
    callees = []
    for e in edges:
        if e.get("type") != "CALLS":
            continue
        if e.get("target") == function_id:
            src = e["source"]
            callers.append({"id": src, "name": _node_short_name(src), "file": _node_file(src)})
        elif e.get("source") == function_id:
            tgt = e["target"]
            callees.append({"id": tgt, "name": _node_short_name(tgt), "file": _node_file(tgt)})

    # Business rules linked to this function or its file
    rules = _load_rules(repo)
    matched_rules = []
    for r in rules:
        r_func = r.get("function_id", "")
        r_file = r.get("file", "")
        if (r_func and r_func == function_id) or (r_file and r_file == file_path and not r_func):
            matched_rules.append({
                "id": r.get("id", ""),
                "description": r.get("description", ""),
                "severity": r.get("severity", "medium"),
                "rule_type": r.get("rule_type", "policy"),
            })

    # Decision points inside this function
    dps = graph.get("decision_points", [])
    matched_dps = []
    for dp in dps:
        if dp.get("function_id") == function_id:
            matched_dps.append({
                "condition": dp.get("condition", ""),
                "condition_type": dp.get("condition_type", dp.get("type", "")),
                "explanation": dp.get("explanation", ""),
                "line": dp.get("line", 0),
                "question": dp.get("question_for_human", dp.get("question", "")),
            })

    pagerank = (graph_node or {}).get("pagerank", (node or {}).get("pagerank", 0.0))

    elapsed = time.monotonic() - t0
    return {
        "function_id": function_id,
        "name": _node_short_name(function_id),
        "file": file_path,
        "description": description,
        "parameters": params,
        "pagerank": round(pagerank, 6),
        "callers": callers,
        "callees": callees,
        "business_rules": matched_rules,
        "decision_points": matched_dps,
        "response_time_ms": round(elapsed * 1000, 1),
    }


# ---------------------------------------------------------------------------
# 2. get_blast_radius (BFS with depth, exact matching — no substring bugs)
# ---------------------------------------------------------------------------

@router.get("/blast-radius")
def get_blast_radius(
    repo: str = Query(..., description="Repository name"),
    files: str = Query(..., description="Comma-separated file paths to check"),
    depth: int = Query(1, ge=1, le=10, description="BFS hop depth"),
):
    """Compute downstream impact: which files call/import the modified files."""
    t0 = time.monotonic()
    repo = validate_repo_name(repo)

    input_files = [f.strip() for f in files.split(",") if f.strip()]
    if not input_files:
        raise HTTPException(status_code=400, detail="'files' parameter must not be empty")

    graph = _load_graph(repo)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Repository '{repo}' not found or not indexed")

    edges = graph.get("edges", [])
    nodes = graph.get("nodes", [])

    # Build set of target node IDs for the input files (exact prefix match)
    input_file_set = set(input_files)
    target_ids: set[str] = set()
    for f in input_files:
        target_ids.add(f)  # File node itself
    for n in nodes:
        nid = n.get("id", "")
        nfile = _node_file(nid)
        if nfile in input_file_set:
            target_ids.add(nid)

    # Build reverse adjacency: target -> [source, ...] for CALLS/IMPORTS edges
    reverse_adj: dict[str, list[str]] = {}
    for e in edges:
        if e.get("type") not in ("CALLS", "IMPORTS"):
            continue
        tgt = e.get("target", "")
        src = e.get("source", "")
        if tgt and src:
            reverse_adj.setdefault(tgt, []).append(src)

    # BFS traversal
    visited_files: set[str] = set(input_files)
    frontier = set(target_ids)
    downstream: list[dict] = []

    for hop in range(1, depth + 1):
        next_frontier: set[str] = set()
        for node_id in frontier:
            for src in reverse_adj.get(node_id, []):
                src_file = _node_file(src)
                if src_file in visited_files:
                    continue
                visited_files.add(src_file)
                next_frontier.add(src)
                downstream.append({
                    "file": src_file,
                    "hop": hop,
                    "called_by": node_id,
                })
        frontier = next_frontier
        if not frontier:
            break

    # Deduplicate by file (keep earliest hop)
    seen_files: dict[str, dict] = {}
    for d in downstream:
        f = d["file"]
        if f not in seen_files:
            seen_files[f] = d
    downstream_deduped = sorted(seen_files.values(), key=lambda x: (x["hop"], x["file"]))

    count = len(downstream_deduped)
    if count == 0:
        risk = "LOW"
    elif count <= 2:
        risk = "MEDIUM"
    elif count <= 5:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    elapsed = time.monotonic() - t0
    return {
        "modified_files": input_files,
        "depth": depth,
        "risk_level": risk,
        "downstream_count": count,
        "downstream": downstream_deduped,
        "response_time_ms": round(elapsed * 1000, 1),
    }


# ---------------------------------------------------------------------------
# 3. search_by_concept (semantic search via ChromaDB)
# ---------------------------------------------------------------------------

@router.get("/search-by-concept")
def search_by_concept(
    q: str = Query(..., min_length=1, max_length=1000, description="Natural language query"),
    repo: str = Query(..., description="Repository name"),
    limit: int = Query(10, ge=1, le=50),
):
    """Semantic search across code knowledge graph using embeddings."""
    t0 = time.monotonic()
    repo = validate_repo_name(repo)

    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo, _DATA_DIR)
        info = embedder.collection_info()
        if info.get("count", 0) == 0:
            return {
                "query": q, "repo": repo, "count": 0, "results": [],
                "error": "Embeddings not available for this repository",
            }

        raw_results = embedder.query(text=q, n_results=limit)
    except Exception as e:
        logger.warning("Semantic search failed for '%s': %s", repo, e)
        return {
            "query": q, "repo": repo, "count": 0, "results": [],
            "error": f"Search unavailable: {e}",
        }

    # Enrich results with data from enriched_nodes.json
    enriched = _load_enriched(repo)

    results = []
    for r in raw_results:
        node_id = r.get("id", "")
        meta = r.get("metadata", {})
        enode = enriched.get(node_id, {})

        description = enode.get("llm_summary") or enode.get("summary") or enode.get("docstring") or ""
        if not description:
            description = r.get("text", "")[:200]

        results.append({
            "id": node_id,
            "name": meta.get("name") or enode.get("name") or _node_short_name(node_id),
            "type": meta.get("type") or enode.get("type", "unknown"),
            "file": meta.get("file") or enode.get("file") or _node_file(node_id),
            "score": round(r.get("score", 0.0), 4),
            "pagerank": round(float(meta.get("pagerank", 0) or enode.get("pagerank", 0)), 6),
            "description": description[:300],
        })

    elapsed = time.monotonic() - t0
    return {
        "query": q,
        "repo": repo,
        "count": len(results),
        "results": results,
        "response_time_ms": round(elapsed * 1000, 1),
    }
