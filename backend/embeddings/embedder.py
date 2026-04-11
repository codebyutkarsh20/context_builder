"""
embedder.py — Build enriched node cache from parsed code data.

Creates a text-rich node map (enriched_nodes.json) that other modules use for
context assembly, function lookups, and knowledge graph queries.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enriched node builder — creates text-rich node cache from parsed data
# ---------------------------------------------------------------------------

def build_enriched_nodes(
    parsed: list[dict],
    graph_data: dict,
    decision_points: list[dict] | None = None,
    domain_concepts: list[dict] | None = None,
    business_rules: list | None = None,
) -> dict[str, dict]:
    """
    Build an enriched node map with text content (docstrings, summaries, params)
    from the transient parsed data. Used for function lookups, context assembly,
    and graph API enrichment.

    Returns {node_id: {id, type, name, file, docstring, params, ...}}
    """
    enriched: dict[str, dict] = {}

    # PageRank lookup from graph_data
    pr_map: dict[str, float] = {}
    for node in graph_data.get("nodes", []):
        pr_map[node["id"]] = node.get("pagerank", 0.0)

    for pf in parsed:
        rel = pf["path"]

        # File node
        enriched[rel] = {
            "id": rel,
            "type": "file",
            "name": Path(rel).name,
            "file": rel,
            "docstring": pf.get("docstring", "") or "",
            "imports": [imp.get("module", "") for imp in pf.get("imports", []) if imp.get("module")],
            "classes": [c["name"] for c in pf.get("classes", [])],
            "functions": [f["name"] for f in pf.get("functions", [])],
            "pagerank": pr_map.get(rel, 0.0),
        }

        # Top-level function nodes
        for fn in pf.get("functions", []):
            fid = f"{rel}::{fn['name']}"
            enriched[fid] = {
                "id": fid,
                "type": "function",
                "name": fn["name"],
                "file": rel,
                "params": fn.get("params", []),
                "return_type": fn.get("return_type"),
                "docstring": fn.get("docstring", "") or "",
                "decorators": fn.get("decorators", []),
                "complexity": fn.get("complexity", 1),
                "pagerank": pr_map.get(fid, 0.0),
            }

        # Class + method nodes
        for cls in pf.get("classes", []):
            cid = f"{rel}::{cls['name']}"
            enriched[cid] = {
                "id": cid,
                "type": "class",
                "name": cls["name"],
                "file": rel,
                "bases": cls.get("bases", []),
                "docstring": cls.get("docstring", "") or "",
                "methods": [m["name"] for m in cls.get("methods", [])],
                "pagerank": pr_map.get(cid, 0.0),
            }
            for method in cls.get("methods", []):
                mid = f"{cid}::{method['name']}"
                enriched[mid] = {
                    "id": mid,
                    "type": "function",
                    "name": method["name"],
                    "file": rel,
                    "params": method.get("params", []),
                    "return_type": method.get("return_type"),
                    "docstring": method.get("docstring", "") or "",
                    "complexity": method.get("complexity", 1),
                    "pagerank": pr_map.get(mid, 0.0),
                }

    # Decision points
    for dp in (decision_points or []):
        enriched[dp["id"]] = {
            "id": dp["id"],
            "type": "decision_point",
            "name": dp.get("condition", ""),
            "file": dp.get("file", ""),
            "condition_type": dp.get("condition_type", ""),
            "function_id": dp.get("function_id", ""),
            "explanation": dp.get("explanation", ""),
        }

    # Domain concepts
    for dc in (domain_concepts or []):
        enriched[dc["id"]] = {
            "id": dc["id"],
            "type": "domain_concept",
            "name": dc.get("name", ""),
            "concept_type": dc.get("type", "entity"),
            "description": dc.get("description", ""),
            "related_classes": dc.get("related_classes", []),
        }

    # Business rules
    for r in (business_rules or []):
        content = getattr(r, "content", str(r)) if not isinstance(r, dict) else r.get("content", str(r))
        rule_type = getattr(r, "rule_type", "") if not isinstance(r, dict) else r.get("rule_type", "")
        source_file = getattr(r, "source_file", "") if not isinstance(r, dict) else r.get("source_file", "")
        rid = f"rule::{source_file}::{content[:50]}"
        enriched[rid] = {
            "id": rid,
            "type": "business_rule",
            "name": content[:100],
            "file": source_file,
            "rule_type": rule_type,
            "content": content,
        }

    logger.info("Built enriched node cache: %d nodes", len(enriched))
    return enriched
