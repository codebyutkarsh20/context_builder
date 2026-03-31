import os
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from graph.neo4j_client import neo4j_client
from graph import queries
from api.utils import validate_repo_name as _validate_repo

_VALID_LAYERS = {"code", "business"}

router = APIRouter(tags=["graph"])


def _load_graph_cache(repo: str):
    """Load graph data from JSON cache written by CLI (--no-neo4j mode)."""
    import json
    from pathlib import Path
    repo = _validate_repo(repo)
    data_dir = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
    p = data_dir / repo / "graph.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@router.get("/graph")
def get_graph(
    repo: str = Query(..., description="Repository name"),
    layer: Optional[str] = Query(None, description="code | business | all"),
    node_type: Optional[str] = Query(None, description="File | Class | Function | BusinessRule | DomainConcept"),
    limit: int = Query(2000, le=10000),
):
    """Return nodes and edges for the knowledge graph visualization."""
    _validate_repo(repo)
    if layer is not None and layer not in _VALID_LAYERS:
        raise HTTPException(status_code=400, detail=f"Invalid layer '{layer}'. Must be one of: {sorted(_VALID_LAYERS)}")
    if neo4j_client.is_connected():
        raw_nodes = neo4j_client.run(queries.get_graph_nodes(repo, layer, node_type, limit))
        raw_edges = neo4j_client.run(queries.get_graph_edges(repo, layer, limit))

        # Normalize nodes: extract type from labels list, fill missing name/file
        _LABEL_ORDER = ["BusinessRule", "DomainConcept", "DecisionPoint", "Class", "Function", "File"]
        def _pick_type(labels: list) -> str:
            if not labels:
                return "File"
            for preferred in _LABEL_ORDER:
                if preferred in labels:
                    return preferred
            return labels[0]

        nodes = []
        for n in raw_nodes:
            labels = n.get("labels") or []
            node_type_val = _pick_type(labels)
            node_id = n.get("id", "")
            name = n.get("name") or n.get("path") or (node_id.split("::")[-1] if "::" in node_id else node_id)
            nodes.append({
                "id": node_id,
                "name": name,
                "type": node_type_val,
                "file": n.get("path") or (node_id.split("::")[0] if "::" in node_id else None),
                "summary": n.get("summary"),
                "docstring": n.get("docstring"),
                "pagerank": n.get("pagerank"),
                "language": n.get("language"),
            })

        # Normalize edges: rename source_id/target_id → source/target
        edges = []
        for e in raw_edges:
            src = e.get("source_id") or e.get("source")
            tgt = e.get("target_id") or e.get("target")
            if src and tgt:
                edges.append({
                    "id": e.get("id") or f"{src}_{e.get('type','')}_{tgt}",
                    "source": src,
                    "target": tgt,
                    "type": e.get("type", "RELATED_TO"),
                    "weight": e.get("weight"),
                })

        # If Neo4j returned no non-CONTAINS edges, supplement from graph.json cache
        # (JS/TS repos may only have CONTAINS edges in Neo4j but full edges in cache)
        non_contains = [e for e in edges if e.get("type") != "CONTAINS"]
        if not non_contains:
            cache = _load_graph_cache(repo)
            if cache:
                cached_edges = cache.get("edges", [])
                node_ids = {n["id"] for n in nodes}
                for e in cached_edges:
                    if e.get("source") in node_ids and e.get("target") in node_ids:
                        if e not in edges:
                            edges.append(e)

        return {"nodes": nodes, "edges": edges, "repo": repo}
    cache = _load_graph_cache(repo)
    if cache:
        _TYPE_MAP = {"file": "File", "class": "Class", "function": "Function",
                     "businessrule": "BusinessRule", "domainconcept": "DomainConcept"}

        def _norm_node(n: dict) -> dict:
            raw_type = n.get("type", "")
            norm_type = _TYPE_MAP.get(raw_type.lower(), raw_type.capitalize() if raw_type else "File")
            node_id = n.get("id", "")
            name = n.get("name") or n.get("label") or (node_id.split("::")[-1] if "::" in node_id else node_id)
            return {**n, "type": norm_type, "name": name}

        raw_nodes = [_norm_node(n) for n in cache.get("nodes", [])]
        # Sort by pagerank descending so we get the most important nodes first
        raw_nodes.sort(key=lambda n: n.get("pagerank", 0.0), reverse=True)

        if node_type:
            raw_nodes = [n for n in raw_nodes if n.get("type") == node_type]
        nodes = raw_nodes[:limit]

        node_ids = {n["id"] for n in nodes}
        raw_edges = cache.get("edges", [])
        edges = [e for e in raw_edges if e.get("source") in node_ids and e.get("target") in node_ids][:limit]
        return {"nodes": nodes, "edges": edges}
    return {"nodes": [], "edges": []}


def _normalize_hotspot(item: dict, rank: int) -> dict:
    """Normalize hotspot dict from graph cache to the expected API shape."""
    raw_type = item.get("type", "")
    type_map = {"file": "File", "class": "Class", "function": "Function",
                "businessrule": "BusinessRule", "domainconcept": "DomainConcept"}
    normalized_type = type_map.get(raw_type.lower(), raw_type.capitalize() if raw_type else "Function")
    return {
        "id": item.get("id", ""),
        "name": item.get("name") or item.get("label") or item.get("id", ""),
        "type": normalized_type,
        "file": item.get("file") or item.get("path"),
        "pagerank": item.get("pagerank", 0.0),
        "rank": rank,
    }


@router.get("/graph/hotspots")
def get_hotspots(
    repo: str = Query(...),
    top_n: int = Query(20, le=100),
):
    """Return top-N nodes ranked by PageRank (most connected/referenced)."""
    if neo4j_client.is_connected():
        return neo4j_client.run(queries.get_hotspots(repo, top_n))
    cache = _load_graph_cache(repo)
    if cache:
        raw = cache.get("hotspots", [])[:top_n]
        return [_normalize_hotspot(h, i + 1) for i, h in enumerate(raw)]
    return []


@router.get("/graph/node/{node_id:path}")
def get_node_detail(node_id: str, repo: str = Query(...)):
    """Return detailed info for a single node including neighbors."""
    repo = _validate_repo(repo)
    if neo4j_client.is_connected():
        result = neo4j_client.run(queries.get_node_detail(node_id, repo))
        if not result:
            raise HTTPException(status_code=404, detail="Node not found")
        return result[0]
    # Fallback: search graph.json cache
    cache = _load_graph_cache(repo)
    if cache:
        for node in cache.get("nodes", []):
            if node.get("id") == node_id:
                edges = cache.get("edges", [])
                neighbors = [
                    e.get("target") for e in edges if e.get("source") == node_id
                ] + [
                    e.get("source") for e in edges if e.get("target") == node_id
                ]
                return {**node, "neighbors": neighbors}
    raise HTTPException(status_code=404, detail="Node not found")


@router.get("/graph/stats")
def get_graph_stats(repo: str = Query(...)):
    """Return aggregate stats: node counts by type, edge counts by type."""
    if neo4j_client.is_connected():
        result = neo4j_client.run(queries.get_graph_stats(repo))
        if result:
            r = result[0]
            # Normalize Neo4j column names to the field names the frontend expects
            files = r.get("file_count", r.get("files", 0))
            classes = r.get("class_count", r.get("classes", 0))
            functions = r.get("function_count", r.get("functions", 0))
            loc = r.get("lines_of_code") or 0
            tech = r.get("tech_stack") or []
            # Count language breakdown from nodes if available
            lang_map = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
                        ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
                        ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".vue": "Vue"}
            # Fetch language counts from enriched nodes cache
            cache = _load_graph_cache(repo)
            lang_counts: dict = {}
            if cache:
                from collections import Counter
                lc: Counter = Counter()
                for n in cache.get("nodes", []):
                    if n.get("type", "").lower() != "file":
                        continue
                    nid = n.get("id", "")
                    for ext, lang in lang_map.items():
                        if nid.endswith(ext):
                            lc[lang] += 1
                            break
                lang_counts = dict(lc)
            total_nodes = r.get("total_nodes", files + classes + functions)
            total_edges = r.get("total_edges", r.get("call_edge_count", 0) + r.get("import_edge_count", 0))
            return {
                "repo": repo,
                "files": files,
                "classes": classes,
                "functions": functions,
                "lines_of_code": loc,
                "languages": lang_counts or {"TypeScript": files},
                "node_type_counts": {"File": files, "Class": classes, "Function": functions},
                "tech_stack": tech,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
            }
        return {}
    # Fallback: use graph.json cache
    cache = _load_graph_cache(repo)
    if cache and "stats" in cache:
        s = cache["stats"]
        files = s.get("files", 0)
        classes = s.get("classes", 0)
        functions = s.get("functions", 0)
        loc = s.get("lines_of_code", 0)
        tech = s.get("tech_stack", [])
        nodes = cache.get("nodes", [])
        edges = cache.get("edges", [])
        # Count by type from nodes
        type_counts = {}
        for n in nodes:
            t = n.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        # Count languages from file extensions
        from collections import Counter
        lang_map = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
                    ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
                    ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".vue": "Vue"}
        lang_counts = Counter()
        for n in nodes:
            if n.get("type", "").lower() != "file":
                continue
            nid = n.get("id", "")
            for ext, lang in lang_map.items():
                if nid.endswith(ext):
                    lang_counts[lang] += 1
                    break
        return {
            "repo": repo, "files": files, "classes": classes,
            "functions": functions, "lines_of_code": loc,
            "languages": dict(lang_counts) or {"Unknown": files},
            "node_type_counts": type_counts or {
                "File": files, "Class": classes, "Function": functions
            },
            "tech_stack": tech, "total_nodes": len(nodes),
            "total_edges": len(edges),
        }
    return {}


@router.get("/graph/rag-status")
def get_rag_status(repo: str = Query(...)):
    """Check if Graph RAG is available for a repository."""
    repo = _validate_repo(repo)
    data_dir = __import__("pathlib").Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
    enriched_path = data_dir / repo / "enriched_nodes.json"

    result = {
        "repo": repo,
        "enriched_nodes": enriched_path.exists(),
        "enriched_count": 0,
        "chromadb_available": False,
        "chromadb_count": 0,
    }

    if enriched_path.exists():
        import json
        enriched = json.loads(enriched_path.read_text())
        result["enriched_count"] = len(enriched)

    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo, data_dir)
        info = embedder.collection_info()
        result["chromadb_available"] = info.get("count", 0) > 0
        result["chromadb_count"] = info.get("count", 0)
    except Exception:
        pass

    return result


@router.get("/graph/decisions")
def get_decisions(
    repo: str = Query(...),
    limit: int = Query(100, le=500),
):
    """Return decision points for a repository."""
    repo = _validate_repo(repo)
    if neo4j_client.is_connected():
        return neo4j_client.run(
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(fn:Function)-[:HAS_DECISION]->(dp:DecisionPoint) "
            "RETURN dp.id AS id, dp.line AS line, dp.condition AS condition, "
            "       dp.condition_type AS condition_type, dp.explanation AS explanation, "
            "       dp.question_for_human AS question_for_human, dp.file AS file, "
            "       fn.name AS function_name "
            "ORDER BY dp.condition_type, dp.file "
            "LIMIT $limit",
            {"repo": repo, "limit": limit},
        )
    # Fallback: graph.json cache
    cache = _load_graph_cache(repo)
    if cache:
        return cache.get("decision_points", [])[:limit]
    return []


@router.get("/graph/domain")
def get_domain_concepts(
    repo: str = Query(...),
    limit: int = Query(50, le=200),
):
    """Return domain concepts for a repository."""
    repo = _validate_repo(repo)
    if neo4j_client.is_connected():
        return neo4j_client.run(
            "MATCH (dc:DomainConcept) "
            "OPTIONAL MATCH (dc)-[:REPRESENTS]->(c:Class) "
            "WITH dc, collect(c.name) AS classes "
            "RETURN dc.id AS id, dc.name AS name, dc.type AS type, "
            "       dc.description AS description, classes AS related_classes "
            "ORDER BY dc.name "
            "LIMIT $limit",
            {"limit": limit},
        )
    # Fallback: graph.json cache
    cache = _load_graph_cache(repo)
    if cache:
        return cache.get("domain_concepts", [])[:limit]
    return []
