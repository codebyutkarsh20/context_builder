import os
from fastapi import APIRouter, Query, HTTPException
from graph.neo4j_client import neo4j_client
from graph import queries
from api.utils import validate_repo_name

router = APIRouter(tags=["search"])

_TYPE_DISPLAY = {
    "file": "File", "class": "Class", "function": "Function",
    "businessrule": "BusinessRule", "domainconcept": "DomainConcept",
}


def _normalize_type(raw: str) -> str:
    return _TYPE_DISPLAY.get(raw.lower(), raw.capitalize()) if raw else "Unknown"


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=1000),
    repo: str = Query(...),
    limit: int = Query(20, le=100),
):
    """Full-text search across node names, summaries, and business rules."""
    validate_repo_name(repo)
    if not neo4j_client.is_connected():
        # Fallback: search graph.json cache
        import json
        from pathlib import Path
        data_dir = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))
        cache_path = data_dir / repo / "graph.json"
        if not cache_path.exists():
            return []
        data = json.loads(cache_path.read_text())
        q_lower = q.lower()
        results = []
        for node in data.get("nodes", []):
            name = (node.get("name") or node.get("label") or node.get("id", "")).lower()
            summary = (node.get("summary") or node.get("docstring") or "").lower()

            # Graduated scoring: exact > starts-with > contains-name > contains-summary
            if name == q_lower:
                score = 1.0
            elif name.startswith(q_lower):
                score = 0.8
            elif q_lower in name:
                score = 0.6
            elif q_lower in summary:
                score = 0.4
            else:
                continue

            raw_type = node.get("type", "unknown")
            snippet = node.get("summary") or node.get("docstring") or ""
            results.append({
                "id": node.get("id", ""),
                "name": node.get("name") or node.get("label") or node.get("id", ""),
                "type": _normalize_type(raw_type),
                "file": node.get("file") or node.get("path", ""),
                "snippet": snippet[:200] if snippet else "",
                "score": score,
            })

        # Sort by score descending, then name ascending
        results.sort(key=lambda r: (-r["score"], r["name"]))
        return results[:limit]
    return neo4j_client.run(queries.search_nodes(repo, q, limit))
