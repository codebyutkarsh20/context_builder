import os
import re
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from graph.neo4j_client import neo4j_client
from graph import queries
from api.utils import validate_repo_name as _validate_repo

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

router = APIRouter(tags=["context"])


def _get_stats_from_cache(repo: str) -> dict:
    """Read stats from graph.json or context.md when Neo4j is unavailable."""
    import json
    from pathlib import Path
    repo = _validate_repo(repo)
    base = _DATA_DIR / repo
    g = base / "graph.json"
    if g.exists():
        data = json.loads(g.read_text())
        nodes = data.get("nodes", [])
        return {
            "file_count": sum(1 for n in nodes if n.get("type") in ("File", "file")),
            "class_count": sum(1 for n in nodes if n.get("type") in ("Class", "class")),
            "function_count": sum(1 for n in nodes if n.get("type") in ("Function", "function")),
            "call_edge_count": len(data.get("edges", [])),
            "summary_count": sum(1 for n in nodes if n.get("summary")),
        }
    ctx = base / "context.md"
    if ctx.exists():
        text = ctx.read_text()
        files = len(re.findall(r'^\| `[^`]+`', text, re.MULTILINE))
        classes = len(re.findall(r'^- \*\*\w+', text, re.MULTILINE))
        functions = len(re.findall(r'^- `\w+\(', text, re.MULTILINE))
        return {"file_count": files, "class_count": classes, "function_count": functions,
                "call_edge_count": 0, "summary_count": 0}
    return {}


@router.get("/context/layers")
def get_context_layers(repo: str = Query(...)):
    """Return the 6-layer context breakdown with token counts and completeness."""
    repo = _validate_repo(repo)
    if neo4j_client.is_connected():
        stats = neo4j_client.run(queries.get_graph_stats(repo))
        s = stats[0] if stats else {}
    else:
        s = _get_stats_from_cache(repo)

    if not s:
        raise HTTPException(status_code=404, detail="Repo not found or not yet analyzed")

    files = s.get("file_count", 0)
    funcs = s.get("function_count", 0)
    classes = s.get("class_count", 0)
    edges = s.get("call_edge_count", 0)
    summaries = s.get("summary_count", 0)
    total_tokens = (files * 15 + files * 80 + (funcs + classes) * 25
                    + edges * 10 + classes * 60 + summaries * 120)

    layers = [
        {
            "layer": 1,
            "name": "Repository Structure",
            "description": "Directory tree, entry points, tech stack",
            "node_count": files,
            "token_estimate": files * 15,
            "completeness": 1.0 if files > 0 else 0.0,
        },
        {
            "layer": 2,
            "name": "File Index",
            "description": "Per-file purpose, imports, exports",
            "node_count": files,
            "token_estimate": files * 80,
            "completeness": 1.0 if files > 0 else 0.0,
        },
        {
            "layer": 3,
            "name": "Symbol Map",
            "description": "Function and class signatures (no bodies)",
            "node_count": funcs + classes,
            "token_estimate": (funcs + classes) * 25,
            "completeness": 1.0 if funcs > 0 else 0.0,
        },
        {
            "layer": 4,
            "name": "Call Graph",
            "description": "Who calls what, PageRank-ranked hotspots",
            "node_count": edges,
            "token_estimate": edges * 10,
            "completeness": 1.0 if edges > 0 else 0.0,
        },
        {
            "layer": 5,
            "name": "Data Models",
            "description": "Pydantic/ORM/dataclass schemas",
            "node_count": classes,
            "token_estimate": classes * 60,
            "completeness": 1.0 if classes > 0 else 0.0,
        },
        {
            "layer": 6,
            "name": "Business Summaries",
            "description": "LLM-generated natural language per module",
            "node_count": summaries,
            "token_estimate": summaries * 120,
            "completeness": min(1.0, summaries / max(files, 1)),
        },
    ]
    return {
        "repo": repo,
        "layers": layers,
        "total_tokens": total_tokens,
        "token_budget": 200_000,
    }


@router.get("/context/summary")
def get_context_summary(repo: str = Query(...)):
    """Return the compiled summary.md content."""
    repo = _validate_repo(repo)
    path = _DATA_DIR / repo / "summary.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not yet generated")
    return {"repo": repo, "content": path.read_text()}


@router.get("/context/full")
def get_context_full(repo: str = Query(...)):
    """Return the full context.md content."""
    repo = _validate_repo(repo)
    path = _DATA_DIR / repo / "context.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Context not yet generated")
    return {"repo": repo, "content": path.read_text()}
