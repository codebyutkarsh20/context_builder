import os
import re
import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from analyzer.structure import StructureAnalyzer
from analyzer.code_parser import CodeParser
from analyzer.call_graph import CallGraphBuilder
from graph.builder import GraphBuilder
from compiler.context_doc import ContextCompiler

logger = logging.getLogger(__name__)
router = APIRouter(tags=["repos"])

_SAFE_REPO_NAME = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# If set, all repo paths must be under this directory (prevents path traversal).
_REPOS_BASE_DIR: Optional[Path] = (
    Path(os.environ["REPOS_BASE_DIR"]).resolve()
    if "REPOS_BASE_DIR" in os.environ
    else None
)

# Directory where context files and graph caches are written.
_DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

# In-memory job store (replace with Redis for production)
jobs: dict[str, dict] = {}
_MAX_JOBS = 100  # Prevent unbounded growth


def _cleanup_old_jobs():
    """Remove oldest completed/failed jobs when limit is exceeded."""
    if len(jobs) <= _MAX_JOBS:
        return
    completed = [
        (jid, j) for jid, j in jobs.items()
        if j.get("status") in ("done", "failed")
    ]
    completed.sort(key=lambda x: x[1].get("_created", 0))
    to_remove = len(jobs) - _MAX_JOBS
    for jid, _ in completed[:to_remove]:
        del jobs[jid]


class AnalyzeRequest(BaseModel):
    repo_path: str
    repo_name: Optional[str] = None
    include_git_history: bool = True
    generate_llm_summaries: bool = False  # requires ANTHROPIC_API_KEY


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | failed
    progress: int  # 0-100
    stage: str
    repo_name: Optional[str] = None
    error: Optional[str] = None


def _sanitize_repo_name(name: str) -> str:
    """Sanitize repo name to prevent path traversal."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "-", name)
    sanitized = sanitized.strip("-.")
    return sanitized or "unnamed"


async def run_analysis(job_id: str, req: AnalyzeRequest):
    def update(stage: str, progress: int):
        jobs[job_id].update({"stage": stage, "progress": progress, "status": "running"})

    try:
        repo_path = Path(req.repo_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Path does not exist: {repo_path}")
        # Path traversal guard
        if _REPOS_BASE_DIR and not str(repo_path).startswith(str(_REPOS_BASE_DIR)):
            raise ValueError(f"Path is outside allowed base directory: {repo_path}")

        repo_name = _sanitize_repo_name(req.repo_name or repo_path.name)
        jobs[job_id]["repo_name"] = repo_name

        update("Analyzing structure", 10)
        structure = StructureAnalyzer(repo_path).analyze()

        update("Parsing code (Tree-sitter)", 30)
        parser = CodeParser(repo_path)
        parsed = parser.parse_all()

        update("Building call graph", 45)
        cg = CallGraphBuilder(parsed)
        graph_data = cg.build()

        # Data access detection
        update("Detecting data access patterns", 50)
        from analyzer.data_access import detect_data_access
        data_access = detect_data_access(parsed)
        node_map = {n["id"]: n for n in graph_data["nodes"]}
        for func_id, access in data_access.items():
            if func_id in node_map:
                node_map[func_id]["reads_from"] = access["reads_from"]
                node_map[func_id]["writes_to"] = access["writes_to"]

        # Decision points + domain concepts
        update("Extracting decision points", 55)
        from enricher.decision_points import extract_decision_points
        from enricher.domain_concepts import extract_domain_concepts
        decision_points = extract_decision_points(parsed)
        domain_concepts = extract_domain_concepts(parsed)

        # Git decision mining
        from analyzer.git_analyzer import GitAnalyzer
        git_analyzer = GitAnalyzer(repo_path)
        git_decisions = git_analyzer.extract_decision_context(
            llm_enhance=bool(os.environ.get("ANTHROPIC_API_KEY"))
        )

        from graph.neo4j_client import neo4j_client
        use_neo4j = neo4j_client.is_connected()

        if use_neo4j:
            update("Writing to Neo4j", 65)
            try:
                builder = GraphBuilder(repo_name, repo_path)
                builder.ingest(
                    structure, parsed, graph_data,
                    decision_points=decision_points,
                    domain_concepts=domain_concepts,
                )
            except Exception as exc:
                logger.warning("Neo4j ingestion failed, continuing without: %s", exc)
                use_neo4j = False
        else:
            update("Skipping Neo4j (not connected)", 65)

        if req.generate_llm_summaries and use_neo4j:
            update("Generating LLM summaries", 75)
            from enricher.summarizer import Summarizer
            Summarizer(repo_name).enrich()

        update("Compiling context document", 90)
        if use_neo4j:
            compiler = ContextCompiler(repo_name, repo_path=repo_path)
            compiler.compile()
        else:
            from enricher.business_logic import BusinessLogicExtractor
            git_data = git_analyzer.analyze()
            extractor = BusinessLogicExtractor(repo_name, parsed)
            rules = extractor.extract_all()
            from cli import _compile_without_neo4j
            _compile_without_neo4j(
                repo_name, structure, parsed, graph_data, rules,
                repo_path=repo_path,
                decision_points=decision_points,
                domain_concepts=domain_concepts,
                git_decisions=git_decisions,
                out_dir=_DATA_DIR / repo_name,
            )

        # Write graph.json cache for dashboard (used when Neo4j is unavailable)
        import json
        out = _DATA_DIR / repo_name
        out.mkdir(parents=True, exist_ok=True)
        cache = {
            "nodes": graph_data.get("nodes", []),
            "edges": graph_data.get("edges", []),
            "hotspots": graph_data.get("hotspots", []),
            "decision_points": decision_points,
            "domain_concepts": domain_concepts,
            "git_decisions": git_decisions,
            "stats": {
                "repo": repo_name,
                "repo_path": str(repo_path),
                "files": len(parsed),
                "classes": sum(len(f.get("classes", [])) for f in parsed),
                "functions": sum(len(f.get("functions", [])) for f in parsed),
                "lines_of_code": sum(f.get("loc", 0) for f in parsed),
                "tech_stack": structure.get("tech_stack", []),
            },
        }
        (out / "graph.json").write_text(json.dumps(cache, default=str))

        # Save enriched nodes + build embeddings
        try:
            from enricher.business_logic import BusinessLogicExtractor as _BLE
            rules_for_enrichment = []
            if not use_neo4j:
                rules_for_enrichment = rules  # Already extracted above
            from embeddings.embedder import build_enriched_nodes, NodeEmbedder
            enriched = build_enriched_nodes(parsed, graph_data, decision_points, domain_concepts, rules_for_enrichment)

            # Merge LLM summaries from Neo4j into enriched nodes for better embeddings
            if use_neo4j:
                try:
                    summaries = neo4j_client.run(
                        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n) "
                        "WHERE n.summary IS NOT NULL "
                        "RETURN n.id AS id, n.summary AS summary",
                        {"repo": repo_name},
                    )
                    merged = 0
                    for row in summaries:
                        nid = row.get("id", "")
                        if nid in enriched:
                            enriched[nid]["llm_summary"] = row["summary"]
                            merged += 1
                    if merged:
                        logger.info("Merged %d LLM summaries into enriched nodes", merged)
                except Exception as neo_err:
                    logger.debug("Could not merge Neo4j summaries: %s", neo_err)

            (out / "enriched_nodes.json").write_text(json.dumps(enriched, default=str))
            embedder = NodeEmbedder(repo_name, _DATA_DIR)
            import concurrent.futures
            _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _fut = _pool.submit(embedder.build_embeddings, enriched)
            try:
                _fut.result(timeout=120)  # 2-min cap; embeddings are non-fatal
            except concurrent.futures.TimeoutError:
                logger.warning("Embedding generation timed out after 120s (non-fatal)")
            finally:
                _pool.shutdown(wait=False)  # Don't block on hung embedding thread
        except Exception as emb_err:
            logger.warning("Embedding generation failed (non-fatal): %s", emb_err)

        # Generate default lint rules based on detected tech stack (Step 16)
        try:
            from agent.lint_rules import generate_default_rules
            lint_rules = generate_default_rules(repo_name)
            if lint_rules:
                logger.info("Generated %d default lint rules for '%s'", len(lint_rules), repo_name)
        except Exception as lint_err:
            logger.debug("Lint rule generation failed (non-fatal): %s", lint_err)

        jobs[job_id].update({"status": "done", "progress": 100, "stage": "Complete"})
        logger.info("Analysis complete for '%s': %d files, %d functions",
                     repo_name, len(parsed),
                     sum(len(f.get("functions", [])) for f in parsed))

    except Exception as e:
        logger.exception("Analysis failed for job %s: %s", job_id, e)
        jobs[job_id].update({"status": "failed", "stage": "Error", "error": str(e)})


@router.post("/analyze", response_model=JobStatus, status_code=202)
async def analyze_repo(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    # Pre-validate repo path synchronously
    repo_path = Path(req.repo_path).resolve()
    if _REPOS_BASE_DIR and not str(repo_path).startswith(str(_REPOS_BASE_DIR)):
        raise HTTPException(status_code=400, detail="Path is outside the allowed base directory")
    if not repo_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {repo_path}")

    _cleanup_old_jobs()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "stage": "Queued",
        "repo_name": None,
        "error": None,
        "_created": time.time(),
    }
    background_tasks.add_task(run_analysis, job_id, req)
    return JobStatus(**{k: v for k, v in jobs[job_id].items() if not k.startswith("_")})


@router.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**{k: v for k, v in jobs[job_id].items() if not k.startswith("_")})


@router.get("/repos")
def list_repos():
    """List all analyzed repos — from Neo4j if connected, else from disk."""
    from graph.neo4j_client import neo4j_client
    from graph.queries import list_repos_query
    if neo4j_client.is_connected():
        return neo4j_client.run(list_repos_query())
    # Fallback: scan DATA_DIR
    base = _DATA_DIR
    if not base.exists():
        return []
    import json as _json
    result = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        entry: dict = {
            "name": d.name,
            "has_context": (d / "context.md").exists(),
            "has_summary": (d / "summary.md").exists(),
        }
        graph_file = d / "graph.json"
        if graph_file.exists():
            try:
                data = _json.loads(graph_file.read_text())
                stats = data.get("stats", {})
                rp = stats.get("repo_path", "")
                if rp:
                    entry["repo_path"] = rp
                entry["files"] = stats.get("files", 0)
                entry["functions"] = stats.get("functions", 0)
                entry["lines_of_code"] = stats.get("lines_of_code", 0)
                entry["tech_stack"] = stats.get("tech_stack", [])
            except Exception:
                pass
        result.append(entry)
    return result


@router.delete("/repos/{repo_name}")
def delete_repo(repo_name: str):
    """Delete all data for a repository (disk + Neo4j + ChromaDB)."""
    import shutil

    if not _SAFE_REPO_NAME.match(repo_name):
        raise HTTPException(status_code=400, detail="Invalid repo name")

    repo_dir = _DATA_DIR / repo_name
    deleted = []

    # 1. Delete from disk
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
        deleted.append("disk")
        logger.info("Deleted disk data for repo '%s'", repo_name)

    # 2. Delete from Neo4j
    try:
        from graph.neo4j_client import neo4j_client
        if neo4j_client.is_connected():
            neo4j_client.run(
                "MATCH (n) WHERE n.repo = $repo DETACH DELETE n",
                {"repo": repo_name},
            )
            deleted.append("neo4j")
            logger.info("Deleted Neo4j data for repo '%s'", repo_name)
    except Exception as e:
        logger.warning("Failed to delete Neo4j data for '%s': %s", repo_name, e)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' not found")

    return {"deleted": repo_name, "cleaned": deleted}


@router.get("/repos/{repo_name}")
def get_repo_detail(repo_name: str):
    """Get detailed info about a single repository."""
    import json as _json

    if not _SAFE_REPO_NAME.match(repo_name):
        raise HTTPException(status_code=400, detail="Invalid repo name")

    repo_dir = _DATA_DIR / repo_name
    if not repo_dir.exists():
        raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' not found")

    entry: dict = {
        "name": repo_name,
        "has_context": (repo_dir / "context.md").exists(),
        "has_summary": (repo_dir / "summary.md").exists(),
        "has_embeddings": (repo_dir / "chromadb").exists(),
        "has_enriched": (repo_dir / "enriched_nodes.json").exists(),
    }

    graph_file = repo_dir / "graph.json"
    if graph_file.exists():
        try:
            data = _json.loads(graph_file.read_text())
            stats = data.get("stats", {})
            entry.update({
                "repo_path": stats.get("repo_path", ""),
                "files": stats.get("files", 0),
                "classes": stats.get("classes", 0),
                "functions": stats.get("functions", 0),
                "lines_of_code": stats.get("lines_of_code", 0),
                "tech_stack": stats.get("tech_stack", []),
                "nodes": len(data.get("nodes", [])),
                "edges": len(data.get("edges", [])),
                "decision_points": len(data.get("decision_points", [])),
                "domain_concepts": len(data.get("domain_concepts", [])),
            })
        except Exception:
            pass

    return entry
