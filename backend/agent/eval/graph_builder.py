"""
eval/graph_builder.py — Build knowledge graph for an eval repo before running the agent.

Runs the same pipeline as `cli.py build --no-neo4j` but:
  - No Neo4j (agent works without it)
  - No LLM summaries (too expensive per-repo)
  - Cache-aware: skips if graph.json already exists for this repo+SHA
  - Logs progress to stdout so the eval runner can surface it

Used by EvalRunner when --build-graph is set.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def _get_repo_sha(repo_path: Path) -> str:
    """Get the current git SHA of a repo (short form)."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def build_eval_graph(
    repo_name: str,
    repo_path: Path,
    data_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Build the knowledge graph for a repo and return the data directory.

    Cache-aware: skips the build if graph.json already exists for the same SHA.
    Rebuilds automatically if the repo SHA has changed (e.g., different eval bugs
    pointing to the same repo_name at different commits).

    Parameters
    ----------
    repo_name : str
        Name used as the data directory key (must match what the agent expects).
    repo_path : Path
        Local path to the cloned repo at the correct SHA.
    data_dir : Path or None
        Root data directory. Defaults to DATA_DIR env var (/tmp/context_builder).
    force : bool
        If True, rebuild unconditionally.

    Returns
    -------
    Path
        The data directory for this repo (data_dir / repo_name).
    """
    base = data_dir or DATA_DIR
    out_dir = base / repo_name
    graph_path = out_dir / "graph.json"
    sha_stamp = out_dir / ".build_sha"

    current_sha = _get_repo_sha(repo_path)

    # Cache hit — skip if graph exists AND SHA matches
    if not force and graph_path.exists():
        cached_sha = sha_stamp.read_text().strip() if sha_stamp.exists() else ""
        if cached_sha and cached_sha == current_sha:
            logger.info(
                "Graph cache hit for %s @ %s — skipping build.",
                repo_name, current_sha,
            )
            return out_dir
        elif cached_sha != current_sha:
            logger.info(
                "SHA changed for %s (%s → %s) — rebuilding graph.",
                repo_name, cached_sha or "unknown", current_sha,
            )

    logger.info("Building knowledge graph for %s at %s", repo_name, repo_path)
    t0 = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse code (Tree-sitter) ─────────────────────────────────────
    logger.info("[1/7] Parsing code...")
    try:
        from analyzer.code_parser import CodeParser
        parser = CodeParser(repo_path)
        parsed = parser.parse_all()
        logger.info("  Parsed %d files", len(parsed))
    except Exception as e:
        logger.error("Code parsing failed: %s", e)
        raise

    # ── Step 2: Build call graph + PageRank ──────────────────────────────────
    logger.info("[2/7] Building call graph...")
    try:
        from analyzer.call_graph import CallGraphBuilder
        cg = CallGraphBuilder(parsed)
        graph_data = cg.build()
        logger.info(
            "  %d nodes, %d edges, top hotspot: %s",
            len(graph_data["nodes"]), len(graph_data["edges"]),
            graph_data["hotspots"][0]["label"] if graph_data.get("hotspots") else "n/a",
        )
    except Exception as e:
        logger.error("Call graph build failed: %s", e)
        raise

    # ── Step 3: Leiden community detection ───────────────────────────────────
    logger.info("[3/7] Running community detection...")
    communities: list = []
    try:
        from graph.community import build_communities, annotate_graph_with_communities
        communities = build_communities(graph_data)
        graph_data = annotate_graph_with_communities(graph_data, communities)
        names = [c["name"] for c in communities]
        logger.info("  %d communities: %s", len(communities), ", ".join(names[:6]))
        (out_dir / "communities.json").write_text(json.dumps(communities, default=str))
    except Exception as e:
        logger.warning("Community detection skipped: %s", e)

    # Write graph.json (with community annotations)
    (out_dir / "graph.json").write_text(json.dumps(graph_data, default=str))
    logger.info("  graph.json written (%d bytes)", (out_dir / "graph.json").stat().st_size)

    # ── Step 4: Data access + decision points + domain concepts ─────────────
    logger.info("[4/7] Extracting code enrichments...")
    decision_points: list = []
    domain_concepts: list = []
    try:
        from analyzer.data_access import detect_data_access
        data_access = detect_data_access(parsed)
        node_map = {n["id"]: n for n in graph_data["nodes"]}
        for func_id, access in data_access.items():
            if func_id in node_map:
                node_map[func_id]["reads_from"] = access["reads_from"]
                node_map[func_id]["writes_to"] = access["writes_to"]
        logger.info("  %d functions with data access patterns", len(data_access))
    except Exception as e:
        logger.warning("Data access detection skipped: %s", e)

    try:
        from enricher.decision_points import extract_decision_points
        decision_points = extract_decision_points(parsed)
        logger.info("  %d decision points", len(decision_points))
    except Exception as e:
        logger.warning("Decision point extraction skipped: %s", e)

    try:
        from enricher.domain_concepts import extract_domain_concepts
        domain_concepts = extract_domain_concepts(parsed)
        logger.info("  %d domain concepts", len(domain_concepts))
    except Exception as e:
        logger.warning("Domain concept extraction skipped: %s", e)

    # ── Step 5: Business rules ────────────────────────────────────────────────
    logger.info("[5/7] Extracting business rules...")
    rules: list = []
    try:
        from enricher.business_logic import BusinessLogicExtractor, persist_rules_to_file
        extractor = BusinessLogicExtractor(repo_name, parsed)
        rules = extractor.extract_all()
        persist_rules_to_file(rules, out_dir / "business_rules.json")
        logger.info("  %d business rules extracted", len(rules))
    except Exception as e:
        logger.warning("Business rules extraction skipped: %s", e)
        (out_dir / "business_rules.json").write_text("[]")

    # ── Step 6: Enriched node cache ───────────────────────────────────────────
    logger.info("[6/7] Building enriched node cache...")
    enriched: list = []
    try:
        from embeddings.embedder import build_enriched_nodes
        enriched = build_enriched_nodes(parsed, graph_data, decision_points, domain_concepts, rules)
        (out_dir / "enriched_nodes.json").write_text(json.dumps(enriched, default=str))
        logger.info("  %d enriched nodes", len(enriched))
    except Exception as e:
        logger.warning("Enriched node build skipped: %s", e)

    # ── Step 7: ChromaDB vector embeddings ────────────────────────────────────
    logger.info("[7/7] Building vector embeddings (ChromaDB)...")
    try:
        from embeddings.embedder import NodeEmbedder
        embedder = NodeEmbedder(repo_name, base)
        embed_count = embedder.build_embeddings(enriched)
        logger.info("  Embedded %d nodes", embed_count)
    except Exception as e:
        logger.warning("ChromaDB embedding skipped: %s", e)

    # Write SHA stamp so next run can detect if the repo changed
    if current_sha:
        sha_stamp.write_text(current_sha)

    elapsed = time.time() - t0
    logger.info(
        "Graph build complete for %s @ %s in %.1fs — output: %s",
        repo_name, current_sha or "unknown", elapsed, out_dir,
    )
    return out_dir
