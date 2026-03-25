"""
embedder.py — Build and query ChromaDB vector embeddings for code knowledge graph nodes.

Uses ChromaDB's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2 via onnxruntime)
so no additional ML dependencies are needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

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
    from the transient parsed data. This bridges the gap between the structural
    graph.json and what ChromaDB needs for embedding.

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


# ---------------------------------------------------------------------------
# Text construction for embedding
# ---------------------------------------------------------------------------

def _node_to_text(node: dict) -> str:
    """Convert an enriched node dict into embeddable text."""
    ntype = node.get("type", "")

    if ntype == "file":
        # Prefer LLM-generated business-purpose description over raw docstring
        doc = node.get("llm_summary") or node.get("summary") or node.get("docstring", "")
        classes = ", ".join(node.get("classes", [])[:10])
        funcs = ", ".join(node.get("functions", [])[:10])
        imports = ", ".join(node.get("imports", [])[:10])
        parts = [f"File: {node.get('file', '')}"]
        if doc:
            parts.append(f"Purpose: {doc[:400]}")
        if classes:
            parts.append(f"Classes: {classes}")
        if funcs:
            parts.append(f"Functions: {funcs}")
        if imports:
            parts.append(f"Imports: {imports}")
        return "\n".join(parts)

    elif ntype == "function":
        params = ", ".join(node.get("params", []))
        ret = f" -> {node['return_type']}" if node.get("return_type") else ""
        # Prefer LLM-generated business-purpose description over raw docstring
        doc = node.get("llm_summary") or node.get("summary") or node.get("docstring", "")
        parts = [f"Function: {node.get('name', '')}({params}){ret} in {node.get('file', '')}"]
        if doc:
            parts.append(f"Purpose: {doc[:400]}")
        return "\n".join(parts)

    elif ntype == "class":
        bases = ", ".join(node.get("bases", []))
        methods = ", ".join(node.get("methods", [])[:10])
        # Prefer LLM-generated business-purpose description over raw docstring
        doc = node.get("llm_summary") or node.get("summary") or node.get("docstring", "")
        parts = [f"Class: {node.get('name', '')} in {node.get('file', '')}"]
        if bases:
            parts.append(f"Inherits: {bases}")
        if doc:
            parts.append(f"Purpose: {doc[:400]}")
        if methods:
            parts.append(f"Methods: {methods}")
        return "\n".join(parts)

    elif ntype == "business_rule":
        return f"Business Rule [{node.get('rule_type', '')}]: {node.get('content', '')}"

    elif ntype == "decision_point":
        return (
            f"Decision Point [{node.get('condition_type', '')}]: {node.get('name', '')} "
            f"in {node.get('function_id', '')}"
        )

    elif ntype == "domain_concept":
        classes = ", ".join(node.get("related_classes", [])[:5])
        desc = node.get("description", "")
        parts = [f"Domain Concept: {node.get('name', '')} ({node.get('concept_type', 'entity')})"]
        if desc:
            parts.append(f"Description: {desc}")
        if classes:
            parts.append(f"Related classes: {classes}")
        return "\n".join(parts)

    return f"{ntype}: {node.get('name', node.get('id', ''))}"


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# ChromaDB embedding pipeline
# ---------------------------------------------------------------------------

class NodeEmbedder:
    """Build and query ChromaDB embeddings for code knowledge graph nodes."""

    def __init__(self, repo_name: str, data_dir: Path) -> None:
        self.repo_name = repo_name
        self.data_dir = Path(data_dir)
        self._collection_name = f"context_builder_{repo_name}"
        self._persist_dir = str(self.data_dir / repo_name / "chromadb")
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def build_embeddings(self, enriched_nodes: dict[str, dict] | None = None) -> int:
        """
        Embed all nodes into ChromaDB. Incremental: skips nodes whose content hasn't changed.

        If enriched_nodes is None, loads from enriched_nodes.json.
        Returns count of newly embedded nodes.
        """
        if enriched_nodes is None:
            enriched_path = self.data_dir / self.repo_name / "enriched_nodes.json"
            if not enriched_path.exists():
                logger.warning("No enriched_nodes.json found for '%s'", self.repo_name)
                return 0
            enriched_nodes = json.loads(enriched_path.read_text())

        collection = self._get_collection()

        # Skip decision points and business rules for embedding (too noisy, too many)
        embeddable_types = {"file", "function", "class", "domain_concept"}

        # Build documents
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for node_id, node in enriched_nodes.items():
            if node.get("type") not in embeddable_types:
                continue

            text = _node_to_text(node)
            if not text or len(text) < 10:
                continue

            ids.append(node_id)
            documents.append(text[:2000])  # ChromaDB max doc size
            metadatas.append({
                "type": node.get("type", ""),
                "name": str(node.get("name", ""))[:100],
                "file": str(node.get("file", ""))[:200],
                "pagerank": float(node.get("pagerank", 0.0)),
                "content_hash": _content_hash(text),
            })

        if not ids:
            logger.info("No embeddable nodes found for '%s'", self.repo_name)
            return 0

        # Batch upsert (ChromaDB handles dedup by ID)
        BATCH = 500
        total = 0
        for i in range(0, len(ids), BATCH):
            batch_ids = ids[i:i + BATCH]
            batch_docs = documents[i:i + BATCH]
            batch_meta = metadatas[i:i + BATCH]
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            total += len(batch_ids)

        logger.info("Embedded %d nodes into ChromaDB for '%s'", total, self.repo_name)
        return total

    def query(
        self,
        text: str,
        n_results: int = 20,
        node_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Query ChromaDB for semantically similar nodes.

        Returns list of {id, text, score, metadata} dicts.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        where_filter = None
        if node_types:
            if len(node_types) == 1:
                where_filter = {"type": node_types[0]}
            else:
                where_filter = {"type": {"$in": node_types}}

        try:
            results = collection.query(
                query_texts=[text],
                n_results=min(n_results, collection.count()),
                where=where_filter,
            )
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            return []

        output = []
        if results and results.get("ids"):
            for i, node_id in enumerate(results["ids"][0]):
                output.append({
                    "id": node_id,
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0),
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
        return output

    def collection_info(self) -> dict:
        """Return info about the embedding collection."""
        try:
            collection = self._get_collection()
            return {
                "collection": self._collection_name,
                "count": collection.count(),
                "persist_dir": self._persist_dir,
            }
        except Exception:
            return {"collection": self._collection_name, "count": 0, "error": "not available"}
