"""
retriever.py — Multi-strategy retriever for Graph RAG.

Combines graph traversal (graph.json edges) and keyword matching
to find the most relevant subgraph for a question.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.query_analyzer import QueryIntent, analyze_query

logger = logging.getLogger(__name__)


@dataclass
class ScoredNode:
    id: str
    score: float
    source: str  # "graph" | "keyword"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Output of the retrieval pipeline."""
    primary_nodes: list[str]     # Directly matched node IDs
    expanded_nodes: list[str]    # Neighbors from graph traversal
    all_node_ids: list[str]      # Merged, ranked list
    edges: list[dict]            # Relevant edges between retrieved nodes
    intent: QueryIntent
    scores: dict[str, float] = field(default_factory=dict)


class GraphRAGRetriever:
    """Multi-strategy retriever combining graph traversal + keywords."""

    def __init__(self, repo_name: str, data_dir: Path) -> None:
        self.repo_name = repo_name
        self.data_dir = Path(data_dir)
        self._enriched: dict[str, dict] | None = None
        self._edges: list[dict] | None = None
        self._edge_index: dict[str, list[dict]] | None = None

    def retrieve(self, question: str, max_nodes: int = 30) -> RetrievalResult:
        """
        Run the full retrieval pipeline:
        1. Analyze query intent
        2. Keyword search
        3. Graph expansion (1-2 hops)
        4. Merge and rank
        """
        intent = analyze_query(question)

        # Strategy 1: Keyword search
        keyword_results = self._keyword_search(question, intent, n=15)

        # Rank
        merged = self._merge_and_rank([], keyword_results)

        # Take top seeds for graph expansion
        seed_ids = [s.id for s in merged[:10]]

        # Strategy 3: Graph expansion
        expanded_ids = self._graph_expand(seed_ids, hops=2)

        # Final merge: seeds + expansion
        primary_ids = [s.id for s in merged[:max_nodes]]
        all_ids_set = set(primary_ids) | set(expanded_ids)

        # Get relevant edges between retrieved nodes
        edges = self._get_edges_between(all_ids_set)

        all_ids = primary_ids + [eid for eid in expanded_ids if eid not in set(primary_ids)]
        all_ids = all_ids[:max_nodes]

        scores = {s.id: s.score for s in merged}

        return RetrievalResult(
            primary_nodes=primary_ids[:15],
            expanded_nodes=expanded_ids[:15],
            all_node_ids=all_ids,
            edges=edges,
            intent=intent,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Keyword search
    # ------------------------------------------------------------------

    def _keyword_search(self, question: str, intent: QueryIntent, n: int) -> list[ScoredNode]:
        enriched = self._load_enriched()
        if not enriched:
            return []

        terms = [name.lower() for name in intent.mentioned_names if len(name) >= 3]
        if not terms:
            return []

        results: list[ScoredNode] = []
        for node_id, node in enriched.items():
            name = (node.get("name") or "").lower()
            doc = (node.get("docstring") or "").lower()
            file_path = (node.get("file") or "").lower()
            searchable = f"{name} {doc} {file_path}"

            # Score based on term matches
            score = 0.0
            for term in terms:
                if term == name:
                    score += 1.0
                elif name.startswith(term) or term.startswith(name):
                    score += 0.7
                elif term in name:
                    score += 0.5
                elif term in doc:
                    score += 0.3
                elif term in file_path:
                    score += 0.2

            if score > 0:
                # Boost by PageRank
                pr = float(node.get("pagerank", 0))
                if pr > 0:
                    import math
                    score *= (1 + math.log1p(pr * 10000))

                results.append(ScoredNode(
                    id=node_id,
                    score=score,
                    source="keyword",
                    metadata={"type": node.get("type", ""), "name": node.get("name", "")},
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:n]

    # ------------------------------------------------------------------
    # Merge and rank (Reciprocal Rank Fusion)
    # ------------------------------------------------------------------

    def _merge_and_rank(self, *result_lists: list[ScoredNode]) -> list[ScoredNode]:
        """Merge multiple result lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        node_map: dict[str, ScoredNode] = {}
        K = 60  # RRF constant

        for results in result_lists:
            for rank, node in enumerate(results):
                rrf_score = 1.0 / (K + rank)
                # Also factor in the node's own relevance score
                rrf_score *= (0.5 + node.score)
                scores[node.id] = scores.get(node.id, 0) + rrf_score
                if node.id not in node_map:
                    node_map[node.id] = node

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            ScoredNode(
                id=nid,
                score=score,
                source=node_map[nid].source if nid in node_map else "merged",
                metadata=node_map[nid].metadata if nid in node_map else {},
            )
            for nid, score in ranked
        ]

    # ------------------------------------------------------------------
    # Graph expansion
    # ------------------------------------------------------------------

    def _graph_expand(self, seed_ids: list[str], hops: int = 2) -> list[str]:
        """Expand from seed nodes via edges. Returns neighbor node IDs."""
        edge_index = self._load_edge_index()
        if not edge_index:
            return []

        visited: set[str] = set(seed_ids)
        frontier: set[str] = set(seed_ids)
        expanded: list[str] = []

        hop1_types = {"CONTAINS", "CALLS", "IMPORTS", "INHERITS", "HAS_DECISION"}
        hop2_types = {"CALLS", "IMPORTS"}

        for hop in range(hops):
            allowed_types = hop1_types if hop == 0 else hop2_types
            next_frontier: set[str] = set()

            for node_id in frontier:
                for edge in edge_index.get(node_id, []):
                    src = edge.get("source", "")
                    tgt = edge.get("target", "")
                    if not src or not tgt:
                        continue
                    neighbor = tgt if src == node_id else src
                    if neighbor not in visited and edge.get("type") in allowed_types:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        expanded.append(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        # Also ensure parent files are included for any function/class
        enriched = self._load_enriched()
        file_ids = set()
        for nid in list(visited):
            node = enriched.get(nid, {})
            if node.get("type") in ("function", "class") and node.get("file"):
                fid = node["file"]
                if fid not in visited:
                    file_ids.add(fid)

        expanded.extend(file_ids)
        return expanded[:30]  # cap expansion

    def _get_edges_between(self, node_ids: set[str]) -> list[dict]:
        """Return edges where both endpoints are in the retrieved set."""
        edges = self._load_edges()
        return [
            e for e in edges
            if e.get("source") in node_ids and e.get("target") in node_ids
        ][:50]

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_enriched(self) -> dict[str, dict]:
        if self._enriched is not None:
            return self._enriched
        path = self.data_dir / self.repo_name / "enriched_nodes.json"
        if path.exists():
            self._enriched = json.loads(path.read_text())
        else:
            self._enriched = {}
        return self._enriched

    def _load_edges(self) -> list[dict]:
        if self._edges is not None:
            return self._edges
        path = self.data_dir / self.repo_name / "graph.json"
        if path.exists():
            data = json.loads(path.read_text())
            self._edges = data.get("edges", [])
        else:
            self._edges = []
        return self._edges

    def _load_edge_index(self) -> dict[str, list[dict]]:
        if self._edge_index is not None:
            return self._edge_index
        edges = self._load_edges()
        idx: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            idx[e.get("source", "")].append(e)
            idx[e.get("target", "")].append(e)
        self._edge_index = dict(idx)
        return self._edge_index
