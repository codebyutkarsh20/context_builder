"""GraphBuilder: ingests parsed repository data into Neo4j."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds and maintains the code knowledge graph for a single repository."""

    def __init__(self, repo_name: str, repo_path: Path) -> None:
        self.repo_name = repo_name
        self.repo_path = str(repo_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        structure: dict[str, Any],
        parsed: list[Any],
        graph_data: dict[str, Any],
        *,
        decision_points: list[dict] | None = None,
        domain_concepts: list[dict] | None = None,
    ) -> None:
        """Ingest all repository data into Neo4j.

        Parameters
        ----------
        structure:
            Output of the repo analyser: contains tech_stack, entry_points,
            file_count, etc.
        parsed:
            List of ParsedFile objects (or dicts) produced by the compiler.
            Each item exposes: id, path, language, loc, docstring,
            classes (list), functions (list).
        graph_data:
            Output of the graph analyser.  Expected keys:
                ``edges``    – list of {source, target, type} dicts
                ``hotspots`` – dict mapping node_id → pagerank score
        """
        self._upsert_repo(structure)

        for parsed_file in parsed:
            self._upsert_file(parsed_file)
            self._upsert_classes(parsed_file)
            self._upsert_functions(parsed_file)

        self._upsert_edges(graph_data)
        self._apply_pagerank(graph_data)

        if decision_points:
            self._upsert_decision_points(decision_points)
        if domain_concepts:
            self._upsert_domain_concepts(domain_concepts)

        logger.info(
            "Graph ingest complete for repo '%s': %d files processed.",
            self.repo_name,
            len(parsed),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # --- Repo ----------------------------------------------------------

    def _upsert_repo(self, structure: dict[str, Any]) -> None:
        query = (
            "MERGE (r:Repo {name: $name}) "
            "SET r.path        = $path, "
            "    r.tech_stack  = $tech_stack, "
            "    r.entry_points = $entry_points, "
            "    r.file_count  = $file_count, "
            "    r.readme      = $readme"
        )
        file_stats = structure.get("file_stats", {})
        params = {
            "name": self.repo_name,
            "path": self.repo_path,
            "tech_stack": structure.get("tech_stack", []),
            "entry_points": structure.get("entry_points", []),
            "file_count": file_stats.get("total_files", structure.get("file_count", 0)),
            "readme": structure.get("readme_content", ""),
        }
        neo4j_client.run(query, params)

    # --- Files ---------------------------------------------------------

    def _upsert_file(self, pf: Any) -> None:
        pf_dict = pf if isinstance(pf, dict) else vars(pf)
        file_id = pf_dict.get("id") or pf_dict.get("path")

        # MERGE the File node
        query = (
            "MERGE (f:File {id: $id}) "
            "SET f.path      = $path, "
            "    f.language  = $language, "
            "    f.loc       = $loc, "
            "    f.docstring = $docstring "
            "WITH f "
            "MATCH (r:Repo {name: $repo_name}) "
            "MERGE (r)-[:CONTAINS]->(f)"
        )
        params = {
            "id": file_id,
            "path": pf_dict.get("path", ""),
            "language": pf_dict.get("language", ""),
            "loc": pf_dict.get("loc", 0),
            "docstring": pf_dict.get("docstring", ""),
            "repo_name": self.repo_name,
        }
        neo4j_client.run(query, params)

    # --- Classes -------------------------------------------------------

    def _upsert_classes(self, pf: Any) -> None:
        pf_dict = pf if isinstance(pf, dict) else vars(pf)
        file_id = pf_dict.get("id") or pf_dict.get("path")
        classes = pf_dict.get("classes", []) or []

        for cls in classes:
            cls_dict = cls if isinstance(cls, dict) else vars(cls)
            class_id = cls_dict.get("id") or f"{file_id}::{cls_dict.get('name', '')}"

            query = (
                "MERGE (c:Class {id: $id}) "
                "SET c.name      = $name, "
                "    c.file      = $file, "
                "    c.bases     = $bases, "
                "    c.docstring = $docstring "
                "WITH c "
                "MATCH (f:File {id: $file_id}) "
                "MERGE (f)-[:CONTAINS]->(c)"
            )
            params = {
                "id": class_id,
                "name": cls_dict.get("name", ""),
                "file": file_id,
                "bases": cls_dict.get("bases", []),
                "docstring": cls_dict.get("docstring", ""),
                "file_id": file_id,
            }
            neo4j_client.run(query, params)

            # Ingest methods belonging to this class
            for method in cls_dict.get("methods", []) or []:
                self._upsert_method(method, file_id=file_id, class_id=class_id)

    # --- Functions / Methods -------------------------------------------

    def _upsert_functions(self, pf: Any) -> None:
        """Ingest top-level functions (not belonging to a class)."""
        pf_dict = pf if isinstance(pf, dict) else vars(pf)
        file_id = pf_dict.get("id") or pf_dict.get("path")
        functions = pf_dict.get("functions", []) or []

        for fn in functions:
            fn_dict = fn if isinstance(fn, dict) else vars(fn)
            fn_id = fn_dict.get("id") or f"{file_id}::{fn_dict.get('name', '')}"

            query = (
                "MERGE (fn:Function {id: $id}) "
                "SET fn.name        = $name, "
                "    fn.file        = $file, "
                "    fn.params      = $params, "
                "    fn.return_type = $return_type, "
                "    fn.docstring   = $docstring, "
                "    fn.decorators  = $decorators "
                "WITH fn "
                "MATCH (f:File {id: $file_id}) "
                "MERGE (f)-[:CONTAINS]->(fn)"
            )
            params = {
                "id": fn_id,
                "name": fn_dict.get("name", ""),
                "file": file_id,
                "params": fn_dict.get("params", []),
                "return_type": fn_dict.get("return_type", ""),
                "docstring": fn_dict.get("docstring", ""),
                "decorators": fn_dict.get("decorators", []),
                "file_id": file_id,
            }
            neo4j_client.run(query, params)

    def _upsert_method(
        self,
        method: Any,
        *,
        file_id: str,
        class_id: str,
    ) -> None:
        """Ingest a method and attach it to its parent class."""
        fn_dict = method if isinstance(method, dict) else vars(method)
        fn_id = fn_dict.get("id") or f"{class_id}::{fn_dict.get('name', '')}"

        query = (
            "MERGE (fn:Function {id: $id}) "
            "SET fn.name        = $name, "
            "    fn.file        = $file, "
            "    fn.params      = $params, "
            "    fn.return_type = $return_type, "
            "    fn.docstring   = $docstring, "
            "    fn.decorators  = $decorators "
            "WITH fn "
            "MATCH (c:Class {id: $class_id}) "
            "MERGE (c)-[:CONTAINS]->(fn)"
        )
        params = {
            "id": fn_id,
            "name": fn_dict.get("name", ""),
            "file": file_id,
            "params": fn_dict.get("params", []),
            "return_type": fn_dict.get("return_type", ""),
            "docstring": fn_dict.get("docstring", ""),
            "decorators": fn_dict.get("decorators", []),
            "class_id": class_id,
        }
        neo4j_client.run(query, params)

    # --- Edges ---------------------------------------------------------

    def _upsert_edges(self, graph_data: dict[str, Any]) -> None:
        """Create IMPORTS and CALLS edges from graph_data."""
        edges: list[dict[str, Any]] = graph_data.get("edges", []) or []

        import_query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[:IMPORTS]->(b)"
        )
        call_query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[:CALLS]->(b)"
        )

        for edge in edges:
            edge_type = (edge.get("type") or "").upper()
            source = edge.get("source")
            target = edge.get("target")

            if not source or not target:
                logger.debug("Skipping edge with missing source/target: %s", edge)
                continue

            edge_params = {"source": source, "target": target}

            if edge_type == "IMPORTS":
                neo4j_client.run(import_query, edge_params)
            elif edge_type == "CALLS":
                neo4j_client.run(call_query, edge_params)
            else:
                logger.debug("Unknown edge type '%s'; skipping.", edge_type)

    # --- PageRank ------------------------------------------------------

    def _apply_pagerank(self, graph_data: dict[str, Any]) -> None:
        """Write pre-computed pagerank scores onto nodes."""
        hotspots: dict[str, float] = graph_data.get("hotspots", {}) or {}

        if not hotspots:
            return

        query = (
            "MATCH (n {id: $id}) "
            "SET n.pagerank = $score"
        )
        for node_id, score in hotspots.items():
            neo4j_client.run(query, {"id": node_id, "score": float(score)})

    # --- Decision Points -----------------------------------------------

    def _upsert_decision_points(self, decision_points: list[dict]) -> None:
        """Create DecisionPoint nodes linked to their functions."""
        dp_query = (
            "MERGE (dp:DecisionPoint {id: $id}) "
            "SET dp.line             = $line, "
            "    dp.condition        = $condition, "
            "    dp.condition_type   = $condition_type, "
            "    dp.explanation      = $explanation, "
            "    dp.question_for_human = $question, "
            "    dp.file             = $file "
            "WITH dp "
            "MATCH (fn:Function {id: $function_id}) "
            "MERGE (fn)-[:HAS_DECISION]->(dp)"
        )

        for dp in decision_points:
            params = {
                "id": dp["id"],
                "line": dp.get("line", 0),
                "condition": dp.get("condition", ""),
                "condition_type": dp.get("condition_type", ""),
                "explanation": dp.get("explanation", ""),
                "question": dp.get("question_for_human", ""),
                "file": dp.get("file", ""),
                "function_id": dp.get("function_id", ""),
            }
            neo4j_client.run(dp_query, params)

        logger.info("Upserted %d decision points into Neo4j.", len(decision_points))

    # --- Domain Concepts -----------------------------------------------

    def _upsert_domain_concepts(self, domain_concepts: list[dict]) -> None:
        """Create DomainConcept nodes linked to related classes."""
        dc_query = (
            "MERGE (dc:DomainConcept {id: $id}) "
            "SET dc.name        = $name, "
            "    dc.type        = $type, "
            "    dc.description = $description"
        )
        link_query = (
            "MATCH (dc:DomainConcept {id: $dc_id}), (c:Class) "
            "WHERE c.name = $class_name "
            "MERGE (dc)-[:REPRESENTS]->(c)"
        )

        for dc in domain_concepts:
            params = {
                "id": dc["id"],
                "name": dc.get("name", ""),
                "type": dc.get("type", "entity"),
                "description": dc.get("description", ""),
            }
            neo4j_client.run(dc_query, params)

            for class_name in dc.get("related_classes", []):
                neo4j_client.run(link_query, {"dc_id": dc["id"], "class_name": class_name})

        logger.info("Upserted %d domain concepts into Neo4j.", len(domain_concepts))
