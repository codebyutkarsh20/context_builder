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
        incremental: bool = True,
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
        incremental:
            When True (default), snapshot existing node IDs before ingest and
            delete any nodes that were not seen in this run (i.e. from deleted
            files/classes/functions).
        """
        # Reset seen-ID tracking for this run
        self._seen_ids: dict[str, set[str]] = {"File": set(), "Class": set(), "Function": set()}

        # Snapshot existing nodes before touching the graph (incremental only)
        snapshot = self._snapshot_existing_node_ids() if incremental else {}

        self._upsert_repo(structure)

        # Batch upserts — collect all nodes first, then write in bulk UNWIND
        # queries. ~100x faster than individual MERGE calls for large repos.
        self._batch_upsert_files(parsed)
        self._batch_upsert_classes(parsed)
        self._batch_upsert_functions(parsed)

        self._batch_upsert_edges(graph_data)
        self._batch_apply_pagerank(graph_data)

        if decision_points:
            self._batch_upsert_decision_points(decision_points)
        if domain_concepts:
            self._batch_upsert_domain_concepts(domain_concepts)

        # Remove nodes that disappeared from the repo since the last ingest
        if incremental and snapshot:
            deleted = self._delete_stale_nodes(snapshot, self._seen_ids)
            total_deleted = sum(deleted.values())
            if total_deleted > 0:
                logger.info(
                    "Incremental update: deleted %d stale nodes (%s)",
                    total_deleted,
                    ", ".join(f"{v} {k}" for k, v in deleted.items() if v > 0),
                )

        logger.info(
            "Graph ingest complete for repo '%s': %d files processed.",
            self.repo_name,
            len(parsed),
        )

    # ------------------------------------------------------------------
    # Incremental update helpers
    # ------------------------------------------------------------------

    def _snapshot_existing_node_ids(self) -> dict[str, set[str]]:
        """Return all current node IDs grouped by label: {File: {...}, Class: {...}, Function: {...}}"""
        snapshot: dict[str, set[str]] = {"File": set(), "Class": set(), "Function": set()}

        for label in ("File", "Class", "Function"):
            query = f"""
                MATCH (r:Repo {{name: $repo_name}})-[:CONTAINS*1..3]->(n:{label})
                RETURN n.id AS id
            """
            try:
                results = neo4j_client.run(query, {"repo_name": self.repo_name})
                for record in (results or []):
                    nid = record.get("id") or record.get("n.id")
                    if nid:
                        snapshot[label].add(str(nid))
            except Exception as e:
                logger.warning("Failed to snapshot %s nodes: %s", label, e)

        return snapshot

    def _track_seen(self, label: str, node_id: str) -> None:
        """Record that this node was seen in the current ingest run."""
        self._seen_ids.setdefault(label, set()).add(str(node_id))

    def _delete_stale_nodes(
        self,
        snapshot: dict[str, set[str]],
        seen_ids: dict[str, set[str]],
    ) -> dict[str, int]:
        """Delete nodes that were in the graph but not seen in this ingest run."""
        deleted_counts: dict[str, int] = {}

        for label in ("Function", "Class", "File"):  # Delete children before parents
            stale_ids = snapshot.get(label, set()) - seen_ids.get(label, set())
            if not stale_ids:
                deleted_counts[label] = 0
                continue

            logger.info(
                "Deleting %d stale %s nodes for repo '%s'",
                len(stale_ids), label, self.repo_name,
            )

            # Delete in batches of 100 to avoid large transactions
            stale_list = list(stale_ids)
            total_deleted = 0
            for i in range(0, len(stale_list), 100):
                batch = stale_list[i:i + 100]
                try:
                    # DETACH DELETE removes the node and all its relationships
                    query = f"""
                        MATCH (n:{label})
                        WHERE n.id IN $ids
                        DETACH DELETE n
                    """
                    neo4j_client.run(query, {"ids": batch})
                    total_deleted += len(batch)
                except Exception as e:
                    logger.warning("Failed to delete stale %s batch: %s", label, e)

            deleted_counts[label] = total_deleted

        return deleted_counts

    def get_stale_nodes(self, days_old: int = 7) -> list[dict]:
        """Return nodes not seen in the last N days (may indicate deleted code)."""
        query = """
            MATCH (r:Repo {name: $repo_name})-[:CONTAINS*1..3]->(n)
            WHERE n.last_seen IS NOT NULL
              AND n.last_seen < datetime() - duration({days: $days})
            RETURN labels(n)[0] AS label, n.id AS id, n.name AS name, n.last_seen AS last_seen
            LIMIT 100
        """
        try:
            results = neo4j_client.run(query, {"repo_name": self.repo_name, "days": days_old})
            return [dict(r) for r in (results or [])]
        except Exception as e:
            logger.warning("Failed to query stale nodes: %s", e)
            return []

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

    # --- Batched upserts (preferred for large repos) --------------------

    _BATCH_SIZE = 500

    def _batch_upsert_files(self, parsed: list[Any]) -> None:
        """Batch-upsert all File nodes using UNWIND."""
        batch: list[dict] = []
        for pf in parsed:
            pf_dict = pf if isinstance(pf, dict) else vars(pf)
            file_id = pf_dict.get("id") or pf_dict.get("path")
            batch.append({
                "id": file_id,
                "path": pf_dict.get("path", ""),
                "language": pf_dict.get("language", ""),
                "loc": pf_dict.get("loc", 0),
                "docstring": (pf_dict.get("docstring") or "")[:500],
            })
            self._track_seen("File", file_id)

        query = (
            "UNWIND $batch AS row "
            "MERGE (f:File {id: row.id}) "
            "SET f.path = row.path, f.language = row.language, "
            "    f.loc = row.loc, f.docstring = row.docstring, "
            "    f.last_seen = datetime() "
            "WITH f, row "
            "MATCH (r:Repo {name: $repo_name}) "
            "MERGE (r)-[:CONTAINS]->(f)"
        )
        for i in range(0, len(batch), self._BATCH_SIZE):
            neo4j_client.run(query, {"batch": batch[i:i + self._BATCH_SIZE], "repo_name": self.repo_name})
        logger.info("Batch-upserted %d files.", len(batch))

    def _batch_upsert_classes(self, parsed: list[Any]) -> None:
        """Batch-upsert all Class nodes using UNWIND."""
        batch: list[dict] = []
        for pf in parsed:
            pf_dict = pf if isinstance(pf, dict) else vars(pf)
            file_id = pf_dict.get("id") or pf_dict.get("path")
            for cls in pf_dict.get("classes", []) or []:
                cls_dict = cls if isinstance(cls, dict) else vars(cls)
                class_id = cls_dict.get("id") or f"{file_id}::{cls_dict.get('name', '')}"
                batch.append({
                    "id": class_id,
                    "name": cls_dict.get("name", ""),
                    "file_id": file_id,
                    "bases": cls_dict.get("bases", []),
                    "docstring": (cls_dict.get("docstring") or "")[:500],
                    "line_start": cls_dict.get("line_start", 0),
                    "line_end": cls_dict.get("line_end", 0),
                    "is_test": cls_dict.get("is_test", False),
                })
                self._track_seen("Class", class_id)

        query = (
            "UNWIND $batch AS row "
            "MERGE (c:Class {id: row.id}) "
            "SET c.name = row.name, c.file = row.file_id, "
            "    c.bases = row.bases, c.docstring = row.docstring, "
            "    c.line_start = row.line_start, c.line_end = row.line_end, "
            "    c.is_test = row.is_test, c.last_seen = datetime() "
            "WITH c, row "
            "MATCH (f:File {id: row.file_id}) "
            "MERGE (f)-[:CONTAINS]->(c)"
        )
        for i in range(0, len(batch), self._BATCH_SIZE):
            neo4j_client.run(query, {"batch": batch[i:i + self._BATCH_SIZE]})
        logger.info("Batch-upserted %d classes.", len(batch))

    def _batch_upsert_functions(self, parsed: list[Any]) -> None:
        """Batch-upsert all Function nodes (top-level + methods) using UNWIND."""
        # Collect top-level functions
        fn_batch: list[dict] = []
        method_batch: list[dict] = []
        for pf in parsed:
            pf_dict = pf if isinstance(pf, dict) else vars(pf)
            file_id = pf_dict.get("id") or pf_dict.get("path")
            for fn in pf_dict.get("functions", []) or []:
                fn_dict = fn if isinstance(fn, dict) else vars(fn)
                fn_id = fn_dict.get("id") or f"{file_id}::{fn_dict.get('name', '')}"
                fn_batch.append(self._fn_to_row(fn_dict, fn_id, file_id))
                self._track_seen("Function", fn_id)
            # Collect methods from classes
            for cls in pf_dict.get("classes", []) or []:
                cls_dict = cls if isinstance(cls, dict) else vars(cls)
                class_id = cls_dict.get("id") or f"{file_id}::{cls_dict.get('name', '')}"
                for method in cls_dict.get("methods", []) or []:
                    m_dict = method if isinstance(method, dict) else vars(method)
                    m_id = m_dict.get("id") or f"{class_id}::{m_dict.get('name', '')}"
                    row = self._fn_to_row(m_dict, m_id, file_id)
                    row["parent_id"] = class_id
                    method_batch.append(row)
                    self._track_seen("Function", m_id)

        # Top-level functions → linked to File
        fn_query = (
            "UNWIND $batch AS row "
            "MERGE (fn:Function {id: row.id}) "
            "SET fn.name = row.name, fn.file = row.file_id, "
            "    fn.params = row.params, fn.return_type = row.return_type, "
            "    fn.docstring = row.docstring, fn.decorators = row.decorators, "
            "    fn.line_start = row.line_start, fn.line_end = row.line_end, "
            "    fn.is_test = row.is_test, fn.last_seen = datetime() "
            "WITH fn, row "
            "MATCH (f:File {id: row.file_id}) "
            "MERGE (f)-[:CONTAINS]->(fn)"
        )
        for i in range(0, len(fn_batch), self._BATCH_SIZE):
            neo4j_client.run(fn_query, {"batch": fn_batch[i:i + self._BATCH_SIZE]})

        # Methods → linked to Class
        method_query = (
            "UNWIND $batch AS row "
            "MERGE (fn:Function {id: row.id}) "
            "SET fn.name = row.name, fn.file = row.file_id, "
            "    fn.params = row.params, fn.return_type = row.return_type, "
            "    fn.docstring = row.docstring, fn.decorators = row.decorators, "
            "    fn.line_start = row.line_start, fn.line_end = row.line_end, "
            "    fn.is_test = row.is_test, fn.last_seen = datetime() "
            "WITH fn, row "
            "MATCH (c:Class {id: row.parent_id}) "
            "MERGE (c)-[:CONTAINS]->(fn)"
        )
        for i in range(0, len(method_batch), self._BATCH_SIZE):
            neo4j_client.run(method_query, {"batch": method_batch[i:i + self._BATCH_SIZE]})

        logger.info("Batch-upserted %d functions + %d methods.", len(fn_batch), len(method_batch))

    @staticmethod
    def _fn_to_row(fn_dict: dict, fn_id: str, file_id: str) -> dict:
        return {
            "id": fn_id,
            "name": fn_dict.get("name", ""),
            "file_id": file_id,
            "params": fn_dict.get("params", []),
            "return_type": fn_dict.get("return_type", ""),
            "docstring": (fn_dict.get("docstring") or "")[:500],
            "decorators": fn_dict.get("decorators", []),
            "line_start": fn_dict.get("line_start", 0),
            "line_end": fn_dict.get("line_end", 0),
            "is_test": fn_dict.get("is_test", False),
        }

    def _batch_upsert_edges(self, graph_data: dict[str, Any]) -> None:
        """Batch-upsert edges using UNWIND, grouped by type."""
        edges = graph_data.get("edges", []) or []
        by_type: dict[str, list[dict]] = {}
        for edge in edges:
            etype = (edge.get("type") or "").upper()
            if etype == "CONTAINS":
                continue  # Handled during node upserts
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                by_type.setdefault(etype, []).append({"source": source, "target": target})

        type_to_rel = {"IMPORTS": "IMPORTS", "CALLS": "CALLS", "INHERITS": "INHERITS", "TESTED_BY": "TESTED_BY"}
        for etype, items in by_type.items():
            rel = type_to_rel.get(etype)
            if not rel:
                continue
            query = (
                f"UNWIND $batch AS row "
                f"MATCH (a {{id: row.source}}), (b {{id: row.target}}) "
                f"MERGE (a)-[:{rel}]->(b)"
            )
            for i in range(0, len(items), self._BATCH_SIZE):
                neo4j_client.run(query, {"batch": items[i:i + self._BATCH_SIZE]})
            logger.info("Batch-upserted %d %s edges.", len(items), etype)

    def _batch_apply_pagerank(self, graph_data: dict[str, Any]) -> None:
        """Write pre-computed pagerank scores using UNWIND."""
        raw = graph_data.get("hotspots") or []
        if isinstance(raw, dict):
            scores = [{"id": k, "score": float(v)} for k, v in raw.items()]
        elif isinstance(raw, list) and raw:
            scores = [{"id": e["id"], "score": float(e.get("pagerank", 0))} for e in raw if "id" in e]
        else:
            nodes = graph_data.get("nodes") or []
            scores = [{"id": n["id"], "score": float(n.get("pagerank", 0))}
                      for n in nodes if "id" in n and n.get("pagerank") is not None]
        if not scores:
            return
        query = "UNWIND $batch AS row MATCH (n {id: row.id}) SET n.pagerank = row.score"
        for i in range(0, len(scores), self._BATCH_SIZE):
            neo4j_client.run(query, {"batch": scores[i:i + self._BATCH_SIZE]})
        logger.info("Batch-applied pagerank to %d nodes.", len(scores))

    def _batch_upsert_decision_points(self, decision_points: list[dict]) -> None:
        """Batch-upsert DecisionPoint nodes using UNWIND."""
        batch = [{
            "id": dp["id"],
            "line": dp.get("line", 0),
            "condition": dp.get("condition", ""),
            "condition_type": dp.get("condition_type", ""),
            "explanation": dp.get("explanation", ""),
            "question": dp.get("question_for_human", ""),
            "file": dp.get("file", ""),
            "function_id": dp.get("function_id", ""),
        } for dp in decision_points]

        query = (
            "UNWIND $batch AS row "
            "MERGE (dp:DecisionPoint {id: row.id}) "
            "SET dp.line = row.line, dp.condition = row.condition, "
            "    dp.condition_type = row.condition_type, dp.explanation = row.explanation, "
            "    dp.question_for_human = row.question, dp.file = row.file "
            "WITH dp, row "
            "MATCH (fn:Function {id: row.function_id}) "
            "MERGE (fn)-[:HAS_DECISION]->(dp)"
        )
        for i in range(0, len(batch), self._BATCH_SIZE):
            neo4j_client.run(query, {"batch": batch[i:i + self._BATCH_SIZE]})
        logger.info("Batch-upserted %d decision points.", len(batch))

    def _batch_upsert_domain_concepts(self, domain_concepts: list[dict]) -> None:
        """Batch-upsert DomainConcept nodes using UNWIND."""
        batch = [{
            "id": dc["id"],
            "name": dc.get("name", ""),
            "type": dc.get("type", "entity"),
            "description": dc.get("description", ""),
        } for dc in domain_concepts]

        query = (
            "UNWIND $batch AS row "
            "MERGE (dc:DomainConcept {id: row.id}) "
            "SET dc.name = row.name, dc.type = row.type, dc.description = row.description"
        )
        for i in range(0, len(batch), self._BATCH_SIZE):
            neo4j_client.run(query, {"batch": batch[i:i + self._BATCH_SIZE]})

        # Link to classes separately
        link_batch: list[dict] = []
        for dc in domain_concepts:
            for cls_name in dc.get("related_classes", []):
                link_batch.append({"dc_id": dc["id"], "class_name": cls_name})
        if link_batch:
            link_query = (
                "UNWIND $batch AS row "
                "MATCH (dc:DomainConcept {id: row.dc_id}), (c:Class) "
                "WHERE c.name = row.class_name "
                "MERGE (dc)-[:REPRESENTS]->(c)"
            )
            for i in range(0, len(link_batch), self._BATCH_SIZE):
                neo4j_client.run(link_query, {"batch": link_batch[i:i + self._BATCH_SIZE]})

        logger.info("Batch-upserted %d domain concepts.", len(batch))

    # --- Legacy single-item upserts (kept for compatibility) -----------

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
            "    f.docstring = $docstring, "
            "    f.last_seen = datetime() "
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
        self._track_seen("File", file_id)

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
                "    c.docstring = $docstring, "
                "    c.line_start = $line_start, "
                "    c.line_end   = $line_end, "
                "    c.is_test    = $is_test, "
                "    c.last_seen = datetime() "
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
                "line_start": cls_dict.get("line_start", 0),
                "line_end": cls_dict.get("line_end", 0),
                "is_test": cls_dict.get("is_test", False),
                "file_id": file_id,
            }
            neo4j_client.run(query, params)
            self._track_seen("Class", class_id)

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
                "    fn.decorators  = $decorators, "
                "    fn.line_start  = $line_start, "
                "    fn.line_end    = $line_end, "
                "    fn.is_test     = $is_test, "
                "    fn.last_seen   = datetime() "
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
                "line_start": fn_dict.get("line_start", 0),
                "line_end": fn_dict.get("line_end", 0),
                "is_test": fn_dict.get("is_test", False),
                "file_id": file_id,
            }
            neo4j_client.run(query, params)
            self._track_seen("Function", fn_id)

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
            "    fn.decorators  = $decorators, "
            "    fn.line_start  = $line_start, "
            "    fn.line_end    = $line_end, "
            "    fn.is_test     = $is_test, "
            "    fn.last_seen   = datetime() "
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
            "line_start": fn_dict.get("line_start", 0),
            "line_end": fn_dict.get("line_end", 0),
            "is_test": fn_dict.get("is_test", False),
            "class_id": class_id,
        }
        neo4j_client.run(query, params)
        self._track_seen("Function", fn_id)

    # --- Edges ---------------------------------------------------------

    def _upsert_edges(self, graph_data: dict[str, Any]) -> None:
        """Create IMPORTS, CALLS, INHERITS, and TESTED_BY edges from graph_data.

        CONTAINS edges are intentionally excluded here — they are created
        as part of the node upsert methods (_upsert_file, _upsert_classes,
        _upsert_method) to maintain structural integrity.
        """
        edges: list[dict[str, Any]] = graph_data.get("edges", []) or []

        import_query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[:IMPORTS]->(b)"
        )
        call_query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[:CALLS]->(b)"
        )
        inherits_query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[:INHERITS]->(b)"
        )
        tested_by_query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            "MERGE (a)-[:TESTED_BY]->(b)"
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
            elif edge_type == "INHERITS":
                neo4j_client.run(inherits_query, edge_params)
            elif edge_type == "TESTED_BY":
                neo4j_client.run(tested_by_query, edge_params)
            elif edge_type == "CONTAINS":
                pass  # Handled during node upsert; skip to avoid duplicates.
            else:
                logger.debug("Unknown edge type '%s'; skipping.", edge_type)

    # --- PageRank ------------------------------------------------------

    def _apply_pagerank(self, graph_data: dict[str, Any]) -> None:
        """Write pre-computed pagerank scores onto nodes.

        Handles both data shapes produced by CallGraphBuilder:
          - ``hotspots`` is a list of dicts: [{id, pagerank, ...}, ...]
          - ``nodes``    is a list of dicts: [{id, pagerank, ...}, ...]
        Falls back to the full ``nodes`` list when hotspots is empty.
        """
        raw = graph_data.get("hotspots") or []

        # CallGraphBuilder._compute_hotspots returns list[dict], not dict[str, float].
        # Build a (node_id → score) mapping regardless of the input shape.
        if isinstance(raw, dict):
            # Legacy dict format: {node_id: score}
            scores: dict[str, float] = {k: float(v) for k, v in raw.items()}
        elif isinstance(raw, list) and raw:
            # Current format: [{id: ..., pagerank: ...}, ...]
            scores = {entry["id"]: float(entry.get("pagerank", 0)) for entry in raw if "id" in entry}
        else:
            # hotspots empty — fall back to full nodes list for pagerank scores
            nodes = graph_data.get("nodes") or []
            scores = {
                n["id"]: float(n.get("pagerank", 0))
                for n in nodes
                if "id" in n and n.get("pagerank") is not None
            }

        if not scores:
            return

        query = (
            "MATCH (n {id: $id}) "
            "SET n.pagerank = $score"
        )
        for node_id, score in scores.items():
            neo4j_client.run(query, {"id": node_id, "score": score})

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
