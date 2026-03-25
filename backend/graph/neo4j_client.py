"""Neo4j client singleton for the context builder."""

from __future__ import annotations

import logging
import os
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Singleton Neo4j driver wrapper."""

    def __init__(self) -> None:
        self._driver = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to Neo4j using environment variables.

        Environment variables:
            NEO4J_URI      – bolt URI (default: bolt://localhost:7687)
            NEO4J_USER     – username  (default: neo4j)
            NEO4J_PASSWORD – password  (default: contextbuilder)
        """
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "contextbuilder")

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Eagerly verify connectivity so errors surface here, not later.
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", uri)
        except (ServiceUnavailable, AuthError) as exc:
            self._driver = None
            logger.error("Failed to connect to Neo4j: %s", exc)
            raise

    def close(self) -> None:
        """Close the Neo4j driver and release all connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    def is_connected(self) -> bool:
        """Return True if the driver is initialised and reachable."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def run(
        self,
        query: str | tuple[str, dict],
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts.

        Accepts two calling conventions:

            client.run("MATCH (n) RETURN n", {"key": "val"})
            client.run(("MATCH (n) RETURN n", {"key": "val"}))

        Parameters
        ----------
        query:
            Either a plain Cypher string *or* a ``(cypher, params)`` tuple
            produced by the query-builder helpers in ``queries.py``.
        params:
            Optional parameter dict used when *query* is a plain string.
            Ignored when *query* is already a tuple.
        """
        if isinstance(query, tuple):
            cypher, params = query[0], query[1] if len(query) > 1 else {}
        else:
            cypher = query
            params = params or {}

        if self._driver is None:
            raise RuntimeError("Neo4j driver is not connected. Call connect() first.")

        with self._driver.session() as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints and full-text index if absent.

        Safe to call multiple times (uses IF NOT EXISTS where supported).
        """
        constraints: list[str] = [
            # Node-key / uniqueness constraints
            (
                "CREATE CONSTRAINT file_id_unique IF NOT EXISTS "
                "FOR (f:File) REQUIRE f.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT class_id_unique IF NOT EXISTS "
                "FOR (c:Class) REQUIRE c.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT function_id_unique IF NOT EXISTS "
                "FOR (fn:Function) REQUIRE fn.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT repo_name_unique IF NOT EXISTS "
                "FOR (r:Repo) REQUIRE r.name IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT decision_point_id_unique IF NOT EXISTS "
                "FOR (dp:DecisionPoint) REQUIRE dp.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT domain_concept_id_unique IF NOT EXISTS "
                "FOR (dc:DomainConcept) REQUIRE dc.id IS UNIQUE"
            ),
        ]

        fulltext_index = (
            "CREATE FULLTEXT INDEX nodeSearch IF NOT EXISTS "
            "FOR (n:File|Class|Function|BusinessRule|DomainConcept|DecisionPoint) "
            "ON EACH [n.name, n.summary, n.content]"
        )

        for stmt in constraints:
            try:
                self.run(stmt)
                logger.debug("Constraint applied: %s", stmt.split("CONSTRAINT")[1].split("IF")[0].strip())
            except Exception as exc:
                logger.warning("Could not apply constraint: %s", exc)

        try:
            self.run(fulltext_index)
            logger.debug("Full-text index 'nodeSearch' ensured.")
        except Exception as exc:
            logger.warning("Could not create full-text index: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

neo4j_client = Neo4jClient()
