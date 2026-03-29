"""Business layer Neo4j persistence helpers.

Supplements the inline persistence in BusinessLogicExtractor._persist_rule()
with stale-edge cleanup and a convenience wrapper for callers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


def cleanup_stale_enforced_by_edges() -> int:
    """Delete ENFORCED_BY edges whose target Function node no longer exists.

    Called after every graph build to remove dangling edges left behind by
    renamed or deleted functions.

    Returns the number of edges deleted.
    """
    if not neo4j_client.is_connected():
        return 0

    try:
        result = neo4j_client.run(
            "MATCH (br:BusinessRule)-[e:ENFORCED_BY]->(f:Function) "
            "WHERE NOT EXISTS { MATCH (f2:Function {id: f.id}) } "
            "DELETE e "
            "RETURN count(e) AS deleted"
        )
        deleted = result[0].get("deleted", 0) if result else 0
        if deleted:
            logger.info("Cleaned up %d stale ENFORCED_BY edge(s)", deleted)
        return deleted
    except Exception as exc:
        logger.warning("Stale ENFORCED_BY cleanup failed (non-fatal): %s", exc)
        return 0


def persist_business_rules_to_neo4j(repo_name: str, parsed: list[dict]) -> int:
    """Extract and persist BusinessRule nodes + ENFORCED_BY edges to Neo4j.

    Convenience wrapper around BusinessLogicExtractor.extract() used by
    build pipelines that want a single call after Function nodes exist.

    Parameters
    ----------
    repo_name:
        Repo name as stored in the Repo node.
    parsed:
        Parsed file records (same format as CodeParser.parse_all() output).

    Returns
    -------
    int
        Number of new BusinessRule nodes created.
    """
    from enricher.business_logic import BusinessLogicExtractor

    extractor = BusinessLogicExtractor(repo_name, parsed)
    count = extractor.extract()
    cleanup_stale_enforced_by_edges()
    return count
