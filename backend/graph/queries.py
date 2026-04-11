"""Cypher query builders for the context builder graph API.

Each function returns either a plain Cypher string or a (cypher, params) tuple
that can be passed directly to ``neo4j_client.run()``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Repo listing
# ---------------------------------------------------------------------------


def list_repos_query() -> str:
    """Return a Cypher query that fetches all Repo nodes."""
    return (
        "MATCH (r:Repo) "
        "RETURN r.name AS name, r.path AS path, "
        "       r.tech_stack AS tech_stack, r.entry_points AS entry_points, "
        "       r.file_count AS file_count "
        "ORDER BY r.name"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

#: Maps layer names to the Neo4j label expressions used in queries.
_LAYER_LABELS: dict[str, str] = {
    "code": "File|Class|Function",
    "business": "BusinessRule|DomainConcept",
}


def get_graph_nodes(
    repo: str,
    layer: str | None,
    node_type: str | None,
    limit: int,
) -> tuple[str, dict]:
    """Return (cypher, params) that fetches graph nodes with optional filters.

    Parameters
    ----------
    repo:
        Repository name to scope the query.
    layer:
        ``"code"`` → File, Class, Function nodes;
        ``"business"`` → BusinessRule, DomainConcept nodes;
        ``None`` → all node types.
    node_type:
        Exact Neo4j label to filter on, e.g. ``"File"``.  Takes precedence
        over *layer* when both are supplied.
    limit:
        Maximum number of nodes to return.
    """
    # label_filter is only used in the node_type fast-path below.
    # The layer branch uses a parameterised WHERE clause instead, because
    # Neo4j does not support inline multi-label patterns ("n:A|B").
    label_filter = f":{node_type}" if node_type else ""

    params: dict = {"repo": repo, "limit": limit}

    if node_type:
        # Fast path: single label via pattern.
        cypher = (
            f"MATCH (r:Repo {{name: $repo}})-[:CONTAINS*1..]->(n:{node_type}) "
            "RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
            "       n.path AS path, n.language AS language, "
            "       n.docstring AS docstring, n.summary AS summary, "
            "       n.pagerank AS pagerank "
            "ORDER BY coalesce(n.pagerank, 0.0) DESC "
            "LIMIT $limit"
        )
    elif layer and layer in _LAYER_LABELS:
        labels = _LAYER_LABELS[layer].split("|")
        # Build WHERE clause: any(l IN labels(n) WHERE l IN [...])
        params["allowed_labels"] = labels
        cypher = (
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n) "
            "WHERE any(l IN labels(n) WHERE l IN $allowed_labels) "
            "RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
            "       n.path AS path, n.language AS language, "
            "       n.docstring AS docstring, n.summary AS summary, "
            "       n.pagerank AS pagerank "
            "ORDER BY coalesce(n.pagerank, 0.0) DESC "
            "LIMIT $limit"
        )
    else:
        cypher = (
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n) "
            "RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
            "       n.path AS path, n.language AS language, "
            "       n.docstring AS docstring, n.summary AS summary, "
            "       n.pagerank AS pagerank "
            "ORDER BY coalesce(n.pagerank, 0.0) DESC "
            "LIMIT $limit"
        )

    return cypher, params


# ---------------------------------------------------------------------------
# Graph edges
# ---------------------------------------------------------------------------


def get_graph_edges(
    repo: str,
    layer: str | None,
    limit: int,
) -> tuple[str, dict]:
    """Return (cypher, params) that fetches edges as {source_id, target_id, type}.

    Parameters
    ----------
    repo:
        Repository name to scope the query.
    layer:
        Optional layer filter (same semantics as :func:`get_graph_nodes`).
    limit:
        Maximum number of edges to return.
    """
    params: dict = {"repo": repo, "limit": limit}

    if layer and layer in _LAYER_LABELS:
        labels = _LAYER_LABELS[layer].split("|")
        params["allowed_labels"] = labels
        cypher = (
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(a) "
            "WHERE any(l IN labels(a) WHERE l IN $allowed_labels) "
            "MATCH (a)-[rel]->(b) "
            "WHERE any(l IN labels(b) WHERE l IN $allowed_labels) "
            "  AND type(rel) <> 'CONTAINS' "
            "RETURN a.id AS source_id, b.id AS target_id, type(rel) AS type "
            "LIMIT $limit"
        )
    else:
        cypher = (
            "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(a) "
            "MATCH (a)-[rel]->(b) "
            "WHERE type(rel) <> 'CONTAINS' "
            "RETURN a.id AS source_id, b.id AS target_id, type(rel) AS type "
            "LIMIT $limit"
        )

    return cypher, params


# ---------------------------------------------------------------------------
# Hotspots
# ---------------------------------------------------------------------------


def get_hotspots(repo: str, top_n: int) -> tuple[str, dict]:
    """Return (cypher, params) for the top-N nodes by PageRank score.

    Parameters
    ----------
    repo:
        Repository name.
    top_n:
        Number of hotspot nodes to return.
    """
    cypher = (
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n) "
        "WHERE n.pagerank IS NOT NULL "
        "RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
        "       n.path AS path, n.pagerank AS pagerank "
        "ORDER BY n.pagerank DESC "
        "LIMIT $top_n"
    )
    return cypher, {"repo": repo, "top_n": top_n}


# ---------------------------------------------------------------------------
# Node detail
# ---------------------------------------------------------------------------


def get_node_detail(node_id: str, repo: str) -> tuple[str, dict]:
    """Return (cypher, params) for a single node and its 1-hop neighbourhood.

    The result set contains the focal node plus all directly connected nodes
    and the relationship types linking them.

    Parameters
    ----------
    node_id:
        The ``id`` property of the target node.
    repo:
        Repository name used to verify ownership.
    """
    cypher = (
        # Verify the node belongs to the given repo.
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n {id: $node_id}) "
        # Collect 1-hop neighbours (both directions).
        "OPTIONAL MATCH (n)-[out_rel]->(neighbour_out) "
        "OPTIONAL MATCH (neighbour_in)-[in_rel]->(n) "
        "RETURN "
        "  n.id          AS id, "
        "  n.name        AS name, "
        "  labels(n)     AS labels, "
        "  n.path        AS path, "
        "  n.language    AS language, "
        "  n.loc         AS loc, "
        "  n.docstring   AS docstring, "
        "  n.summary     AS summary, "
        "  n.content     AS content, "
        "  n.pagerank    AS pagerank, "
        "  collect(DISTINCT { "
        "    id: neighbour_out.id, "
        "    name: neighbour_out.name, "
        "    labels: labels(neighbour_out), "
        "    direction: 'outgoing', "
        "    rel_type: type(out_rel) "
        "  }) AS outgoing_neighbours, "
        "  collect(DISTINCT { "
        "    id: neighbour_in.id, "
        "    name: neighbour_in.name, "
        "    labels: labels(neighbour_in), "
        "    direction: 'incoming', "
        "    rel_type: type(in_rel) "
        "  }) AS incoming_neighbours"
    )
    return cypher, {"node_id": node_id, "repo": repo}


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------


def get_graph_stats(repo: str) -> tuple[str, dict]:
    """Return (cypher, params) for aggregate statistics about a repository graph.

    Result keys
    -----------
    file_count, class_count, function_count,
    call_edge_count, import_edge_count, summary_count
    """
    cypher = (
        "MATCH (r:Repo {name: $repo}) "
        # File count + sum loc in one pass
        "OPTIONAL MATCH (r)-[:CONTAINS*1..]->(f:File) "
        "WITH r, count(DISTINCT f) AS file_count, sum(coalesce(f.loc, 0)) AS lines_of_code "
        "OPTIONAL MATCH (r)-[:CONTAINS*1..]->(c:Class) "
        "WITH r, file_count, lines_of_code, count(DISTINCT c) AS class_count "
        "OPTIONAL MATCH (r)-[:CONTAINS*1..]->(fn:Function) "
        "WITH r, file_count, lines_of_code, class_count, count(DISTINCT fn) AS function_count "
        # Edge counts
        "OPTIONAL MATCH (r)-[:CONTAINS*1..]->(a)-[:CALLS]->(b) "
        "WITH r, file_count, lines_of_code, class_count, function_count, "
        "     count(DISTINCT [a.id, b.id]) AS call_edge_count "
        "OPTIONAL MATCH (r)-[:CONTAINS*1..]->(x)-[:IMPORTS]->(y) "
        "WITH r, file_count, lines_of_code, class_count, function_count, call_edge_count, "
        "     count(DISTINCT [x.id, y.id]) AS import_edge_count "
        "RETURN "
        "  file_count, "
        "  class_count, "
        "  function_count, "
        "  call_edge_count, "
        "  import_edge_count, "
        "  lines_of_code, "
        "  r.tech_stack AS tech_stack"
    )
    return cypher, {"repo": repo}


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------


def search_nodes(repo: str, query: str, limit: int) -> tuple[str, dict]:
    """Return (cypher, params) for a full-text search over nodes.

    Uses the ``nodeSearch`` full-text index which covers the ``name``,
    ``summary``, and ``content`` properties of File, Class, Function,
    BusinessRule, and DomainConcept nodes.

    Results are scoped to *repo* and ordered by relevance score descending.

    Parameters
    ----------
    repo:
        Repository name to scope results.
    query:
        Lucene-compatible search string.
    limit:
        Maximum number of results.
    """
    cypher = (
        "CALL db.index.fulltext.queryNodes('nodeSearch', $q) "
        "YIELD node, score "
        # Scope to the requested repo
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(node) "
        "RETURN "
        "  node.id       AS id, "
        "  node.name     AS name, "
        "  labels(node)  AS labels, "
        "  node.path     AS path, "
        "  node.summary  AS summary, "
        "  node.pagerank AS pagerank, "
        "  score "
        "ORDER BY score DESC "
        "LIMIT $limit"
    )
    return cypher, {"repo": repo, "q": query, "limit": limit}


# ---------------------------------------------------------------------------
# Test coverage (TESTED_BY edges)
# ---------------------------------------------------------------------------


def get_tests_for(node_id: str, repo: str) -> tuple[str, dict]:
    """Return (cypher, params) for test files covering a given source node.

    Traverses incoming TESTED_BY edges to find test files that import or
    exercise the target function or file.

    Parameters
    ----------
    node_id:
        The ``id`` property of the source node to find tests for.
    repo:
        Repository name used to scope results.
    """
    cypher = (
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(test) "
        "-[:TESTED_BY]->(n {id: $node_id}) "
        "RETURN test.id AS id, test.name AS name, test.path AS path, "
        "       labels(test) AS labels "
        "ORDER BY test.name"
    )
    return cypher, {"node_id": node_id, "repo": repo}


# ---------------------------------------------------------------------------
# Execution flows (entry points with variable-length CALLS paths)
# ---------------------------------------------------------------------------


def get_execution_flows(repo: str, limit: int = 20) -> tuple[str, dict]:
    """Return (cypher, params) for entry-point functions and their call depth.

    Entry points are functions with framework decorators or no incoming CALLS.
    Each result includes the function, its decorators, and the length of the
    longest outgoing CALLS chain (depth).

    Parameters
    ----------
    repo:
        Repository name.
    limit:
        Maximum number of flows to return.
    """
    cypher = (
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(ep:Function) "
        "WHERE NOT ()-[:CALLS]->(ep) OR size(ep.decorators) > 0 "
        "OPTIONAL MATCH path = (ep)-[:CALLS*1..10]->(callee) "
        "WITH ep, max(length(path)) AS depth, "
        "     count(DISTINCT callee) AS node_count, "
        "     collect(DISTINCT callee.path) AS touched_files "
        "RETURN ep.id AS id, ep.name AS name, ep.path AS path, "
        "       ep.decorators AS decorators, ep.is_test AS is_test, "
        "       coalesce(depth, 0) AS depth, "
        "       coalesce(node_count, 0) AS node_count, "
        "       size(touched_files) AS file_count "
        "ORDER BY node_count DESC, depth DESC "
        "LIMIT $limit"
    )
    return cypher, {"repo": repo, "limit": limit}


# ---------------------------------------------------------------------------
# Dead code (uncalled, untested functions)
# ---------------------------------------------------------------------------


def get_dead_code(repo: str, limit: int = 50) -> tuple[str, dict]:
    """Return (cypher, params) for functions with no callers and no tests.

    Excludes dunder methods, test functions, and entry points with decorators.

    Parameters
    ----------
    repo:
        Repository name.
    limit:
        Maximum number of dead-code nodes to return.
    """
    cypher = (
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(fn:Function) "
        "WHERE NOT ()-[:CALLS]->(fn) "
        "  AND NOT ()-[:TESTED_BY]->(fn) "
        "  AND NOT fn.name STARTS WITH '__' "
        "  AND coalesce(fn.is_test, false) = false "
        "  AND size(coalesce(fn.decorators, [])) = 0 "
        "RETURN fn.id AS id, fn.name AS name, fn.path AS path, "
        "       labels(fn) AS labels, fn.line_start AS line_start "
        "ORDER BY fn.path, fn.line_start "
        "LIMIT $limit"
    )
    return cypher, {"repo": repo, "limit": limit}


# ---------------------------------------------------------------------------
# Change impact (affected nodes for given changed files)
# ---------------------------------------------------------------------------


def get_change_impact(
    repo: str, changed_files: list[str], limit: int = 30,
) -> tuple[str, dict]:
    """Return (cypher, params) for nodes in changed files with caller/test counts.

    Parameters
    ----------
    repo:
        Repository name.
    changed_files:
        List of file paths that were changed.
    limit:
        Maximum number of affected nodes to return.
    """
    cypher = (
        "MATCH (r:Repo {name: $repo})-[:CONTAINS*1..]->(n) "
        "WHERE n.path IN $files AND (n:Function OR n:Class) "
        "OPTIONAL MATCH (caller)-[:CALLS]->(n) "
        "OPTIONAL MATCH (test)-[:TESTED_BY]->(n) "
        "WITH n, count(DISTINCT caller) AS caller_count, "
        "     count(DISTINCT test) AS test_count "
        "RETURN n.id AS id, n.name AS name, n.path AS path, "
        "       labels(n) AS labels, "
        "       n.line_start AS line_start, n.line_end AS line_end, "
        "       caller_count, test_count, "
        "       CASE WHEN test_count = 0 THEN true ELSE false END AS untested "
        "ORDER BY caller_count DESC "
        "LIMIT $limit"
    )
    return cypher, {"repo": repo, "files": changed_files, "limit": limit}
