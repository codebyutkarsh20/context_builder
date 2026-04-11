"""
graph/community.py — Leiden community detection for the knowledge graph.

Clusters the call graph into named communities for scope-aware localisation.
Communities let the Scout tier map a bug ticket to a cluster in one shot,
eliminating whole-codebase search.

Edge weights: CALLS=1.0, INHERITS=0.8, IMPORTS=0.7
Algorithm: Leiden (via leidenalg + python-igraph)
Fallback: networkx greedy modularity if leidenalg not installed
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_WEIGHTS: dict[str, float] = {
    "CALLS": 1.0,
    "INHERITS": 0.8,
    "IMPORTS": 0.7,
}

_MIN_COMMUNITY_SIZE: int = 3
_MAX_COMMUNITIES: int = 15

# Common directory / file name tokens that carry no domain meaning.
_PATH_STOPWORDS: frozenset[str] = frozenset(
    {
        "src", "lib", "app", "main", "index", "init", "__init__",
        "utils", "helpers", "common", "shared", "base", "core",
        "tests", "test", "spec", "mocks", "mock", "fixtures",
        "models", "views", "controllers", "routes", "handlers",
        "migrations", "static", "assets", "scripts", "config",
        "settings", "constants", "types", "interfaces", "schemas",
        "py", "js", "ts", "go", "java", "rb",
        # Generic structural tokens that dominate in monorepos
        "backend", "frontend", "api", "v1", "v2",
        "helper", "type", "interface", "constant", "constants",
    }
)


# ===========================================================================
# Naming helpers
# ===========================================================================


def _extract_path_tokens(file_path: str) -> list[str]:
    """Extract meaningful tokens from a file path.

    Splits on directory separators, underscores, hyphens, and dots; lower-cases
    everything; and drops stopwords and single-character tokens.

    Example
    -------
    ``"auth/session_manager.py"``  →  ``["auth", "session", "manager"]``
    """
    if not file_path:
        return []

    p = Path(file_path)
    # Directory components + file stem (without extension)
    parts = list(p.parts[:-1]) + [p.stem]

    tokens: list[str] = []
    for part in parts:
        for tok in re.split(r"[^a-zA-Z0-9]+", part):
            tok = tok.lower()
            if len(tok) > 1 and tok not in _PATH_STOPWORDS:
                tokens.append(tok)

    return tokens


def _derive_community_name(
    file_paths: list[str],
    all_communities_tokens: Counter[str],
    top_n: int = 2,
) -> str:
    """Derive a human-readable community name from a list of file paths.

    Counts token frequencies within this community, then applies an IDF-style
    penalty for tokens that appear across many communities (generic tokens like
    "api" or "backend").  The score for each token is::

        score = count_in_this_community / log(1 + count_in_all_communities)

    Falls back to ``"misc"`` when no meaningful tokens are found.

    Parameters
    ----------
    file_paths:
        File paths belonging to this community.
    all_communities_tokens:
        Counter mapping each token to how many communities it appears in.
        Used to penalise ubiquitous path segments.
    top_n:
        Number of top-scoring tokens to join into the community name.
    """
    import math

    counter: Counter[str] = Counter()
    for fp in file_paths:
        for tok in _extract_path_tokens(fp):
            counter[tok] += 1

    if not counter:
        return "misc"

    scored = {
        tok: count / math.log(1 + all_communities_tokens.get(tok, 1))
        for tok, count in counter.items()
    }
    top_tokens = sorted(scored, key=scored.__getitem__, reverse=True)[:top_n]
    return "-".join(top_tokens) if top_tokens else "misc"


def _ensure_unique_names(communities: list[dict]) -> list[dict]:
    """Append a numeric suffix to duplicate community names.

    The first occurrence keeps its original name; subsequent duplicates become
    ``"auth-2"``, ``"auth-3"``, etc.
    """
    name_counts: Counter[str] = Counter(c["name"] for c in communities)
    seen: Counter[str] = Counter()
    for community in communities:
        name = community["name"]
        if name_counts[name] > 1:
            seen[name] += 1
            if seen[name] > 1:
                community["name"] = f"{name}-{seen[name]}"
    return communities


# ===========================================================================
# Graph helpers
# ===========================================================================


def _build_node_map(graph_data: dict) -> dict[str, dict]:
    """Return a ``{node_id: node_dict}`` mapping for O(1) lookup."""
    return {n["id"]: n for n in graph_data.get("nodes", []) if "id" in n}


def _collect_file_paths(node_ids: list[str], node_map: dict[str, dict]) -> list[str]:
    """Return deduplicated file paths for a list of node IDs."""
    files: list[str] = []
    seen: set[str] = set()
    for nid in node_ids:
        node = node_map.get(nid, {})
        fp = node.get("file") or node.get("path") or ""
        if fp and fp not in seen:
            files.append(fp)
            seen.add(fp)
    return files


# ===========================================================================
# Leiden detection (primary path)
# ===========================================================================


def _run_leiden(
    graph_data: dict,
    resolution: float,
) -> list[list[str]] | None:
    """Attempt community detection with leidenalg + igraph.

    Returns a list of communities (each a list of node IDs) or ``None`` when
    the required libraries are not installed.
    """
    try:
        import igraph as ig  # type: ignore[import]
        import leidenalg  # type: ignore[import]
    except ImportError:
        logger.info(
            "leidenalg / python-igraph not installed — falling back to networkx."
        )
        return None

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        return []

    node_ids: list[str] = [n["id"] for n in nodes if "id" in n]
    id_to_idx: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    g = ig.Graph(n=len(node_ids), directed=True)
    g.vs["name"] = node_ids

    edge_list: list[tuple[int, int]] = []
    weight_list: list[float] = []
    seen_edges: set[tuple[int, int]] = set()

    for edge in edges:
        src_idx = id_to_idx.get(edge.get("source", ""))
        tgt_idx = id_to_idx.get(edge.get("target", ""))
        if src_idx is None or tgt_idx is None or src_idx == tgt_idx:
            continue
        key = (src_idx, tgt_idx)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        w = EDGE_WEIGHTS.get((edge.get("type") or "").upper(), 0.5)
        edge_list.append(key)
        weight_list.append(w)

    if edge_list:
        g.add_edges(edge_list)
        g.es["weight"] = weight_list

    # ModularityVertexPartition supports directed graphs; RBConfigurationVertexPartition
    # is undirected-only.  Use CPMVertexPartition when a resolution parameter is
    # needed on a directed graph; fall back to ModularityVertexPartition otherwise.
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights="weight" if edge_list else None,
        resolution_parameter=resolution,
        seed=42,
    )

    return [[node_ids[i] for i in part] for part in partition]


# ===========================================================================
# NetworkX fallback (secondary path)
# ===========================================================================


def _run_networkx_greedy(graph_data: dict) -> list[list[str]]:
    """Community detection using networkx greedy modularity.

    Weights are respected via the ``weight`` edge attribute.  Falls back to
    treating each connected component as a single community if the algorithm
    raises an exception.
    """
    import networkx as nx  # hard dependency listed in requirements.txt
    from networkx.algorithms.community import greedy_modularity_communities

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    G: nx.Graph = nx.Graph()
    G.add_nodes_from(n["id"] for n in nodes if "id" in n)

    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if not src or not tgt or src == tgt:
            continue
        if not G.has_node(src) or not G.has_node(tgt):
            continue
        w = EDGE_WEIGHTS.get((edge.get("type") or "").upper(), 0.5)
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] = G[src][tgt].get("weight", 0.0) + w
        else:
            G.add_edge(src, tgt, weight=w)

    communities: list[list[str]] = []
    for component in nx.connected_components(G):
        sub = G.subgraph(component).copy()
        if len(sub) == 1:
            communities.append(list(sub.nodes))
            continue
        try:
            for comm in greedy_modularity_communities(sub, weight="weight"):
                communities.append(list(comm))
        except Exception as exc:
            logger.warning("greedy_modularity_communities failed on component: %s", exc)
            communities.append(list(sub.nodes))

    return communities


# ===========================================================================
# Post-processing
# ===========================================================================


def _merge_small_communities(
    communities: list[list[str]],
    min_size: int,
) -> tuple[list[list[str]], bool]:
    """Merge sub-threshold communities into a single misc bucket.

    Returns the processed list and a boolean indicating whether any communities
    were merged (i.e. whether the last element is a misc bucket).
    """
    large: list[list[str]] = []
    misc_nodes: list[str] = []

    for comm in communities:
        if len(comm) >= min_size:
            large.append(comm)
        else:
            misc_nodes.extend(comm)

    had_misc = bool(misc_nodes)
    if had_misc:
        large.append(misc_nodes)

    return large, had_misc


def _cap_communities(
    communities: list[list[str]],
    max_communities: int,
) -> list[list[str]]:
    """Ensure at most *max_communities* communities.

    The smallest excess communities are merged into the last bucket.
    """
    if len(communities) <= max_communities:
        return communities

    sorted_comms = sorted(communities, key=len, reverse=True)
    kept = sorted_comms[: max_communities - 1]
    overflow: list[str] = []
    for comm in sorted_comms[max_communities - 1 :]:
        overflow.extend(comm)
    if overflow:
        kept.append(overflow)
    return kept


def _build_community_dicts(
    communities: list[list[str]],
    node_map: dict[str, dict],
    misc_index: int | None,
) -> list[dict]:
    """Convert raw community node-ID lists into structured community dicts.

    Parameters
    ----------
    misc_index:
        Index of the misc bucket in *communities*, or ``None`` if there is
        no misc bucket.
    """
    # Pre-compute per-community token sets so _derive_community_name can apply
    # IDF weighting: tokens that appear in many communities are penalised.
    community_file_paths: list[list[str]] = [
        _collect_file_paths(node_ids, node_map) for node_ids in communities
    ]
    all_communities_tokens: Counter[str] = Counter()
    for fp_list in community_file_paths:
        # Count each token once per community (not per file) for IDF denominator.
        community_token_set: set[str] = set()
        for fp in fp_list:
            community_token_set.update(_extract_path_tokens(fp))
        for tok in community_token_set:
            all_communities_tokens[tok] += 1

    result: list[dict] = []
    for idx, node_ids in enumerate(communities):
        file_paths = community_file_paths[idx]
        name = (
            "misc"
            if idx == misc_index
            else _derive_community_name(file_paths, all_communities_tokens)
        )

        file_counter: Counter[str] = Counter()
        for nid in node_ids:
            fp = (
                node_map.get(nid, {}).get("file")
                or node_map.get(nid, {}).get("path")
                or ""
            )
            if fp:
                file_counter[fp] += 1

        result.append(
            {
                "id": idx,
                "name": name,
                "node_ids": node_ids,
                "size": len(node_ids),
                "dominant_files": [fp for fp, _ in file_counter.most_common(5)],
            }
        )
    return result


# ===========================================================================
# Public API
# ===========================================================================


def build_communities(
    graph_data: dict,
    resolution: float = 1.0,
) -> list[dict]:
    """Detect communities in the knowledge graph using the Leiden algorithm.

    Clusters the call graph into named communities for scope-aware localisation.
    Each community exposes the node IDs that belong to it, a human-readable
    name derived from dominant path tokens, the top-5 most referenced files,
    and the community size.

    Parameters
    ----------
    graph_data:
        Dict with keys ``"nodes"`` (list of node dicts) and ``"edges"`` (list
        of edge dicts).  Node dicts must have at least an ``"id"`` key; edge
        dicts must have ``"source"``, ``"target"``, and ``"type"`` keys.
    resolution:
        Resolution parameter forwarded to the Leiden algorithm.  Higher values
        produce more, smaller communities.  Has no effect on the networkx
        fallback.  Default: ``1.0``.

    Returns
    -------
    list[dict]
        Each dict has the shape::

            {
                "id":             int,        # 0-based index
                "name":           str,        # e.g. "auth-session"
                "node_ids":       list[str],  # member node IDs
                "size":           int,        # len(node_ids)
                "dominant_files": list[str],  # top-5 files by node count
            }

    Notes
    -----
    - Communities with fewer than 3 nodes are merged into a single ``"misc"``
      community.
    - The result is capped at 15 communities; excess smallest ones are merged
      into the last bucket.
    - Names are deduplicated with a numeric suffix when two communities share
      the same token-derived name.
    - If ``leidenalg`` or ``python-igraph`` is not installed the function falls
      back to networkx ``greedy_modularity_communities``.
    """
    if not graph_data.get("nodes"):
        logger.warning("build_communities called with empty graph_data — returning [].")
        return []

    node_map = _build_node_map(graph_data)

    # --- Run detection (Leiden preferred, networkx as fallback) -------------
    raw = _run_leiden(graph_data, resolution)
    algorithm_used = "leiden"
    if raw is None:
        raw = _run_networkx_greedy(graph_data)
        algorithm_used = "networkx-greedy"

    logger.info(
        "community detection (%s): %d raw partitions from %d nodes.",
        algorithm_used,
        len(raw),
        len(graph_data.get("nodes", [])),
    )

    if not raw:
        return []

    # --- Post-processing ----------------------------------------------------
    merged, had_misc = _merge_small_communities(raw, _MIN_COMMUNITY_SIZE)
    capped = _cap_communities(merged, _MAX_COMMUNITIES)

    # The misc bucket is always the last element when had_misc is True.
    # _cap_communities may move it to a different position (if it were not one
    # of the top-N largest), but because misc is built by aggregation it is
    # typically large enough to stay.  We recompute its index after capping.
    misc_index: int | None = None
    if had_misc:
        # After merge, the misc bucket's node set is known.  Find its new index.
        misc_set = set(merged[-1]) if merged else set()
        for i, comm in enumerate(capped):
            if set(comm) == misc_set:
                misc_index = i
                break
        # If capping merged more nodes into the last bucket, the last element
        # is still the overflow/misc bucket.
        if misc_index is None and capped:
            misc_index = len(capped) - 1

    communities = _build_community_dicts(capped, node_map, misc_index)
    communities = _ensure_unique_names(communities)

    logger.info(
        "Final communities (%d): %s",
        len(communities),
        [f"{c['name']}({c['size']})" for c in communities],
    )

    return communities


def annotate_graph_with_communities(
    graph_data: dict,
    communities: list[dict],
) -> dict:
    """Return a copy of *graph_data* with community annotations added to nodes.

    Each node dict gains two extra keys:

    - ``"community_id"``   – integer community index
    - ``"community_name"`` – human-readable community name (e.g. ``"auth-session"``)

    Nodes that do not belong to any community retain ``None`` for both keys.
    The original *graph_data* dict is not mutated.

    Parameters
    ----------
    graph_data:
        Original graph dict.
    communities:
        Output of :func:`build_communities`.

    Returns
    -------
    dict
        New dict with the same ``"edges"`` reference and a new ``"nodes"``
        list where every node is a shallow copy with the community fields added.
    """
    membership: dict[str, tuple[int, str]] = {}
    for community in communities:
        for nid in community["node_ids"]:
            membership[nid] = (community["id"], community["name"])

    annotated_nodes: list[dict] = []
    for node in graph_data.get("nodes", []):
        nid = node.get("id")
        comm_id, comm_name = membership.get(nid, (None, None))
        annotated_nodes.append(
            {
                **node,
                "community_id": comm_id,
                "community_name": comm_name,
            }
        )

    return {**graph_data, "nodes": annotated_nodes}


def get_community_for_files(
    communities: list[dict],
    file_paths: list[str],
) -> str | None:
    """Return the community name that best matches the given file paths.

    Scores each community by counting how many of its ``dominant_files`` overlap
    with (or are contained in) *file_paths*, then returns the name of the
    highest-scoring community.  Falls back to a node-ID-level match when no
    file-level overlap is found.

    Parameters
    ----------
    communities:
        Output of :func:`build_communities`.
    file_paths:
        File paths to match (e.g. the files touched by a bug ticket).

    Returns
    -------
    str | None
        Community name with the highest overlap, or ``None`` if *communities*
        is empty or no match is found at all.
    """
    if not communities or not file_paths:
        return None

    query_set = {str(fp).strip() for fp in file_paths if fp}

    best_name: str | None = None
    best_score: int = 0

    for community in communities:
        score = 0
        for dom_file in community.get("dominant_files", []):
            for qf in query_set:
                if dom_file == qf or dom_file in qf or qf in dom_file:
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best_name = community["name"]

    # Secondary fallback: node-ID-level overlap
    if best_score == 0:
        for community in communities:
            score = sum(1 for nid in community.get("node_ids", []) if nid in query_set)
            if score > best_score:
                best_score = score
                best_name = community["name"]

    return best_name if best_score > 0 else None


def build_community_index(communities: list[dict]) -> dict[str, list[str]]:
    """Build a ``{community_name: [representative_files]}`` index for quick lookup.

    "Representative files" are the ``dominant_files`` stored on each community
    (top-5 most referenced files within that community).

    Parameters
    ----------
    communities:
        Output of :func:`build_communities`.

    Returns
    -------
    dict[str, list[str]]
        Mapping from community name to its representative file list.  When two
        communities share the same name (which should not occur after
        :func:`build_communities` deduplicates names), their file lists are
        merged and deduplicated.

    Example
    -------
    ::

        {
            "auth-session":      ["auth/session.py", "auth/login.py"],
            "payments-checkout": ["payments/checkout.py", ...],
            "misc":              [...],
        }
    """
    index: dict[str, list[str]] = defaultdict(list)

    for community in communities:
        name = community.get("name", "misc")
        for fp in community.get("dominant_files", []):
            if fp not in index[name]:
                index[name].append(fp)

    return dict(index)
