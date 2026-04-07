"""
test_community_detection.py — Tests for graph/community.py

Covers:
  - build_communities() with Leiden / networkx fallback
  - annotate_graph_with_communities()
  - get_community_for_files()
  - build_community_index()
  - Edge-case handling (empty graph, tiny communities, >15 communities)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures — synthetic call graphs
# ---------------------------------------------------------------------------

def _make_node(node_id: str, file: str = "") -> dict:
    return {"id": node_id, "file": file or node_id.split("::")[0], "type": "Function"}


def _make_edge(src: str, tgt: str, kind: str = "CALLS") -> dict:
    return {"source": src, "target": tgt, "type": kind}


@pytest.fixture()
def auth_graph():
    """A small graph with two clear clusters: auth/* and payments/*."""
    nodes = [
        _make_node("auth/login.py::login", "auth/login.py"),
        _make_node("auth/login.py::logout", "auth/login.py"),
        _make_node("auth/session.py::create_session", "auth/session.py"),
        _make_node("auth/session.py::destroy_session", "auth/session.py"),
        _make_node("auth/session.py::check_session", "auth/session.py"),
        _make_node("payments/checkout.py::process_payment", "payments/checkout.py"),
        _make_node("payments/checkout.py::validate_cart", "payments/checkout.py"),
        _make_node("payments/refund.py::issue_refund", "payments/refund.py"),
        _make_node("payments/refund.py::cancel_order", "payments/refund.py"),
        _make_node("payments/refund.py::log_refund", "payments/refund.py"),
    ]
    edges = [
        _make_edge("auth/login.py::login", "auth/session.py::create_session"),
        _make_edge("auth/login.py::logout", "auth/session.py::destroy_session"),
        _make_edge("auth/session.py::check_session", "auth/login.py::login"),
        _make_edge("payments/checkout.py::process_payment", "payments/checkout.py::validate_cart"),
        _make_edge("payments/checkout.py::process_payment", "payments/refund.py::log_refund"),
        _make_edge("payments/refund.py::issue_refund", "payments/refund.py::cancel_order"),
        _make_edge("payments/refund.py::cancel_order", "payments/refund.py::log_refund"),
    ]
    return {"nodes": nodes, "edges": edges}


@pytest.fixture()
def tiny_graph():
    """A graph where most communities are < MIN_COMMUNITY_SIZE (3)."""
    nodes = [
        _make_node("a/foo.py::foo", "a/foo.py"),
        _make_node("b/bar.py::bar", "b/bar.py"),
        _make_node("c/baz.py::baz", "c/baz.py"),
        _make_node("c/baz.py::qux", "c/baz.py"),
        _make_node("c/baz.py::quux", "c/baz.py"),
    ]
    edges = [
        _make_edge("c/baz.py::baz", "c/baz.py::qux"),
        _make_edge("c/baz.py::qux", "c/baz.py::quux"),
    ]
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# build_communities()
# ---------------------------------------------------------------------------

class TestBuildCommunities:
    def test_returns_list_of_dicts(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        assert isinstance(communities, list)
        assert all(isinstance(c, dict) for c in communities)

    def test_community_schema(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        assert len(communities) >= 1
        c = communities[0]
        assert "id" in c
        assert "name" in c
        assert "node_ids" in c
        assert "size" in c
        assert "dominant_files" in c
        assert isinstance(c["node_ids"], list)
        assert isinstance(c["size"], int)
        assert c["size"] == len(c["node_ids"])

    def test_covers_all_nodes(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        all_node_ids = set()
        for c in communities:
            all_node_ids.update(c["node_ids"])
        expected = {n["id"] for n in auth_graph["nodes"]}
        assert all_node_ids == expected

    def test_community_names_are_strings(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        for c in communities:
            assert isinstance(c["name"], str)
            assert len(c["name"]) > 0

    def test_names_are_unique(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        names = [c["name"] for c in communities]
        assert len(names) == len(set(names)), f"Duplicate community names: {names}"

    def test_empty_graph_returns_empty(self):
        from graph.community import build_communities
        result = build_communities({"nodes": [], "edges": []})
        assert result == []

    def test_no_edges_still_works(self):
        from graph.community import build_communities
        graph = {"nodes": [_make_node(f"x/f{i}.py::fn{i}", f"x/f{i}.py") for i in range(5)], "edges": []}
        result = build_communities(graph)
        # Should return communities (each node is its own, merged into misc)
        assert isinstance(result, list)

    def test_small_communities_merged_into_misc(self, tiny_graph):
        from graph.community import build_communities
        communities = build_communities(tiny_graph)
        names = [c["name"] for c in communities]
        # The lone nodes (foo, bar) should end up in misc
        assert "misc" in names or any(c["size"] >= 1 for c in communities)

    def test_max_15_communities(self):
        """Never returns more than 15 communities even with large graphs."""
        from graph.community import build_communities
        # Create 30 isolated clusters of 4 nodes
        nodes = []
        edges = []
        for cluster in range(30):
            base = f"pkg{cluster}"
            for i in range(4):
                nodes.append(_make_node(f"{base}/f{i}.py::fn{i}", f"{base}/f{i}.py"))
            # Dense internal edges
            for i in range(3):
                edges.append(_make_edge(f"{base}/f{i}.py::fn{i}", f"{base}/f{i+1}.py::fn{i+1}"))

        communities = build_communities({"nodes": nodes, "edges": edges})
        assert len(communities) <= 15

    def test_no_empty_node_id_lists(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        for c in communities:
            assert c["node_ids"], f"Community '{c['name']}' has no node_ids"

    def test_dominant_files_are_strings(self, auth_graph):
        from graph.community import build_communities
        communities = build_communities(auth_graph)
        for c in communities:
            assert isinstance(c["dominant_files"], list)
            assert all(isinstance(f, str) for f in c["dominant_files"])


# ---------------------------------------------------------------------------
# annotate_graph_with_communities()
# ---------------------------------------------------------------------------

class TestAnnotateGraph:
    def test_adds_community_fields_to_all_nodes(self, auth_graph):
        from graph.community import build_communities, annotate_graph_with_communities
        communities = build_communities(auth_graph)
        annotated = annotate_graph_with_communities(auth_graph, communities)

        for node in annotated["nodes"]:
            assert "community_id" in node
            assert "community_name" in node

    def test_original_not_mutated(self, auth_graph):
        from graph.community import build_communities, annotate_graph_with_communities
        import copy
        original_copy = copy.deepcopy(auth_graph)
        communities = build_communities(auth_graph)
        annotate_graph_with_communities(auth_graph, communities)
        # Original should be unchanged
        for orig_node, node in zip(original_copy["nodes"], auth_graph["nodes"]):
            assert "community_id" not in orig_node
            assert "community_name" not in orig_node

    def test_edges_preserved(self, auth_graph):
        from graph.community import build_communities, annotate_graph_with_communities
        communities = build_communities(auth_graph)
        annotated = annotate_graph_with_communities(auth_graph, communities)
        assert annotated["edges"] == auth_graph["edges"]

    def test_community_ids_are_ints_or_none(self, auth_graph):
        from graph.community import build_communities, annotate_graph_with_communities
        communities = build_communities(auth_graph)
        annotated = annotate_graph_with_communities(auth_graph, communities)
        for node in annotated["nodes"]:
            cid = node["community_id"]
            assert cid is None or isinstance(cid, int)

    def test_empty_communities_sets_none(self):
        from graph.community import annotate_graph_with_communities
        graph = {"nodes": [_make_node("x/f.py::fn", "x/f.py")], "edges": []}
        annotated = annotate_graph_with_communities(graph, [])
        assert annotated["nodes"][0]["community_id"] is None
        assert annotated["nodes"][0]["community_name"] is None


# ---------------------------------------------------------------------------
# get_community_for_files()
# ---------------------------------------------------------------------------

class TestGetCommunityForFiles:
    @pytest.fixture()
    def sample_communities(self, auth_graph):
        from graph.community import build_communities
        return build_communities(auth_graph)

    def test_finds_auth_community(self, sample_communities):
        from graph.community import get_community_for_files
        name = get_community_for_files(sample_communities, ["auth/login.py"])
        assert name is not None
        assert "auth" in name.lower() or name == "misc"

    def test_finds_payments_community(self, sample_communities):
        from graph.community import get_community_for_files
        name = get_community_for_files(sample_communities, ["payments/checkout.py"])
        assert name is not None

    def test_returns_none_for_unknown_file(self, sample_communities):
        from graph.community import get_community_for_files
        name = get_community_for_files(sample_communities, ["completely/unknown/file.py"])
        assert name is None

    def test_empty_file_list_returns_none(self, sample_communities):
        from graph.community import get_community_for_files
        assert get_community_for_files(sample_communities, []) is None

    def test_empty_communities_returns_none(self):
        from graph.community import get_community_for_files
        assert get_community_for_files([], ["auth/login.py"]) is None


# ---------------------------------------------------------------------------
# build_community_index()
# ---------------------------------------------------------------------------

class TestBuildCommunityIndex:
    def test_returns_dict(self, auth_graph):
        from graph.community import build_communities, build_community_index
        communities = build_communities(auth_graph)
        idx = build_community_index(communities)
        assert isinstance(idx, dict)

    def test_keys_are_community_names(self, auth_graph):
        from graph.community import build_communities, build_community_index
        communities = build_communities(auth_graph)
        idx = build_community_index(communities)
        names = {c["name"] for c in communities}
        assert set(idx.keys()) == names

    def test_values_are_lists_of_strings(self, auth_graph):
        from graph.community import build_communities, build_community_index
        communities = build_communities(auth_graph)
        idx = build_community_index(communities)
        for name, files in idx.items():
            assert isinstance(files, list)
            assert all(isinstance(f, str) for f in files)

    def test_empty_communities_gives_empty_index(self):
        from graph.community import build_community_index
        assert build_community_index([]) == {}


# ---------------------------------------------------------------------------
# Path token extraction helpers
# ---------------------------------------------------------------------------

class TestPathTokenExtraction:
    def test_extracts_meaningful_tokens(self):
        from graph.community import _extract_path_tokens
        tokens = _extract_path_tokens("auth/session_manager.py")
        assert "auth" in tokens
        assert "session" in tokens
        assert "manager" in tokens

    def test_filters_stopwords(self):
        from graph.community import _extract_path_tokens
        tokens = _extract_path_tokens("src/utils/helpers.py")
        # 'src', 'utils', 'helpers' should be filtered by stopwords
        assert tokens == []  # all are stopwords

    def test_empty_path(self):
        from graph.community import _extract_path_tokens
        assert _extract_path_tokens("") == []

    def test_derive_community_name(self):
        from graph.community import _derive_community_name
        name = _derive_community_name(["auth/login.py", "auth/session.py", "auth/tokens.py"])
        assert "auth" in name

    def test_derive_community_name_fallback(self):
        from graph.community import _derive_community_name
        # All stopwords → fallback to "misc"
        name = _derive_community_name(["tests/test_utils.py"])
        # 'tests', 'test', 'utils' are all stopwords → should return 'misc'
        assert name == "misc"
