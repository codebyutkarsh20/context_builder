"""
test_keyword_search_null_fields.py — Tests for _keyword_search None/null field handling.

Covers acceptance criteria:
- name=None does not raise AttributeError
- docstring=None does not raise AttributeError
- both name and docstring=None treated as empty strings
- normal (non-null) fields produce correct results (regression guard)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_retriever(enriched_nodes: dict):
    """Create a Retriever instance with mocked enriched nodes."""
    from rag.retriever import Retriever

    retriever = Retriever.__new__(Retriever)
    retriever._enriched = enriched_nodes
    retriever._edges = []
    retriever._edge_index = {}
    retriever.data_dir = Path("/tmp/fake")
    retriever.repo_name = "fake_repo"
    return retriever


def _make_intent(mentioned_names: list):
    """Create a minimal QueryIntent-like object."""
    intent = MagicMock()
    intent.mentioned_names = mentioned_names
    return intent


class TestKeywordSearchNullFields:
    """Tests for safe None handling in _keyword_search."""

    def test_null_name_does_not_raise(self):
        """AC1: name=None should not raise AttributeError."""
        enriched = {
            "node_1": {
                "name": None,
                "docstring": "This is a valid docstring about authentication",
                "file": "auth/login.py",
                "type": "function",
                "pagerank": 0.0,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["authentication"])

        # Should NOT raise AttributeError
        try:
            results = retriever._keyword_search("authentication", intent, n=10)
        except AttributeError as e:
            pytest.fail(f"_keyword_search raised AttributeError with null name: {e}")

        # Results should be a list (possibly empty, possibly with the node matched on docstring)
        assert isinstance(results, list)

    def test_null_docstring_does_not_raise(self):
        """AC2: docstring=None should not raise AttributeError."""
        enriched = {
            "node_2": {
                "name": "authenticate_user",
                "docstring": None,
                "file": "auth/login.py",
                "type": "function",
                "pagerank": 0.0,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["authenticate"])

        # Should NOT raise AttributeError
        try:
            results = retriever._keyword_search("authenticate", intent, n=10)
        except AttributeError as e:
            pytest.fail(f"_keyword_search raised AttributeError with null docstring: {e}")

        assert isinstance(results, list)

    def test_both_null_name_and_docstring_treated_as_empty_strings(self):
        """AC3: Both name=None and docstring=None should be treated as empty strings."""
        enriched = {
            "node_with_nulls": {
                "name": None,
                "docstring": None,
                "file": "utils/helpers.py",
                "type": "function",
                "pagerank": 0.0,
            },
            "node_with_file_match": {
                "name": None,
                "docstring": None,
                "file": "utils/keyword_matcher.py",
                "type": "function",
                "pagerank": 0.0,
            },
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["keyword"])

        # Should complete without error
        try:
            results = retriever._keyword_search("keyword search", intent, n=10)
        except AttributeError as e:
            pytest.fail(f"_keyword_search raised AttributeError with both null: {e}")

        assert isinstance(results, list)
        # The node with 'keyword' in file path should match
        matched_ids = [r.id for r in results]
        assert "node_with_file_match" in matched_ids, (
            "Expected node with 'keyword' in file path to be matched even when name and docstring are None"
        )
        # The node without any match should not appear
        assert "node_with_nulls" not in matched_ids

    def test_normal_non_null_fields_regression(self):
        """AC4: When all fields are valid (non-null), results should be correct (no regression)."""
        enriched = {
            "node_exact": {
                "name": "compute_score",
                "docstring": "Computes the relevance score for a document.",
                "file": "scoring/ranker.py",
                "type": "function",
                "pagerank": 0.0,
            },
            "node_partial": {
                "name": "compute_metrics",
                "docstring": "Computes various metrics.",
                "file": "metrics/calculator.py",
                "type": "function",
                "pagerank": 0.0,
            },
            "node_unrelated": {
                "name": "load_data",
                "docstring": "Loads data from disk.",
                "file": "io/loader.py",
                "type": "function",
                "pagerank": 0.0,
            },
        }
        retriever = _make_retriever(enriched)
        # Search for 'compute_score' — exact match should score highest
        intent = _make_intent(["compute_score"])

        results = retriever._keyword_search("compute score", intent, n=10)

        assert isinstance(results, list)
        assert len(results) >= 1

        # Exact match should be first
        assert results[0].id == "node_exact", (
            f"Expected node_exact to rank first (exact name match), got {results[0].id}"
        )
        # Unrelated node should not appear
        matched_ids = [r.id for r in results]
        assert "node_unrelated" not in matched_ids

    def test_null_name_matches_on_docstring(self):
        """When name is None, keyword in docstring should still produce a match."""
        enriched = {
            "node_doc_match": {
                "name": None,
                "docstring": "Handles the authentication flow for users.",
                "file": "auth/flow.py",
                "type": "function",
                "pagerank": 0.0,
            },
            "node_no_match": {
                "name": None,
                "docstring": "Renders the home page template.",
                "file": "views/home.py",
                "type": "function",
                "pagerank": 0.0,
            },
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["authentication"])

        results = retriever._keyword_search("authentication", intent, n=10)

        assert isinstance(results, list)
        matched_ids = [r.id for r in results]
        assert "node_doc_match" in matched_ids, (
            "Expected docstring match when name is None"
        )
        assert "node_no_match" not in matched_ids

    def test_null_docstring_matches_on_name(self):
        """When docstring is None, keyword in name should still produce a match."""
        enriched = {
            "node_name_match": {
                "name": "parse_token",
                "docstring": None,
                "file": "auth/tokens.py",
                "type": "function",
                "pagerank": 0.0,
            },
            "node_no_match": {
                "name": "render_page",
                "docstring": None,
                "file": "views/page.py",
                "type": "function",
                "pagerank": 0.0,
            },
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["token"])

        results = retriever._keyword_search("token parsing", intent, n=10)

        assert isinstance(results, list)
        matched_ids = [r.id for r in results]
        assert "node_name_match" in matched_ids, (
            "Expected name match when docstring is None"
        )
        assert "node_no_match" not in matched_ids

    def test_short_terms_filtered_out(self):
        """Terms shorter than 3 characters should be ignored (existing behavior)."""
        enriched = {
            "node_1": {
                "name": "do_it",
                "docstring": None,
                "file": "utils/do.py",
                "type": "function",
                "pagerank": 0.0,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["do", "it"])  # Both < 3 chars

        results = retriever._keyword_search("do it", intent, n=10)

        # No terms with len >= 3, so result should be empty
        assert results == []

    def test_empty_enriched_returns_empty(self):
        """Empty enriched nodes should return empty list."""
        retriever = _make_retriever({})
        intent = _make_intent(["anything"])

        results = retriever._keyword_search("anything", intent, n=10)
        assert results == []
