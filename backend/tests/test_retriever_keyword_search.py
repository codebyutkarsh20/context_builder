"""
Unit tests for rag/retriever.py — _keyword_search null-safety.

Covers:
  - None name field does not raise AttributeError
  - None docstring field does not raise AttributeError
  - None file field does not raise AttributeError
  - All fields None does not raise AttributeError
  - Normal keyword matching still works
  - PageRank None does not raise
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import GraphRAGRetriever, ScoredNode
from rag.query_analyzer import QueryIntent


def _make_retriever(enriched_data: dict) -> GraphRAGRetriever:
    """Create a GraphRAGRetriever with pre-loaded enriched data."""
    retriever = GraphRAGRetriever.__new__(GraphRAGRetriever)
    retriever.repo_name = "test-repo"
    retriever.data_dir = Path("/tmp/test")
    retriever._enriched = enriched_data
    retriever._edges = []
    retriever._edge_index = {}
    return retriever


def _make_intent(names: list[str]) -> QueryIntent:
    """Create a QueryIntent with specified mentioned_names."""
    intent = QueryIntent()
    intent.mentioned_names = names
    return intent


class TestKeywordSearchNullSafety:

    def test_none_name_does_not_raise(self):
        """name=None should not cause AttributeError."""
        enriched = {
            "node1": {
                "name": None,
                "docstring": "some docstring",
                "file": "app.py",
                "type": "function",
                "pagerank": 0.5,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["some"])
        # Should not raise AttributeError
        results = retriever._keyword_search("some query", intent, n=10)
        assert isinstance(results, list)

    def test_none_docstring_does_not_raise(self):
        """docstring=None should not cause AttributeError."""
        enriched = {
            "node1": {
                "name": "process_payment",
                "docstring": None,
                "file": "app.py",
                "type": "function",
                "pagerank": 0.5,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["process_payment"])
        results = retriever._keyword_search("process_payment", intent, n=10)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].id == "node1"

    def test_none_file_does_not_raise(self):
        """file=None should not cause AttributeError."""
        enriched = {
            "node1": {
                "name": "validate_user",
                "docstring": "validates user input",
                "file": None,
                "type": "function",
                "pagerank": 0.1,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["validate_user"])
        results = retriever._keyword_search("validate_user", intent, n=10)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_all_fields_none_does_not_raise(self):
        """All null fields should not cause any AttributeError."""
        enriched = {
            "node1": {
                "name": None,
                "docstring": None,
                "file": None,
                "type": None,
                "pagerank": None,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["anything"])
        results = retriever._keyword_search("anything", intent, n=10)
        assert isinstance(results, list)
        assert len(results) == 0  # No match since name is None → empty string

    def test_none_pagerank_does_not_raise(self):
        """pagerank=None should not cause AttributeError or TypeError."""
        enriched = {
            "node1": {
                "name": "get_user",
                "docstring": "gets the user",
                "file": "users.py",
                "type": "function",
                "pagerank": None,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["get_user"])
        results = retriever._keyword_search("get_user", intent, n=10)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_normal_keyword_match(self):
        """Normal keyword matching still returns correct results."""
        enriched = {
            "node1": {
                "name": "authenticate_user",
                "docstring": "authenticates a user by password",
                "file": "auth.py",
                "type": "function",
                "pagerank": 0.8,
            },
            "node2": {
                "name": "get_products",
                "docstring": "returns all products",
                "file": "products.py",
                "type": "function",
                "pagerank": 0.2,
            },
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["authenticate_user"])
        results = retriever._keyword_search("authenticate user", intent, n=10)
        ids = [r.id for r in results]
        assert "node1" in ids
        assert "node2" not in ids

    def test_multiple_null_nodes_mixed_with_valid(self):
        """Mix of null and valid nodes: only valid nodes match, no crash."""
        enriched = {
            "null_node": {
                "name": None,
                "docstring": None,
                "file": None,
                "type": None,
                "pagerank": None,
            },
            "valid_node": {
                "name": "process_order",
                "docstring": "processes an order",
                "file": "orders.py",
                "type": "function",
                "pagerank": 0.3,
            },
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent(["process_order"])
        results = retriever._keyword_search("process order", intent, n=10)
        ids = [r.id for r in results]
        assert "valid_node" in ids
        assert "null_node" not in ids

    def test_empty_enriched_returns_empty_list(self):
        """Empty enriched dict returns empty list."""
        retriever = _make_retriever({})
        intent = _make_intent(["something"])
        results = retriever._keyword_search("something", intent, n=10)
        assert results == []

    def test_empty_mentioned_names_returns_empty_list(self):
        """No mentioned names → no terms → returns empty list."""
        enriched = {
            "node1": {
                "name": "some_func",
                "docstring": "does something",
                "file": "app.py",
                "type": "function",
                "pagerank": 0.5,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent([])
        results = retriever._keyword_search("", intent, n=10)
        assert results == []

    def test_none_in_mentioned_names_does_not_raise(self):
        """None values in mentioned_names should be safely filtered."""
        enriched = {
            "node1": {
                "name": "some_func",
                "docstring": "does something",
                "file": "app.py",
                "type": "function",
                "pagerank": 0.5,
            }
        }
        retriever = _make_retriever(enriched)
        intent = _make_intent([None, "some_func"])  # type: ignore[list-item]
        # Should not raise AttributeError when None.lower() would be called
        results = retriever._keyword_search("some func", intent, n=10)
        assert isinstance(results, list)
