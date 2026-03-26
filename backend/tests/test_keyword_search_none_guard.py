"""Tests for _keyword_search None-guard fix in GraphRAGRetriever."""
import types
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so we can instantiate GraphRAGRetriever without real files
# ---------------------------------------------------------------------------

def _make_retriever(enriched_nodes: dict):
    """Return a GraphRAGRetriever instance with pre-loaded enriched nodes."""
    # Import here so the test file doesn't fail at collection time if the
    # module path isn't on sys.path in all environments.
    from backend.rag.retriever import GraphRAGRetriever

    retriever = GraphRAGRetriever.__new__(GraphRAGRetriever)
    # Initialise the private caches directly to avoid file-system access.
    retriever._enriched = enriched_nodes
    retriever._edges = []
    retriever._edge_index = {}
    retriever.data_dir = MagicMock()
    retriever.repo_name = "test_repo"
    return retriever


def _make_intent(mentioned_names):
    """Return a minimal QueryIntent-like object."""
    intent = MagicMock()
    intent.mentioned_names = mentioned_names
    return intent


# ---------------------------------------------------------------------------
# Acceptance criterion 1: null *name* field must not raise AttributeError
# ---------------------------------------------------------------------------

def test_keyword_search_null_name_does_not_raise():
    """_keyword_search must not raise when a node's 'name' field is None."""
    enriched = {
        "node::1": {
            "name": None,
            "docstring": "some docstring text about payment processing",
            "file": "payment.py",
            "type": "function",
            "pagerank": 0,
        }
    }
    retriever = _make_retriever(enriched)
    intent = _make_intent(["payment"])

    # Must not raise
    results = retriever._keyword_search("payment", intent, n=10)
    # The node should still be found via the docstring / file path match
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: null *docstring* field must not raise AttributeError
# ---------------------------------------------------------------------------

def test_keyword_search_null_docstring_does_not_raise():
    """_keyword_search must not raise when a node's 'docstring' field is None."""
    enriched = {
        "node::2": {
            "name": "process_payment",
            "docstring": None,
            "file": "payments.py",
            "type": "function",
            "pagerank": 0,
        }
    }
    retriever = _make_retriever(enriched)
    intent = _make_intent(["process_payment"])

    results = retriever._keyword_search("process payment", intent, n=10)
    assert isinstance(results, list)
    # The node matches by name, so it should appear in results
    ids = [r.id for r in results]
    assert "node::2" in ids


# ---------------------------------------------------------------------------
# Acceptance criterion 3: all-null name+docstring → empty result, no exception
# ---------------------------------------------------------------------------

def test_keyword_search_all_null_fields_returns_empty():
    """When every node has null name and docstring, return [] without raising."""
    enriched = {
        "node::3": {
            "name": None,
            "docstring": None,
            "file": None,
            "type": "function",
            "pagerank": 0,
        },
        "node::4": {
            "name": None,
            "docstring": None,
            "file": None,
            "type": "class",
            "pagerank": 0,
        },
    }
    retriever = _make_retriever(enriched)
    intent = _make_intent(["something"])

    results = retriever._keyword_search("something", intent, n=10)
    assert results == []


# ---------------------------------------------------------------------------
# Acceptance criterion 4: mixed valid/null nodes → valid ones still matched
# ---------------------------------------------------------------------------

def test_keyword_search_mixed_nodes_matches_valid():
    """Nodes with valid fields are still matched when other nodes have nulls."""
    enriched = {
        "node::null": {
            "name": None,
            "docstring": None,
            "file": None,
            "type": "function",
            "pagerank": 0,
        },
        "node::valid": {
            "name": "checkout_flow",
            "docstring": "Handles the checkout flow for the shopping cart",
            "file": "checkout.py",
            "type": "function",
            "pagerank": 0,
        },
    }
    retriever = _make_retriever(enriched)
    intent = _make_intent(["checkout_flow"])

    results = retriever._keyword_search("checkout flow", intent, n=10)
    assert isinstance(results, list)
    ids = [r.id for r in results]
    assert "node::valid" in ids
    assert "node::null" not in ids


# ---------------------------------------------------------------------------
# Bonus: None entries inside mentioned_names must also be tolerated
# ---------------------------------------------------------------------------

def test_keyword_search_none_in_mentioned_names_does_not_raise():
    """None entries in intent.mentioned_names must be skipped gracefully."""
    enriched = {
        "node::5": {
            "name": "authenticate_user",
            "docstring": "Authenticates a user by credentials",
            "file": "auth.py",
            "type": "function",
            "pagerank": 0,
        }
    }
    retriever = _make_retriever(enriched)
    # Inject a None into the list of mentioned names
    intent = _make_intent([None, "authenticate_user", None])

    results = retriever._keyword_search("authenticate user", intent, n=10)
    assert isinstance(results, list)
    ids = [r.id for r in results]
    assert "node::5" in ids
