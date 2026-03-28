"""Tests for the _merge_and_rank (RRF) function in backend/rag/retriever.py."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scored_node(node_id: str, score: float, source: str = "test"):
    """Create a minimal ScoredNode-like object for testing."""
    node = MagicMock()
    node.id = node_id
    node.score = score
    node.source = source
    node.metadata = {}
    return node


def get_retriever_instance():
    """Return an instance of the GraphRAGRetriever with dependencies mocked out."""
    # Import here so the module is loaded after any patches are applied.
    import importlib
    import sys

    # We need to import the retriever module.  Guard against missing optional
    # dependencies by mocking them if necessary.
    try:
        from backend.rag.retriever import GraphRAGRetriever  # type: ignore
    except ModuleNotFoundError:
        pytest.skip("backend.rag.retriever not importable in this environment")

    retriever = GraphRAGRetriever.__new__(GraphRAGRetriever)
    return retriever


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMergeAndRankRRF:
    """Unit-tests for _merge_and_rank in GraphRAGRetriever."""

    def setup_method(self):
        self.retriever = get_retriever_instance()

    # ------------------------------------------------------------------
    # 1. Default K=60 must not raise any error
    # ------------------------------------------------------------------

    def test_default_k60_no_error(self):
        """RRF with default K=60 must complete without any exception."""
        list1 = [
            make_scored_node("a", 0.9),
            make_scored_node("b", 0.7),
        ]
        list2 = [
            make_scored_node("b", 0.8),
            make_scored_node("c", 0.6),
        ]
        # Must not raise ZeroDivisionError or any other exception
        result = self.retriever._merge_and_rank(list1, list2)
        assert isinstance(result, list)
        assert len(result) == 3  # a, b, c

    # ------------------------------------------------------------------
    # 2. K=0 must raise ValueError — not silently crash
    # ------------------------------------------------------------------

    def test_k_zero_raises_value_error(self):
        """K=0 must raise a ValueError with a meaningful message, not a ZeroDivisionError."""
        list1 = [make_scored_node("a", 0.9)]
        with pytest.raises(ValueError, match="K must be a positive integer"):
            self.retriever._merge_and_rank(list1, K=0)

    def test_k_zero_does_not_raise_zero_division_error(self):
        """Confirm the original bug (ZeroDivisionError) is NOT the exception raised for K=0."""
        list1 = [make_scored_node("a", 0.9)]
        with pytest.raises(ValueError):
            # Should be a ValueError, not a ZeroDivisionError
            self.retriever._merge_and_rank(list1, K=0)

    # ------------------------------------------------------------------
    # 3. Negative K must also raise ValueError
    # ------------------------------------------------------------------

    def test_negative_k_raises_value_error(self):
        """Negative K values must be rejected with a ValueError."""
        list1 = [make_scored_node("x", 0.5)]
        with pytest.raises(ValueError, match="K must be a positive integer"):
            self.retriever._merge_and_rank(list1, K=-1)

    # ------------------------------------------------------------------
    # 4. Scores are correctly fused and ordered
    # ------------------------------------------------------------------

    def test_fused_scores_ordered_descending(self):
        """Documents appearing near the top of multiple lists should rank highest."""
        # Node 'b' appears first in list2 and second in list1 — it should
        # accumulate more RRF score than 'a' (first in list1 only) or 'c'
        # (first in list2 only, low own-score).
        list1 = [
            make_scored_node("a", 0.5),  # rank 0 in list1
            make_scored_node("b", 0.5),  # rank 1 in list1
        ]
        list2 = [
            make_scored_node("b", 0.5),  # rank 0 in list2
            make_scored_node("c", 0.5),  # rank 1 in list2
        ]
        result = self.retriever._merge_and_rank(list1, list2, K=60)
        ids_in_order = [node.id for node in result]
        # 'b' appears in both lists so it should have the highest fused score
        assert ids_in_order[0] == "b", (
            f"Expected 'b' to rank first (appears in both lists), got: {ids_in_order}"
        )
        # Scores must be in descending order
        scores = [node.score for node in result]
        assert scores == sorted(scores, reverse=True), (
            f"Scores are not in descending order: {scores}"
        )

    def test_single_list_passthrough(self):
        """A single result list should be returned in its original rank order."""
        nodes = [
            make_scored_node("x", 0.9),
            make_scored_node("y", 0.5),
            make_scored_node("z", 0.1),
        ]
        result = self.retriever._merge_and_rank(nodes, K=60)
        ids = [n.id for n in result]
        assert ids == ["x", "y", "z"], (
            f"Single-list ordering not preserved: {ids}"
        )

    def test_empty_lists_return_empty(self):
        """Passing empty result lists should return an empty list."""
        result = self.retriever._merge_and_rank([], [], K=60)
        assert result == []

    def test_rrf_score_values_are_positive(self):
        """All fused RRF scores must be strictly positive."""
        nodes = [make_scored_node(f"node_{i}", float(i) / 10) for i in range(5)]
        result = self.retriever._merge_and_rank(nodes, K=60)
        for node in result:
            assert node.score > 0, f"Expected positive score, got {node.score} for {node.id}"

    # ------------------------------------------------------------------
    # 5. Mathematical correctness spot-check
    # ------------------------------------------------------------------

    def test_rrf_formula_correctness(self):
        """Manually verify the RRF formula: score = 1/(K+rank) * (0.5 + node.score)."""
        K = 60
        node = make_scored_node("doc", score=0.5)
        result = self.retriever._merge_and_rank([node], K=K)
        assert len(result) == 1
        expected_score = (1.0 / (K + 0)) * (0.5 + 0.5)
        assert abs(result[0].score - expected_score) < 1e-9, (
            f"Expected score {expected_score}, got {result[0].score}"
        )
