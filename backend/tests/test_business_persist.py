"""Tests for graph/business/persist.py — Neo4j business rule persistence and stale edge cleanup."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPersistBusinessRulesToNeo4j:
    def test_calls_extractor_extract(self):
        with (
            patch("enricher.business_logic.BusinessLogicExtractor") as MockExt,
            patch("graph.business.persist.cleanup_stale_enforced_by_edges"),
        ):
            mock_ext = MockExt.return_value
            mock_ext.extract.return_value = 5
            from graph.business.persist import persist_business_rules_to_neo4j
            count = persist_business_rules_to_neo4j("my-repo", [])
            assert count == 5
            mock_ext.extract.assert_called_once()

    def test_calls_stale_cleanup_after_extract(self):
        with (
            patch("enricher.business_logic.BusinessLogicExtractor") as MockExt,
            patch("graph.business.persist.cleanup_stale_enforced_by_edges") as mock_cleanup,
        ):
            MockExt.return_value.extract.return_value = 0
            from graph.business.persist import persist_business_rules_to_neo4j
            persist_business_rules_to_neo4j("my-repo", [])
            mock_cleanup.assert_called_once()


class TestCleanupStaleEnforcedByEdges:
    def test_returns_zero_when_neo4j_not_connected(self):
        with patch("graph.business.persist.neo4j_client") as mock_client:
            mock_client.is_connected.return_value = False
            from graph.business.persist import cleanup_stale_enforced_by_edges
            result = cleanup_stale_enforced_by_edges()
            assert result == 0
            mock_client.run.assert_not_called()

    def test_returns_deleted_count_from_neo4j(self):
        with patch("graph.business.persist.neo4j_client") as mock_client:
            mock_client.is_connected.return_value = True
            mock_client.run.return_value = [{"deleted": 3}]
            from graph.business.persist import cleanup_stale_enforced_by_edges
            result = cleanup_stale_enforced_by_edges()
            assert result == 3

    def test_handles_neo4j_error_gracefully(self):
        with patch("graph.business.persist.neo4j_client") as mock_client:
            mock_client.is_connected.return_value = True
            mock_client.run.side_effect = RuntimeError("Neo4j down")
            from graph.business.persist import cleanup_stale_enforced_by_edges
            result = cleanup_stale_enforced_by_edges()
            assert result == 0  # non-fatal
