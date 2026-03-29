"""Tests for Neo4j uniqueness constraints — BusinessRule and FailureRecord."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.neo4j_client import Neo4jClient


def _client_with_mock_driver():
    """Return a Neo4jClient with a mock driver attached."""
    client = Neo4jClient()
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])
    client._driver = mock_driver
    return client, mock_session


class TestEnsureConstraints:
    def test_business_rule_uniqueness_constraint_applied(self):
        client, mock_session = _client_with_mock_driver()
        client.ensure_constraints()
        calls_made = [str(c) for c in mock_session.run.call_args_list]
        assert any("business_rule_id_unique" in c for c in calls_made)

    def test_failure_record_uniqueness_constraint_applied(self):
        client, mock_session = _client_with_mock_driver()
        client.ensure_constraints()
        calls_made = [str(c) for c in mock_session.run.call_args_list]
        assert any("failure_record_id_unique" in c for c in calls_made)

    def test_existing_constraints_not_duplicated(self):
        """IF NOT EXISTS guard means re-running is safe."""
        client, mock_session = _client_with_mock_driver()
        client.ensure_constraints()
        count_before = mock_session.run.call_count
        client.ensure_constraints()
        count_after = mock_session.run.call_count
        # Same number of calls each run (idempotent)
        assert count_after == count_before * 2
