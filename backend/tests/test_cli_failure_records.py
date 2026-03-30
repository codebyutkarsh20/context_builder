"""
Tests for mine_failure_records integration in CLI build command.

Verifies:
  - mine_failure_records is importable and callable
  - Returns empty list when ENABLE_FAILURE_RECORDS is not set
  - persist_failure_records is callable with records and repo name
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCLIFailureRecordsIntegration:
    """Verify mine_failure_records is importable and callable from CLI context."""

    def test_mine_failure_records_returns_empty_when_disabled(self, tmp_path):
        """mine_failure_records returns [] when ENABLE_FAILURE_RECORDS is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_FAILURE_RECORDS", None)
            from graph.business.failure_records import mine_failure_records
            result = mine_failure_records(tmp_path)
            assert result == []

    def test_mine_failure_records_returns_empty_for_non_git(self, tmp_path):
        """mine_failure_records returns [] for non-git directories even when enabled."""
        with patch.dict(os.environ, {"ENABLE_FAILURE_RECORDS": "true"}):
            from graph.business.failure_records import mine_failure_records
            result = mine_failure_records(tmp_path)
            assert result == []

    def test_persist_failure_records_handles_no_connection(self):
        """persist_failure_records returns 0 when Neo4j is not connected."""
        with patch("graph.neo4j_client.neo4j_client") as mock_client:
            mock_client.is_connected.return_value = False
            from graph.business.failure_records import persist_failure_records
            records = [{"id": "fr-1", "commit_hash": "abc", "function_hits": []}]
            count = persist_failure_records(records, "test-repo")
            assert count == 0
