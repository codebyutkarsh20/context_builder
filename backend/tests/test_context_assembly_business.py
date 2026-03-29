"""Tests for _load_business_rules() in agent/pipeline.py.

Covers business rule loading from flat file, FailureRecord query from Neo4j,
empty context warning injection, and fallback when Neo4j is disconnected.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _patch_data_dir(tmp_path):
    """Patch DATA_DIR in pipeline module to tmp_path."""
    return patch("agent.pipeline.DATA_DIR", tmp_path)


def _write_rules(tmp_path, repo_name, rules):
    repo_dir = tmp_path / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "business_rules.json").write_text(json.dumps(rules))


class TestLoadBusinessRules:
    def test_returns_warning_when_no_rules_file(self, tmp_path):
        with _patch_data_dir(tmp_path):
            from agent.pipeline import _load_business_rules
            result = _load_business_rules("myrepo", ["api/auth.py"])
        assert "WARNING" in result
        assert "high-risk" in result

    def test_returns_warning_when_no_relevant_rules(self, tmp_path):
        _write_rules(tmp_path, "myrepo", [
            {"id": "x", "description": "rule for other.py", "file": "other.py",
             "function_id": "", "severity": "medium", "source": "docstring"}
        ])
        with _patch_data_dir(tmp_path):
            from agent.pipeline import _load_business_rules
            result = _load_business_rules("myrepo", ["api/auth.py"])
        assert "WARNING" in result

    def test_returns_rule_when_file_matches(self, tmp_path):
        _write_rules(tmp_path, "myrepo", [
            {"id": "r1", "description": "email must be validated", "file": "api/auth.py",
             "function_id": "validate_email", "severity": "high", "source": "docstring"}
        ])
        with _patch_data_dir(tmp_path):
            with patch("agent.pipeline.neo4j_client") as mock_neo:
                mock_neo.is_connected.return_value = False
                from agent.pipeline import _load_business_rules
                result = _load_business_rules("myrepo", ["api/auth.py"])
        assert "email must be validated" in result
        assert "⚠️ DO NOT VIOLATE" in result  # HIGH severity

    def test_medium_severity_no_warning_marker(self, tmp_path):
        _write_rules(tmp_path, "myrepo", [
            {"id": "r2", "description": "log all requests", "file": "api/auth.py",
             "function_id": "", "severity": "medium", "source": "todo"}
        ])
        with _patch_data_dir(tmp_path):
            with patch("agent.pipeline.neo4j_client") as mock_neo:
                mock_neo.is_connected.return_value = False
                from agent.pipeline import _load_business_rules
                result = _load_business_rules("myrepo", ["api/auth.py"])
        assert "log all requests" in result
        assert "⚠️ DO NOT VIOLATE" not in result

    def test_failure_records_from_neo4j_included(self, tmp_path):
        _write_rules(tmp_path, "myrepo", [])
        with _patch_data_dir(tmp_path):
            with patch("agent.pipeline.neo4j_client") as mock_neo:
                mock_neo.is_connected.return_value = True
                mock_neo.run.return_value = [
                    {"message": "fixes #5: login crash", "date": "2026-01-10",
                     "issue_ref": "#5", "severity": "high"}
                ]
                from agent.pipeline import _load_business_rules
                result = _load_business_rules("myrepo", ["api/auth.py"])
        assert "fixes #5: login crash" in result
        assert "PAST FAILURES" in result

    def test_neo4j_failure_records_not_fatal(self, tmp_path):
        _write_rules(tmp_path, "myrepo", [
            {"id": "r3", "description": "must check token", "file": "api/auth.py",
             "function_id": "", "severity": "medium", "source": "docstring"}
        ])
        with _patch_data_dir(tmp_path):
            with patch("agent.pipeline.neo4j_client") as mock_neo:
                mock_neo.is_connected.return_value = True
                mock_neo.run.side_effect = RuntimeError("Neo4j timeout")
                from agent.pipeline import _load_business_rules
                result = _load_business_rules("myrepo", ["api/auth.py"])
        # Still returns rule content despite Neo4j failure
        assert "must check token" in result

    def test_warning_always_non_empty(self, tmp_path):
        """_load_business_rules never returns empty string."""
        with _patch_data_dir(tmp_path):
            with patch("agent.pipeline.neo4j_client") as mock_neo:
                mock_neo.is_connected.return_value = False
                from agent.pipeline import _load_business_rules
                result = _load_business_rules("myrepo", ["totally/unknown.py"])
        assert len(result) > 0
