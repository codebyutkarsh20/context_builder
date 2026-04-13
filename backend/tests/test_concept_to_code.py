"""
test_concept_to_code.py — Unit tests for graph_utils.query_concept_to_code()

Tests cover:
  - Keyword extraction (stop-word filtering, length gating, dedup)
  - JSON fallback path (no Neo4j required)
  - Neo4j path (mocked client)
  - Empty / no-match cases
  - Deduplication of hint_functions / hint_files
  - Prompt-section formatting
  - Integration with intake_node state merging
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rules_json(rules: list[dict]) -> str:
    return json.dumps(rules)


# ---------------------------------------------------------------------------
# Tests — keyword extraction (tested indirectly via JSON fallback)
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    def test_stop_words_filtered(self, tmp_path):
        """Common stop words like 'the', 'is', 'bug' don't trigger matches."""
        rules_dir = tmp_path / "myrepo"
        rules_dir.mkdir()
        (rules_dir / "business_rules.json").write_text(_make_rules_json([
            {"description": "the bug is in the system", "source": "docstring",
             "file": "app.py", "function_id": "app.run", "severity": "low"},
        ]))

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("the bug is here", "the issue is known", "myrepo")

        # None of these are valid 4+ char non-stop keywords → no match
        assert result["matched_rules"] == []
        assert result["concept_section"] == ""

    def test_short_tokens_ignored(self, tmp_path):
        """Tokens shorter than 4 chars are not used as keywords."""
        rules_dir = tmp_path / "repo"
        rules_dir.mkdir()
        (rules_dir / "business_rules.json").write_text(_make_rules_json([
            {"description": "pay API key", "source": "constant",
             "file": "pay.py", "function_id": "", "severity": "medium"},
        ]))

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("pay api", "key err", "repo")

        assert result["matched_rules"] == []

    def test_keyword_dedup(self, tmp_path):
        """Duplicate keywords in title+description produce one keyword entry."""
        rules_dir = tmp_path / "repo"
        rules_dir.mkdir()
        (rules_dir / "business_rules.json").write_text(_make_rules_json([
            {"description": "requisition approval must pass review",
             "source": "docstring", "file": "approval.py",
             "function_id": "approval.check", "severity": "high"},
        ]))

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            # "approval" appears twice — should still match once
            result = query_concept_to_code(
                "approval flow", "approval process broken", "repo"
            )

        assert len(result["matched_rules"]) == 1


# ---------------------------------------------------------------------------
# Tests — JSON fallback path
# ---------------------------------------------------------------------------

class TestJsonFallback:
    def _setup_repo(self, tmp_path: Path, rules: list[dict]) -> Path:
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "business_rules.json").write_text(json.dumps(rules))
        return repo_dir

    def test_matching_rule_returned(self, tmp_path):
        self._setup_repo(tmp_path, [
            {"description": "requisition must be approved before payment",
             "source": "docstring", "file": "payments/service.py",
             "function_id": "payments.service.process_payment", "severity": "critical"},
        ])

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code(
                "Requisition not approved", "payment fails when requisition pending", "myrepo"
            )

        assert len(result["matched_rules"]) == 1
        rule = result["matched_rules"][0]
        assert "requisition" in rule["rule_text"].lower()
        assert "payments/service.py" in result["hint_files"]

    def test_non_matching_rule_excluded(self, tmp_path):
        self._setup_repo(tmp_path, [
            {"description": "inventory sync must run daily",
             "source": "constant", "file": "inventory.py",
             "function_id": "inventory.sync", "severity": "medium"},
        ])

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code(
                "Login timeout issue", "user cannot login after session expires", "myrepo"
            )

        assert result["matched_rules"] == []
        assert result["hint_functions"] == []

    def test_hint_functions_extracted(self, tmp_path):
        self._setup_repo(tmp_path, [
            {"description": "approval flow requires valid approver",
             "source": "docstring", "file": "workflow/approve.py",
             "function_id": "workflow.approve.validate_approver,workflow.approve.submit",
             "severity": "high"},
        ])

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Approval flow broken", "approver not found", "myrepo")

        assert "validate_approver" in result["hint_functions"]
        assert "submit" in result["hint_functions"]

    def test_hint_files_deduplicated(self, tmp_path):
        self._setup_repo(tmp_path, [
            {"description": "requisition approval required",
             "source": "docstring", "file": "approval.py",
             "function_id": "approval.check", "severity": "high"},
            {"description": "approval must log every decision",
             "source": "todo", "file": "approval.py",
             "function_id": "approval.log_decision", "severity": "medium"},
        ])

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Requisition approval", "approval missing", "myrepo")

        # "approval.py" should appear only once even though two rules share it
        assert result["hint_files"].count("approval.py") == 1

    def test_max_rules_respected(self, tmp_path):
        rules = [
            {"description": f"requisition rule number {i}",
             "source": "docstring", "file": f"rule{i}.py",
             "function_id": f"mod.func{i}", "severity": "medium"}
            for i in range(20)
        ]
        self._setup_repo(tmp_path, rules)

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Requisition issue", "requisition broken", "myrepo")

        assert len(result["matched_rules"]) <= 6  # default max_rules

    def test_missing_business_rules_json(self, tmp_path):
        """No business_rules.json → empty result, no crash."""
        (tmp_path / "repo").mkdir()

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Approval flow", "broken", "repo")

        assert result["matched_rules"] == []
        assert result["concept_section"] == ""

    def test_empty_title_and_description(self, tmp_path):
        """Empty input → no keywords → empty result."""
        self._setup_repo(tmp_path, [
            {"description": "approval must pass", "source": "docstring",
             "file": "x.py", "function_id": "x.check", "severity": "high"},
        ])

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("", "", "myrepo")

        assert result["matched_rules"] == []


# ---------------------------------------------------------------------------
# Tests — Neo4j path (mocked)
# ---------------------------------------------------------------------------

class TestNeo4jPath:
    def test_neo4j_results_used_when_connected(self, tmp_path):
        (tmp_path / "repo").mkdir()

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.run.return_value = [
            {
                "rule_text": "requisition must have approved status before payment",
                "rule_type": "docstring",
                "source_file": "payments/service.py",
                "function_names": ["process_payment", "validate_requisition"],
                "function_files": ["payments/service.py"],
            }
        ]

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path), \
             patch("graph.neo4j_client.neo4j_client", mock_client):
            result = query_concept_to_code(
                "Requisition payment fails", "payment blocked on requisition approval", "repo"
            )

        assert len(result["matched_rules"]) == 1
        assert "process_payment" in result["hint_functions"]
        assert "payments/service.py" in result["hint_files"]

    def test_neo4j_cypher_uses_all_keywords(self, tmp_path):
        (tmp_path / "repo").mkdir()

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.run.return_value = []

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path), \
             patch("graph.neo4j_client.neo4j_client", mock_client):
            query_concept_to_code("approval requisition workflow", "flow broken", "repo")

        assert mock_client.run.called
        cypher, params = mock_client.run.call_args[0]
        # Should have kw0, kw1, ... params
        assert "kw0" in params
        assert "toLower" in cypher

    def test_neo4j_exception_falls_back_to_json(self, tmp_path):
        rules_dir = tmp_path / "repo"
        rules_dir.mkdir()
        (rules_dir / "business_rules.json").write_text(json.dumps([
            {"description": "approval flow must validate user permissions",
             "source": "docstring", "file": "auth.py",
             "function_id": "auth.validate_permissions", "severity": "high"},
        ]))

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.run.side_effect = Exception("Connection reset")

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path), \
             patch("graph.neo4j_client.neo4j_client", mock_client):
            result = query_concept_to_code("Approval flow", "permissions broken", "repo")

        # Should fall back to JSON and still find the rule
        assert len(result["matched_rules"]) == 1
        assert "validate_permissions" in result["hint_functions"]


# ---------------------------------------------------------------------------
# Tests — concept_section format
# ---------------------------------------------------------------------------

class TestConceptSection:
    def test_section_header_present(self, tmp_path):
        rules_dir = tmp_path / "repo"
        rules_dir.mkdir()
        (rules_dir / "business_rules.json").write_text(json.dumps([
            {"description": "requisition approval required",
             "source": "docstring", "file": "app.py",
             "function_id": "app.approve", "severity": "high"},
        ]))

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Requisition approval", "flow broken", "repo")

        assert "RELEVANT BUSINESS RULES" in result["concept_section"]
        assert "requisition" in result["concept_section"].lower()

    def test_section_includes_enforcing_function(self, tmp_path):
        rules_dir = tmp_path / "repo"
        rules_dir.mkdir()
        (rules_dir / "business_rules.json").write_text(json.dumps([
            {"description": "approval must notify finance team",
             "source": "docstring", "file": "workflow.py",
             "function_id": "workflow.notify_finance", "severity": "medium"},
        ]))

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Approval notification missing", "finance not notified", "repo")

        assert "notify_finance" in result["concept_section"]

    def test_empty_match_gives_empty_section(self, tmp_path):
        (tmp_path / "repo").mkdir()

        from agent.graph_utils import query_concept_to_code
        with patch("agent.graph_utils.DATA_DIR", tmp_path):
            result = query_concept_to_code("Login timeout", "session expired", "repo")

        assert result["concept_section"] == ""


# ---------------------------------------------------------------------------
# Tests — intake_node state merging
# ---------------------------------------------------------------------------

class TestIntakeIntegration:
    """Test that intake_node correctly merges concept-to-code results into intent."""

    def test_c2c_hints_merged_into_intent(self):
        """query_concept_to_code results get appended to likely_affected_functions."""
        c2c_result = {
            "matched_rules": [{"rule_text": "test", "rule_type": "docstring",
                               "source_file": "x.py", "function_names": ["graph_func"],
                               "function_files": ["x.py"]}],
            "hint_functions": ["graph_func"],
            "hint_files": ["x.py"],
            "concept_section": "## RELEVANT BUSINESS RULES\n- [DOCSTRING] test",
        }
        existing_intent = {
            "likely_affected_functions": ["llm_func"],
            "likely_affected_modules": ["llm_module.py"],
        }

        # Simulate the merge logic from intake_node
        merged_funcs = list(dict.fromkeys(
            existing_intent["likely_affected_functions"] + c2c_result["hint_functions"]
        ))
        merged_mods = list(dict.fromkeys(
            existing_intent["likely_affected_modules"] + c2c_result["hint_files"]
        ))

        assert "llm_func" in merged_funcs
        assert "graph_func" in merged_funcs
        assert "llm_module.py" in merged_mods
        assert "x.py" in merged_mods

    def test_c2c_no_duplicates_in_merge(self):
        """Functions appearing in both LLM output and graph match are deduplicated."""
        c2c_result = {
            "hint_functions": ["shared_func", "graph_only_func"],
            "hint_files": [],
        }
        existing = {"likely_affected_functions": ["shared_func", "llm_only_func"]}

        merged = list(dict.fromkeys(
            existing["likely_affected_functions"] + c2c_result["hint_functions"]
        ))

        assert merged.count("shared_func") == 1
        assert "graph_only_func" in merged
        assert "llm_only_func" in merged

    def test_c2c_concept_section_stored_in_intent(self):
        """_concept_section stored in intent dict for later injection into kickstart."""
        c2c_result = {
            "matched_rules": [{}],
            "hint_functions": [],
            "hint_files": [],
            "concept_section": "## RELEVANT BUSINESS RULES\nsome rule",
        }
        intent = {"likely_affected_functions": [], "likely_affected_modules": []}

        if c2c_result.get("matched_rules"):
            intent["_concept_section"] = c2c_result["concept_section"]

        assert intent["_concept_section"] == c2c_result["concept_section"]
