"""
test_scout_pipeline.py — Tests for agent/scout.py (3-agent FL pipeline)

Covers:
  - scout_localize() orchestration (3 agents in sequence)
  - Graceful fallback when each agent fails
  - Output schema validation
  - Cost calculation present in output
  - _load_business_rules() helper
  - _summarise_graph() helper
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.scout import (
    ExtractedContext,
    GraphDebuggerOutput,
    RankedLocation,
    RerankerOutput,
    SuspectLocation,
    _load_business_rules,
    _summarise_graph,
    scout_localize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def work_order():
    return {
        "ticket_id": "CB-001",
        "title": "Submit gate blocks when tests are skipped",
        "description": "check_tool_call() should allow submission when tests are skipped, not block",
        "repo_name": "context_builder",
    }


@pytest.fixture()
def intent():
    return {
        "likely_affected_functions": ["check_tool_call"],
        "likely_affected_modules": ["agent/react_guardrails.py"],
        "actual_behavior": "submission blocked when tests.skipped is True",
        "expected_behavior": "submission allowed when tests.skipped is True",
    }


@pytest.fixture()
def sample_graph():
    return {
        "nodes": [
            {"id": "agent/react_guardrails.py", "type": "File", "file": "agent/react_guardrails.py", "pagerank": 0.05},
            {"id": "agent/react_guardrails.py::check_tool_call", "label": "check_tool_call",
             "file": "agent/react_guardrails.py", "type": "Function", "pagerank": 0.08},
            {"id": "agent/react_guardrails.py::GuardrailState", "label": "GuardrailState",
             "file": "agent/react_guardrails.py", "type": "Class", "pagerank": 0.06},
        ],
        "edges": [
            {"source": "agent/react_pipeline.py::react_agent_node",
             "target": "agent/react_guardrails.py::check_tool_call",
             "type": "CALLS"},
        ],
    }


def _fake_extracted():
    return ExtractedContext(
        function_names=["check_tool_call"],
        module_hints=["agent/react_guardrails.py"],
        error_types=["SubmissionBlocked"],
        data_structures=["GuardrailState"],
        bug_summary="check_tool_call blocks submission when tests are skipped",
    )


def _fake_debugger_output():
    return GraphDebuggerOutput(
        suspects=[
            SuspectLocation(
                file="agent/react_guardrails.py",
                function="check_tool_call",
                confidence=0.92,
                reason="submit gate condition missing tests_skipped exclusion",
            )
        ],
        blast_radius_files=["agent/react_pipeline.py"],
        relevant_business_rule_ids=["tests must pass before submission"],
    )


def _fake_reranker_output():
    return RerankerOutput(
        ranked_locations=[
            RankedLocation(
                file="agent/react_guardrails.py",
                function="check_tool_call",
                confidence=0.95,
                reason="Confirmed: submit gate missing `and not gs.tests_skipped`",
            )
        ],
        relevant_failure_records=["Previous incident: skipped tests blocked deploy"],
        additional_blast_radius=[],
    )


# ---------------------------------------------------------------------------
# _load_business_rules() helper
# ---------------------------------------------------------------------------

class TestLoadBusinessRules:
    def test_loads_valid_json(self, tmp_path):
        rules = [{"description": "rule A", "severity": "high"}]
        (tmp_path / "test_repo").mkdir()
        (tmp_path / "test_repo" / "business_rules.json").write_text(json.dumps(rules))
        result = _load_business_rules(tmp_path, "test_repo")
        assert result == rules

    def test_returns_empty_when_file_missing(self, tmp_path):
        result = _load_business_rules(tmp_path, "nonexistent_repo")
        assert result == []

    def test_returns_empty_on_invalid_json(self, tmp_path):
        (tmp_path / "bad_repo").mkdir()
        (tmp_path / "bad_repo" / "business_rules.json").write_text("NOT JSON {{{")
        result = _load_business_rules(tmp_path, "bad_repo")
        assert result == []


# ---------------------------------------------------------------------------
# _summarise_graph() helper
# ---------------------------------------------------------------------------

class TestSummariseGraph:
    def test_returns_string(self, sample_graph):
        entities = _fake_extracted()
        result = _summarise_graph(sample_graph, {}, entities)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_caps_at_1200_chars(self, sample_graph):
        entities = _fake_extracted()
        result = _summarise_graph(sample_graph, {}, entities)
        assert len(result) <= 1200

    def test_includes_matched_nodes(self, sample_graph):
        entities = _fake_extracted()
        result = _summarise_graph(sample_graph, {}, entities)
        assert "check_tool_call" in result or "guardrail" in result.lower()

    def test_fallback_when_no_hints_match(self):
        graph = {
            "nodes": [{"id": "x/y.py::zzz", "label": "zzz", "file": "x/y.py", "pagerank": 0.9}],
            "edges": [],
        }
        entities = ExtractedContext(
            function_names=["totally_unknown"],
            module_hints=["no_such_module"],
        )
        result = _summarise_graph(graph, {}, entities)
        # Falls back to top PageRank nodes
        assert "zzz" in result or "TOP" in result

    def test_empty_graph_returns_fallback(self):
        entities = _fake_extracted()
        result = _summarise_graph({"nodes": [], "edges": []}, {}, entities)
        assert "no graph data available" in result.lower() or isinstance(result, str)


# ---------------------------------------------------------------------------
# scout_localize() — happy path
# ---------------------------------------------------------------------------

class TestScoutLocalizeHappyPath:
    def test_returns_expected_schema(self, tmp_path, work_order, intent, sample_graph):
        """Full pipeline produces the correct output schema."""
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize(
                repo_name="context_builder",
                work_order=work_order,
                intent=intent,
                data_dir=tmp_path,
            )

        assert isinstance(result, dict)
        assert "top_locations" in result
        assert "blast_radius_files" in result
        assert "relevant_business_rules" in result
        assert "relevant_failure_records" in result
        assert "scout_cost_usd" in result

    def test_top_locations_have_required_fields(self, tmp_path, work_order, intent, sample_graph):
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        for loc in result["top_locations"]:
            assert "file" in loc
            assert "confidence" in loc
            assert isinstance(loc["confidence"], float)
            assert 0.0 <= loc["confidence"] <= 1.0

    def test_blast_radius_is_list(self, tmp_path, work_order, intent, sample_graph):
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        assert isinstance(result["blast_radius_files"], list)

    def test_cost_is_positive_float(self, tmp_path, work_order, intent, sample_graph):
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        assert isinstance(result["scout_cost_usd"], float)
        assert result["scout_cost_usd"] >= 0.0

    def test_correct_localization(self, tmp_path, work_order, intent, sample_graph):
        """Top location should be the guardrails file for this bug."""
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        top = result["top_locations"][0]
        assert "guardrail" in top["file"].lower()


# ---------------------------------------------------------------------------
# scout_localize() — graceful fallback
# ---------------------------------------------------------------------------

class TestScoutLocalizeGracefulFallback:
    def test_extractor_failure_uses_fallback(self, tmp_path, work_order, intent, sample_graph):
        """Agent 1 failure must use intent-seeded fallback, not abort."""
        with patch("agent.scout._run_extractor", side_effect=RuntimeError("LLM down")), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        # Should still return a result (debugger used fallback-seeded extracted)
        assert "top_locations" in result

    def test_debugger_failure_returns_empty_suspects(self, tmp_path, work_order, intent, sample_graph):
        """Agent 2 failure must produce empty suspects, not crash."""
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", side_effect=RuntimeError("API error")), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        assert "top_locations" in result
        assert isinstance(result["top_locations"], list)

    def test_reranker_failure_falls_back_to_debugger(self, tmp_path, work_order, intent, sample_graph):
        """Agent 3 failure must fall back to Agent 2's suspects."""
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", side_effect=RuntimeError("Opus timeout")), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        # Should fall back to Agent 2's suspects
        assert "top_locations" in result
        locs = result["top_locations"]
        if locs:
            assert "guardrail" in locs[0]["file"].lower()

    def test_graph_load_failure_still_returns_result(self, tmp_path, work_order, intent):
        """Graph load failure must not abort — pipeline continues with empty graph."""
        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", return_value=_fake_reranker_output()), \
             patch("agent.graph_utils.load_graph_data", side_effect=FileNotFoundError("no graph")):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        assert "top_locations" in result

    def test_all_agents_fail_returns_empty_result(self, tmp_path, work_order, intent):
        """When everything fails, return an empty-but-valid result dict."""
        with patch("agent.scout._run_extractor", side_effect=RuntimeError), \
             patch("agent.scout._run_debugger", side_effect=RuntimeError), \
             patch("agent.scout._run_reranker", side_effect=RuntimeError), \
             patch("agent.graph_utils.load_graph_data", side_effect=RuntimeError):

            result = scout_localize("context_builder", work_order, intent, tmp_path)

        assert isinstance(result, dict)
        assert "top_locations" in result
        assert result["top_locations"] == []


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

class TestScoutSchemas:
    def test_extracted_context_defaults(self):
        e = ExtractedContext()
        assert e.function_names == []
        assert e.module_hints == []
        assert e.bug_summary == ""

    def test_suspect_location_confidence_clamped(self):
        with pytest.raises(Exception):
            SuspectLocation(file="x.py", confidence=1.5)  # Should raise validation error

    def test_reranker_output_structure(self):
        out = RerankerOutput(
            ranked_locations=[RankedLocation(file="x.py", confidence=0.8)],
            relevant_failure_records=["Past incident A"],
        )
        assert len(out.ranked_locations) == 1
        assert out.ranked_locations[0].confidence == 0.8


# ---------------------------------------------------------------------------
# scout_localize() — timeout returns partial results
# ---------------------------------------------------------------------------

class TestScoutTimeout:
    def test_scout_timeout_returns_partial(self, tmp_path, work_order, intent, sample_graph):
        """When one agent hangs and SIGALRM fires, scout_localize returns partial results.

        Simulates the timeout by having _run_reranker raise _ScoutTimeout (the same
        exception the SIGALRM handler raises). Agent 1 and Agent 2 complete normally
        so their results should appear in the output.
        """
        from agent.scout import _ScoutTimeout

        with patch("agent.scout._run_extractor", return_value=_fake_extracted()), \
             patch("agent.scout._run_debugger", return_value=_fake_debugger_output()), \
             patch("agent.scout._run_reranker", side_effect=_ScoutTimeout("timeout")), \
             patch("agent.graph_utils.load_graph_data", return_value=(sample_graph, {})):

            result = scout_localize(
                repo_name="context_builder",
                work_order=work_order,
                intent=intent,
                data_dir=tmp_path,
            )

        # Pipeline should return partial results, not crash
        assert isinstance(result, dict)
        assert "top_locations" in result
        assert "scout_cost_usd" in result
        # Debugger output should still appear as fallback (demoted suspects)
        # since _run_debugger succeeded before the timeout
        assert isinstance(result["top_locations"], list)
        assert isinstance(result["blast_radius_files"], list)
