"""
Tests for the new unified eval pipeline (agent.eval package).

Tests:
  - Dataset loading and validation
  - Scoring functions (existing + new metrics)
  - Regression detection
  - Report generation
  - Runner work_order construction
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BUG = {
    "ticket_id": "TEST-001",
    "title": "Test bug",
    "description": "Something is broken",
    "repo_url": "https://github.com/test/repo",
    "repo_sha": "abc1234567890",
    "fix_sha": "def1234567890",
    "expected_files": ["src/module.py"],
    "expected_root_cause": "missing null check validation",
    "difficulty": "single-file",
    "source": "open-source",
    "priority": "high",
}

SAMPLE_BUG_MULTI = {
    **SAMPLE_BUG,
    "ticket_id": "TEST-002",
    "difficulty": "multi-file",
    "expected_files": ["src/module.py", "src/utils.py"],
    "expected_patch_files": ["src/module.py", "src/utils.py"],
}

PASSING_RESULT = {
    "status": "done",
    "localization": {
        "fault_files": ["src/module.py"],
        "root_cause_hypothesis": "The issue is a missing null check on the validation path",
    },
    "repair": {
        "patches": [
            {"file_path": "src/module.py", "original_code": "x", "patched_code": "y", "explanation": "fix"}
        ],
    },
    "review": {
        "verdict": "APPROVE",
        "confidence": 0.92,
    },
    "test_result": "passed\n5 tests passed",
    "cost_usd": 0.45,
    "tool_call_count": 18,
}

FAILING_RESULT = {
    "status": "failed",
    "localization": {
        "fault_files": ["src/wrong.py"],
        "root_cause_hypothesis": "Something unrelated",
    },
    "repair": {"patches": []},
    "review": {"verdict": "CHANGES_REQUESTED", "confidence": 0.3},
    "test_result": "",
    "error": "Pipeline crashed",
}


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_load_valid_dataset(self, tmp_path: Path):
        from agent.eval.dataset import load_eval_dataset

        bugs = [SAMPLE_BUG]
        dataset_file = tmp_path / "bugs.json"
        dataset_file.write_text(json.dumps(bugs))

        result = load_eval_dataset(dataset_file)
        assert len(result) == 1
        assert result[0]["ticket_id"] == "TEST-001"

    def test_load_fills_defaults(self, tmp_path: Path):
        from agent.eval.dataset import load_eval_dataset

        bugs = [SAMPLE_BUG]
        dataset_file = tmp_path / "bugs.json"
        dataset_file.write_text(json.dumps(bugs))

        result = load_eval_dataset(dataset_file)
        assert result[0]["language"] == "python"
        assert result[0]["category"] == "unknown"
        assert result[0]["tags"] == []
        assert result[0]["repo_name"] == "repo"

    def test_load_missing_file(self):
        from agent.eval.dataset import load_eval_dataset

        with pytest.raises(FileNotFoundError):
            load_eval_dataset(Path("/nonexistent/bugs.json"))

    def test_load_invalid_schema(self, tmp_path: Path):
        from agent.eval.dataset import load_eval_dataset

        bugs = [{"ticket_id": "BAD"}]  # Missing required fields
        dataset_file = tmp_path / "bugs.json"
        dataset_file.write_text(json.dumps(bugs))

        with pytest.raises(ValueError, match="validation failed"):
            load_eval_dataset(dataset_file)

    def test_load_invalid_difficulty(self, tmp_path: Path):
        from agent.eval.dataset import load_eval_dataset

        bug = {**SAMPLE_BUG, "difficulty": "impossible"}
        dataset_file = tmp_path / "bugs.json"
        dataset_file.write_text(json.dumps([bug]))

        with pytest.raises(ValueError, match="difficulty"):
            load_eval_dataset(dataset_file)

    def test_load_not_array(self, tmp_path: Path):
        from agent.eval.dataset import load_eval_dataset

        dataset_file = tmp_path / "bugs.json"
        dataset_file.write_text(json.dumps({"not": "an array"}))

        with pytest.raises(ValueError, match="JSON array"):
            load_eval_dataset(dataset_file)


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_passing_case(self):
        from agent.eval.scoring import score_case

        score = score_case(PASSING_RESULT, SAMPLE_BUG, "fixed")
        assert score["localization_hit"] is True
        assert score["root_cause_match"] is True
        assert score["fix_generated"] is True
        assert score["review_approved"] is True
        assert score["full_pass"] is True
        assert score["pipeline"] == "fixed"
        assert score["confidence"] == 0.92

    def test_score_failing_case(self):
        from agent.eval.scoring import score_case

        score = score_case(FAILING_RESULT, SAMPLE_BUG, "react")
        assert score["localization_hit"] is False
        assert score["fix_generated"] is False
        assert score["review_approved"] is False
        assert score["full_pass"] is False

    def test_localization_suffix_matching(self):
        from agent.eval.scoring import score_case

        result = {
            **PASSING_RESULT,
            "localization": {"fault_files": ["full/path/to/src/module.py"]},
        }
        score = score_case(result, SAMPLE_BUG, "fixed")
        assert score["localization_hit"] is True

    def test_patch_correctness_exact_match(self):
        from agent.eval.scoring import score_case

        score = score_case(PASSING_RESULT, SAMPLE_BUG, "fixed")
        assert score["patch_correctness"] == 1.0

    def test_patch_correctness_no_patches(self):
        from agent.eval.scoring import score_case

        score = score_case(FAILING_RESULT, SAMPLE_BUG, "fixed")
        assert score["patch_correctness"] == 0.0

    def test_multi_file_complete_single_file_always_true(self):
        from agent.eval.scoring import score_case

        score = score_case(PASSING_RESULT, SAMPLE_BUG, "fixed")
        assert score["multi_file_complete"] is True

    def test_multi_file_complete_missing_file(self):
        from agent.eval.scoring import score_case

        # Only patches one of two expected files
        result = {
            **PASSING_RESULT,
            "repair": {
                "patches": [
                    {"file_path": "src/module.py", "original_code": "x", "patched_code": "y", "explanation": "fix"}
                ]
            },
        }
        score = score_case(result, SAMPLE_BUG_MULTI, "fixed")
        assert score["multi_file_complete"] is False

    def test_multi_file_complete_all_files(self):
        from agent.eval.scoring import score_case

        result = {
            **PASSING_RESULT,
            "repair": {
                "patches": [
                    {"file_path": "src/module.py", "original_code": "x", "patched_code": "y", "explanation": "fix"},
                    {"file_path": "src/utils.py", "original_code": "a", "patched_code": "b", "explanation": "fix"},
                ]
            },
        }
        score = score_case(result, SAMPLE_BUG_MULTI, "fixed")
        assert score["multi_file_complete"] is True

    def test_test_pass_true(self):
        from agent.eval.scoring import score_case

        score = score_case(PASSING_RESULT, SAMPLE_BUG, "fixed")
        assert score["test_pass"] is True

    def test_test_pass_false(self):
        from agent.eval.scoring import score_case

        score = score_case(FAILING_RESULT, SAMPLE_BUG, "fixed")
        assert score["test_pass"] is False

    def test_cost_tracking(self):
        from agent.eval.scoring import score_case

        score = score_case(PASSING_RESULT, SAMPLE_BUG, "react")
        assert score["cost_usd"] == 0.45
        assert score["tool_call_count"] == 18


class TestBuildSummary:
    def test_summary_basic(self):
        from agent.eval.scoring import score_case, build_summary

        scores = [
            score_case(PASSING_RESULT, SAMPLE_BUG, "fixed"),
            score_case(FAILING_RESULT, SAMPLE_BUG, "fixed"),
        ]
        summary = build_summary(scores, pipeline="fixed")

        assert summary["total"] == 2
        assert summary["pass_rate"] == 0.5
        assert summary["localization_accuracy"] == 0.5
        assert summary["fix_rate"] == 0.5
        assert len(summary["failures"]) == 1

    def test_summary_empty(self):
        from agent.eval.scoring import build_summary

        summary = build_summary([], pipeline="fixed")
        assert summary["total"] == 0
        assert summary["pass_rate"] == 0.0

    def test_summary_filters_by_pipeline(self):
        from agent.eval.scoring import score_case, build_summary

        scores = [
            score_case(PASSING_RESULT, SAMPLE_BUG, "fixed"),
            score_case(PASSING_RESULT, SAMPLE_BUG, "react"),
        ]
        fixed_summary = build_summary(scores, pipeline="fixed")
        react_summary = build_summary(scores, pipeline="react")

        assert fixed_summary["total"] == 1
        assert react_summary["total"] == 1


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestRegression:
    def test_no_regression(self):
        from agent.eval.regression import detect_regressions

        prev = {"pass_rate": 0.80, "localization_accuracy": 0.90, "fix_rate": 0.85, "approval_rate": 0.80}
        curr = {"pass_rate": 0.82, "localization_accuracy": 0.90, "fix_rate": 0.85, "approval_rate": 0.82}

        result = detect_regressions(prev, curr)
        assert result["regressions"] == []

    def test_regression_detected(self):
        from agent.eval.regression import detect_regressions

        prev = {"pass_rate": 0.80, "localization_accuracy": 0.90, "fix_rate": 0.85, "approval_rate": 0.80}
        curr = {"pass_rate": 0.60, "localization_accuracy": 0.90, "fix_rate": 0.85, "approval_rate": 0.80}

        result = detect_regressions(prev, curr)
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["metric"] == "pass_rate"

    def test_improvement_detected(self):
        from agent.eval.regression import detect_regressions

        prev = {"pass_rate": 0.60, "localization_accuracy": 0.70}
        curr = {"pass_rate": 0.80, "localization_accuracy": 0.90}

        result = detect_regressions(prev, curr)
        assert len(result["improvements"]) == 2


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:
    def _make_report(self):
        from agent.eval.runner import EvalRunReport, EvalCaseResult
        from agent.eval.scoring import score_case, build_summary

        scores = [
            score_case(PASSING_RESULT, SAMPLE_BUG, "fixed"),
            score_case(FAILING_RESULT, SAMPLE_BUG, "react"),
        ]

        results = [
            EvalCaseResult(
                ticket_id="TEST-001", pipeline="fixed",
                score=scores[0], duration_seconds=120, cost_usd=0.45,
            ),
            EvalCaseResult(
                ticket_id="TEST-001", pipeline="react",
                score=scores[1], duration_seconds=90, cost_usd=0.30, error="crashed",
            ),
        ]

        report = EvalRunReport(
            run_id="test-123",
            timestamp=1700000000.0,
            dataset_path="eval/bugs.json",
            total_bugs=1,
            pipelines=["fixed", "react"],
            results=results,
            summary={
                "fixed": build_summary([scores[0]], pipeline="fixed"),
                "react": build_summary([scores[1]], pipeline="react"),
            },
        )
        return report

    def test_markdown_report_contains_sections(self):
        from agent.eval.report import generate_markdown_report

        report = self._make_report()
        md = generate_markdown_report(report)

        assert "# Eval Report" in md
        assert "## Summary" in md
        assert "## Per-Bug Breakdown" in md
        assert "## Failure Analysis" in md
        assert "## Cost & Performance" in md

    def test_json_report_serializable(self):
        from agent.eval.report import generate_json_report

        report = self._make_report()
        json_str = generate_json_report(report)
        data = json.loads(json_str)

        assert data["run_id"] == "test-123"
        assert data["total_bugs"] == 1
        assert len(data["results"]) == 2

    def test_report_to_dict(self):
        report = self._make_report()
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["run_id"] == "test-123"
        assert isinstance(d["results"], list)


# ---------------------------------------------------------------------------
# Runner helpers tests
# ---------------------------------------------------------------------------

class TestRunnerHelpers:
    def test_bug_to_work_order(self):
        from agent.eval.runner import _bug_to_work_order

        wo = _bug_to_work_order(SAMPLE_BUG, Path("/tmp/repo"))
        assert wo["ticket_id"] == "TEST-001"
        assert wo["repo_path"] == "/tmp/repo"
        # repo_name comes from bug dict; SAMPLE_BUG has no repo_name so falls back to ticket_id
        assert wo["repo_name"] == "test-001"

    def test_bug_to_work_order_with_repo_name(self):
        from agent.eval.runner import _bug_to_work_order

        bug = {**SAMPLE_BUG, "repo_name": "myrepo"}
        wo = _bug_to_work_order(bug, Path("/tmp/repo"))
        assert wo["repo_name"] == "myrepo"

    def test_error_score(self):
        from agent.eval.runner import _error_score

        score = _error_score(SAMPLE_BUG, "fixed", "clone failed")
        assert score["full_pass"] is False
        assert score["error"] == "clone failed"
        assert score["pipeline"] == "fixed"

    def test_pick_winner_one_pass(self):
        from agent.eval.runner import _pick_winner, EvalCaseResult

        r1 = EvalCaseResult("T", "fixed", {"full_pass": True}, cost_usd=0.5, duration_seconds=100)
        r2 = EvalCaseResult("T", "react", {"full_pass": False}, cost_usd=0.3, duration_seconds=80)

        assert _pick_winner(r1, r2, "fixed", "react") == "fixed"

    def test_pick_winner_both_pass_cheaper_wins(self):
        from agent.eval.runner import _pick_winner, EvalCaseResult

        r1 = EvalCaseResult("T", "fixed", {"full_pass": True}, cost_usd=0.5, duration_seconds=100)
        r2 = EvalCaseResult("T", "react", {"full_pass": True}, cost_usd=0.3, duration_seconds=80)

        assert "react" in _pick_winner(r1, r2, "fixed", "react")

    def test_pick_winner_neither_pass(self):
        from agent.eval.runner import _pick_winner, EvalCaseResult

        r1 = EvalCaseResult("T", "fixed", {"full_pass": False}, cost_usd=0.5, duration_seconds=100)
        r2 = EvalCaseResult("T", "react", {"full_pass": False}, cost_usd=0.3, duration_seconds=80)

        assert _pick_winner(r1, r2, "fixed", "react") == "neither"
