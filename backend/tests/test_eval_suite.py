"""
Tests for eval_suite.py — scoring logic, regression detection, and timeout.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from unittest.mock import patch, MagicMock
from agent.eval_suite import (
    _score_localization_hit,
    _score_root_cause_match,
    _score_fix_generated,
    _score_review_approved,
    score_single_run,
    _detect_regressions,
    _append_to_history,
    load_eval_bugs,
)


# ── Scoring helpers ──────────────────────────────────────────────────

class TestLocalizationHit:

    def test_exact_match(self):
        result = {"localization": {"fault_files": ["api/profile.py"]}}
        assert _score_localization_hit(result, ["api/profile.py"]) is True

    def test_suffix_match(self):
        result = {"localization": {"fault_files": ["backend/api/profile.py"]}}
        assert _score_localization_hit(result, ["api/profile.py"]) is True

    def test_no_match(self):
        result = {"localization": {"fault_files": ["api/users.py"]}}
        assert _score_localization_hit(result, ["api/profile.py"]) is False

    def test_empty_localization(self):
        result = {"localization": {}}
        assert _score_localization_hit(result, ["api/profile.py"]) is False

    def test_none_localization(self):
        result = {}
        assert _score_localization_hit(result, ["api/profile.py"]) is False

    def test_case_insensitive(self):
        result = {"localization": {"fault_files": ["API/Profile.py"]}}
        assert _score_localization_hit(result, ["api/profile.py"]) is True


class TestRootCauseMatch:

    def test_keyword_match_above_threshold(self):
        result = {"localization": {"root_cause_hypothesis": "The null check for avatar field is missing, default value not set"}}
        assert _score_root_cause_match(result, "missing null check avatar field default value") is True

    def test_keyword_match_below_threshold(self):
        result = {"localization": {"root_cause_hypothesis": "Something unrelated"}}
        assert _score_root_cause_match(result, "missing null check avatar field default value") is False

    def test_empty_hypothesis(self):
        result = {"localization": {"root_cause_hypothesis": ""}}
        assert _score_root_cause_match(result, "missing null check") is False

    def test_empty_expected(self):
        result = {"localization": {"root_cause_hypothesis": "some hypothesis"}}
        assert _score_root_cause_match(result, "") is False


class TestFixGenerated:

    def test_patches_present(self):
        result = {"repair": {"patches": [{"file_path": "a.py", "original_code": "x", "patched_code": "y"}]}}
        assert _score_fix_generated(result) is True

    def test_no_patches(self):
        result = {"repair": {"patches": []}}
        assert _score_fix_generated(result) is False

    def test_no_repair(self):
        result = {}
        assert _score_fix_generated(result) is False


class TestReviewApproved:

    def test_approved(self):
        result = {"review": {"verdict": "APPROVE"}}
        assert _score_review_approved(result) is True

    def test_changes_requested(self):
        result = {"review": {"verdict": "CHANGES_REQUESTED"}}
        assert _score_review_approved(result) is False

    def test_escalated(self):
        result = {"review": {"verdict": "ESCALATE"}}
        assert _score_review_approved(result) is False


class TestScoreSingleRun:

    def test_full_pass(self):
        result = {
            "localization": {"fault_files": ["api/profile.py"], "root_cause_hypothesis": "null check avatar"},
            "repair": {"patches": [{"file_path": "a.py"}]},
            "review": {"verdict": "APPROVE", "confidence": 0.95},
            "status": "done",
        }
        bug = {
            "ticket_id": "T-1",
            "expected_files": ["api/profile.py"],
            "expected_root_cause": "null check avatar",
        }
        score = score_single_run(result, bug)
        assert score["localization_hit"] is True
        assert score["fix_generated"] is True
        assert score["review_approved"] is True
        assert score["confidence"] == 0.95

    def test_full_fail(self):
        result = {"status": "failed", "error": "timeout"}
        bug = {"ticket_id": "T-2", "expected_files": ["a.py"], "expected_root_cause": "x"}
        score = score_single_run(result, bug)
        assert score["localization_hit"] is False
        assert score["fix_generated"] is False
        assert score["review_approved"] is False


# ── Regression detection ─────────────────────────────────────────────

class TestRegressionDetection:

    def test_no_regression(self):
        previous = {"pass_rate": 0.8, "localization_accuracy": 0.9, "fix_rate": 0.7, "approval_rate": 0.6}
        current = {"pass_rate": 0.85, "localization_accuracy": 0.9, "fix_rate": 0.75, "approval_rate": 0.65}
        result = _detect_regressions(previous, current)
        assert result["regressions"] == []
        assert len(result["improvements"]) > 0

    def test_regression_detected(self):
        previous = {"pass_rate": 0.8, "localization_accuracy": 0.9, "fix_rate": 0.7, "approval_rate": 0.6}
        current = {"pass_rate": 0.5, "localization_accuracy": 0.9, "fix_rate": 0.7, "approval_rate": 0.6}
        result = _detect_regressions(previous, current)
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["metric"] == "pass_rate"

    def test_small_drop_not_regression(self):
        previous = {"pass_rate": 0.8, "localization_accuracy": 0.9, "fix_rate": 0.7, "approval_rate": 0.6}
        current = {"pass_rate": 0.78, "localization_accuracy": 0.88, "fix_rate": 0.68, "approval_rate": 0.58}
        result = _detect_regressions(previous, current)
        assert result["regressions"] == []


# ── History tracking ─────────────────────────────────────────────────

class TestHistoryTracking:

    def test_append_creates_file(self, tmp_path):
        history_file = tmp_path / "eval_history.json"
        summary = {"pass_rate": 0.8, "localization_accuracy": 0.9, "fix_rate": 0.7, "approval_rate": 0.6, "avg_confidence": 0.85}
        _append_to_history(history_file, summary, 1000.0)
        assert history_file.exists()
        data = json.loads(history_file.read_text())
        assert len(data) == 1
        assert data[0]["pass_rate"] == 0.8

    def test_append_preserves_history(self, tmp_path):
        history_file = tmp_path / "eval_history.json"
        history_file.write_text('[{"timestamp": 500, "pass_rate": 0.5}]')
        summary = {"pass_rate": 0.8, "localization_accuracy": 0.9, "fix_rate": 0.7, "approval_rate": 0.6, "avg_confidence": 0.85}
        _append_to_history(history_file, summary, 1000.0)
        data = json.loads(history_file.read_text())
        assert len(data) == 2


# ── Bug loader ───────────────────────────────────────────────────────

class TestLoadEvalBugs:

    def test_loads_sample_bugs(self):
        bugs = load_eval_bugs("nonexistent_repo_for_test")
        assert len(bugs) >= 1
        assert "ticket_id" in bugs[0]
