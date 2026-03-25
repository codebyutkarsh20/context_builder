"""
Unit tests for enricher/decision_points.py — decision point extraction and classification.

Covers:
  - classify_condition: threshold, role_check, status_check, feature_flag,
    error_guard, logic_branch
  - _make_decision_point: valid creation, trivial error_guard filtering, short conditions
  - extract_decision_points: from parsed file structures
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enricher.decision_points import (
    _make_decision_point,
    classify_condition,
    extract_decision_points,
)


# ---------------------------------------------------------------------------
# classify_condition
# ---------------------------------------------------------------------------

class TestClassifyCondition:

    # ── Threshold ────────────────────────────────────────────────────────

    def test_constant_with_limit_suffix(self):
        assert classify_condition("x > MAX_RETRIES") == "threshold"

    def test_constant_with_timeout_suffix(self):
        assert classify_condition("elapsed > REQUEST_TIMEOUT") == "threshold"

    def test_constant_with_rate_suffix(self):
        assert classify_condition("count >= RATE_LIMIT") == "threshold"

    def test_numeric_literal_with_comparison(self):
        assert classify_condition("age > 18") == "threshold"

    def test_numeric_literal_less_than(self):
        assert classify_condition("score < 50") == "threshold"

    def test_numeric_equal_check(self):
        assert classify_condition("attempts == 3") == "threshold"

    def test_float_threshold(self):
        assert classify_condition("confidence >= 0.8") == "threshold"

    # ── Role check ────────────────────────────────────────────────────────

    def test_is_admin(self):
        assert classify_condition("user.is_admin") == "role_check"

    def test_has_permission(self):
        assert classify_condition("user.has_perm('edit')") == "role_check"

    def test_is_staff(self):
        assert classify_condition("request.user.is_staff") == "role_check"

    def test_user_role(self):
        assert classify_condition("user.role == 'admin'") == "role_check"

    def test_access_level(self):
        assert classify_condition("access_level >= 5") == "role_check"

    # ── Status check ─────────────────────────────────────────────────────

    def test_is_active(self):
        assert classify_condition("user.is_active") == "status_check"

    def test_status_equals(self):
        assert classify_condition("order.status == 'pending'") == "status_check"

    def test_is_enabled(self):
        assert classify_condition("account.is_enabled") == "status_check"

    def test_is_verified(self):
        assert classify_condition("email.is_verified") == "status_check"

    def test_is_archived(self):
        assert classify_condition("item.is_archived") == "status_check"

    def test_phase_check(self):
        assert classify_condition("pipeline.phase == 'review'") == "status_check"

    # ── Feature flag ─────────────────────────────────────────────────────

    def test_feature_flag_variable(self):
        # "feature_flag" keyword (not is_enabled which matches status first)
        assert classify_condition("feature_flag.check('new_ui')") == "feature_flag"

    def test_feature_enabled_function(self):
        assert classify_condition("feature_enabled('dark_mode')") == "feature_flag"

    def test_is_feature(self):
        # "flag" is standalone word → matches \bflag\b in _FEATURE_FLAG_PATTERNS
        assert classify_condition("flag == 'new_ui'") == "feature_flag"

    def test_ab_test(self):
        assert classify_condition("ab_test.variant == 'B'") == "feature_flag"

    # ── Error guard ──────────────────────────────────────────────────────

    def test_is_none_check(self):
        assert classify_condition("user is None") == "error_guard"

    def test_is_not_none_check(self):
        assert classify_condition("result is not None") == "error_guard"

    def test_not_none(self):
        assert classify_condition("not None") == "error_guard"

    def test_raise_check(self):
        assert classify_condition("raise ValueError") == "error_guard"

    def test_assert_check(self):
        assert classify_condition("assert x is not None") == "error_guard"

    # ── Logic branch (fallback) ───────────────────────────────────────────

    def test_simple_boolean(self):
        assert classify_condition("some_flag") == "logic_branch"

    def test_string_equality(self):
        assert classify_condition("name == 'hello'") == "logic_branch"

    def test_arbitrary_expression(self):
        assert classify_condition("a and b") == "logic_branch"

    # ── Priority — constants take precedence over roles ───────────────────

    def test_constant_before_role(self):
        # RATE_LIMIT is a constant pattern → threshold wins
        assert classify_condition("RATE_LIMIT > user.role") == "threshold"


# ---------------------------------------------------------------------------
# _make_decision_point
# ---------------------------------------------------------------------------

class TestMakeDecisionPoint:

    def test_basic_threshold_dp(self):
        cond = {"condition_text": "count >= MAX_RETRIES", "line": 10, "branch_count": 2}
        dp = _make_decision_point("app.py", "app.py::process", cond)
        assert dp is not None
        assert dp["condition_type"] == "threshold"
        assert dp["file"] == "app.py"
        assert dp["function_id"] == "app.py::process"
        assert dp["line"] == 10

    def test_returns_none_for_empty_condition(self):
        cond = {"condition_text": "", "line": 5, "branch_count": 2}
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert dp is None

    def test_returns_none_for_too_short_condition(self):
        cond = {"condition_text": "x", "line": 5, "branch_count": 2}
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert dp is None

    def test_filters_trivial_error_guard(self):
        # error_guard with branch_count <= 1 is filtered out
        cond = {"condition_text": "x is None", "line": 5, "branch_count": 1}
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert dp is None

    def test_keeps_error_guard_with_multiple_branches(self):
        cond = {"condition_text": "x is None", "line": 5, "branch_count": 2}
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert dp is not None
        assert dp["condition_type"] == "error_guard"

    def test_dp_id_includes_line_number(self):
        cond = {"condition_text": "status == 'active'", "line": 42, "branch_count": 2}
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert "L42" in dp["id"]

    def test_explanation_and_question_initially_none(self):
        cond = {"condition_text": "user.is_admin", "line": 10, "branch_count": 2}
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert dp["explanation"] is None
        assert dp["question_for_human"] is None

    def test_references_constant_field(self):
        cond = {
            "condition_text": "count >= RATE_LIMIT",
            "line": 5,
            "branch_count": 2,
            "references_constant": True,
        }
        dp = _make_decision_point("app.py", "app.py::fn", cond)
        assert dp["references_constant"] is True


# ---------------------------------------------------------------------------
# extract_decision_points
# ---------------------------------------------------------------------------

class TestExtractDecisionPoints:

    def _make_parsed(self, path, functions=None, classes=None):
        return {
            "path": path,
            "functions": functions or [],
            "classes": classes or [],
        }

    def test_empty_input_returns_empty(self):
        assert extract_decision_points([]) == []

    def test_top_level_function_dp(self):
        parsed = [self._make_parsed("app.py", functions=[
            {
                "name": "process",
                "conditionals": [
                    {"condition_text": "count >= MAX_RETRIES", "line": 5, "branch_count": 2},
                ],
            }
        ])]
        dps = extract_decision_points(parsed)
        assert len(dps) == 1
        assert dps[0]["condition_type"] == "threshold"
        assert dps[0]["file"] == "app.py"

    def test_class_method_dp(self):
        parsed = [self._make_parsed("app.py", classes=[
            {
                "name": "OrderProcessor",
                "methods": [
                    {
                        "name": "approve",
                        "conditionals": [
                            {"condition_text": "user.is_admin", "line": 10, "branch_count": 2},
                        ],
                    }
                ],
            }
        ])]
        dps = extract_decision_points(parsed)
        assert len(dps) == 1
        assert dps[0]["condition_type"] == "role_check"
        assert "OrderProcessor" in dps[0]["function_id"]
        assert "approve" in dps[0]["function_id"]

    def test_trivial_error_guards_filtered(self):
        parsed = [self._make_parsed("app.py", functions=[
            {
                "name": "fn",
                "conditionals": [
                    # trivial error guard (branch_count=1) — should be filtered
                    {"condition_text": "x is None", "line": 5, "branch_count": 1},
                    # substantial threshold — should be kept
                    {"condition_text": "x >= MAX_SIZE", "line": 10, "branch_count": 2},
                ],
            }
        ])]
        dps = extract_decision_points(parsed)
        types = [dp["condition_type"] for dp in dps]
        assert "threshold" in types
        # trivial error guard should NOT appear
        trivial = [dp for dp in dps if dp["condition"] == "x is None"]
        assert trivial == []

    def test_multiple_files(self):
        parsed = [
            self._make_parsed("a.py", functions=[
                {
                    "name": "fn_a",
                    "conditionals": [
                        {"condition_text": "status == 'active'", "line": 1, "branch_count": 2},
                    ],
                }
            ]),
            self._make_parsed("b.py", functions=[
                {
                    "name": "fn_b",
                    "conditionals": [
                        {"condition_text": "user.is_admin", "line": 1, "branch_count": 2},
                    ],
                }
            ]),
        ]
        dps = extract_decision_points(parsed)
        assert len(dps) == 2
        files = {dp["file"] for dp in dps}
        assert files == {"a.py", "b.py"}

    def test_function_id_format_for_top_level(self):
        parsed = [self._make_parsed("utils/helpers.py", functions=[
            {
                "name": "my_func",
                "conditionals": [
                    {"condition_text": "flag.is_enabled('x')", "line": 5, "branch_count": 2},
                ],
            }
        ])]
        dps = extract_decision_points(parsed)
        assert dps[0]["function_id"] == "utils/helpers.py::my_func"

    def test_no_conditionals_produces_no_dps(self):
        parsed = [self._make_parsed("app.py", functions=[
            {"name": "fn", "conditionals": []},
        ])]
        dps = extract_decision_points(parsed)
        assert dps == []
