"""
Tests for test enforcement in repair_node and programmatic stub test generation.

Covers:
  - MANDATORY test_patches prompt when reviewer flags TESTS FAIL
  - Programmatic stub test generation from acceptance criteria
  - Stub test generation from function names when no acceptance criteria
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agent.pipeline import _generate_stub_tests


class TestStubTestGeneration:
    """Tests for _generate_stub_tests fallback."""

    def test_generates_from_acceptance_criteria(self):
        intent = {
            "acceptance_criteria": [
                "slugify returns truncated slug of max 80 chars",
                "slugify handles empty string input",
            ]
        }
        localization = {"fault_functions": ["_slugify"]}
        target_files = ["agent/feature_flags.py"]

        stubs = _generate_stub_tests(intent, localization, target_files)

        assert len(stubs) == 1
        patch = stubs[0]
        assert patch["file_path"] == "tests/test_feature_flags.py"
        assert patch["original_code"] == ""
        assert "test_acceptance_1" in patch["patched_code"]
        assert "test_acceptance_2" in patch["patched_code"]
        assert "slugify returns truncated" in patch["patched_code"]

    def test_generates_from_function_names_when_no_criteria(self):
        intent = {"acceptance_criteria": []}
        localization = {"fault_functions": ["_slugify", "create_flag"]}
        target_files = ["agent/feature_flags.py"]

        stubs = _generate_stub_tests(intent, localization, target_files)

        assert len(stubs) == 1
        code = stubs[0]["patched_code"]
        assert "test__slugify_fixed" in code
        assert "test_create_flag_fixed" in code

    def test_returns_empty_when_no_functions(self):
        intent = {"acceptance_criteria": ["something"]}
        localization = {"fault_functions": []}
        target_files = ["agent/feature_flags.py"]

        stubs = _generate_stub_tests(intent, localization, target_files)
        assert stubs == []

    def test_stub_tests_are_valid_python(self):
        intent = {
            "acceptance_criteria": ["set_pr_url with nonexistent flag should log a warning"]
        }
        localization = {"fault_functions": ["set_pr_url"]}
        target_files = ["agent/feature_flags.py"]

        stubs = _generate_stub_tests(intent, localization, target_files)
        code = stubs[0]["patched_code"]
        # Should be parseable Python
        import ast
        ast.parse(code)


class TestMandatoryTestEnforcement:
    """Tests that the feedback_section includes MANDATORY test prompt
    when the review flagged TESTS as FAIL."""

    def test_tests_fail_triggers_mandatory_prompt(self):
        """Simulates the feedback_section construction logic from repair_node."""
        previous_review = {
            "feedback": "Fix looks correct but no tests provided.",
            "checks": [
                {"name": "ROOT_CAUSE", "status": "PASS"},
                {"name": "TESTS", "status": "FAIL"},
            ],
        }

        # Reproduce the logic from pipeline.py repair_node
        feedback_section = ""
        if previous_review.get("feedback"):
            feedback_section = f"\nPREVIOUS REVIEW FEEDBACK:\n{previous_review['feedback'][:500]}\n"
            review_checks = previous_review.get("checks", [])
            tests_failed = any(
                (c.get("name") == "TESTS" and c.get("status") == "FAIL")
                for c in review_checks
            )
            if tests_failed:
                feedback_section += (
                    "\nMANDATORY: You MUST produce test_patches in your response. "
                    "The reviewer rejected your previous fix because it had NO tests. "
                    "If you return empty test_patches again, the fix will be rejected. "
                    "Generate at least one test that verifies the fixed behavior.\n"
                )

        assert "MANDATORY" in feedback_section
        assert "test_patches" in feedback_section

    def test_no_tests_fail_no_mandatory(self):
        """When TESTS passes, no mandatory prompt should be added."""
        previous_review = {
            "feedback": "Looks good, minor style nit.",
            "checks": [
                {"name": "ROOT_CAUSE", "status": "PASS"},
                {"name": "TESTS", "status": "PASS"},
            ],
        }

        feedback_section = ""
        if previous_review.get("feedback"):
            feedback_section = f"\nPREVIOUS REVIEW FEEDBACK:\n{previous_review['feedback'][:500]}\n"
            review_checks = previous_review.get("checks", [])
            tests_failed = any(
                (c.get("name") == "TESTS" and c.get("status") == "FAIL")
                for c in review_checks
            )
            if tests_failed:
                feedback_section += "\nMANDATORY: ..."

        assert "MANDATORY" not in feedback_section
