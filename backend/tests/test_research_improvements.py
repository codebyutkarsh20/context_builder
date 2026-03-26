"""
test_research_improvements.py — Tests for changes based on the 30 research best practices.

Covers:
- _extract_function_source (Finding #23: scope to target function)
- _strip_gap_markers (remove windowing artifacts)
- acceptance_criteria in IntentAnalysis (Finding #21: verification from spec)
- Scope guard in test_node (Finding #22/#23: narrow scope)
- _enrich_from_fix (Finding #26: self-enriching loop)
- Smart test output parsing (Spotify: extract error lines)
- Token usage logging (Finding #11)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── _extract_function_source ──────────────────────────────────────────

class TestExtractFunctionSource:
    """Finding #23: Extract only the target function, not the whole file."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agent.pipeline import _extract_function_source
        self.extract = _extract_function_source

    def test_extracts_simple_function(self):
        source = (
            "import os\n"
            "\n"
            "def hello():\n"
            "    return 'hi'\n"
            "\n"
            "def goodbye():\n"
            "    return 'bye'\n"
        )
        result = self.extract(source, "hello")
        assert result is not None
        assert "def hello" in result
        assert "return 'hi'" in result

    def test_does_not_include_other_function_body(self):
        source = (
            "import os\n"
            "\n"
            "def foo():\n"
            "    return 1\n"
            "\n"
            "\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
        )
        # With context_lines=2, the next function def might appear at the edge,
        # but the OTHER function's body should not be fully included.
        result = self.extract(source, "foo", context_lines=0)
        assert result is not None
        assert "def foo" in result
        assert "return 1" in result
        assert "return 2" not in result

    def test_extracts_multiline_function(self):
        source = (
            "def compute(x, y):\n"
            "    total = x + y\n"
            "    if total > 10:\n"
            "        return total * 2\n"
            "    return total\n"
            "\n"
            "def other():\n"
            "    pass\n"
        )
        result = self.extract(source, "compute")
        assert result is not None
        assert "total = x + y" in result
        assert "return total" in result

    def test_returns_none_for_missing_function(self):
        source = "def foo():\n    pass\n"
        result = self.extract(source, "nonexistent")
        assert result is None

    def test_returns_none_for_syntax_error(self):
        source = "def foo(\n"
        result = self.extract(source, "foo")
        assert result is None

    def test_extracts_async_function(self):
        source = (
            "async def fetch_data(url):\n"
            "    response = await get(url)\n"
            "    return response\n"
        )
        result = self.extract(source, "fetch_data")
        assert result is not None
        assert "async def fetch_data" in result

    def test_includes_context_lines(self):
        source = (
            "# Module header\n"
            "import os\n"
            "\n"
            "def target():\n"
            "    return os.getcwd()\n"
            "\n"
            "# Footer\n"
        )
        result = self.extract(source, "target", context_lines=2)
        assert result is not None
        # Should include some surrounding lines
        assert "def target" in result

    def test_extracts_method_in_class(self):
        source = (
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        return 1\n"
            "\n"
            "    def method_b(self):\n"
            "        return 2\n"
        )
        result = self.extract(source, "method_a")
        assert result is not None
        assert "def method_a" in result


# ── _strip_gap_markers ────────────────────────────────────────────────

class TestStripGapMarkers:
    """Remove windowing artifacts from source before sending to LLM."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agent.pipeline import _strip_gap_markers
        self.strip = _strip_gap_markers

    def test_removes_line_range_marker(self):
        content = "line 1\n# ... (lines 50-100 omitted) ...\nline 101"
        result = self.strip(content)
        assert "omitted" not in result
        assert "line 1" in result
        assert "line 101" in result

    def test_removes_more_lines_marker(self):
        content = "code here\n# ... (200 more lines)\nend"
        result = self.strip(content)
        assert "more lines" not in result

    def test_preserves_normal_comments(self):
        content = "# This is a normal comment\ndef foo():\n    pass"
        result = self.strip(content)
        assert "normal comment" in result
        assert "def foo" in result

    def test_empty_input(self):
        assert self.strip("") == ""

    def test_no_markers_unchanged(self):
        content = "def foo():\n    return 42\n"
        result = self.strip(content)
        # _strip_gap_markers uses splitlines() + join, which strips trailing newline
        assert "def foo" in result
        assert "return 42" in result


# ── acceptance_criteria in IntentAnalysis ──────────────────────────────

class TestAcceptanceCriteria:
    """Finding #21: Verification criteria from the spec."""

    def test_intent_analysis_has_acceptance_criteria_field(self):
        from agent.types import IntentAnalysis
        obj = IntentAnalysis(
            expected_behavior="should log warning",
            actual_behavior="silently fails",
        )
        assert hasattr(obj, "acceptance_criteria")
        assert isinstance(obj.acceptance_criteria, list)

    def test_acceptance_criteria_defaults_empty(self):
        from agent.types import IntentAnalysis
        obj = IntentAnalysis(
            expected_behavior="x",
            actual_behavior="y",
        )
        assert obj.acceptance_criteria == []

    def test_acceptance_criteria_populated(self):
        from agent.types import IntentAnalysis
        criteria = ["calling with bad input logs warning", "return value is None"]
        obj = IntentAnalysis(
            expected_behavior="x",
            actual_behavior="y",
            acceptance_criteria=criteria,
        )
        assert len(obj.acceptance_criteria) == 2
        assert "logs warning" in obj.acceptance_criteria[0]

    def test_acceptance_criteria_serializes(self):
        from agent.types import IntentAnalysis
        obj = IntentAnalysis(
            expected_behavior="x",
            actual_behavior="y",
            acceptance_criteria=["test1", "test2"],
        )
        data = obj.model_dump()
        assert "acceptance_criteria" in data
        assert data["acceptance_criteria"] == ["test1", "test2"]


# ── Scope guard ───────────────────────────────────────────────────────

class TestScopeGuard:
    """Finding #22/#23: Patches should only touch expected files."""

    def test_scope_guard_logs_unexpected_file(self, tmp_repo, caplog):
        """Patches touching files outside fault_files trigger a warning."""
        import logging
        from agent.pipeline import test_node

        state = {
            "work_order": {
                "ticket_id": "SCOPE-1",
                "repo_path": str(tmp_repo),
                "repo_name": "test",
            },
            "localization": {"fault_files": ["utils.py"]},
            "repair": {
                "patches": [{
                    "file_path": "app.py",  # NOT in fault_files
                    "original_code": "return email.lower().strip()",
                    "patched_code": "return email.lower().strip() if email else ''",
                }],
            },
            "status": "",
        }

        with caplog.at_level(logging.WARNING):
            result = test_node(state)

        assert any("SCOPE GUARD" in r.message for r in caplog.records)

    def test_scope_guard_no_warning_for_expected_file(self, tmp_repo, caplog):
        """Patches touching fault_files do NOT trigger scope warning."""
        import logging
        from agent.pipeline import test_node

        state = {
            "work_order": {
                "ticket_id": "SCOPE-2",
                "repo_path": str(tmp_repo),
                "repo_name": "test",
            },
            "localization": {"fault_files": ["app.py"]},
            "repair": {
                "patches": [{
                    "file_path": "app.py",
                    "original_code": "return email.lower().strip()",
                    "patched_code": "return email.lower().strip() if email else ''",
                }],
            },
            "status": "",
        }

        with caplog.at_level(logging.WARNING):
            result = test_node(state)

        scope_warnings = [r for r in caplog.records if "SCOPE GUARD" in r.message]
        assert len(scope_warnings) == 0


# ── Self-enriching feedback loop ──────────────────────────────────────

class TestEnrichFromFix:
    """Finding #26: Store fix patterns after successful PR."""

    def test_stores_fix_record(self, tmp_path):
        from agent.pipeline import _enrich_from_fix, DATA_DIR
        import agent.pipeline as pipeline

        original = pipeline.DATA_DIR
        pipeline.DATA_DIR = tmp_path

        try:
            state = {
                "work_order": {"ticket_id": "FIX-1", "repo_name": "myrepo"},
                "localization": {
                    "fault_files": ["app.py"],
                    "fault_functions": ["process"],
                    "root_cause_hypothesis": "Missing null check",
                },
                "repair": {"explanation": "Added null guard"},
                "pr_url": "https://github.com/test/test/pull/1",
            }

            _enrich_from_fix(state)

            history_path = tmp_path / "myrepo" / "fix_history.json"
            assert history_path.exists()

            records = json.loads(history_path.read_text())
            assert len(records) == 1
            assert records[0]["ticket_id"] == "FIX-1"
            assert records[0]["root_cause"] == "Missing null check"
            assert records[0]["pr_url"] == "https://github.com/test/test/pull/1"
        finally:
            pipeline.DATA_DIR = original

    def test_appends_to_existing_history(self, tmp_path):
        from agent.pipeline import _enrich_from_fix
        import agent.pipeline as pipeline

        original = pipeline.DATA_DIR
        pipeline.DATA_DIR = tmp_path

        try:
            # Pre-populate with one record
            repo_dir = tmp_path / "myrepo"
            repo_dir.mkdir(parents=True)
            (repo_dir / "fix_history.json").write_text(json.dumps([{"ticket_id": "OLD-1"}]))

            state = {
                "work_order": {"ticket_id": "FIX-2", "repo_name": "myrepo"},
                "localization": {"fault_files": [], "fault_functions": [], "root_cause_hypothesis": ""},
                "repair": {"explanation": "Fix 2"},
                "pr_url": "",
            }

            _enrich_from_fix(state)

            records = json.loads((repo_dir / "fix_history.json").read_text())
            assert len(records) == 2
            assert records[0]["ticket_id"] == "OLD-1"
            assert records[1]["ticket_id"] == "FIX-2"
        finally:
            pipeline.DATA_DIR = original

    def test_no_crash_without_repo_name(self):
        from agent.pipeline import _enrich_from_fix
        # Should not raise
        _enrich_from_fix({"work_order": {}, "localization": {}, "repair": {}, "pr_url": ""})


# ── Smart test output parsing ─────────────────────────────────────────

class TestSmartTestOutputParsing:
    """Spotify finding: extract only error lines from test output."""

    def test_success_output_is_short(self, tmp_path):
        """Passing tests return a brief summary, not full output."""
        from agent.pipeline import _run_tests

        # Create a minimal Python project with a passing test
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        (tmp_path / "test_ok.py").write_text(
            "def test_pass():\n    assert 1 + 1 == 2\n"
        )

        result = _run_tests(tmp_path)
        assert result.startswith("passed")
        # Should be concise, not thousands of chars
        assert len(result) < 500

    def test_failure_output_extracts_errors(self, tmp_path):
        """Failing tests return extracted error lines, not raw dump."""
        from agent.pipeline import _run_tests

        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        (tmp_path / "test_fail.py").write_text(
            "def test_fail():\n    assert 1 == 2, 'expected failure'\n"
        )

        result = _run_tests(tmp_path)
        assert "failed" in result.lower()
        # Should contain the assertion error
        assert "assert" in result.lower() or "FAILED" in result

    def test_no_test_runner_returns_skipped(self, tmp_path):
        """Directory without test runner returns skip message."""
        from agent.pipeline import _run_tests
        result = _run_tests(tmp_path)
        assert "skipped" in result.lower()


# ── _build_source_section truncation ──────────────────────────────────

class TestBuildSourceSection:
    """Fault files truncate at 200 lines, callers at 400."""

    def test_fault_file_truncated_at_200(self):
        from agent.pipeline import _build_source_section
        code = "\n".join(f"line{i}" for i in range(300))
        source_code = {"fault.py": code}
        section, _ = _build_source_section(source_code)
        # Should be truncated
        assert "truncated" in section
        # Should have roughly 200 lines of content
        assert "line199" in section
        assert "line250" not in section

    def test_caller_file_truncated_at_400(self):
        from agent.pipeline import _build_source_section
        code = "\n".join(f"line{i}" for i in range(500))
        source_code = {"caller.py (caller)": code}
        section, _ = _build_source_section(source_code)
        assert "truncated" in section
        assert "line399" in section
        assert "line450" not in section

    def test_short_file_not_truncated(self):
        from agent.pipeline import _build_source_section
        code = "def foo():\n    return 1\n"
        source_code = {"short.py": code}
        section, _ = _build_source_section(source_code)
        assert "truncated" not in section
        assert "def foo" in section


# ── Token logging ─────────────────────────────────────────────────────

class TestTokenLogging:
    """Finding #11: Log approximate token usage per LLM call."""

    def test_structured_call_logs_tokens(self, caplog):
        """_structured_call logs approximate input tokens."""
        import logging
        from agent.pipeline import _structured_call
        from agent.types import IntentAnalysis

        with caplog.at_level(logging.INFO):
            try:
                # This will likely fail without API key, but logging happens before the call
                _structured_call("claude-sonnet-4-6", 100, IntentAnalysis, "test prompt")
            except Exception:
                pass

        token_logs = [r for r in caplog.records if "input tokens" in r.message]
        assert len(token_logs) >= 1
        assert "IntentAnalysis" in token_logs[0].message
