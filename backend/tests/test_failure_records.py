"""Tests for graph/business/failure_records.py — FailureRecord mining."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-load the module so patches hit module-level names
import graph.business.failure_records as fr_mod


# ---------------------------------------------------------------------------
# _classify_fix_commit
# ---------------------------------------------------------------------------

class TestClassifyFixCommit:
    def test_fixes_hash_pattern(self):
        is_fix, ref, conf = fr_mod._classify_fix_commit("fixes #123: null pointer in login")
        assert is_fix
        assert ref == "#123"
        assert conf == 1.0

    def test_closes_hash_pattern(self):
        is_fix, ref, conf = fr_mod._classify_fix_commit("closes #456 — auth regression")
        assert is_fix
        assert ref == "#456"

    def test_hotfix_keyword(self):
        is_fix, ref, conf = fr_mod._classify_fix_commit("hotfix: payment timeout")
        assert is_fix
        assert conf == 1.0

    def test_incident_keyword(self):
        is_fix, ref, conf = fr_mod._classify_fix_commit("incident: db connection leak")
        assert is_fix

    def test_generic_bug_keyword_low_confidence(self):
        is_fix, ref, conf = fr_mod._classify_fix_commit("fix typo in README")
        assert is_fix
        assert conf == 0.5
        assert ref is None

    def test_non_fix_commit(self):
        is_fix, ref, conf = fr_mod._classify_fix_commit("feat: add dark mode")
        assert not is_fix
        assert conf == 0.0

    def test_jira_ref_extracted(self):
        # Temporarily set env and reload pattern
        with patch.dict(os.environ, {"JIRA_PROJECT_PREFIX": "ACME"}):
            import importlib
            import graph.business.failure_records as fresh_fr
            importlib.reload(fresh_fr)
            is_fix, ref, conf = fresh_fr._classify_fix_commit("fixes ACME-789: login crash")
            assert ref == "ACME-789"


# ---------------------------------------------------------------------------
# _parse_diff_hunks
# ---------------------------------------------------------------------------

class TestParseDiffHunks:
    def test_single_hunk(self):
        diff = "@@ -10,5 +12,8 @@ def foo():"
        hunks = fr_mod._parse_diff_hunks(diff)
        assert len(hunks) == 1
        assert hunks[0] == (12, 19)  # start=12, end=12+(8-1)=19

    def test_hunk_without_count(self):
        diff = "@@ -5 +7 @@ single line"
        hunks = fr_mod._parse_diff_hunks(diff)
        assert hunks[0] == (7, 7)

    def test_multiple_hunks(self):
        diff = "@@ -1,3 +1,4 @@\n@@ -20,2 +21,2 @@"
        hunks = fr_mod._parse_diff_hunks(diff)
        assert len(hunks) == 2


# ---------------------------------------------------------------------------
# _match_hunk_to_function
# ---------------------------------------------------------------------------

class TestMatchHunkToFunction:
    def test_hunk_within_function(self):
        functions = [("process_payment", 10, 50), ("validate", 60, 80)]
        result = fr_mod._match_hunk_to_function(functions, hunk_start=20, hunk_end=30)
        assert result == "process_payment"

    def test_hunk_outside_all_functions(self):
        functions = [("foo", 10, 20)]
        result = fr_mod._match_hunk_to_function(functions, hunk_start=50, hunk_end=60)
        assert result is None

    def test_empty_functions_list(self):
        result = fr_mod._match_hunk_to_function([], hunk_start=1, hunk_end=5)
        assert result is None


# ---------------------------------------------------------------------------
# mine_failure_records
# ---------------------------------------------------------------------------

class TestMineFailureRecords:
    def _env_enabled(self):
        return patch.dict(os.environ, {"ENABLE_FAILURE_RECORDS": "true"})

    def test_returns_empty_when_flag_off(self, tmp_path):
        result = fr_mod.mine_failure_records(tmp_path)
        assert result == []

    def test_returns_empty_for_non_git_repo(self, tmp_path):
        with self._env_enabled():
            with patch.object(fr_mod, "_is_git_repo", return_value=False):
                result = fr_mod.mine_failure_records(tmp_path)
                assert result == []

    def test_fix_commit_produces_record(self, tmp_path):
        log_output = (
            "COMMIT|abc123|2026-01-10|fixes #42: null pointer in login\n"
            "api/auth.py\n"
        )
        file_content = "def login(user, pwd):\n    return True\n"
        diff_output = "@@ -1,2 +1,3 @@\n+    # fix\n"

        def fake_run_git(args, cwd):
            if "log" in args:
                return log_output, "", 0
            if "show" in args and ":" in args[-1]:
                return file_content, "", 0
            if "diff" in args:
                return diff_output, "", 0
            return "", "", 0

        with self._env_enabled():
            with (
                patch.object(fr_mod, "_is_git_repo", return_value=True),
                patch.object(fr_mod, "_run_git", side_effect=fake_run_git),
            ):
                result = fr_mod.mine_failure_records(tmp_path)
                assert len(result) == 1
                assert result[0]["issue_ref"] == "#42"
                assert result[0]["confidence"] == 1.0

    def test_squash_commit_skipped(self, tmp_path):
        """Commits touching >20 files are skipped."""
        files = "\n".join(f"file{i}.py" for i in range(25))
        log_output = f"COMMIT|bbb|2026-01-11|fixes #99: big cleanup\n{files}\n"

        def fake_run_git(args, cwd):
            if "log" in args:
                return log_output, "", 0
            return "", "", 0

        with self._env_enabled():
            with (
                patch.object(fr_mod, "_is_git_repo", return_value=True),
                patch.object(fr_mod, "_run_git", side_effect=fake_run_git),
            ):
                result = fr_mod.mine_failure_records(tmp_path)
                assert result == []

    def test_non_python_file_links_to_file(self, tmp_path):
        log_output = "COMMIT|ccc|2026-01-12|fixes #10: fix JS bug\napp.js\n"

        def fake_run_git(args, cwd):
            if "log" in args:
                return log_output, "", 0
            return "", "", 0

        with self._env_enabled():
            with (
                patch.object(fr_mod, "_is_git_repo", return_value=True),
                patch.object(fr_mod, "_run_git", side_effect=fake_run_git),
            ):
                result = fr_mod.mine_failure_records(tmp_path)
                assert len(result) == 1
                assert result[0]["function_hits"][0]["function"] is None

    def test_max_commits_cap_applied(self, tmp_path):
        """max_commits=2 limits processing to 2 fix commits."""
        lines = ""
        for i in range(10):
            lines += f"COMMIT|hash{i}|2026-01-{i+1:02d}|fixes #{i}: bug\nfile{i}.py\n"

        def fake_run_git(args, cwd):
            if "log" in args:
                return lines, "", 0
            if "show" in args:
                return "def foo():\n    pass\n", "", 0
            if "diff" in args:
                return "@@ -1,1 +1,2 @@\n", "", 0
            return "", "", 0

        with self._env_enabled():
            with (
                patch.object(fr_mod, "_is_git_repo", return_value=True),
                patch.object(fr_mod, "_run_git", side_effect=fake_run_git),
            ):
                result = fr_mod.mine_failure_records(tmp_path, max_commits=2)
                assert len(result) <= 2

    def test_no_issue_ref_sets_low_confidence_label(self, tmp_path):
        log_output = "COMMIT|ddd|2026-01-13|fix typo in error message\nutils.py\n"

        def fake_run_git(args, cwd):
            if "log" in args:
                return log_output, "", 0
            return "", "", 0

        with self._env_enabled():
            with (
                patch.object(fr_mod, "_is_git_repo", return_value=True),
                patch.object(fr_mod, "_run_git", side_effect=fake_run_git),
            ):
                result = fr_mod.mine_failure_records(tmp_path)
                assert len(result) == 1
                assert "(unverified)" in result[0]["message"]
                assert result[0]["confidence"] == 0.5
