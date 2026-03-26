"""Tests for backend/agent/feature_flags.py"""
import json
import logging
import os
import pytest

import backend.agent.feature_flags as ff


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_flags_dir(tmp_path, monkeypatch):
    """Redirect all flag file I/O to a temporary directory."""
    monkeypatch.setattr(ff, "FLAGS_DIR", str(tmp_path))
    return tmp_path


def _write_repo_flags(repo_name: str, flags: list, tmp_path):
    """Helper: write a flags JSON file directly into the temp dir."""
    path = os.path.join(str(tmp_path), f"{repo_name}.json")
    with open(path, "w") as fh:
        json.dump(flags, fh)


def _read_repo_flags(repo_name: str, tmp_path) -> list:
    """Helper: read the flags JSON file from the temp dir."""
    path = os.path.join(str(tmp_path), f"{repo_name}.json")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# set_pr_url – happy path
# ---------------------------------------------------------------------------

class TestSetPrUrlHappyPath:
    def test_pr_url_is_stored_when_flag_exists(self, isolated_flags_dir):
        """set_pr_url writes the PR URL into the matching flag."""
        flags = [{"name": "my-feature", "enabled": False, "pr_url": ""}]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        ff.set_pr_url("my-repo", "my-feature", "https://github.com/org/repo/pull/42")

        updated = _read_repo_flags("my-repo", isolated_flags_dir)
        assert updated[0]["pr_url"] == "https://github.com/org/repo/pull/42"

    def test_other_flags_are_unchanged_when_target_flag_exists(self, isolated_flags_dir):
        """set_pr_url must not mutate flags other than the targeted one."""
        flags = [
            {"name": "flag-a", "enabled": True, "pr_url": ""},
            {"name": "flag-b", "enabled": False, "pr_url": "https://old-url"},
        ]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        ff.set_pr_url("my-repo", "flag-a", "https://new-url")

        updated = _read_repo_flags("my-repo", isolated_flags_dir)
        assert updated[1]["pr_url"] == "https://old-url"

    def test_returns_none_on_success(self, isolated_flags_dir):
        """set_pr_url returns None (implicitly) when the flag exists."""
        flags = [{"name": "feat", "enabled": False, "pr_url": ""}]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        result = ff.set_pr_url("my-repo", "feat", "https://url")

        assert result is None


# ---------------------------------------------------------------------------
# set_pr_url – flag not found (the bug that was fixed)
# ---------------------------------------------------------------------------

class TestSetPrUrlFlagNotFound:
    def test_warning_logged_when_flag_does_not_exist(self, isolated_flags_dir, caplog):
        """set_pr_url must log a warning when the flag name is not found."""
        flags = [{"name": "existing-flag", "enabled": False, "pr_url": ""}]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        with caplog.at_level(logging.WARNING, logger="backend.agent.feature_flags"):
            ff.set_pr_url("my-repo", "non-existent-flag", "https://url")

        assert "non-existent-flag" in caplog.text
        assert "my-repo" in caplog.text

    def test_warning_message_matches_expected_pattern(self, isolated_flags_dir, caplog):
        """The warning message format must match the toggle_flag pattern."""
        flags = [{"name": "real-flag", "enabled": False, "pr_url": ""}]
        _write_repo_flags("repo-x", flags, isolated_flags_dir)

        with caplog.at_level(logging.WARNING, logger="backend.agent.feature_flags"):
            ff.set_pr_url("repo-x", "ghost-flag", "https://url")

        assert any(
            "Flag ghost-flag not found for repo repo-x" in r.getMessage()
            for r in caplog.records
        )

    def test_flags_file_unchanged_when_flag_does_not_exist(self, isolated_flags_dir):
        """set_pr_url must not modify the flags file when the flag is absent."""
        flags = [{"name": "other-flag", "enabled": True, "pr_url": "https://original"}]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        ff.set_pr_url("my-repo", "missing-flag", "https://new-url")

        updated = _read_repo_flags("my-repo", isolated_flags_dir)
        assert updated[0]["pr_url"] == "https://original"

    def test_returns_none_when_flag_does_not_exist(self, isolated_flags_dir):
        """set_pr_url returns None when the flag is not found (no exception)."""
        flags = [{"name": "flag-a", "enabled": False, "pr_url": ""}]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        result = ff.set_pr_url("my-repo", "flag-b", "https://url")

        assert result is None


# ---------------------------------------------------------------------------
# set_pr_url – edge cases
# ---------------------------------------------------------------------------

class TestSetPrUrlEdgeCases:
    def test_empty_flags_list_logs_warning(self, isolated_flags_dir, caplog):
        """When the flags list is empty, a warning must still be emitted."""
        _write_repo_flags("empty-repo", [], isolated_flags_dir)

        with caplog.at_level(logging.WARNING, logger="backend.agent.feature_flags"):
            ff.set_pr_url("empty-repo", "any-flag", "https://url")

        assert "any-flag" in caplog.text
        assert "empty-repo" in caplog.text

    def test_pr_url_overwritten_when_already_set(self, isolated_flags_dir):
        """set_pr_url replaces an existing non-empty pr_url value."""
        flags = [{"name": "my-feature", "enabled": True, "pr_url": "https://old"}]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        ff.set_pr_url("my-repo", "my-feature", "https://new")

        updated = _read_repo_flags("my-repo", isolated_flags_dir)
        assert updated[0]["pr_url"] == "https://new"

    def test_only_first_matching_flag_name_is_updated(self, isolated_flags_dir):
        """If duplicate flag names exist (malformed data), the loop stops at the first match."""
        flags = [
            {"name": "dup", "enabled": False, "pr_url": ""},
            {"name": "dup", "enabled": False, "pr_url": ""},
        ]
        _write_repo_flags("my-repo", flags, isolated_flags_dir)

        ff.set_pr_url("my-repo", "dup", "https://url")

        updated = _read_repo_flags("my-repo", isolated_flags_dir)
        # First entry is updated; because of early return, second is untouched.
        assert updated[0]["pr_url"] == "https://url"
        assert updated[1]["pr_url"] == ""
