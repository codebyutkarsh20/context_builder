"""Tests for set_pr_url in feature_flags.py"""
import pytest
from unittest.mock import patch, MagicMock, call

import backend.agent.feature_flags as ff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flag(name="my-flag", enabled=False, pr_url=None):
    flag = {"name": name, "enabled": enabled}
    if pr_url is not None:
        flag["pr_url"] = pr_url
    return flag


# ---------------------------------------------------------------------------
# set_pr_url – happy path
# ---------------------------------------------------------------------------

class TestSetPrUrlHappyPath:
    def test_pr_url_is_stored_on_matching_flag(self):
        """When the flag exists, pr_url should be updated in the flag dict."""
        flags = [_make_flag("feature-x")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags") as mock_save:
            ff.set_pr_url("my-repo", "feature-x", "https://github.com/pr/1")
            assert flags[0]["pr_url"] == "https://github.com/pr/1"

    def test_save_flags_called_on_match(self):
        """_save_flags must be called with the updated flags when a match is found."""
        flags = [_make_flag("feature-x")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags") as mock_save:
            ff.set_pr_url("my-repo", "feature-x", "https://github.com/pr/1")
            mock_save.assert_called_once_with("my-repo", flags)

    def test_only_matching_flag_is_modified(self):
        """Flags whose name does not match must not have pr_url set."""
        flags = [_make_flag("feature-x"), _make_flag("feature-y")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"):
            ff.set_pr_url("my-repo", "feature-x", "https://github.com/pr/1")
            assert "pr_url" not in flags[1]

    def test_no_warning_logged_on_match(self):
        """No warning should be emitted when the flag is found."""
        flags = [_make_flag("feature-x")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"), \
             patch.object(ff.logger, "warning") as mock_warn:
            ff.set_pr_url("my-repo", "feature-x", "https://github.com/pr/1")
            mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# set_pr_url – missing flag (the bug that was fixed)
# ---------------------------------------------------------------------------

class TestSetPrUrlMissingFlag:
    def test_warning_logged_when_flag_not_found(self):
        """A warning must be logged when the specified flag name does not exist."""
        flags = [_make_flag("other-flag")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"), \
             patch.object(ff.logger, "warning") as mock_warn:
            ff.set_pr_url("my-repo", "nonexistent-flag", "https://github.com/pr/99")
            mock_warn.assert_called_once_with(
                "Flag %s not found for repo %s",
                "nonexistent-flag",
                "my-repo",
            )

    def test_save_flags_not_called_when_flag_not_found(self):
        """_save_flags must NOT be called when no matching flag is found."""
        flags = [_make_flag("other-flag")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags") as mock_save, \
             patch.object(ff.logger, "warning"):
            ff.set_pr_url("my-repo", "nonexistent-flag", "https://github.com/pr/99")
            mock_save.assert_not_called()

    def test_existing_flags_unchanged_when_flag_not_found(self):
        """Existing flag data must remain unmodified when the target flag is absent."""
        flags = [_make_flag("other-flag")]
        original_flag_copy = dict(flags[0])
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"), \
             patch.object(ff.logger, "warning"):
            ff.set_pr_url("my-repo", "nonexistent-flag", "https://github.com/pr/99")
            assert flags[0] == original_flag_copy

    def test_warning_contains_flag_name(self):
        """Warning message must include the flag name for easy diagnosis."""
        flags = []
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"), \
             patch.object(ff.logger, "warning") as mock_warn:
            ff.set_pr_url("repo-a", "missing-flag", "https://github.com/pr/5")
            args = mock_warn.call_args[0]
            assert "missing-flag" in args

    def test_warning_contains_repo_name(self):
        """Warning message must include the repo name for easy diagnosis."""
        flags = []
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"), \
             patch.object(ff.logger, "warning") as mock_warn:
            ff.set_pr_url("repo-a", "missing-flag", "https://github.com/pr/5")
            args = mock_warn.call_args[0]
            assert "repo-a" in args


# ---------------------------------------------------------------------------
# set_pr_url – edge cases
# ---------------------------------------------------------------------------

class TestSetPrUrlEdgeCases:
    def test_empty_flags_list_logs_warning(self):
        """An empty flags list should trigger the not-found warning."""
        with patch.object(ff, "_load_flags", return_value=[]), \
             patch.object(ff, "_save_flags"), \
             patch.object(ff.logger, "warning") as mock_warn:
            ff.set_pr_url("my-repo", "any-flag", "https://github.com/pr/1")
            mock_warn.assert_called_once()

    def test_first_matching_flag_updated_when_duplicates_exist(self):
        """If duplicate flag names exist, the first match is updated and save is called."""
        flags = [_make_flag("dup-flag"), _make_flag("dup-flag")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags") as mock_save:
            ff.set_pr_url("my-repo", "dup-flag", "https://github.com/pr/2")
            assert flags[0]["pr_url"] == "https://github.com/pr/2"
            mock_save.assert_called_once()

    def test_overwrite_existing_pr_url(self):
        """set_pr_url should overwrite an already-set pr_url."""
        flags = [_make_flag("feature-z", pr_url="https://github.com/pr/old")]
        with patch.object(ff, "_load_flags", return_value=flags), \
             patch.object(ff, "_save_flags"):
            ff.set_pr_url("my-repo", "feature-z", "https://github.com/pr/new")
            assert flags[0]["pr_url"] == "https://github.com/pr/new"
