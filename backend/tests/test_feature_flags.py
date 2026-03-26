import logging
import pytest
from unittest.mock import patch, MagicMock
from backend.agent import feature_flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flags(names):
    """Return a minimal flags list for the given flag names."""
    return [{"name": n, "enabled": False, "pr_url": None} for n in names]


# ---------------------------------------------------------------------------
# set_pr_url — happy path
# ---------------------------------------------------------------------------

def test_set_pr_url_updates_existing_flag():
    """set_pr_url persists the PR URL when the flag exists."""
    flags = _make_flags(["my-flag"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags") as mock_write:
        feature_flags.set_pr_url("my-repo", "my-flag", "https://github.com/org/repo/pull/1")

    mock_write.assert_called_once()
    written_flags = mock_write.call_args[0][1]
    assert written_flags[0]["pr_url"] == "https://github.com/org/repo/pull/1"


def test_set_pr_url_writes_to_correct_repo():
    """set_pr_url calls _write_flags with the correct repo_name."""
    flags = _make_flags(["flag-a"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags") as mock_write:
        feature_flags.set_pr_url("target-repo", "flag-a", "https://example.com/pr/42")

    assert mock_write.call_args[0][0] == "target-repo"


def test_set_pr_url_does_not_mutate_other_flags():
    """set_pr_url only updates the matching flag, leaving others unchanged."""
    flags = _make_flags(["flag-x", "flag-y"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags"):
        feature_flags.set_pr_url("repo", "flag-x", "https://example.com/pr/1")

    assert flags[1]["pr_url"] is None


# ---------------------------------------------------------------------------
# set_pr_url — failure case (flag not found → warning logged)
# ---------------------------------------------------------------------------

def test_set_pr_url_logs_warning_when_flag_not_found(caplog):
    """set_pr_url logs a warning when flag_name is not present in the store."""
    flags = _make_flags(["existing-flag"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags") as mock_write, \
         caplog.at_level(logging.WARNING, logger="backend.agent.feature_flags"):
        feature_flags.set_pr_url("my-repo", "nonexistent-flag", "https://example.com/pr/99")

    assert mock_write.call_count == 0, "_write_flags must not be called when the flag is missing"
    assert "nonexistent-flag" in caplog.text
    assert "my-repo" in caplog.text


def test_set_pr_url_does_not_write_when_flag_not_found():
    """set_pr_url must not persist anything when the flag does not exist."""
    flags = _make_flags(["other-flag"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags") as mock_write:
        feature_flags.set_pr_url("repo", "missing", "https://example.com/pr/7")

    mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# set_pr_url — edge cases
# ---------------------------------------------------------------------------

def test_set_pr_url_empty_flags_list_logs_warning(caplog):
    """set_pr_url logs a warning when the flags store is completely empty."""
    with patch.object(feature_flags, "_read_flags", return_value=[]), \
         patch.object(feature_flags, "_write_flags"), \
         caplog.at_level(logging.WARNING, logger="backend.agent.feature_flags"):
        feature_flags.set_pr_url("repo", "any-flag", "https://example.com/pr/0")

    assert "any-flag" in caplog.text


def test_set_pr_url_returns_none_on_success():
    """set_pr_url returns None (implicitly) on the happy path."""
    flags = _make_flags(["flag-ok"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags"):
        result = feature_flags.set_pr_url("repo", "flag-ok", "https://example.com/pr/5")

    assert result is None


def test_set_pr_url_returns_none_when_flag_not_found():
    """set_pr_url returns None even when the flag is not found."""
    flags = _make_flags(["some-flag"])

    with patch.object(feature_flags, "_read_flags", return_value=flags), \
         patch.object(feature_flags, "_write_flags"):
        result = feature_flags.set_pr_url("repo", "not-here", "https://example.com/pr/3")

    assert result is None
