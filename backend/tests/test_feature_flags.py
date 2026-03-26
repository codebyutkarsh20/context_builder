"""
Unit tests for agent/feature_flags.py — feature flag CRUD operations.

Covers:
  - _slugify
  - create_flag (basic, deduplication)
  - list_flags (empty, populated)
  - toggle_flag (enable, disable, not-found)
  - get_flag (found, not-found)
  - set_pr_url
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.feature_flags as ff
from agent.feature_flags import (
    _slugify,
    create_flag,
    get_flag,
    list_flags,
    set_pr_url,
    toggle_flag,
)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a tmp directory for every test."""
    monkeypatch.setattr(ff, "DATA_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

class TestSlugify:

    def test_alphanumeric_unchanged(self):
        assert _slugify("hello123") == "hello123"

    def test_spaces_replaced(self):
        assert _slugify("hello world") == "hello_world"

    def test_special_chars_replaced(self):
        result = _slugify("fix: null pointer @ line 42!")
        assert all(c.isalnum() or c in "_-" for c in result)

    def test_collapses_multiple_underscores(self):
        assert "__" not in _slugify("hello   world")

    def test_strips_leading_trailing_underscores(self):
        result = _slugify("!!!hello!!!")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_truncated_to_80_chars(self):
        result = _slugify("a" * 200)
        assert len(result) <= 80

    def test_empty_string_returns_flag(self):
        assert _slugify("") == "flag"

    def test_hyphens_allowed(self):
        result = _slugify("my-feature")
        assert "my" in result
        assert "feature" in result


# ---------------------------------------------------------------------------
# create_flag
# ---------------------------------------------------------------------------

class TestCreateFlag:

    def test_creates_flag_and_returns_name(self, tmp_path):
        name = create_flag("repo", "PROJ-1", "Fix null error", ["app.py"])
        assert "PROJ" in name
        assert "Fix_null_error" in name or "fix_null_error" in name.lower()

    def test_flag_stored_on_disk(self, tmp_path):
        create_flag("repo", "PROJ-1", "Fix null error", ["app.py"])
        flags_file = tmp_path / "repo" / "feature_flags.json"
        assert flags_file.exists()
        data = json.loads(flags_file.read_text())
        assert len(data) == 1

    def test_flag_initially_disabled(self, tmp_path):
        name = create_flag("repo", "PROJ-1", "Fix null error", ["app.py"])
        flag = get_flag("repo", name)
        assert flag is not None
        assert flag["enabled"] is False

    def test_flag_contains_ticket_id(self, tmp_path):
        name = create_flag("repo", "PROJ-99", "Fix something", ["x.py"])
        flag = get_flag("repo", name)
        assert flag["ticket_id"] == "PROJ-99"

    def test_flag_contains_files_changed(self, tmp_path):
        create_flag("repo", "T-1", "Fix", ["a.py", "b.py"])
        flags = list_flags("repo")
        assert len(flags) == 1
        assert "a.py" in flags[0]["files_changed"]

    def test_flag_contains_created_at(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", [])
        flag = get_flag("repo", name)
        assert "created_at" in flag
        assert flag["created_at"]  # non-empty

    def test_deduplication_same_name(self, tmp_path):
        name1 = create_flag("repo", "T-1", "Fix null error", ["a.py"])
        name2 = create_flag("repo", "T-1", "Fix null error", ["a.py"])
        assert name1 == name2
        # Still only one entry on disk
        flags = list_flags("repo")
        assert len(flags) == 1

    def test_creates_parent_dir(self, tmp_path):
        # Directory doesn't exist yet — create_flag should make it
        create_flag("new-repo", "T-1", "Bug fix", [])
        assert (tmp_path / "new-repo").is_dir()

    def test_multiple_flags_different_tickets(self, tmp_path):
        create_flag("repo", "T-1", "Fix A", ["a.py"])
        create_flag("repo", "T-2", "Fix B", ["b.py"])
        flags = list_flags("repo")
        assert len(flags) == 2


# ---------------------------------------------------------------------------
# list_flags
# ---------------------------------------------------------------------------

class TestListFlags:

    def test_empty_when_no_flags_file(self, tmp_path):
        result = list_flags("no-such-repo")
        assert result == []

    def test_returns_all_flags(self, tmp_path):
        for i in range(3):
            create_flag("repo", f"T-{i}", f"Fix {i}", [])
        flags = list_flags("repo")
        assert len(flags) == 3

    def test_returns_list_of_dicts(self, tmp_path):
        create_flag("repo", "T-1", "Fix", [])
        flags = list_flags("repo")
        assert isinstance(flags[0], dict)

    def test_handles_corrupt_json(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "feature_flags.json").write_text("not valid json")
        result = list_flags("repo")
        assert result == []


# ---------------------------------------------------------------------------
# toggle_flag
# ---------------------------------------------------------------------------

class TestToggleFlag:

    def test_enable_flag(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", [])
        result = toggle_flag("repo", name, True)
        assert result is not None
        assert result["enabled"] is True

    def test_disable_flag(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", [])
        toggle_flag("repo", name, True)
        result = toggle_flag("repo", name, False)
        assert result["enabled"] is False

    def test_returns_none_for_unknown_flag(self, tmp_path):
        result = toggle_flag("repo", "nonexistent_flag", True)
        assert result is None

    def test_toggle_persists_to_disk(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", [])
        toggle_flag("repo", name, True)
        # Re-load from disk
        flag = get_flag("repo", name)
        assert flag["enabled"] is True


# ---------------------------------------------------------------------------
# get_flag
# ---------------------------------------------------------------------------

class TestGetFlag:

    def test_returns_flag_when_found(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", ["app.py"])
        flag = get_flag("repo", name)
        assert flag is not None
        assert flag["name"] == name

    def test_returns_none_when_not_found(self, tmp_path):
        result = get_flag("repo", "no_such_flag")
        assert result is None

    def test_returns_correct_flag_among_multiple(self, tmp_path):
        n1 = create_flag("repo", "T-1", "Fix A", ["a.py"])
        n2 = create_flag("repo", "T-2", "Fix B", ["b.py"])
        flag = get_flag("repo", n2)
        assert flag["ticket_id"] == "T-2"


# ---------------------------------------------------------------------------
# set_pr_url
# ---------------------------------------------------------------------------

class TestSetPrUrl:

    def test_sets_pr_url(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", [])
        set_pr_url("repo", name, "https://github.com/org/repo/pull/42")
        flag = get_flag("repo", name)
        assert flag["pr_url"] == "https://github.com/org/repo/pull/42"

    def test_no_op_for_missing_flag(self, tmp_path):
        # Should log a warning when flag is not found
        with unittest.mock.patch("agent.feature_flags.logger") as mock_logger:
            set_pr_url("repo", "nonexistent", "https://github.com/x/y/pull/1")
            mock_logger.warning.assert_called_once_with(
                "Flag %s not found for repo %s", "nonexistent", "repo"
            )

    def test_no_warning_for_existing_flag(self, tmp_path):
        # Should NOT log a warning when flag is found
        name = create_flag("repo", "T-2", "Valid flag", [])
        with unittest.mock.patch("agent.feature_flags.logger") as mock_logger:
            set_pr_url("repo", name, "https://example.com/pr/42")
            mock_logger.warning.assert_not_called()

    def test_no_side_effects_for_missing_flag(self, tmp_path):
        # flags list should remain unchanged after a missing-flag call
        name = create_flag("repo", "T-3", "Existing flag", [])
        flags_before = list_flags("repo")
        set_pr_url("repo", "nonexistent", "https://github.com/x/y/pull/99")
        flags_after = list_flags("repo")
        assert flags_before == flags_after

    def test_pr_url_persists(self, tmp_path):
        name = create_flag("repo", "T-1", "Fix", [])
        set_pr_url("repo", name, "https://example.com/pr/99")
        # Reload
        flags = list_flags("repo")
        assert flags[0]["pr_url"] == "https://example.com/pr/99"
