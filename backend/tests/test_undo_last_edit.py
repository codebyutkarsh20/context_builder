"""
test_undo_last_edit.py — Tests for the per-edit rollback tool.

Edit history is captured by string_replace and create_file. undo_last_edit
pops the most recent entry and restores the file to its prior state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.react_tools import (
    create_file,
    get_edit_history,
    reset_edit_history,
    set_react_context,
    set_sandbox_path,
    string_replace,
    undo_last_edit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    """Set up a fake sandbox with a single test file."""
    set_react_context("test_repo", str(tmp_path))
    set_sandbox_path(tmp_path, "branch", "main")
    reset_edit_history()
    yield tmp_path
    reset_edit_history()


# ---------------------------------------------------------------------------
# Edit history capture
# ---------------------------------------------------------------------------

class TestEditHistoryCapture:
    def test_string_replace_records_history(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("hello world")
        result = string_replace.invoke({
            "file_path": "x.py",
            "old_string": "hello",
            "new_string": "goodbye",
        })
        assert result.startswith("OK")
        history = get_edit_history()
        assert len(history) == 1
        assert history[0]["file_path"] == "x.py"
        assert history[0]["before_content"] == "hello world"
        assert history[0]["after_content"] == "goodbye world"
        assert history[0]["tool"] == "string_replace"

    def test_create_file_records_history(self, sandbox):
        result = create_file.invoke({
            "file_path": "newfile.py",
            "content": "print('hi')",
        })
        assert result.startswith("OK")
        history = get_edit_history()
        assert len(history) == 1
        assert history[0]["before_content"] is None  # didn't exist before
        assert history[0]["after_content"] == "print('hi')"
        assert history[0]["tool"] == "create_file"

    def test_create_file_overwrites_records_prior_content(self, sandbox):
        f = sandbox / "existing.py"
        f.write_text("OLD CONTENT")
        result = create_file.invoke({
            "file_path": "existing.py",
            "content": "NEW CONTENT",
        })
        assert result.startswith("OK")
        history = get_edit_history()
        assert len(history) == 1
        assert history[0]["before_content"] == "OLD CONTENT"

    def test_failed_string_replace_does_not_record(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("hello world")
        result = string_replace.invoke({
            "file_path": "x.py",
            "old_string": "NOT_PRESENT",
            "new_string": "won't matter",
        })
        assert result.startswith("ERROR")
        # No edit happened → no history entry
        assert get_edit_history() == []

    def test_multiple_edits_recorded_in_order(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("a b c")
        string_replace.invoke({"file_path": "x.py", "old_string": "a", "new_string": "1"})
        string_replace.invoke({"file_path": "x.py", "old_string": "b", "new_string": "2"})
        string_replace.invoke({"file_path": "x.py", "old_string": "c", "new_string": "3"})
        history = get_edit_history()
        assert len(history) == 3
        assert [h["before_content"] for h in history] == ["a b c", "1 b c", "1 2 c"]


# ---------------------------------------------------------------------------
# undo_last_edit behavior
# ---------------------------------------------------------------------------

class TestUndoLastEdit:
    def test_undo_with_no_history_returns_error(self, sandbox):
        result = undo_last_edit.invoke({})
        assert result.startswith("ERROR")
        assert "empty" in result.lower() or "no edits" in result.lower()

    def test_undo_string_replace_restores_content(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("hello world")
        string_replace.invoke({"file_path": "x.py", "old_string": "hello", "new_string": "goodbye"})
        assert f.read_text() == "goodbye world"

        result = undo_last_edit.invoke({})
        assert result.startswith("OK")
        assert f.read_text() == "hello world"

    def test_undo_create_file_deletes_new_file(self, sandbox):
        create_file.invoke({"file_path": "fresh.py", "content": "x = 1"})
        f = sandbox / "fresh.py"
        assert f.exists()

        result = undo_last_edit.invoke({})
        assert result.startswith("OK")
        assert "deleted" in result.lower()
        assert not f.exists()

    def test_undo_create_file_overwrite_restores_prior(self, sandbox):
        f = sandbox / "existing.py"
        f.write_text("ORIGINAL")
        create_file.invoke({"file_path": "existing.py", "content": "REPLACED"})
        assert f.read_text() == "REPLACED"

        result = undo_last_edit.invoke({})
        assert result.startswith("OK")
        assert f.read_text() == "ORIGINAL"

    def test_undo_pops_history(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("aaa")
        string_replace.invoke({"file_path": "x.py", "old_string": "aaa", "new_string": "bbb"})
        assert len(get_edit_history()) == 1
        undo_last_edit.invoke({})
        assert len(get_edit_history()) == 0

    def test_undo_undoes_most_recent_only(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("a b c")
        string_replace.invoke({"file_path": "x.py", "old_string": "a", "new_string": "1"})
        string_replace.invoke({"file_path": "x.py", "old_string": "b", "new_string": "2"})
        string_replace.invoke({"file_path": "x.py", "old_string": "c", "new_string": "3"})
        # File now has "1 2 3"
        assert f.read_text() == "1 2 3"

        # Undo once → reverts the c→3 edit
        undo_last_edit.invoke({})
        assert f.read_text() == "1 2 c"

        # Undo again → reverts the b→2 edit
        undo_last_edit.invoke({})
        assert f.read_text() == "1 b c"

        # Undo again → reverts the a→1 edit
        undo_last_edit.invoke({})
        assert f.read_text() == "a b c"

        assert get_edit_history() == []

    def test_undo_after_history_emptied_returns_error(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("hello")
        string_replace.invoke({"file_path": "x.py", "old_string": "hello", "new_string": "bye"})
        undo_last_edit.invoke({})  # consumes the only entry
        result = undo_last_edit.invoke({})
        assert result.startswith("ERROR")


# ---------------------------------------------------------------------------
# State reset between runs
# ---------------------------------------------------------------------------

class TestEditHistoryReset:
    def test_set_react_context_clears_history(self, sandbox, tmp_path):
        f = sandbox / "x.py"
        f.write_text("hello")
        string_replace.invoke({"file_path": "x.py", "old_string": "hello", "new_string": "bye"})
        assert len(get_edit_history()) == 1

        # Simulate a new run
        new_sandbox = tmp_path / "new"
        new_sandbox.mkdir()
        set_react_context("repo2", str(new_sandbox))
        assert get_edit_history() == []

    def test_reset_helper_works(self, sandbox):
        f = sandbox / "x.py"
        f.write_text("hello")
        string_replace.invoke({"file_path": "x.py", "old_string": "hello", "new_string": "bye"})
        reset_edit_history()
        assert get_edit_history() == []

    def test_reset_when_already_clean(self):
        reset_edit_history()
        reset_edit_history()
        assert get_edit_history() == []


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_undo_in_edit_tools(self):
        from agent.react_tools import EDIT_TOOLS
        names = [t.name for t in EDIT_TOOLS]
        assert "undo_last_edit" in names

    def test_undo_has_docstring(self):
        assert undo_last_edit.description
        assert len(undo_last_edit.description) > 100
