"""
test_learn_from_fix.py — Tests for per-repo persistent learnings.

The learn_from_fix module captures lessons from completed runs and injects
them into future runs on the same repo. Tests cover:
  - Recording: successful/failed runs, Haiku path + rule-based fallback
  - Storage: append, trim-to-cap, round-trip parse
  - Loading: compact markdown section, respects max_entries cap
  - Cross-repo isolation: lesson for repoA never appears in repoB prompt
  - State reset / idempotency
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.learn_from_fix as lff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point the module at a temp data dir so we don't touch real state."""
    monkeypatch.setattr(lff, "DATA_DIR", tmp_path)
    # Ensure feature flag is on for tests
    monkeypatch.setattr(lff, "LEARN_FROM_FIX_ENABLED", True)
    yield tmp_path


def _minimal_state(
    ticket_id: str = "TEST-001",
    repo_name: str = "myrepo",
    successful: bool = True,
    **extra,
) -> dict:
    """Build a minimal state dict like what finalize_node would see."""
    state = {
        "work_order": {
            "ticket_id": ticket_id,
            "repo_name": repo_name,
            "title": "Fix the thing",
            "description": "The thing is broken.",
        },
        "intent": {"fix_type": "bug_fix"},
        "submitted": successful,
        "run_outcome": {
            "tests_passed": successful,
            "tool_call_count": 20,
        },
        "review": {"verdict": "APPROVE" if successful else "REJECT"},
    }
    state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Fallback lesson path (no LLM call)
# ---------------------------------------------------------------------------

class TestFallbackLesson:
    def test_fallback_for_successful_run(self):
        state = _minimal_state(successful=True)
        # Patch get_current_plan to return a non-None plan
        with patch("agent.react_tools.get_current_plan", return_value={
            "root_cause": "missing validation",
            "target_files": ["app/validate.py"],
            "approach": "add len() check",
        }):
            lesson = lff._fallback_lesson(state, successful=True)
        assert "**Pattern**" in lesson
        assert "**Lesson**" in lesson
        assert "**Tactic**" in lesson
        assert "TEST-001" in lesson
        assert "missing validation" in lesson

    def test_fallback_for_failed_run(self):
        state = _minimal_state(successful=False)
        state["escalate_reason"] = "tests did not pass"
        with patch("agent.react_tools.get_current_plan", return_value={
            "root_cause": "wrong hypothesis",
            "target_files": ["x.py"],
            "approach": "",
        }):
            lesson = lff._fallback_lesson(state, successful=False)
        assert "Failed" in lesson
        assert "wrong hypothesis" in lesson
        assert "reconsider" in lesson.lower()

    def test_fallback_with_no_plan(self):
        """No plan (e.g. agent escalated before produce_plan) — still produces output."""
        state = _minimal_state(successful=False)
        with patch("agent.react_tools.get_current_plan", return_value=None):
            lesson = lff._fallback_lesson(state, successful=False)
        assert "**Pattern**" in lesson  # Format preserved even with no plan


# ---------------------------------------------------------------------------
# record_lesson — happy path (via fallback, no real Haiku)
# ---------------------------------------------------------------------------

class TestRecordLesson:
    def test_creates_lessons_file_on_first_run(self, tmp_path):
        state = _minimal_state(repo_name="demo", ticket_id="BUG-1")
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "root", "target_files": ["f.py"], "approach": "a",
            }):
                entry = lff.record_lesson(state)
        lessons_file = tmp_path / "demo" / "agent_lessons.md"
        assert lessons_file.exists()
        text = lessons_file.read_text()
        assert "BUG-1" in text
        assert "SUCCESS" in text
        assert entry is not None and "BUG-1" in entry

    def test_failed_run_records_FAIL_status(self, tmp_path):
        state = _minimal_state(repo_name="demo", ticket_id="BUG-2", successful=False)
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                lff.record_lesson(state)
        text = (tmp_path / "demo" / "agent_lessons.md").read_text()
        assert "FAIL" in text
        assert "BUG-2" in text

    def test_appends_multiple_lessons(self, tmp_path):
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                lff.record_lesson(_minimal_state(ticket_id="BUG-1", repo_name="r"))
                lff.record_lesson(_minimal_state(ticket_id="BUG-2", repo_name="r"))
                lff.record_lesson(_minimal_state(ticket_id="BUG-3", repo_name="r"))
        text = (tmp_path / "r" / "agent_lessons.md").read_text()
        assert "BUG-1" in text
        assert "BUG-2" in text
        assert "BUG-3" in text

    def test_caps_at_max_lessons_stored(self, tmp_path, monkeypatch):
        """When adding the Nth lesson past the cap, the oldest is evicted."""
        monkeypatch.setattr(lff, "MAX_LESSONS_STORED", 3)
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                for i in range(5):
                    lff.record_lesson(_minimal_state(ticket_id=f"BUG-{i}", repo_name="r"))

        text = (tmp_path / "r" / "agent_lessons.md").read_text()
        # First two should be evicted
        assert "BUG-0" not in text
        assert "BUG-1" not in text
        # Last three remain
        assert "BUG-2" in text
        assert "BUG-3" in text
        assert "BUG-4" in text

    def test_disabled_feature_returns_none(self, monkeypatch):
        monkeypatch.setattr(lff, "LEARN_FROM_FIX_ENABLED", False)
        state = _minimal_state()
        assert lff.record_lesson(state) is None

    def test_missing_repo_name_skips(self):
        state = _minimal_state()
        state["work_order"].pop("repo_name")
        assert lff.record_lesson(state) is None

    def test_haiku_path_used_when_available(self, tmp_path):
        """When the Haiku extractor returns content, that's used (not the fallback)."""
        haiku_output = (
            "**Pattern**: auth flow\n"
            "**Lesson**: remember to check token expiry before validating\n"
            "**Tactic**: read auth middleware first"
        )
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=haiku_output):
            lff.record_lesson(_minimal_state(ticket_id="BUG-H", repo_name="r"))
        text = (tmp_path / "r" / "agent_lessons.md").read_text()
        assert "token expiry" in text
        assert "auth middleware" in text


# ---------------------------------------------------------------------------
# load_lessons
# ---------------------------------------------------------------------------

class TestLoadLessons:
    def test_returns_empty_when_no_lessons_file(self):
        result = lff.load_lessons("nonexistent_repo")
        assert result == ""

    def test_returns_formatted_section(self, tmp_path):
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "root", "target_files": ["f.py"], "approach": "a",
            }):
                lff.record_lesson(_minimal_state(ticket_id="BUG-A", repo_name="r"))
        section = lff.load_lessons("r")
        assert "## LESSONS FROM PAST RUNS" in section
        assert "BUG-A" in section

    def test_respects_max_injected(self, tmp_path, monkeypatch):
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                for i in range(10):
                    lff.record_lesson(_minimal_state(ticket_id=f"BUG-{i}", repo_name="r"))
        section = lff.load_lessons("r", max_entries=3)
        # Only the 3 newest should appear
        assert "BUG-9" in section
        assert "BUG-8" in section
        assert "BUG-7" in section
        # Earlier ones shouldn't
        assert "BUG-5" not in section

    def test_disabled_feature_returns_empty(self, tmp_path, monkeypatch):
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                lff.record_lesson(_minimal_state(ticket_id="BUG-X", repo_name="r"))
        # Now disable the flag and try to load
        monkeypatch.setattr(lff, "LEARN_FROM_FIX_ENABLED", False)
        assert lff.load_lessons("r") == ""

    def test_truncates_very_large_lessons_file(self, tmp_path, monkeypatch):
        """When the formatted section exceeds 3K chars, it's truncated with a marker."""
        # Force a large amount of stored lessons by raising the cap, then inject all
        monkeypatch.setattr(lff, "MAX_LESSONS_STORED", 50)
        with patch("agent.learn_from_fix._extract_lesson_via_haiku",
                   return_value="**Pattern**: x\n**Lesson**: " + "y" * 500 + "\n**Tactic**: z"):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                for i in range(30):
                    lff.record_lesson(_minimal_state(ticket_id=f"BUG-{i}", repo_name="r"))
        section = lff.load_lessons("r", max_entries=30)
        if len(section) >= 3000:
            assert "truncated" in section


# ---------------------------------------------------------------------------
# Cross-repo isolation — CRITICAL for correctness
# ---------------------------------------------------------------------------

class TestCrossRepoIsolation:
    def test_lessons_are_per_repo(self, tmp_path):
        with patch("agent.learn_from_fix._extract_lesson_via_haiku", return_value=""):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "django-specific", "target_files": ["x.py"], "approach": "a",
            }):
                lff.record_lesson(_minimal_state(ticket_id="D-1", repo_name="django"))

            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "flask-specific", "target_files": ["x.py"], "approach": "a",
            }):
                lff.record_lesson(_minimal_state(ticket_id="F-1", repo_name="flask"))

        django_section = lff.load_lessons("django")
        flask_section = lff.load_lessons("flask")

        assert "D-1" in django_section
        assert "F-1" not in django_section
        assert "F-1" in flask_section
        assert "D-1" not in flask_section


# ---------------------------------------------------------------------------
# Markdown parse round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_parse_format_roundtrip(self, tmp_path):
        """Format → parse → format must be stable (no data loss on re-append)."""
        with patch("agent.learn_from_fix._extract_lesson_via_haiku",
                   return_value="**Pattern**: p\n**Lesson**: l\n**Tactic**: t"):
            with patch("agent.react_tools.get_current_plan", return_value={
                "root_cause": "r", "target_files": ["f.py"], "approach": "a",
            }):
                lff.record_lesson(_minimal_state(ticket_id="BUG-RT", repo_name="r"))
        path = tmp_path / "r" / "agent_lessons.md"
        text1 = path.read_text()
        entries = lff._parse_lessons(text1)
        text2 = lff._format_lessons(entries)
        # Re-parsing the re-formatted output must give the same entries
        entries2 = lff._parse_lessons(text2)
        assert len(entries) == len(entries2)
        for e1, e2 in zip(entries, entries2):
            assert e1["ticket_id"] == e2["ticket_id"]
            assert e1["body"] == e2["body"]
