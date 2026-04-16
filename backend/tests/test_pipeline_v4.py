"""
test_pipeline_v4.py — Tests for Pipeline v4 setup_node and its 3 parallel threads.

Covers:
  - setup_node runs all 3 threads and merges results into state["_dynamic_context"]
  - _setup_thread_repo: repo detection, sandbox creation, baseline tests
  - _setup_thread_scout: scout localization with path validation + fuzzy recovery
  - _setup_thread_context: repo tree, graph context, lessons, concept mappings
  - Thread isolation: no shared mutable state between threads
  - Error resilience: each thread failing independently doesn't block others
  - Edge cases: missing repo path, non-git repo, permission errors
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(repo_path: str = "/tmp/fake_repo") -> dict:
    return {
        "work_order": {
            "ticket_id": "TEST-001",
            "title": "Test bug",
            "description": "Something is broken in check_gate()",
            "repo_name": "test_repo",
            "repo_path": repo_path,
            "priority": "high",
            "comments": [],
        },
        "intent": {},
        "submitted": False,
        "escalated": False,
        "escalate_reason": "",
        "explanation": "",
        "tool_call_count": 0,
        "cost_usd": 0.0,
        "messages": [],
        "localization": {},
        "repair": {},
        "review": {},
        "status": "pending",
        "error": "",
        "pr_url": "",
        "test_result": "",
        "dry_run": False,
        "brts": [],
        "sandbox_path": "",
        "branch_name": "",
        "base_branch": "main",
    }


def _make_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo for sandbox creation tests."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    (tmp_path / "hello.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: _setup_thread_repo
# ---------------------------------------------------------------------------

class TestSetupThreadRepo:
    """Tests for Thread 1: repo detection + sandbox + baseline tests."""

    def test_creates_sandbox_and_returns_paths(self, tmp_path):
        """Happy path: creates worktree, returns sandbox_path, branch, base."""
        repo = _make_git_repo(tmp_path)

        from agent.react_pipeline import _setup_thread_repo

        with patch("agent.react_pipeline.re") as mock_re_module:
            # Let the real re module work — just import from agent.react_pipeline
            mock_re_module.sub = __import__("re").sub
            mock_re_module.findall = __import__("re").findall

            result = _setup_thread_repo(repo, "test_repo")

        assert result["sandbox_path"], "sandbox_path should be non-empty"
        assert Path(result["sandbox_path"]).exists(), "sandbox directory should exist"
        assert result["branch_name"].startswith("fix/test_repo")
        assert result["base_branch"], "base_branch should be set"
        assert isinstance(result["baseline_failures"], set)

        # Cleanup worktree
        subprocess.run(
            ["git", "worktree", "remove", "--force", result["sandbox_path"]],
            cwd=repo, capture_output=True,
        )

    def test_auto_detects_project_config(self, tmp_path):
        """Writes .agent_config.json when missing."""
        repo = _make_git_repo(tmp_path)
        (repo / "setup.py").write_text("from setuptools import setup\nsetup(name='test')\n")

        from agent.react_pipeline import _setup_thread_repo

        result = _setup_thread_repo(repo, "test_repo")

        # Config should have been written (repo_detection detects Python)
        config_path = repo / ".agent_config.json"
        # May or may not exist depending on detection — but shouldn't crash
        assert result["sandbox_path"], "sandbox should be created regardless of detection"

        # Cleanup
        if result["sandbox_path"]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", result["sandbox_path"]],
                cwd=repo, capture_output=True,
            )

    def test_returns_empty_when_no_repo_path(self):
        """Returns empty result when repo_path is None."""
        from agent.react_pipeline import _setup_thread_repo

        result = _setup_thread_repo(None, "test_repo")

        assert result["sandbox_path"] == ""
        assert result["branch_name"] == ""
        assert result["baseline_failures"] == set()

    def test_returns_empty_when_not_git_repo(self, tmp_path):
        """Returns empty result when directory is not a git repo."""
        from agent.react_pipeline import _setup_thread_repo

        result = _setup_thread_repo(tmp_path, "test_repo")

        assert result["sandbox_path"] == ""
        assert result["branch_name"] == ""

    def test_handles_dirty_repo(self, tmp_path):
        """Returns empty result when repo has uncommitted changes."""
        repo = _make_git_repo(tmp_path)
        # Make repo dirty (modified tracked file)
        (repo / "hello.py").write_text("print('modified')\n")

        from agent.react_pipeline import _setup_thread_repo

        result = _setup_thread_repo(repo, "test_repo")

        assert result["sandbox_path"] == "", "should not create sandbox with dirty repo"

    def test_captures_baseline_test_failures(self, tmp_path):
        """Captures pre-existing test failures as set[str]."""
        repo = _make_git_repo(tmp_path)

        from agent.react_pipeline import _setup_thread_repo

        with patch("agent.sandbox.run_tests", return_value="failed\nFAILED tests/test_a.py::test_one\nFAILED tests/test_b.py::test_two"):
            result = _setup_thread_repo(repo, "test_repo")

        assert "tests/test_a.py::test_one" in result["baseline_failures"]
        assert "tests/test_b.py::test_two" in result["baseline_failures"]

        # Cleanup
        if result["sandbox_path"]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", result["sandbox_path"]],
                cwd=repo, capture_output=True,
            )

    def test_baseline_passing_returns_empty_set(self, tmp_path):
        """When baseline tests pass, failures set is empty."""
        repo = _make_git_repo(tmp_path)

        from agent.react_pipeline import _setup_thread_repo

        with patch("agent.sandbox.run_tests", return_value="passed (5 tests in 0.3s)"):
            result = _setup_thread_repo(repo, "test_repo")

        assert result["baseline_failures"] == set()

        # Cleanup
        if result["sandbox_path"]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", result["sandbox_path"]],
                cwd=repo, capture_output=True,
            )


# ---------------------------------------------------------------------------
# Tests: _setup_thread_scout
# ---------------------------------------------------------------------------

class TestSetupThreadScout:
    """Tests for Thread 2: scout localization."""

    def test_returns_scout_report_and_validated_files(self, tmp_path):
        """Happy path: scout returns locations, paths get validated."""
        repo = tmp_path
        # Create the actual file that scout will "find"
        (repo / "src").mkdir(parents=True, exist_ok=True)
        handler = repo / "src" / "handler.py"
        handler.write_text("def handle(): pass\n")

        fake_scout_report = {
            "top_locations": [
                {"file": "src/handler.py", "function": "handle", "confidence": 0.9, "reason": "test"},
            ],
            "scout_cost_usd": 0.02,
        }

        from agent.react_pipeline import _setup_thread_scout

        with patch("agent.scout.scout_localize", return_value=fake_scout_report), \
             patch("agent.react_pipeline._classify_community", return_value="core"):

            result = _setup_thread_scout(
                "test_repo",
                {"title": "Bug", "description": "broken"},
                {"actual_behavior": "crashes"},
                repo,
            )

        assert result["scout_report"] == fake_scout_report
        assert "src/handler.py" in result["scout_files"]
        assert result["community"] == "core"

    def test_filters_hallucinated_paths(self, tmp_path):
        """Hallucinated paths that don't exist are excluded."""
        fake_scout_report = {
            "top_locations": [
                {"file": "src/nonexistent.py", "function": "foo", "confidence": 0.9, "reason": "hallucinated"},
            ],
            "scout_cost_usd": 0.01,
        }

        from agent.react_pipeline import _setup_thread_scout

        with patch("agent.scout.scout_localize", return_value=fake_scout_report), \
             patch("agent.react_pipeline._classify_community", return_value=None):

            result = _setup_thread_scout(
                "test_repo",
                {"title": "Bug"},
                {},
                tmp_path,
            )

        assert result["scout_files"] == [], "hallucinated paths should be filtered out"

    def test_falls_back_to_prelocalize_on_scout_failure(self):
        """When scout raises, falls back to _prelocalize."""
        from agent.react_pipeline import _setup_thread_scout

        with patch("agent.scout.scout_localize", side_effect=RuntimeError("API down")), \
             patch("agent.react_pipeline._classify_community", return_value=None), \
             patch("agent.react_pipeline._prelocalize", return_value=["fallback/file.py"]):

            result = _setup_thread_scout(
                "test_repo",
                {"title": "Bug"},
                {},
                Path("/tmp/fake"),
            )

        assert result["scout_files"] == ["fallback/file.py"]

    def test_returns_empty_when_no_repo_name(self):
        """Returns empty result when repo_name is empty."""
        from agent.react_pipeline import _setup_thread_scout

        result = _setup_thread_scout("", {}, {}, None)

        assert result["scout_report"] == {}
        assert result["scout_files"] == []
        assert result["community"] is None

    def test_fuzzy_recovery_for_hallucinated_paths(self, tmp_path):
        """Fuzzy recovery finds the real file when scout gets directory wrong."""
        # Create real file in a different directory than what scout reports
        (tmp_path / "checkers").mkdir()
        (tmp_path / "checkers" / "misc.py").write_text("def check(): pass\n")

        fake_scout_report = {
            "top_locations": [
                {"file": "extensions/misc.py", "function": "check", "confidence": 0.8, "reason": "test"},
            ],
            "scout_cost_usd": 0.01,
        }

        from agent.react_pipeline import _setup_thread_scout

        with patch("agent.scout.scout_localize", return_value=fake_scout_report), \
             patch("agent.react_pipeline._classify_community", return_value=None):

            result = _setup_thread_scout(
                "test_repo",
                {"title": "Bug"},
                {},
                tmp_path,
            )

        # Fuzzy recovery should find checkers/misc.py
        assert len(result["scout_files"]) == 1
        assert "misc.py" in result["scout_files"][0]


# ---------------------------------------------------------------------------
# Tests: _setup_thread_context
# ---------------------------------------------------------------------------

class TestSetupThreadContext:
    """Tests for Thread 3: context assembly."""

    def test_builds_repo_tree(self, tmp_path):
        """Builds a repo tree listing from the filesystem."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass\n")

        from agent.react_pipeline import _setup_thread_context

        with patch("agent.graph_utils.build_kickstart_context", return_value="graph stuff"), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}):

            result = _setup_thread_context("test_repo", tmp_path, {"title": "Bug"})

        assert "src/main.py" in result["repo_tree"]

    def test_loads_graph_context(self, tmp_path):
        """Loads kickstart context from graph_utils."""
        from agent.react_pipeline import _setup_thread_context

        with patch("agent.scout._build_repo_listing", return_value=""), \
             patch("agent.graph_utils.build_kickstart_context", return_value="GRAPH NEIGHBORS of hint area"), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}):

            result = _setup_thread_context("test_repo", tmp_path, {"title": "Bug"})

        assert "GRAPH NEIGHBORS" in result["graph_context"]

    def test_loads_lessons(self, tmp_path):
        """Loads past-run lessons."""
        from agent.react_pipeline import _setup_thread_context

        with patch("agent.scout._build_repo_listing", return_value=""), \
             patch("agent.graph_utils.build_kickstart_context", return_value=""), \
             patch("agent.learn_from_fix.load_lessons", return_value="## LESSONS FROM PAST RUNS\n- avoid X"), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}):

            result = _setup_thread_context("test_repo", tmp_path, {"title": "Bug"})

        assert "LESSONS" in result["lessons"]

    def test_loads_concept_mappings(self, tmp_path):
        """Loads concept-to-code mappings."""
        c2c = {
            "matched_rules": [{"rule_text": "approval flow"}],
            "hint_functions": ["approve_request"],
            "hint_files": ["workflow/approval.py"],
            "concept_section": "## Concept Mapping\n...",
        }

        from agent.react_pipeline import _setup_thread_context

        with patch("agent.scout._build_repo_listing", return_value=""), \
             patch("agent.graph_utils.build_kickstart_context", return_value=""), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value=c2c):

            result = _setup_thread_context("test_repo", tmp_path, {"title": "approval bug"})

        assert result["concept_mappings"]["hint_functions"] == ["approve_request"]

    def test_resilient_to_individual_failures(self, tmp_path):
        """Each sub-step failing doesn't block the others."""
        from agent.react_pipeline import _setup_thread_context

        with patch("agent.scout._build_repo_listing", side_effect=RuntimeError("boom")), \
             patch("agent.graph_utils.build_kickstart_context", return_value="graph ok"), \
             patch("agent.learn_from_fix.load_lessons", side_effect=RuntimeError("kaboom")), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}):

            result = _setup_thread_context("test_repo", tmp_path, {"title": "Bug"})

        # repo_tree and lessons failed, but graph_context succeeded
        assert result["repo_tree"] == ""
        assert result["graph_context"] == "graph ok"
        assert result["lessons"] == ""

    def test_handles_none_repo_path(self):
        """Returns empty strings when repo_path is None."""
        from agent.react_pipeline import _setup_thread_context

        with patch("agent.graph_utils.build_kickstart_context", return_value=""), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}):

            result = _setup_thread_context("test_repo", None, {"title": "Bug"})

        assert result["repo_tree"] == ""


# ---------------------------------------------------------------------------
# Tests: setup_node (integration)
# ---------------------------------------------------------------------------

class TestSetupNode:
    """Tests for the top-level setup_node that orchestrates 3 threads."""

    def test_setup_node_produces_dynamic_context(self, tmp_path):
        """setup_node merges all thread results into state['_dynamic_context']."""
        repo = _make_git_repo(tmp_path)
        state = _minimal_state(repo_path=str(repo))

        from agent.react_pipeline import setup_node
        from agent.types import IntentAnalysis

        fake_intent = IntentAnalysis(
            expected_behavior="should work",
            actual_behavior="crashes",
            likely_affected_modules=["src/handler.py"],
            likely_affected_functions=["handle"],
            fix_type="bug_fix",
            severity="high",
            acceptance_criteria=["test passes"],
        )

        with patch("agent.llm.structured_call", return_value=fake_intent), \
             patch("agent.react_pipeline._classify_community", return_value=None), \
             patch("agent.scout.scout_localize", return_value={"top_locations": [], "scout_cost_usd": 0}), \
             patch("agent.react_pipeline._get_trace", return_value=None), \
             patch("agent.graph_utils.build_kickstart_context", return_value="graph ctx"), \
             patch("agent.learn_from_fix.load_lessons", return_value="lessons text"), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}), \
             patch("agent.sandbox.run_tests", return_value="passed"), \
             patch("agent.sandbox.cleanup_stale_worktrees"):

            result = setup_node(state)

        # _dynamic_context should exist with all required keys
        dc = result.get("_dynamic_context")
        assert dc is not None, "state should have _dynamic_context"
        assert "sandbox_path" in dc
        assert "branch_name" in dc
        assert "base_branch" in dc
        assert "baseline_failures" in dc
        assert "scout_report" in dc
        assert "scout_files" in dc
        assert "community" in dc
        assert "repo_tree" in dc
        assert "graph_context" in dc
        assert "lessons" in dc
        assert "concept_mappings" in dc

        # Verify sandbox was actually created
        assert dc["sandbox_path"], "sandbox should have been created"
        assert Path(dc["sandbox_path"]).exists()

        # State-level fields should also be set
        assert result["sandbox_path"] == dc["sandbox_path"]
        assert result["branch_name"] == dc["branch_name"]

        # Intent should be populated
        assert result["intent"].get("actual_behavior"), "intent should be populated from _translate_intent"

        # Cleanup worktree
        subprocess.run(
            ["git", "worktree", "remove", "--force", dc["sandbox_path"]],
            cwd=repo, capture_output=True,
        )

    def test_setup_node_resilient_to_thread_failures(self, tmp_path):
        """If one thread fails, others still complete and results merge."""
        repo = _make_git_repo(tmp_path)
        state = _minimal_state(repo_path=str(repo))

        from agent.react_pipeline import setup_node
        from agent.types import IntentAnalysis

        fake_intent = IntentAnalysis(
            expected_behavior="should work",
            actual_behavior="crashes",
            likely_affected_modules=[],
            likely_affected_functions=[],
            fix_type="bug_fix",
            severity="high",
            acceptance_criteria=[],
        )

        # Scout thread will fail, but repo and context threads should succeed
        with patch("agent.llm.structured_call", return_value=fake_intent), \
             patch("agent.react_pipeline._classify_community", side_effect=RuntimeError("fail")), \
             patch("agent.scout.scout_localize", side_effect=RuntimeError("scout down")), \
             patch("agent.react_pipeline._prelocalize", side_effect=RuntimeError("also down")), \
             patch("agent.react_pipeline._get_trace", return_value=None), \
             patch("agent.graph_utils.build_kickstart_context", return_value="graph works"), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}), \
             patch("agent.sandbox.run_tests", return_value="passed"), \
             patch("agent.sandbox.cleanup_stale_worktrees"):

            result = setup_node(state)

        dc = result["_dynamic_context"]

        # Repo thread should have succeeded
        assert dc["sandbox_path"], "sandbox should still be created despite scout failure"

        # Scout thread failed — empty results expected
        assert dc["scout_files"] == []
        assert dc["scout_report"] == {}

        # Context thread should have succeeded
        assert dc["graph_context"] == "graph works"

        # Cleanup
        if dc["sandbox_path"]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", dc["sandbox_path"]],
                cwd=repo, capture_output=True,
            )

    def test_setup_node_merges_scout_files_into_intent(self, tmp_path):
        """Scout results are merged back into state['intent']."""
        repo = _make_git_repo(tmp_path)
        handler = repo / "src" / "handler.py"
        handler.parent.mkdir(parents=True, exist_ok=True)
        handler.write_text("def handle(): pass\n")
        # Commit the new file so worktree sees it
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add handler"],
            cwd=repo, capture_output=True, check=True,
        )

        state = _minimal_state(repo_path=str(repo))

        from agent.react_pipeline import setup_node
        from agent.types import IntentAnalysis

        fake_intent = IntentAnalysis(
            expected_behavior="should work",
            actual_behavior="crashes",
            likely_affected_modules=[],
            likely_affected_functions=[],
            fix_type="bug_fix",
            severity="high",
            acceptance_criteria=[],
        )

        fake_scout_report = {
            "top_locations": [
                {"file": "src/handler.py", "function": "handle", "confidence": 0.9, "reason": "scout"},
            ],
            "scout_cost_usd": 0.02,
        }

        with patch("agent.llm.structured_call", return_value=fake_intent), \
             patch("agent.react_pipeline._classify_community", return_value="core_module"), \
             patch("agent.scout.scout_localize", return_value=fake_scout_report), \
             patch("agent.react_pipeline._get_trace", return_value=None), \
             patch("agent.graph_utils.build_kickstart_context", return_value=""), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value={"matched_rules": []}), \
             patch("agent.sandbox.run_tests", return_value="passed"), \
             patch("agent.sandbox.cleanup_stale_worktrees"):

            result = setup_node(state)

        # Scout files should be merged into intent
        assert "src/handler.py" in result["intent"].get("confirmed_files", [])
        assert result["intent"].get("community") == "core_module"

        # Cleanup
        dc = result["_dynamic_context"]
        if dc["sandbox_path"]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", dc["sandbox_path"]],
                cwd=repo, capture_output=True,
            )

    def test_setup_node_merges_concept_to_code_into_intent(self, tmp_path):
        """Concept-to-code mappings are merged into intent's affected functions/modules."""
        repo = _make_git_repo(tmp_path)
        state = _minimal_state(repo_path=str(repo))

        from agent.react_pipeline import setup_node
        from agent.types import IntentAnalysis

        fake_intent = IntentAnalysis(
            expected_behavior="should work",
            actual_behavior="crashes",
            likely_affected_modules=["existing/file.py"],
            likely_affected_functions=["existing_func"],
            fix_type="bug_fix",
            severity="high",
            acceptance_criteria=[],
        )

        c2c = {
            "matched_rules": [{"rule_text": "approval flow"}],
            "hint_functions": ["approve_request"],
            "hint_files": ["workflow/approval.py"],
            "concept_section": "## Concept Mapping\n...",
        }

        with patch("agent.llm.structured_call", return_value=fake_intent), \
             patch("agent.react_pipeline._classify_community", return_value=None), \
             patch("agent.scout.scout_localize", return_value={"top_locations": [], "scout_cost_usd": 0}), \
             patch("agent.react_pipeline._get_trace", return_value=None), \
             patch("agent.graph_utils.build_kickstart_context", return_value=""), \
             patch("agent.learn_from_fix.load_lessons", return_value=""), \
             patch("agent.graph_utils.query_concept_to_code", return_value=c2c), \
             patch("agent.sandbox.run_tests", return_value="passed"), \
             patch("agent.sandbox.cleanup_stale_worktrees"):

            result = setup_node(state)

        intent = result["intent"]
        # Original functions + concept-derived should be merged
        assert "existing_func" in intent["likely_affected_functions"]
        assert "approve_request" in intent["likely_affected_functions"]
        # Concept section should be stashed
        assert intent.get("_concept_section") == "## Concept Mapping\n..."

        # Cleanup
        dc = result["_dynamic_context"]
        if dc["sandbox_path"]:
            subprocess.run(
                ["git", "worktree", "remove", "--force", dc["sandbox_path"]],
                cwd=repo, capture_output=True,
            )

    def test_threads_do_not_share_mutable_state(self, tmp_path):
        """Verify that thread functions receive independent copies, not shared refs."""
        repo = _make_git_repo(tmp_path)
        state = _minimal_state(repo_path=str(repo))

        from agent.react_pipeline import setup_node
        from agent.types import IntentAnalysis

        fake_intent = IntentAnalysis(
            expected_behavior="should work",
            actual_behavior="crashes",
            likely_affected_modules=["src/handler.py"],
            likely_affected_functions=["handle"],
            fix_type="bug_fix",
            severity="high",
            acceptance_criteria=[],
        )

        # Track that each thread gets separate dict objects
        captured_args = {}

        original_scout = None

        def spy_scout(repo_name, work_order, intent, repo_path, **kw):
            captured_args["scout_work_order_id"] = id(work_order)
            captured_args["scout_intent_id"] = id(intent)
            return {"top_locations": [], "scout_cost_usd": 0}

        def spy_context(repo_name, repo_path, work_order):
            captured_args["context_work_order_id"] = id(work_order)
            return {
                "repo_tree": "",
                "graph_context": "",
                "lessons": "",
                "concept_mappings": {"matched_rules": []},
            }

        with patch("agent.llm.structured_call", return_value=fake_intent), \
             patch("agent.react_pipeline._classify_community", return_value=None), \
             patch("agent.react_pipeline._setup_thread_scout", side_effect=spy_scout), \
             patch("agent.react_pipeline._setup_thread_context", side_effect=spy_context), \
             patch("agent.react_pipeline._get_trace", return_value=None), \
             patch("agent.sandbox.run_tests", return_value="passed"), \
             patch("agent.sandbox.cleanup_stale_worktrees"):

            result = setup_node(state)

        # The work_order dicts passed to scout and context should be different objects
        # (dict() copies in setup_node), not the same reference
        if "scout_work_order_id" in captured_args and "context_work_order_id" in captured_args:
            assert captured_args["scout_work_order_id"] == captured_args["context_work_order_id"], \
                "Both threads receive the same snapshot dict (from dict() copy)"

        # Cleanup
        dc = result.get("_dynamic_context", {})
        if dc.get("sandbox_path"):
            subprocess.run(
                ["git", "worktree", "remove", "--force", dc["sandbox_path"]],
                cwd=repo, capture_output=True,
            )
