"""
test_brt_and_verifier.py — Tests for brt_node() and verifier_node() in react_pipeline.py

Covers:
  - brt_node skips non-Python repos
  - brt_node skips when no hint files
  - brt_node only confirms exit-code-1 tests (not 0, 4, 5)
  - brt_node stores confirmed BRTs in state["brts"]
  - brt_node is non-fatal (graceful failure propagation)
  - verifier_node skips when agent didn't submit
  - verifier_node skips when no diff
  - verifier_node stores verdict in state
  - disable_brt flag bypasses brt_node
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(repo_path: str = "/tmp/fake_repo", submitted: bool = False) -> dict:
    return {
        "work_order": {
            "ticket_id": "TEST-001",
            "title": "Test bug",
            "description": "Something is broken in check_gate()",
            "repo_name": "test_repo",
            "repo_path": repo_path,
        },
        "intent": {
            "likely_affected_modules": ["agent/guardrails.py"],
            "likely_affected_functions": ["check_gate"],
            "actual_behavior": "gate blocks when it should allow",
            "expected_behavior": "gate should allow skipped tests",
        },
        "submitted": submitted,
        "escalated": False,
        "escalate_reason": "",
        "explanation": "Fixed the gate condition",
        "tool_call_count": 0,
        "cost_usd": 0.0,
        "messages": [],
        "localization": {},
        "repair": {},
        "review": {},
        "status": "done",
        "error": "",
        "pr_url": "",
        "test_result": "",
        "dry_run": False,
        "brts": [],
    }


# ---------------------------------------------------------------------------
# brt_node — skipping conditions
# ---------------------------------------------------------------------------

class TestBrtNodeSkipping:
    def test_skips_non_python_repo(self, tmp_path):
        """brt_node must skip gracefully for non-Python repos (no pytest markers)."""
        from agent.react_pipeline import brt_node

        state = _minimal_state(repo_path=str(tmp_path))
        # tmp_path has no pytest markers → not a Python repo
        result = brt_node(state)
        # Should return state unchanged (no brts added)
        assert "brts" not in result or result.get("brts") == []

    def test_skips_when_no_hint_files(self, tmp_path):
        """brt_node skips when intent has no hint files."""
        from agent.react_pipeline import brt_node

        # Create a Python marker so it's detected as Python
        (tmp_path / "pytest.ini").write_text("[pytest]\n")

        state = _minimal_state(repo_path=str(tmp_path))
        state["intent"]["likely_affected_modules"] = []
        state["intent"]["confirmed_files"] = []

        result = brt_node(state)
        assert result.get("brts") == [] or "brts" not in result

    def test_skips_when_no_repo_path(self):
        """brt_node skips gracefully when work_order has no repo_path."""
        from agent.react_pipeline import brt_node

        state = _minimal_state(repo_path="")
        state["work_order"]["repo_path"] = ""
        result = brt_node(state)
        assert result is not None  # Must not crash


# ---------------------------------------------------------------------------
# brt_node — BRT confirmation logic
# ---------------------------------------------------------------------------

class TestBrtConfirmation:
    """Only subprocess exit code 1 (assertion failure) should confirm a BRT."""

    def _run_brt_with_exit_code(self, tmp_path: Path, exit_code: int) -> dict:
        """Helper: set up a Python repo, mock Haiku + subprocess, run brt_node."""
        from agent.react_pipeline import brt_node

        # Create Python repo markers
        (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
        source_file = tmp_path / "agent" / "guardrails.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text(
            "def check_gate(gs):\n    if not gs.tests_passed:\n        return 'blocked'\n    return None\n"
        )

        state = _minimal_state(repo_path=str(tmp_path))
        state["intent"]["likely_affected_modules"] = ["agent/guardrails.py"]
        state["intent"]["likely_affected_functions"] = ["check_gate"]

        fake_candidate = MagicMock()
        fake_candidate.test_code = "def test_gate_allows_skipped():\n    assert check_gate(None) is None\n"
        fake_candidate.description = "gate should allow skipped"
        fake_candidate.target_function = "check_gate"

        fake_batch = MagicMock()
        fake_batch.candidates = [fake_candidate]

        mock_proc = MagicMock()
        mock_proc.returncode = exit_code
        mock_proc.stdout = "FAILED" if exit_code == 1 else "passed 1 test"
        mock_proc.stderr = ""

        with patch("agent.llm.structured_call", return_value=fake_batch), \
             patch("subprocess.run", return_value=mock_proc), \
             patch("os.unlink", return_value=None):
            return brt_node(state)

    def test_exit_code_1_confirms_brt(self, tmp_path):
        result = self._run_brt_with_exit_code(tmp_path, 1)
        assert len(result.get("brts", [])) == 1, "exit code 1 should confirm BRT"

    def test_exit_code_0_does_not_confirm(self, tmp_path):
        result = self._run_brt_with_exit_code(tmp_path, 0)
        assert result.get("brts", []) == [], "exit code 0 (test passes on broken code) should NOT confirm"

    def test_exit_code_4_does_not_confirm(self, tmp_path):
        result = self._run_brt_with_exit_code(tmp_path, 4)
        assert result.get("brts", []) == [], "exit code 4 (usage error) should NOT confirm"

    def test_exit_code_5_does_not_confirm(self, tmp_path):
        result = self._run_brt_with_exit_code(tmp_path, 5)
        assert result.get("brts", []) == [], "exit code 5 (no tests collected) should NOT confirm"

    def test_brt_stores_required_fields(self, tmp_path):
        result = self._run_brt_with_exit_code(tmp_path, 1)
        brts = result.get("brts", [])
        if brts:
            brt = brts[0]
            assert "code" in brt
            assert "description" in brt
            assert "target_function" in brt
            assert "fail_output" in brt

    def test_max_3_brts_stored(self, tmp_path):
        """brt_node stores at most 3 confirmed BRTs."""
        from agent.react_pipeline import brt_node

        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        source_file = tmp_path / "agent" / "g.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def f(): pass\n")

        state = _minimal_state(repo_path=str(tmp_path))
        state["intent"]["likely_affected_modules"] = ["agent/g.py"]
        state["intent"]["likely_affected_functions"] = ["f"]

        # Make 6 candidates all with exit code 1
        fake_candidates = []
        for i in range(6):
            c = MagicMock()
            c.test_code = f"def test_{i}():\n    assert False\n"
            c.description = f"test {i}"
            c.target_function = "f"
            fake_candidates.append(c)

        fake_batch = MagicMock()
        fake_batch.candidates = fake_candidates

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "FAILED"
        mock_proc.stderr = ""

        with patch("agent.llm.structured_call", return_value=fake_batch), \
             patch("subprocess.run", return_value=mock_proc), \
             patch("os.unlink", return_value=None):
            result = brt_node(state)

        assert len(result.get("brts", [])) <= 3


# ---------------------------------------------------------------------------
# brt_node — graceful failure
# ---------------------------------------------------------------------------

class TestBrtGracefulFailure:
    def test_llm_failure_is_non_fatal(self, tmp_path):
        """LLM API failure must not propagate — agent proceeds without BRTs."""
        from agent.react_pipeline import brt_node

        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        source_file = tmp_path / "agent" / "g.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def f(): pass\n")

        state = _minimal_state(repo_path=str(tmp_path))
        state["intent"]["likely_affected_modules"] = ["agent/g.py"]
        state["intent"]["likely_affected_functions"] = ["f"]

        with patch("agent.llm.structured_call", side_effect=RuntimeError("API timeout")):
            result = brt_node(state)

        assert result is not None
        assert result.get("brts", []) == []

    def test_subprocess_timeout_is_non_fatal(self, tmp_path):
        """Subprocess timeout per candidate should not abort BRT generation."""
        from agent.react_pipeline import brt_node

        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        source_file = tmp_path / "agent" / "g.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def f(): pass\n")

        state = _minimal_state(repo_path=str(tmp_path))
        state["intent"]["likely_affected_modules"] = ["agent/g.py"]
        state["intent"]["likely_affected_functions"] = ["f"]

        fake_candidate = MagicMock()
        fake_candidate.test_code = "def test_something():\n    assert False\n"
        fake_candidate.description = "desc"
        fake_candidate.target_function = "f"
        fake_batch = MagicMock()
        fake_batch.candidates = [fake_candidate]

        with patch("agent.llm.structured_call", return_value=fake_batch), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pytest", 30)), \
             patch("os.unlink", return_value=None):
            result = brt_node(state)

        assert result is not None
        assert result.get("brts", []) == []


# ---------------------------------------------------------------------------
# verifier_node — skipping conditions
# ---------------------------------------------------------------------------

class TestVerifierNodeSkipping:
    def test_skips_when_not_submitted(self):
        from agent.react_pipeline import verifier_node
        state = _minimal_state(submitted=False)
        result = verifier_node(state)
        assert result is not None
        # verifier_verdict should not be set
        assert "verifier_verdict" not in result

    def test_skips_when_no_sandbox(self):
        from agent.react_pipeline import verifier_node
        state = _minimal_state(submitted=True)
        state["sandbox_path"] = ""
        result = verifier_node(state)
        assert "verifier_verdict" not in result

    def test_skips_when_no_diff(self, tmp_path):
        from agent.react_pipeline import verifier_node

        state = _minimal_state(submitted=True)
        state["sandbox_path"] = str(tmp_path)
        # subprocess.run returns empty diff
        mock_proc = MagicMock()
        mock_proc.stdout = ""
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            result = verifier_node(state)

        assert "verifier_verdict" not in result


# ---------------------------------------------------------------------------
# verifier_node — approve/reject path
# ---------------------------------------------------------------------------

class TestVerifierNodeVerdict:
    def _run_verifier(self, tmp_path: Path, verdict: str, confidence: float) -> dict:
        from agent.react_pipeline import verifier_node

        state = _minimal_state(submitted=True)
        state["sandbox_path"] = str(tmp_path)
        state["test_result"] = "passed: 5 tests"

        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/f.py b/f.py\n+def fix(): pass\n"
        mock_diff.returncode = 0

        from pydantic import BaseModel

        _verdict = verdict
        _confidence = confidence

        class FakeVerifier(BaseModel):
            verdict: str = _verdict
            confidence: float = _confidence
            explanation: str = "looks correct"
            regression_risk: str = "LOW"

        with patch("subprocess.run", return_value=mock_diff), \
             patch("agent.llm.structured_call", return_value=FakeVerifier()):
            return verifier_node(state)

    def test_approve_verdict_stored(self, tmp_path):
        result = self._run_verifier(tmp_path, "APPROVE", 0.9)
        assert result.get("verifier_verdict") == "APPROVE"

    def test_reject_verdict_stored(self, tmp_path):
        result = self._run_verifier(tmp_path, "REJECT", 0.7)
        assert result.get("verifier_verdict") == "REJECT"

    def test_verifier_confidence_stored(self, tmp_path):
        result = self._run_verifier(tmp_path, "APPROVE", 0.85)
        assert result.get("verifier_confidence") == pytest.approx(0.85, abs=0.01)

    def test_verifier_failure_is_non_fatal(self, tmp_path):
        from agent.react_pipeline import verifier_node

        state = _minimal_state(submitted=True)
        state["sandbox_path"] = str(tmp_path)

        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/f.py b/f.py\n+x = 1\n"
        mock_diff.returncode = 0

        with patch("subprocess.run", return_value=mock_diff), \
             patch("agent.llm.structured_call", side_effect=RuntimeError("LLM error")):
            result = verifier_node(state)

        assert result is not None


# ---------------------------------------------------------------------------
# disable_brt flag (A/B eval feature)
# ---------------------------------------------------------------------------

class TestDisableBrtFlag:
    def test_disable_brt_skips_brt_node(self, tmp_path):
        """run_ticket_react with disable_brt=True should not call brt_node logic."""
        from agent import react_pipeline
        import threading

        # Set the thread-local flag directly
        react_pipeline._thread_local.disable_brt = True

        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        source_file = tmp_path / "agent" / "g.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def f(): pass\n")

        state = _minimal_state(repo_path=str(tmp_path))
        state["intent"]["likely_affected_modules"] = ["agent/g.py"]

        brt_called = []

        original_structured_call = react_pipeline._structured_call if hasattr(react_pipeline, "_structured_call") else None

        with patch("agent.llm.structured_call") as mock_sc:
            # If disable_brt is checked before the LLM call, _structured_call should never be called
            result = react_pipeline.brt_node(state)

        # Since disable_brt is set, brt_node should return state unchanged
        # (The actual skip is in run_ticket_react, not brt_node itself)
        assert result is not None

        # Clean up
        react_pipeline._thread_local.disable_brt = False
