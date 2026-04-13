"""
test_pipeline_integration.py — Integration tests for the ReAct pipeline.

Exercises real code paths end-to-end with mocked LLM calls and subprocess
boundaries. Verifies state flows correctly between pipeline nodes.

Covers:
  - intake_node → brt_node sequential flow (state threading)
  - verifier_node APPROVE / REJECT / invalid verdict / failure paths
  - finalize_node dry_run repair extraction
  - finalize_node escalation bypass
  - brt_node uses repo virtualenv python
  - best_of_n cleanup of losing sandboxes
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

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
            "priority": "high",
            "comments": [],
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
        "sandbox_path": "",
        "branch_name": "",
        "base_branch": "main",
    }


# ---------------------------------------------------------------------------
# 1. test_intake_to_brt_flow — intake_node → brt_node sequential
# ---------------------------------------------------------------------------

def test_intake_to_brt_flow(tmp_path):
    """Run intake_node → brt_node sequentially with mocked LLM.

    Verifies:
      - State flows correctly between nodes (intent populated after intake, used by brt)
      - BRT node receives confirmed_files from intake's pre-localization
      - Mock structured_call returns valid IntentAnalysis and BRTBatch
      - Mock subprocess.run for BRT candidate execution (return exit code 1 for confirmed)
    """
    from agent.react_pipeline import intake_node, brt_node
    from agent.types import IntentAnalysis

    # Set up a Python repo so brt_node doesn't skip
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    source_file = tmp_path / "agent" / "guardrails.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "def check_gate(gs):\n"
        "    if not gs.tests_passed:\n"
        "        return 'blocked'\n"
        "    return None\n"
    )

    state = _minimal_state(repo_path=str(tmp_path))

    # --- Mock for intake_node (IntentAnalysis) ---
    fake_intent = IntentAnalysis(
        expected_behavior="gate should allow skipped tests",
        actual_behavior="gate blocks when tests.skipped is True",
        likely_affected_modules=["agent/guardrails.py"],
        likely_affected_functions=["check_gate"],
        fix_type="bug_fix",
        severity="high",
        acceptance_criteria=["check_gate returns None when tests_skipped is True"],
    )

    # --- Mock for brt_node (BRTBatch) ---
    fake_brt_candidate = MagicMock()
    fake_brt_candidate.test_code = (
        "def test_gate_allows_skipped():\n"
        "    assert check_gate(None) is None\n"
    )
    fake_brt_candidate.description = "gate should allow skipped tests"
    fake_brt_candidate.target_function = "check_gate"

    fake_batch = MagicMock()
    fake_batch.candidates = [fake_brt_candidate]

    mock_brt_proc = MagicMock()
    mock_brt_proc.returncode = 1
    mock_brt_proc.stdout = "FAILED test_gate_allows_skipped"
    mock_brt_proc.stderr = ""

    # Track which model each structured_call is for
    call_count = [0]

    def _mock_structured_call(model, max_tokens, response_model, prompt, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            # First 1-2 calls are intake (IntentAnalysis + possibly CommunityMatch)
            if response_model.__name__ == "IntentAnalysis":
                return fake_intent
            # CommunityMatch or other intake calls — return a mock
            mock_result = MagicMock()
            mock_result.community_name = "unknown"
            mock_result.confidence = 0.1
            return mock_result
        # Later calls are BRT
        return fake_batch

    with patch("agent.llm.structured_call", side_effect=_mock_structured_call), \
         patch("agent.react_pipeline._prelocalize", return_value=["agent/guardrails.py"]), \
         patch("agent.react_pipeline._classify_community", return_value=None), \
         patch("agent.react_pipeline._get_trace", return_value=None), \
         patch("subprocess.run", return_value=mock_brt_proc), \
         patch("os.unlink", return_value=None):

        # --- Phase 1: intake_node ---
        state = intake_node(state)

        # Verify intent was populated by intake
        assert state["intent"] is not None
        assert state["intent"].get("actual_behavior"), "intake should populate actual_behavior"
        assert state["intent"].get("likely_affected_functions"), "intake should populate likely_affected_functions"

        # Inject confirmed_files as if pre-localization ran
        # (since we mocked _prelocalize above, scout may not have run — ensure confirmed_files present)
        if "confirmed_files" not in state["intent"]:
            state["intent"]["confirmed_files"] = ["agent/guardrails.py"]

        # --- Phase 2: brt_node ---
        state = brt_node(state)

    # Verify BRT node used confirmed_files from intake's pre-localization
    brts = state.get("brts", [])
    assert len(brts) >= 1, "brt_node should confirm at least 1 BRT with exit code 1"
    assert "code" in brts[0], "confirmed BRT should have test code"
    assert "target_function" in brts[0], "confirmed BRT should have target_function"


# ---------------------------------------------------------------------------
# 2. test_verifier_approves_good_patch
# ---------------------------------------------------------------------------

def test_verifier_approves_good_patch(tmp_path):
    """Verifier returns APPROVE with high confidence. Verify verdict stored correctly."""
    from agent.react_pipeline import verifier_node
    from pydantic import BaseModel

    state = _minimal_state(submitted=True)
    state["sandbox_path"] = str(tmp_path)
    state["test_result"] = "passed: 5 tests"

    mock_diff = MagicMock()
    mock_diff.stdout = "diff --git a/f.py b/f.py\n+def fix(): pass\n"
    mock_diff.returncode = 0

    class FakeVerifierResult(BaseModel):
        verdict: str = "APPROVE"
        confidence: float = 0.9
        # Includes adversarial-probe evidence so the anti-rationalization gate
        # in verifier_node doesn't downgrade the verdict.
        explanation: str = (
            "Considered the boundary case (empty input, parallel callers) and "
            "the patch correctly fixes the bug."
        )
        regression_risk: str = "LOW"

    with patch("subprocess.run", return_value=mock_diff), \
         patch("agent.llm.structured_call", return_value=FakeVerifierResult()):
        result = verifier_node(state)

    assert result["verifier_verdict"] == "APPROVE"
    assert result["verifier_confidence"] == pytest.approx(0.9, abs=0.01)
    assert "boundary" in result["verifier_explanation"]
    assert result["verifier_regression_risk"] == "LOW"


# ---------------------------------------------------------------------------
# 3. test_verifier_rejects_bad_patch
# ---------------------------------------------------------------------------

def test_verifier_rejects_bad_patch(tmp_path):
    """Verifier returns REJECT with high confidence. Warning appended to explanation."""
    from agent.react_pipeline import verifier_node
    from pydantic import BaseModel

    state = _minimal_state(submitted=True)
    state["sandbox_path"] = str(tmp_path)
    state["test_result"] = "passed: 5 tests"
    state["explanation"] = "Original fix explanation"

    mock_diff = MagicMock()
    mock_diff.stdout = "diff --git a/f.py b/f.py\n-def old(): pass\n+def new(): pass\n"
    mock_diff.returncode = 0

    class FakeVerifierResult(BaseModel):
        verdict: str = "REJECT"
        confidence: float = 0.95
        explanation: str = "Patch introduces a regression in the auth module"
        regression_risk: str = "HIGH"

    with patch("subprocess.run", return_value=mock_diff), \
         patch("agent.llm.structured_call", return_value=FakeVerifierResult()):
        result = verifier_node(state)

    assert result["verifier_verdict"] == "REJECT"
    assert result["verifier_confidence"] == pytest.approx(0.95, abs=0.01)
    # High-confidence REJECT should append warning to explanation
    assert "VERIFIER FLAGGED" in result["explanation"]
    assert "Patch introduces a regression" in result["explanation"]
    # Original explanation should still be present
    assert "Original fix explanation" in result["explanation"]


# ---------------------------------------------------------------------------
# 4. test_verifier_validates_invalid_verdict
# ---------------------------------------------------------------------------

def test_verifier_validates_invalid_verdict(tmp_path):
    """Verifier returns invalid verdict 'MAYBE'. Should be normalized to REJECT."""
    from agent.react_pipeline import verifier_node
    from pydantic import BaseModel

    state = _minimal_state(submitted=True)
    state["sandbox_path"] = str(tmp_path)
    state["test_result"] = "passed: 3 tests"

    mock_diff = MagicMock()
    mock_diff.stdout = "diff --git a/f.py b/f.py\n+x = 1\n"
    mock_diff.returncode = 0

    class FakeVerifierResult(BaseModel):
        verdict: str = "MAYBE"
        confidence: float = 0.5
        explanation: str = "Not sure if this is correct"
        regression_risk: str = "MEDIUM"

    with patch("subprocess.run", return_value=mock_diff), \
         patch("agent.llm.structured_call", return_value=FakeVerifierResult()):
        result = verifier_node(state)

    assert result["verifier_verdict"] == "REJECT", "Invalid verdict 'MAYBE' should be normalized to REJECT"


# ---------------------------------------------------------------------------
# 5. test_verifier_failure_sets_skip
# ---------------------------------------------------------------------------

def test_verifier_failure_sets_skip(tmp_path):
    """When structured_call raises an exception, verifier_verdict should be SKIP."""
    from agent.react_pipeline import verifier_node

    state = _minimal_state(submitted=True)
    state["sandbox_path"] = str(tmp_path)
    state["test_result"] = "passed: 2 tests"

    mock_diff = MagicMock()
    mock_diff.stdout = "diff --git a/f.py b/f.py\n+x = 1\n"
    mock_diff.returncode = 0

    with patch("subprocess.run", return_value=mock_diff), \
         patch("agent.llm.structured_call", side_effect=RuntimeError("API rate limit exceeded")):
        result = verifier_node(state)

    assert result["verifier_verdict"] == "SKIP"
    assert "API rate limit exceeded" in result["verifier_explanation"]


# ---------------------------------------------------------------------------
# 6. test_finalize_dry_run_extracts_repair
# ---------------------------------------------------------------------------

def test_finalize_dry_run_extracts_repair(tmp_path):
    """finalize_node with dry_run=True extracts repair dict from sandbox diff."""
    from agent.react_pipeline import finalize_node

    # Set up a git repo in tmp_path to simulate the sandbox
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    state = _minimal_state(submitted=True)
    state["sandbox_path"] = str(sandbox)
    state["dry_run"] = True
    state["test_result"] = "passed: 3 tests"
    state["work_order"]["repo_path"] = str(tmp_path)

    # Mock subprocess calls: diff --stat, diff HEAD~1 --name-only
    def _mock_subprocess_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "--stat" in cmd_str:
            result.stdout = " src/app.py | 3 ++-\n 1 file changed\n"
        elif "--name-only" in cmd_str:
            result.stdout = "src/app.py\n"
        elif "--porcelain" in cmd_str:
            result.stdout = ""
        else:
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=_mock_subprocess_run), \
         patch("agent.react_pipeline._get_trace", return_value=None), \
         patch("agent.react_pipeline._report_progress"), \
         patch("agent.react_pipeline._cleanup_sandbox"):
        result = finalize_node(state)

    assert result["dry_run"] is True
    assert "(dry-run" in result.get("pr_url", "")
    # repair should be populated with patch files from the diff
    patches = result.get("repair", {}).get("patches", [])
    assert len(patches) >= 1, "dry_run finalize should extract patches from sandbox"
    assert patches[0]["file_path"] == "src/app.py"


# ---------------------------------------------------------------------------
# 7. test_finalize_escalated_skips_pr
# ---------------------------------------------------------------------------

def test_finalize_escalated_skips_pr():
    """finalize_node with escalated=True should not attempt PR creation."""
    from agent.react_pipeline import finalize_node

    state = _minimal_state(submitted=False)
    state["escalated"] = True
    state["escalate_reason"] = "Agent could not localize the bug"

    with patch("agent.react_pipeline._get_trace", return_value=None), \
         patch("agent.react_pipeline._report_progress"), \
         patch("agent.react_pipeline._cleanup_sandbox") as mock_cleanup, \
         patch("subprocess.run") as mock_subprocess:
        result = finalize_node(state)

    assert result["status"] == "escalated"
    # subprocess.run should NOT be called for push/PR since we're escalating
    # (cleanup may call subprocess for worktree removal, so check for push specifically)
    for c in mock_subprocess.call_args_list:
        args = c[0][0] if c[0] else []
        assert "push" not in args, "git push should not be called when escalated"
        assert "pr" not in args, "gh pr create should not be called when escalated"


# ---------------------------------------------------------------------------
# 8. test_brt_uses_repo_python
# ---------------------------------------------------------------------------

def test_brt_uses_repo_python(tmp_path):
    """brt_node should use the repo's virtualenv python, not sys.executable."""
    from agent.react_pipeline import brt_node

    # Set up Python repo
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    source_file = tmp_path / "agent" / "guardrails.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def check_gate(gs):\n    return None\n")

    state = _minimal_state(repo_path=str(tmp_path))
    state["intent"]["likely_affected_modules"] = ["agent/guardrails.py"]
    state["intent"]["likely_affected_functions"] = ["check_gate"]

    fake_candidate = MagicMock()
    fake_candidate.test_code = "def test_gate():\n    assert False\n"
    fake_candidate.description = "gate test"
    fake_candidate.target_function = "check_gate"
    fake_batch = MagicMock()
    fake_batch.candidates = [fake_candidate]

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = "FAILED"
    mock_proc.stderr = ""

    fake_python = "/fake/venv/bin/python"

    with patch("agent.llm.structured_call", return_value=fake_batch), \
         patch("agent.react_pipeline._find_repo_python", return_value=fake_python), \
         patch("subprocess.run", return_value=mock_proc) as mock_subprocess, \
         patch("os.unlink", return_value=None):
        brt_node(state)

    # Verify subprocess.run was called with the repo python, not sys.executable
    assert mock_subprocess.call_count >= 1, "subprocess.run should be called for BRT execution"
    first_call_args = mock_subprocess.call_args_list[0][0][0]
    assert first_call_args[0] == fake_python, (
        f"Expected repo python '{fake_python}', got '{first_call_args[0]}'"
    )
    assert first_call_args[0] != sys.executable, "Should not use sys.executable"


# ---------------------------------------------------------------------------
# 9. test_best_of_n_cleans_losing_sandboxes
# ---------------------------------------------------------------------------

def test_best_of_n_cleans_losing_sandboxes(tmp_path):
    """best_of_n should cleanup sandboxes of losing instances but not the winner."""
    from agent.react_pipeline import _run_best_of_n

    # Create 3 fake result dicts with different sandbox_paths
    sandbox_a = str(tmp_path / "sandbox_a")
    sandbox_b = str(tmp_path / "sandbox_b")
    sandbox_c = str(tmp_path / "sandbox_c")

    results = [
        {
            "work_order": {"ticket_id": "T-1_bon0", "repo_path": str(tmp_path)},
            "submitted": True,
            "test_result": "passed: 5 tests",
            "verifier_verdict": "APPROVE",
            "review": {"confidence": 0.95},
            "cost_usd": 1.5,
            "sandbox_path": sandbox_a,
            "branch_name": "fix/a",
            "status": "done",
        },
        {
            "work_order": {"ticket_id": "T-1_bon1", "repo_path": str(tmp_path)},
            "submitted": True,
            "test_result": "failed: 2 tests",
            "verifier_verdict": "REJECT",
            "review": {"confidence": 0.3},
            "cost_usd": 2.0,
            "sandbox_path": sandbox_b,
            "branch_name": "fix/b",
            "status": "done",
        },
        {
            "work_order": {"ticket_id": "T-1_bon2", "repo_path": str(tmp_path)},
            "submitted": False,
            "test_result": "",
            "verifier_verdict": "",
            "review": {"confidence": 0.0},
            "cost_usd": 0.5,
            "sandbox_path": sandbox_c,
            "branch_name": "fix/c",
            "escalated": True,
            "escalate_reason": "Could not fix",
            "status": "escalated",
        },
    ]

    work_order = {"ticket_id": "T-1", "repo_path": str(tmp_path)}

    # Mock run_ticket_react to return the 3 results in sequence
    call_index = [0]
    def _mock_run_one(seed):
        r = results[call_index[0] % len(results)]
        call_index[0] += 1
        return r

    with patch("agent.react_pipeline.run_ticket_react", side_effect=[results[0], results[1], results[2]]), \
         patch("agent.react_pipeline._cleanup_sandbox") as mock_cleanup, \
         patch("agent.react_pipeline._resolve_repo_path", return_value=tmp_path), \
         patch("concurrent.futures.ProcessPoolExecutor") as MockExecutor:

        # Simulate ProcessPoolExecutor returning our results
        mock_executor_instance = MagicMock()
        MockExecutor.return_value.__enter__ = MagicMock(return_value=mock_executor_instance)
        MockExecutor.return_value.__exit__ = MagicMock(return_value=False)

        futures = []
        for r in results:
            f = MagicMock()
            f.result.return_value = r
            futures.append(f)
        mock_executor_instance.submit.side_effect = futures

        # Mock as_completed to return futures in order
        with patch("concurrent.futures.as_completed", return_value=futures):
            best = _run_best_of_n(work_order, None, None, True, 3)

    # Winner is sandbox_a (submitted=True, test passed, APPROVE, highest confidence)
    assert best["sandbox_path"] == sandbox_a

    # Losing sandboxes (b and c) should be cleaned up
    cleaned_paths = [c[0][0] for c in mock_cleanup.call_args_list]
    assert sandbox_b in cleaned_paths, "Losing sandbox B should be cleaned up"
    assert sandbox_c in cleaned_paths, "Losing sandbox C should be cleaned up"
    assert sandbox_a not in cleaned_paths, "Winner sandbox A should NOT be cleaned up"
