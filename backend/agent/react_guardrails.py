"""
react_guardrails.py — Safety guardrails for the ReAct agent loop.

Enforces: sandbox gate, submit gate, cost cap, tool call cap, wall-clock cap.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Limits
MAX_TOOL_CALLS = 40          # Legacy default — use budget_for_difficulty() for new code
MAX_WALL_TIME = 900          # 15 minutes
MAX_COST_USD = 5.00
MAX_TEST_FAILURES = 3

# Adaptive tool call budget based on task complexity.
# Research: most bugs resolve in 15-25 calls; calls 30-40 are usually stuck loops.
# Giving a tighter budget to simple bugs saves cost and forces better focus.
_CALL_BUDGET = {
    "single-file": 30,  # one file, one function — agent should finish in 15-25
    "multi-file":  45,  # cross-module changes — callers, tests, imports need updating
    "complex":     55,  # architecture changes, 5+ files, new abstractions
}
_DEFAULT_BUDGET = 40   # unknown complexity — safe middle ground


def budget_for_difficulty(difficulty: str) -> int:
    """Return the adaptive tool call budget for a given task difficulty."""
    return _CALL_BUDGET.get(difficulty, _DEFAULT_BUDGET)

# Tools that require a sandbox to exist
SANDBOX_REQUIRED_TOOLS = frozenset({
    "string_replace", "create_file", "check_syntax",
    "run_tests", "run_brt",
})

# Tools that are terminal (end the loop)
TERMINAL_TOOLS = frozenset({"submit_fix", "escalate"})


class GuardrailState:
    """Tracks guardrail-relevant state during a ReAct loop run."""

    def __init__(self, max_tool_calls: int = MAX_TOOL_CALLS):
        self.max_tool_calls: int = max_tool_calls
        self.sandbox_created: bool = False
        self.sandbox_path: str = ""
        self.tests_passed: bool = False
        self.tests_attempted: bool = False
        self.tests_skipped: bool = False
        self.test_failure_count: int = 0
        self.review_approved: bool = False
        self.review_count: int = 0
        self.review_verdict: str = ""
        self.tool_call_count: int = 0
        self.cost_usd: float = 0.0
        self.start_time: float = time.monotonic()
        # Anti-pattern tracking
        self.tool_history: list[str] = []  # Last N tool names
        self.grep_count: int = 0
        self.read_file_count: int = 0
        self.run_tests_count: int = 0
        self.string_replace_count: int = 0
        # File state cache — tracks files read for read-before-edit enforcement
        # {relative_path: content_snippet} — snippet is first 200 chars for identity
        self.files_read: dict[str, str] = {}
        # Cross-phase file cache — preserves key file contents across observation masking
        # {relative_path: full_content} — top N files injected into EDIT phase
        self.file_cache: dict[str, str] = {}
        self.FILE_CACHE_MAX = 5

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def time_remaining(self) -> float:
        return max(0, MAX_WALL_TIME - self.elapsed)


def check_limits(gs: GuardrailState) -> str | None:
    """Check global limits. Returns error string if exceeded, None if OK."""
    if gs.tool_call_count >= gs.max_tool_calls:
        return (
            f"ERROR: Tool call limit reached ({gs.max_tool_calls}). "
            "You must call escalate with a reason, or submit_fix if ready."
        )
    if gs.elapsed >= MAX_WALL_TIME:
        return (
            f"ERROR: Time limit reached ({MAX_WALL_TIME // 60} minutes). "
            "You must call escalate or submit_fix now."
        )
    if gs.cost_usd >= MAX_COST_USD:
        return (
            f"ERROR: Cost cap reached (${MAX_COST_USD:.2f}). "
            "You must call escalate or submit_fix now."
        )
    return None


def check_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    gs: GuardrailState,
) -> str | None:
    """Check if a specific tool call is allowed.

    Returns an error string if blocked, None if the call should proceed.
    """
    # Global limits first
    limit_error = check_limits(gs)
    if limit_error and tool_name not in TERMINAL_TOOLS:
        return limit_error

    # Sandbox gate
    if tool_name in SANDBOX_REQUIRED_TOOLS and not gs.sandbox_created:
        return (
            f"ERROR: No sandbox exists — {tool_name} requires one.\n"
            "NEXT STEP: Call create_sandbox() right now, then retry.\n"
            "This is a required setup step, NOT a reason to escalate."
        )

    # Read-before-edit gate — you must read a file before editing it
    if tool_name == "string_replace":
        edit_path = tool_args.get("file_path", "")
        if edit_path and edit_path not in gs.files_read:
            return (
                f"WARNING: You haven't read '{edit_path}' yet. "
                "Call read_file or read_function first to see the current content, "
                "then retry string_replace with the exact old_string from the file."
            )

    # Submit gate — require sandbox + at least attempted tests.
    # Review is recommended but NOT required — the reviewer can be wrong.
    if tool_name == "submit_fix":
        hard_missing = []
        if not gs.sandbox_created:
            hard_missing.append("create_sandbox (no sandbox exists)")
        if not gs.tests_attempted:
            hard_missing.append("run_tests (must attempt tests at least once)")
        if hard_missing:
            return (
                "ERROR: Cannot submit yet. Missing prerequisites:\n"
                + "\n".join(f"  - {m}" for m in hard_missing)
            )
        # Soft warnings — inform but don't block
        warnings = []
        if not gs.tests_passed and not gs.tests_skipped:
            warnings.append("Tests failed — double-check your fix is correct.")
        if not gs.review_approved:
            warnings.append("Review not yet approved — submitting without review approval.")
        if warnings:
            return "WARNING: " + " ".join(warnings) + " Proceeding with submit."

    # ── Soft guidance (warnings only — never block exploration) ─────────
    # The agent needs freedom to explore. These are nudges, not walls.

    explore_tools = {"grep_repo", "read_file", "read_function", "list_files",
                     "get_file_structure", "get_function_info", "get_blast_radius"}

    # Nudge: prefer read_function over grep after several greps
    if tool_name == "grep_repo" and gs.grep_count >= 8:
        return (
            f"WARNING: grep_repo called {gs.grep_count} times. "
            "Consider using read_function(file, function_name) instead — "
            "it gives you the complete function with full context. "
            "grep finds WHERE things are; read_function shows you WHAT they do."
        )

    # Nudge: tests on unmodified code
    if tool_name in ("run_tests", "run_brt") and gs.sandbox_created and gs.string_replace_count == 0:
        return (
            "WARNING: You're running tests but haven't made any edits yet. "
            "Tests will show the existing behavior, not your fix. "
            "Call string_replace() first to apply your fix, THEN run tests."
        )

    # Nudge: test infra issues — don't keep retrying
    if tool_name == "run_tests" and gs.run_tests_count >= 3:
        return (
            f"WARNING: run_tests called {gs.run_tests_count} times. "
            "If tests can't run (missing deps/pytest), that's an environment issue, "
            "not your code. Proceed to request_review and submit_fix."
        )

    # Nudge: review loop — if reviewer keeps rejecting, submit anyway
    # A correct fix shouldn't be blocked by a disagreeing reviewer
    if tool_name == "request_review" and gs.review_count >= 2:
        return (
            f"WARNING: review requested {gs.review_count} times. "
            "If the reviewer keeps requesting changes but your fix is correct, "
            "call submit_fix directly — the reviewer may be wrong."
        )

    return None


def update_from_tool_result(
    tool_name: str,
    tool_args: dict[str, Any],
    result: str,
    gs: GuardrailState,
) -> None:
    """Update guardrail state based on tool execution results."""
    gs.tool_call_count += 1
    gs.tool_history.append(tool_name)

    # Track files read for read-before-edit enforcement + cross-phase cache
    if tool_name in ("read_file", "read_function") and not result.startswith("ERROR"):
        file_path = tool_args.get("file_path", "")
        if file_path:
            gs.files_read[file_path] = result[:200]
            # Cache full content for top N files (avoids re-reading after masking)
            if len(gs.file_cache) < gs.FILE_CACHE_MAX:
                gs.file_cache[file_path] = result[:8000]

    # Track per-tool counts for anti-pattern detection
    if tool_name == "grep_repo":
        gs.grep_count += 1
    elif tool_name == "read_file":
        gs.read_file_count += 1
    elif tool_name == "run_tests":
        gs.run_tests_count += 1
    elif tool_name == "string_replace":
        gs.string_replace_count += 1
        if not result.startswith("ERROR"):
            gs._last_edit_call = gs.tool_call_count  # Track for post-edit explore cap
    elif tool_name == "request_review":
        gs.review_count += 1

    if tool_name == "create_sandbox" and "OK:" in result:
        gs.sandbox_created = True
        gs._sandbox_call_number = gs.tool_call_count  # Track when sandbox was created
        # Extract sandbox path from result
        for line in result.split("\n"):
            if "sandbox_path=" in line:
                gs.sandbox_path = line.split("sandbox_path=")[-1].strip()

    elif tool_name == "run_tests":
        gs.tests_attempted = True
        if result.startswith("passed"):
            gs.tests_passed = True
            gs.test_failure_count = 0
        elif result.startswith("skipped") or result.startswith("error"):
            # "skipped" = no tests collected/ran (exit code 5, missing deps)
            # "error" = test execution failed (import error, bad path, exit code 4)
            # Both count as "attempted but couldn't run" — agent may proceed
            gs.tests_skipped = True
        elif result.startswith("failed"):
            # Actual assertion failures — agent must fix before submitting
            gs.tests_passed = False
            gs.test_failure_count += 1

    elif tool_name == "request_review":
        if "APPROVE" in result:
            gs.review_approved = True
            gs.review_verdict = "APPROVE"
        elif "CHANGES_REQUESTED" in result:
            gs.review_approved = False
            gs.review_verdict = "CHANGES_REQUESTED"
        elif "ESCALATE" in result:
            gs.review_approved = False
            gs.review_verdict = "ESCALATE"
