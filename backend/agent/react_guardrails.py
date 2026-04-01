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
MAX_TOOL_CALLS = 60
MAX_WALL_TIME = 900  # 15 minutes
MAX_COST_USD = 5.00
MAX_TEST_FAILURES = 3

# Tools that require a sandbox to exist
SANDBOX_REQUIRED_TOOLS = frozenset({
    "string_replace", "create_file", "check_syntax",
    "run_tests", "run_linters",
})

# Tools that are terminal (end the loop)
TERMINAL_TOOLS = frozenset({"submit_fix", "escalate"})


class GuardrailState:
    """Tracks guardrail-relevant state during a ReAct loop run."""

    def __init__(self):
        self.sandbox_created: bool = False
        self.sandbox_path: str = ""
        self.tests_passed: bool = False
        self.test_failure_count: int = 0
        self.review_approved: bool = False
        self.review_verdict: str = ""
        self.tool_call_count: int = 0
        self.cost_usd: float = 0.0
        self.start_time: float = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def time_remaining(self) -> float:
        return max(0, MAX_WALL_TIME - self.elapsed)


def check_limits(gs: GuardrailState) -> str | None:
    """Check global limits. Returns error string if exceeded, None if OK."""
    if gs.tool_call_count >= MAX_TOOL_CALLS:
        return (
            f"ERROR: Tool call limit reached ({MAX_TOOL_CALLS}). "
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
            "ERROR: You must call create_sandbox before using "
            f"{tool_name}. Create a sandbox first, then edit/test."
        )

    # Submit gate
    if tool_name == "submit_fix":
        missing = []
        if not gs.sandbox_created:
            missing.append("create_sandbox (no sandbox exists)")
        if not gs.tests_passed:
            missing.append("run_tests (tests must pass)")
        if not gs.review_approved:
            missing.append("request_review (review must approve)")
        if missing:
            return (
                "ERROR: Cannot submit yet. Missing prerequisites:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )

    # Test failure cap (advisory, not blocking)
    if tool_name == "run_tests" and gs.test_failure_count >= MAX_TEST_FAILURES:
        return (
            f"WARNING: Tests have failed {gs.test_failure_count} times. "
            "Consider calling escalate instead of retrying. "
            "If you believe this attempt will succeed, proceed."
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

    if tool_name == "create_sandbox" and "OK:" in result:
        gs.sandbox_created = True
        # Extract sandbox path from result
        for line in result.split("\n"):
            if "sandbox_path=" in line:
                gs.sandbox_path = line.split("sandbox_path=")[-1].strip()

    elif tool_name == "run_tests":
        if result.startswith("passed"):
            gs.tests_passed = True
            gs.test_failure_count = 0  # Reset on success
        elif result.startswith("failed"):
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
