"""
types.py — Pydantic models and state definitions for the AI Deploy Agent pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Structured output models (used with LLM tool_use)
# ---------------------------------------------------------------------------

class IntentAnalysis(BaseModel):
    """Output of intent translation — what the bug means technically."""
    expected_behavior: str = Field(description="What the correct behavior should be")
    actual_behavior: str = Field(description="What is currently happening (the bug)")
    likely_affected_modules: list[str] = Field(default_factory=list, description="Module/file names likely involved")
    likely_affected_functions: list[str] = Field(default_factory=list, description="Function names likely involved")
    fix_type: str = Field(default="bug_fix", description="bug_fix, enhancement, or refactor")
    severity: str = Field(default="medium", description="critical, high, medium, or low")
    acceptance_criteria: list[str] = Field(default_factory=list, description="Testable assertions that prove the fix works. E.g. 'set_pr_url with nonexistent flag should log a warning'. These come from the spec, not the implementation.")


class LocalizationResult(BaseModel):
    """Output of fault localization — where the bug is."""
    fault_files: list[str] = Field(default_factory=list, description="File paths containing the bug")
    fault_functions: list[str] = Field(default_factory=list, description="Function names containing the bug")
    fault_classes: list[str] = Field(default_factory=list, description="Class names involved")
    root_cause_hypothesis: str = Field(default="", description="One paragraph explaining the likely root cause")
    confidence: float = Field(default=0.0, description="Confidence score from 0.0 to 1.0")
    evidence: list[str] = Field(default_factory=list, description="Evidence supporting the hypothesis")


class Patch(BaseModel):
    """A single file patch with actual code changes."""
    file_path: str = Field(description="Path to the file being modified")
    original_code: str = Field(default="", description="The original code snippet that needs to change")
    patched_code: str = Field(default="", description="The replacement code after the fix")
    explanation: str = Field(default="", description="What was changed and why")


class RepairResult(BaseModel):
    """Output of repair agent — the generated fix with real code."""
    patches: list[Patch] = Field(default_factory=list, description="Code patches to apply to existing files")
    test_patches: list[Patch] = Field(default_factory=list, description="New or updated test files to create. For each test file: file_path is the test file path (e.g. 'tests/test_foo.py'), original_code is empty string (new file) or existing content to replace, patched_code is the full test file content.")
    explanation: str = Field(default="", description="Overall fix summary in 2-3 sentences")
    needs_more_files: list[str] = Field(default_factory=list, description="File paths the agent needs to read to complete the fix. If you cannot produce patches because you need to see more source code, list the file paths here.")


class ReviewCheck(BaseModel):
    """Single review check result."""
    name: str = Field(description="Check name: ROOT_CAUSE, BUSINESS_RULES, PATTERNS, COMPLETENESS, or TESTS")
    status: str = Field(description="PASS, FAIL, or WARNING")
    comment: str = Field(default="", description="Explanation for this check result")


class ReviewResult(BaseModel):
    """Output of review agent."""
    verdict: str = Field(default="PENDING", description="APPROVE, CHANGES_REQUESTED, or ESCALATE")
    confidence: float = Field(default=0.0, description="Confidence score from 0.0 to 1.0")
    checks: list[ReviewCheck] = Field(default_factory=list, description="Individual review checks")
    feedback: str = Field(default="", description="Specific feedback if CHANGES_REQUESTED")


# ---------------------------------------------------------------------------
# Other models
# ---------------------------------------------------------------------------

class WorkOrder(BaseModel):
    """Parsed bug ticket — input to the agent pipeline."""
    ticket_id: str
    title: str
    description: str
    repo_name: str
    repo_path: str = ""
    priority: str = "medium"
    affected_component: Optional[str] = None
    reproduction_steps: Optional[str] = None
    comments: list[str] = Field(default_factory=list)


class PipelineStatus(str, Enum):
    PENDING = "pending"
    INTAKE = "intake"
    CONTEXT = "context_assembly"
    EXPLORING = "exploring"        # Agentic exploration phase
    LOCALIZING = "localizing"
    READING_SOURCE = "reading_source"
    REPAIRING = "repairing"
    REVIEWING = "reviewing"
    TESTING = "testing"
    PR_CREATING = "pr_creating"
    DONE = "done"
    ESCALATED = "escalated"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""
    work_order: dict          # WorkOrder fields
    intent: dict              # IntentAnalysis.model_dump()
    context: str              # Assembled context (from Graph RAG or exploration)
    context_nodes: int        # Number of nodes in context
    source_code: dict         # {file_path: code_string} — actual source of localized files
    localization: dict        # LocalizationResult.model_dump()
    repair: dict              # RepairResult.model_dump()
    review: dict              # ReviewResult.model_dump()
    iteration_count: int      # Developer↔Reviewer loop count
    status: str               # PipelineStatus value
    error: str                # Error message if failed
    pr_url: str               # GitHub PR URL if created
    test_result: str          # Test execution output
    sandbox_path: str         # Git worktree path
    branch_name: str          # Fix branch name
    base_branch: str          # Original branch for PR base
    patches_applied: int      # Number of patches applied
    exploration_log: list     # Tool calls + results from exploration phase
    caller_files: list        # Caller file paths discovered for blast radius (Phase 2B)
    dry_run: bool             # Skip PR creation + feature flags, return patch + PR body only
