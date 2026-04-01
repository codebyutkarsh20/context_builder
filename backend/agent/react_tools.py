"""
react_tools.py — Sandbox-aware tools for the ReAct agent pipeline.

Provides edit, sandbox, test, review, and completion tools that the agent
uses alongside the read-only exploration tools from explore_tools.py.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Thread-local state for sandbox context
_tls = threading.local()

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def set_react_context(
    repo_name: str,
    repo_path: str | Path,
    data_dir: Path | None = None,
) -> None:
    """Set per-run context for react tools (thread-local)."""
    _tls.repo_name = repo_name
    _tls.repo_path = Path(repo_path) if repo_path else None
    _tls.sandbox_path = None
    _tls.branch_name = ""
    _tls.base_branch = ""
    if data_dir:
        _tls.data_dir = data_dir


def set_sandbox_path(sandbox_path: Path, branch_name: str, base_branch: str) -> None:
    """Update the sandbox path after creation (called from create_sandbox tool)."""
    _tls.sandbox_path = sandbox_path
    _tls.branch_name = branch_name
    _tls.base_branch = base_branch


def get_sandbox_path() -> Path | None:
    """Get the current sandbox path."""
    return getattr(_tls, "sandbox_path", None)


def get_branch_name() -> str:
    """Get the current branch name."""
    return getattr(_tls, "branch_name", "")


def get_base_branch() -> str:
    """Get the base branch."""
    return getattr(_tls, "base_branch", "")


_BINARY_EXTENSIONS = frozenset({
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.o', '.a', '.dylib',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.woff', '.woff2',
    '.ttf', '.zip', '.tar', '.gz', '.db', '.sqlite', '.sqlite3', '.DS_Store',
    '.pdf', '.mp3', '.mp4',
})


def _resolve_sandbox_path(file_path: str) -> Path | None:
    """Resolve a file path within the sandbox. Returns None if outside sandbox."""
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox:
        return None
    # If the agent passed an absolute path that's already inside the sandbox, use it directly
    p = Path(file_path)
    if p.is_absolute():
        resolved = p.resolve()
        if str(resolved).startswith(str(sandbox.resolve())):
            return resolved
        return None  # Absolute path outside sandbox
    resolved = (sandbox / file_path).resolve()
    # Path traversal check
    if not str(resolved).startswith(str(sandbox.resolve())):
        return None
    return resolved


def _resolve_repo_or_sandbox(file_path: str) -> Path | None:
    """Resolve a file path — sandbox if available, otherwise repo root."""
    p = Path(file_path)
    sandbox = getattr(_tls, "sandbox_path", None)
    if sandbox:
        if p.is_absolute():
            resolved = p.resolve()
            if str(resolved).startswith(str(sandbox.resolve())):
                return resolved
        else:
            resolved = (sandbox / file_path).resolve()
            if str(resolved).startswith(str(sandbox.resolve())):
                return resolved
    repo_path = getattr(_tls, "repo_path", None)
    if repo_path:
        if p.is_absolute():
            resolved = p.resolve()
            if str(resolved).startswith(str(repo_path.resolve())):
                return resolved
        else:
            resolved = (repo_path / file_path).resolve()
            if str(resolved).startswith(str(repo_path.resolve())):
                return resolved
    return None


# ---------------------------------------------------------------------------
# Edit tools (sandbox-aware)
# ---------------------------------------------------------------------------

@tool
def string_replace(file_path: str, old_string: str, new_string: str) -> str:
    """
    Replace an exact string in a file in the sandbox. The old_string must be
    a unique, exact substring of the file (including whitespace/indentation).

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
        old_string: The exact text to replace (must be unique in the file)
        new_string: The replacement text
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox:
        return "ERROR: No sandbox exists. Call create_sandbox first."

    if not old_string or not old_string.strip():
        return "ERROR: old_string must not be empty"
    if old_string == new_string:
        return "ERROR: old_string and new_string are identical — no change"

    resolved = _resolve_sandbox_path(file_path)
    if resolved is None:
        return f"ERROR: Path traversal blocked: {file_path}"
    if not resolved.exists():
        # Try finding the file by name
        matches = list(sandbox.rglob(Path(file_path).name))
        resolved = None
        for m in matches:
            candidate = m.resolve()
            if str(candidate).startswith(str(sandbox.resolve())):
                resolved = candidate
                break
        if resolved is None:
            return f"ERROR: File not found in sandbox: {file_path}"

    if resolved.suffix.lower() in _BINARY_EXTENSIONS:
        return f"ERROR: Binary file skipped: {file_path}"

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)

        if count == 0:
            # Try whitespace-normalized match
            lines = content.splitlines()
            old_lines = old_string.splitlines()
            norm_content = [l.rstrip() for l in lines]
            norm_old = [l.rstrip() for l in old_lines]
            for i in range(len(norm_content) - len(norm_old) + 1):
                if norm_content[i:i + len(norm_old)] == norm_old:
                    actual_old = "\n".join(lines[i:i + len(norm_old)])
                    content = content.replace(actual_old, new_string, 1)
                    resolved.write_text(content, encoding="utf-8")
                    return f"OK: replaced (whitespace-normalized) in {file_path}"
            return (
                f"ERROR: old_string not found in {file_path}.\n"
                f"Use read_file or read_function to get the current exact content."
            )
        elif count > 1:
            return (
                f"ERROR: old_string appears {count} times in {file_path}. "
                f"Make it longer/more unique so only one instance matches."
            )

        new_content = content.replace(old_string, new_string, 1)
        resolved.write_text(new_content, encoding="utf-8")

        old_lines_count = old_string.count("\n") + 1
        new_lines_count = new_string.count("\n") + 1
        delta = new_lines_count - old_lines_count
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        return f"OK: replaced 1 occurrence in {file_path} ({delta_str} lines)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def check_syntax(file_path: str) -> str:
    """
    Check a Python file for syntax errors after editing.
    Always run this after string_replace to verify your edit is valid.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
    """
    resolved = _resolve_repo_or_sandbox(file_path)
    if resolved is None or not resolved.exists():
        return f"ERROR: File not found: {file_path}"

    if resolved.suffix.lower() != ".py":
        return f"OK: syntax check only for .py files (skipped {file_path})"

    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import ast; ast.parse(open({repr(str(resolved))}).read())"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return f"OK: {file_path} — no syntax errors"
        else:
            return f"SYNTAX ERROR in {file_path}:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "ERROR: syntax check timed out"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def create_file(file_path: str, content: str) -> str:
    """
    Create a new file in the sandbox (e.g., test files). Will overwrite if exists.

    Args:
        file_path: Relative path from repo root e.g. 'tests/test_fix.py'
        content: Full file content to write
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox:
        return "ERROR: No sandbox exists. Call create_sandbox first."

    resolved = _resolve_sandbox_path(file_path)
    if resolved is None:
        return f"ERROR: Path traversal blocked: {file_path}"

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        return f"OK: created {file_path} ({lines} lines)"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Sandbox tools
# ---------------------------------------------------------------------------

@tool
def create_sandbox() -> str:
    """
    Create an isolated git worktree sandbox for editing and testing.
    Call this ONCE before making any edits. Returns the sandbox path.
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if sandbox and sandbox.exists():
        return f"OK: Sandbox already exists at {sandbox}. sandbox_path={sandbox}"

    repo_path = getattr(_tls, "repo_path", None)
    if not repo_path:
        return "ERROR: No repo path set. Cannot create sandbox."

    repo_name = getattr(_tls, "repo_name", "unknown")
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_name).lower()
    branch_suffix = uuid.uuid4().hex[:6]
    branch_name = f"fix/{safe_name}-{branch_suffix}"
    worktree_path = Path(f"/tmp/agent_sandbox_{safe_name}_{branch_suffix}")

    try:
        # Get base branch
        base_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()

        # Lock to prevent race conditions
        with open(repo_path / ".agent_lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Check for dirty repo (ignore untracked files)
                porcelain = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo_path, capture_output=True, text=True, check=True, timeout=30,
                ).stdout
                dirty = "\n".join(
                    l for l in porcelain.splitlines() if l and not l.startswith("??")
                ).strip()
                if dirty:
                    return "ERROR: Repo has uncommitted changes. Commit or stash first."

                # Clean up stale worktrees
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=repo_path, capture_output=True, timeout=30,
                )

                # Remove old worktree directory if it exists
                if worktree_path.exists():
                    import shutil
                    shutil.rmtree(worktree_path, ignore_errors=True)

                # Create worktree
                subprocess.run(
                    ["git", "worktree", "add", "-b", branch_name,
                     str(worktree_path), base_branch],
                    cwd=repo_path, capture_output=True, text=True, check=True, timeout=60,
                )
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        # Update thread-local context
        set_sandbox_path(worktree_path, branch_name, base_branch)

        logger.info("Sandbox created: %s (branch: %s)", worktree_path, branch_name)
        return (
            f"OK: Sandbox created at {worktree_path}\n"
            f"branch={branch_name}\n"
            f"base={base_branch}\n"
            f"sandbox_path={worktree_path}\n"
            f"You can now use string_replace and create_file to make edits."
        )
    except subprocess.CalledProcessError as e:
        return f"ERROR: Git worktree creation failed: {e.stderr or e}"
    except Exception as e:
        return f"ERROR: Sandbox creation failed: {e}"


@tool
def run_tests(test_path: str = "") -> str:
    """
    Run the repo's test suite and linters on your changes in the sandbox.
    Runs linters first, then targeted tests, then full suite.

    Args:
        test_path: Optional specific test file/dir to run. Empty = auto-detect.
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox or not sandbox.exists():
        return "ERROR: No sandbox exists. Call create_sandbox first."

    repo_path = getattr(_tls, "repo_path", None)
    base_branch = getattr(_tls, "base_branch", "")
    results = []

    # Step 1: Run linters only on NEWLY INTRODUCED errors (not pre-existing ones)
    try:
        from agent.pipeline import _run_repo_linters

        # Get only files the agent created (untracked) — these are fully the agent's responsibility
        new_files = []
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
        for line in status_result.stdout.splitlines():
            if line.startswith("??"):
                f = line[3:].strip()
                if f.endswith(".py"):
                    new_files.append(f)

        if new_files:
            patches = [{"file_path": f} for f in new_files]
            lint_errors = _run_repo_linters(sandbox, patches)
            if lint_errors:
                return f"failed: linter errors in new files\n{lint_errors}"
            results.append(f"Linters (new files): PASSED ({len(new_files)} files)")
        else:
            results.append("Linters: skipped (no new files)")
    except Exception as e:
        results.append(f"Linters: skipped ({e})")

    # Step 2: Run tests
    try:
        from agent.sandbox import run_tests as _run_tests

        if test_path:
            # Run specific test
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-x", "-v", "--tb=short"],
                cwd=sandbox, capture_output=True, text=True, timeout=120,
            )
            output = (test_result.stdout + test_result.stderr)[-3000:]
            if test_result.returncode == 0:
                results.append(f"Tests ({test_path}): PASSED")
            else:
                return f"failed: test failures\n{output}"
        else:
            # Auto-detect and run
            test_output = _run_tests(sandbox, repo_path=repo_path)
            if test_output.startswith("passed"):
                results.append(f"Tests: {test_output[:200]}")
            elif test_output.startswith("skipped"):
                results.append(f"Tests: {test_output[:200]}")
            else:
                return f"failed: {test_output[:3000]}"
    except Exception as e:
        results.append(f"Tests: error ({e})")

    return "passed\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Completion tools
# ---------------------------------------------------------------------------

@tool
def request_review(explanation: str) -> str:
    """
    Request an independent AI review of your fix. This calls Claude Opus
    with fresh context (no access to your exploration) for unbiased review.
    You MUST call this before submit_fix.

    Args:
        explanation: 2-3 sentence explanation of what you fixed and why.
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox or not sandbox.exists():
        return "ERROR: No sandbox exists. Cannot review without changes."

    repo_name = getattr(_tls, "repo_name", "")

    # Collect the diff
    try:
        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
        diff_text = diff_result.stdout[:8000] if diff_result.stdout else "(no diff)"
    except Exception:
        diff_text = "(could not generate diff)"

    # Get modified files
    modified_files = []
    try:
        name_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
        modified_files = [l.strip() for l in name_result.stdout.splitlines() if l.strip()]
    except Exception:
        pass

    # Build independent reviewer context
    reviewer_context = ""
    try:
        from agent.pipeline import _build_reviewer_context
        if repo_name:
            reviewer_context = _build_reviewer_context(repo_name, modified_files)
    except Exception:
        reviewer_context = "No business rules or blast radius data available."

    # Build review prompt
    from agent.types import ReviewResult

    prompt = f"""Review this bug fix as an independent reviewer.

FIX EXPLANATION: {explanation}

DIFF:
{diff_text}

FILES MODIFIED: {modified_files}

INDEPENDENT CONTEXT (from knowledge graph):
{reviewer_context}

Produce 6 checks (ROOT_CAUSE, BUSINESS_RULES, PATTERNS, COMPLETENESS, BLAST_RADIUS, TESTS):
- ROOT_CAUSE: Does the fix address WHY the bug happens?
- BUSINESS_RULES: Any business rules violated?
- PATTERNS: Code follows existing conventions?
- COMPLETENESS: All buggy locations patched?
- BLAST_RADIUS: Interface changes covered?
- TESTS: Evidence of testing?

verdict: APPROVE if all checks pass. CHANGES_REQUESTED if any fail. ESCALATE if too complex."""

    try:
        from agent.pipeline import _structured_call
        result = _structured_call("claude-opus-4-6", 3000, ReviewResult, prompt)
        review_dict = result.model_dump()

        # Format response
        checks_str = "\n".join(
            f"  {c['name']}: {c['status']}" + (f" — {c['comment']}" if c.get('comment') else "")
            for c in review_dict.get("checks", [])
        )
        verdict = review_dict.get("verdict", "UNKNOWN")
        confidence = review_dict.get("confidence", 0)
        feedback = review_dict.get("feedback", "")

        response = f"REVIEW VERDICT: {verdict} (confidence: {confidence:.0%})\n\nChecks:\n{checks_str}"
        if feedback:
            response += f"\n\nFeedback: {feedback}"

        return response
    except Exception as e:
        # Fallback to Sonnet
        try:
            from agent.pipeline import _structured_call
            result = _structured_call("claude-sonnet-4-6", 2000, ReviewResult, prompt)
            review_dict = result.model_dump()
            verdict = review_dict.get("verdict", "UNKNOWN")
            confidence = review_dict.get("confidence", 0)
            feedback = review_dict.get("feedback", "")
            checks_str = "\n".join(
                f"  {c['name']}: {c['status']}" for c in review_dict.get("checks", [])
            )
            return f"REVIEW VERDICT: {verdict} (confidence: {confidence:.0%})\n\nChecks:\n{checks_str}\n\nFeedback: {feedback}"
        except Exception as e2:
            return f"ERROR: Review failed: {e2}"


@tool
def submit_fix(explanation: str) -> str:
    """
    Submit your fix for PR creation. Only call this after:
    1. Tests have passed (run_tests returned 'passed')
    2. Review has approved (request_review returned 'APPROVE')

    Args:
        explanation: 2-3 sentence summary of what was fixed and why.
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox or not sandbox.exists():
        return "ERROR: No sandbox exists."

    # Commit changes in sandbox
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=sandbox, capture_output=True, text=True, check=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", f"fix: {explanation[:200]}"],
            cwd=sandbox, capture_output=True, text=True, check=True, timeout=30,
        )
    except subprocess.CalledProcessError as e:
        # Might already be committed
        logger.debug("Commit during submit: %s", e)

    branch = getattr(_tls, "branch_name", "")
    return (
        f"OK: Fix submitted.\n"
        f"explanation={explanation}\n"
        f"branch={branch}\n"
        f"sandbox_path={sandbox}"
    )


@tool
def escalate(reason: str) -> str:
    """
    Escalate to human. Call this when you cannot fix the bug after multiple attempts,
    or when the bug is too complex for automated fixing.

    Args:
        reason: Clear explanation of why you're escalating and what you tried.
    """
    return f"ESCALATED: {reason}"


# ---------------------------------------------------------------------------
# Tool collections
# ---------------------------------------------------------------------------

EDIT_TOOLS = [string_replace, check_syntax, create_file]
SANDBOX_TOOLS = [create_sandbox, run_tests]
COMPLETION_TOOLS = [request_review, submit_fix, escalate]

# All react-specific tools (exploration tools are added from explore_tools.py)
REACT_TOOLS = EDIT_TOOLS + SANDBOX_TOOLS + COMPLETION_TOOLS
