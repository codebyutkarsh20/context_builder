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

    p = Path(file_path)

    # Auto-strip absolute sandbox or repo prefix if agent mistakenly used one
    if p.is_absolute():
        sandbox_str = str(sandbox.resolve())
        if str(p).startswith(sandbox_str):
            file_path = str(p)[len(sandbox_str):].lstrip("/")
            p = Path(file_path)
        else:
            repo_path = getattr(_tls, "repo_path", None)
            if repo_path:
                repo_str = str(repo_path.resolve())
                if str(p).startswith(repo_str):
                    file_path = str(p)[len(repo_str):].lstrip("/")
                    p = Path(file_path)
                    logger.debug("Stripped repo prefix from absolute path: %s -> %s", p, file_path)
                else:
                    return None  # Absolute path outside both sandbox and repo
            else:
                return None

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
        file_path: RELATIVE path from repo root e.g. 'app/services/payment.py'.
                   Do NOT use absolute paths like '/tmp/agent_sandbox_.../file.py'.
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


def _diff_scoped_lint(sandbox: Path, modified_files: list[str]) -> str:
    """Run ruff on modified files but only report errors on lines the agent changed.

    Returns empty string if no new lint errors, otherwise the filtered error text.
    """
    import shutil as _shutil

    if not _shutil.which("ruff"):
        return ""  # No ruff available, skip

    # Get the set of changed line numbers per file from git diff
    changed_lines: dict[str, set[int]] = {}
    try:
        diff_output = subprocess.run(
            ["git", "diff", "-U0", "HEAD"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        ).stdout
        current_file = None
        for line in diff_output.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("@@ ") and current_file:
                # Parse hunk header: @@ -old,count +new,count @@
                import re as _re_inner
                m = _re_inner.search(r'\+(\d+)(?:,(\d+))?', line)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2)) if m.group(2) else 1
                    if current_file not in changed_lines:
                        changed_lines[current_file] = set()
                    changed_lines[current_file].update(range(start, start + count))
    except Exception:
        return ""  # Can't determine changed lines, skip lint

    if not changed_lines:
        return ""

    # Run ruff on modified files
    try:
        ruff_result = subprocess.run(
            ["ruff", "check", "--no-fix"] + modified_files,
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return ""

    if ruff_result.returncode == 0:
        return ""

    # Filter: only keep errors on lines the agent changed
    new_errors = []
    for err_line in ruff_result.stdout.splitlines():
        # ruff output format: file.py:line:col: EXXXX message
        parts = err_line.split(":", 3)
        if len(parts) >= 3:
            err_file = parts[0].strip()
            try:
                err_lineno = int(parts[1].strip())
            except ValueError:
                continue
            if err_file in changed_lines and err_lineno in changed_lines[err_file]:
                new_errors.append(err_line)

    if new_errors:
        return "Lint errors on YOUR changed lines:\n" + "\n".join(new_errors)
    return ""


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
    Runs linters first on changed files, then the specified tests.

    IMPORTANT: Always pass a specific test_path targeting tests relevant to your fix.
    Example: test_path='tests/test_helpers.py::TestJSON' or test_path='tests/test_app.py'
    Running with empty test_path triggers auto-detect which often fails on repos needing
    special setup (virtualenvs, fixtures, etc).

    Args:
        test_path: Specific test file, directory, or test ID to run.
                   Examples: 'tests/test_app.py', 'tests/test_helpers.py::TestJSON'
                   Empty = auto-detect (not recommended — prefer targeted tests).
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

        # Collect agent-touched files (new + modified)
        new_files = []
        modified_files = []
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
        for line in status_result.stdout.splitlines():
            if line.startswith("??"):
                f = line[3:].strip()
                if f.endswith(".py"):
                    new_files.append(f)
            elif line and line[0] in ("M", " ") and line[1] in ("M",):
                f = line[3:].strip()
                if f.endswith(".py"):
                    modified_files.append(f)

        # Also check staged modifications
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
        for f in diff_result.stdout.splitlines():
            f = f.strip()
            if f.endswith(".py") and f not in modified_files and f not in new_files:
                modified_files.append(f)

        # Lint new files fully (agent owns 100% of their content)
        if new_files:
            patches = [{"file_path": f} for f in new_files]
            lint_errors = _run_repo_linters(sandbox, patches)
            if lint_errors:
                return f"failed: linter errors in new files\n{lint_errors}"
            results.append(f"Linters (new files): PASSED ({len(new_files)} files)")

        # For modified files, only report errors on CHANGED lines (diff-scoped linting)
        if modified_files:
            diff_lint_errors = _diff_scoped_lint(sandbox, modified_files)
            if diff_lint_errors:
                return f"failed: linter errors in your changes\n{diff_lint_errors}"
            results.append(f"Linters (modified files): PASSED ({len(modified_files)} files)")

        if not new_files and not modified_files:
            results.append("Linters: skipped (no changed files)")
    except Exception as e:
        results.append(f"Linters: skipped ({e})")

    # Step 2: Run tests
    # Both paths go through sandbox.run_tests or raw pytest with consistent
    # classification into exactly one of: passed/skipped/error/failed.
    try:
        from agent.sandbox import run_tests as _run_tests

        if test_path:
            # Try sandbox runner first — it handles .agent_config.json
            # (setup commands, custom test command, env vars, timeouts).
            # Pass test_path as the worktree_path's test target.
            test_output = _run_tests(sandbox, repo_path=repo_path)

            # If sandbox runner found tests, use its result
            if test_output.startswith("passed") or test_output.startswith("failed"):
                classified = _classify_sandbox_output(test_output)
                results.append(classified)
                return classified + "\n" + "\n".join(results)

            # Sandbox runner skipped/errored — fall back to targeted pytest
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-x", "-v", "--tb=short"],
                cwd=sandbox, capture_output=True, text=True, timeout=120,
            )
            output = (test_result.stdout + test_result.stderr)[-3000:]
            classified = _classify_test_output(test_result.returncode, output, test_path)
            results.append(classified)
            return classified + "\n" + "\n".join(results)
        else:
            # Auto-detect via sandbox runner (honors .agent_config.json fully)
            test_output = _run_tests(sandbox, repo_path=repo_path)
            classified = _classify_sandbox_output(test_output)
            results.append(classified)
            return classified + "\n" + "\n".join(results)
    except Exception as e:
        return f"error: test execution failed ({e})"


def _classify_test_output(returncode: int, output: str, test_path: str) -> str:
    """Classify pytest result into exactly one of: passed, skipped, error, failed."""
    if returncode == 0:
        return f"passed: tests ({test_path}) passed"
    elif returncode == 5:
        return f"skipped: no tests collected for {test_path} (exit code 5)"
    elif returncode == 4:
        return f"error: pytest usage error for {test_path} (exit code 4 — missing deps or bad path)"
    else:
        return f"failed: test failures in {test_path}\n{output[-2000:]}"


def _classify_sandbox_output(test_output: str) -> str:
    """Classify sandbox runner output into: passed, skipped, error, or failed.

    sandbox.run_tests returns strings starting with 'passed', 'failed',
    'skipped', or 'error'. Preserve the prefix so guardrails recognize it.
    """
    if test_output.startswith("passed"):
        return test_output[:500]
    elif test_output.startswith("skipped"):
        return test_output[:500]
    elif test_output.startswith("error"):
        # Sandbox runner can return "error: ..." — preserve as error, NOT failed
        return test_output[:500]
    else:
        # Unknown prefix — treat as failed (conservative)
        return f"failed: {test_output[:2000]}"


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

    # Verify there are actual changes to commit
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sandbox, capture_output=True, text=True, check=True, timeout=30,
        )
        has_changes = bool(status.stdout.strip())
    except Exception:
        has_changes = False

    # Check if there are agent-created commits beyond the base branch.
    # Compare against the base branch (stored in thread-local), NOT HEAD~1.
    # HEAD~1 on a fresh worktree shows the base branch's own last commit,
    # which is a false positive.
    base_branch = getattr(_tls, "base_branch", "")
    has_agent_commits = False
    if base_branch:
        try:
            log_check = subprocess.run(
                ["git", "log", f"{base_branch}..HEAD", "--oneline"],
                cwd=sandbox, capture_output=True, text=True, timeout=10,
            )
            has_agent_commits = bool(log_check.stdout.strip())
        except Exception:
            pass

    if not has_changes and not has_agent_commits:
        return "ERROR: No changes to submit. Did you forget to call string_replace?"

    # Commit staged + unstaged changes
    committed = False
    if has_changes:
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=sandbox, capture_output=True, text=True, check=True, timeout=30,
            )
            result = subprocess.run(
                ["git", "commit", "-m", f"fix: {explanation[:200]}"],
                cwd=sandbox, capture_output=True, text=True, timeout=30,
            )
            committed = result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.warning("Commit failed during submit: %s", e.stderr or e)

    if not committed and not has_agent_commits:
        return "ERROR: Commit failed and no prior agent commits exist. Cannot submit."

    branch = getattr(_tls, "branch_name", "")
    commit_status = "committed" if committed else "previous agent commit exists"
    return (
        f"OK: Fix submitted.\n"
        f"commit_status={commit_status}\n"
        f"explanation={explanation}\n"
        f"branch={branch}\n"
        f"sandbox_path={sandbox}"
    )


@tool
def record_localization(
    fault_files: list[str],
    fault_functions: list[str],
    root_cause_hypothesis: str,
) -> str:
    """
    Record your localization findings — which files and functions contain the bug
    and your hypothesis for why the bug occurs. Call this ONCE after you finish
    exploring and before you start editing.

    This is used for tracking and eval scoring. It does not affect the fix.

    Args:
        fault_files: List of file paths that contain the bug, e.g. ['flask/wrappers.py']
        fault_functions: List of function names that are buggy, e.g. ['Response.make_sequence']
        root_cause_hypothesis: 1-2 sentence explanation of WHY the bug happens
    """
    if not fault_files:
        return "ERROR: fault_files must not be empty — provide at least one file path"
    if not root_cause_hypothesis:
        return "ERROR: root_cause_hypothesis must not be empty"

    # Store in thread-local for later extraction by the pipeline
    _tls.localization = {
        "fault_files": fault_files,
        "fault_functions": fault_functions or [],
        "root_cause_hypothesis": root_cause_hypothesis,
        "confidence": 0.9,
    }
    return (
        f"OK: Localization recorded.\n"
        f"  fault_files: {fault_files}\n"
        f"  fault_functions: {fault_functions}\n"
        f"  hypothesis: {root_cause_hypothesis[:200]}"
    )


def get_localization() -> dict:
    """Retrieve the recorded localization (called by react_loop/pipeline)."""
    return getattr(_tls, "localization", {})


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
# Multi-file coordination tools
# ---------------------------------------------------------------------------

@tool
def get_callers(file_path: str, function_name: str = "") -> str:
    """
    Find files that call or import the specified file/function.
    Use this AFTER editing a file to check if callers need updating too.
    Especially important when you change a function signature, rename something,
    or modify return values.

    Args:
        file_path: Relative path to the modified file e.g. 'flask/wrappers.py'
        function_name: Optional specific function name to search for callers of
    """
    repo_name = getattr(_tls, "repo_name", "")
    repo_path = getattr(_tls, "repo_path", None)

    callers: list[str] = []

    # Strategy 1: Query knowledge graph
    try:
        from agent.pipeline import _load_graph_data, _find_callers_from_graph
        graph_data, _ = _load_graph_data(repo_name)
        if graph_data:
            fault_files = [file_path]
            fault_fns = [function_name] if function_name else []
            callers = _find_callers_from_graph(graph_data, fault_files, fault_fns)
    except Exception:
        pass

    # Strategy 2: Fallback to grep
    if not callers and repo_path:
        try:
            from agent.pipeline import _find_callers_via_grep
            callers = _find_callers_via_grep(repo_path, [file_path])
        except Exception:
            pass

    # Strategy 3: Simple grep if pipeline functions unavailable
    if not callers and repo_path:
        search_dir = getattr(_tls, "sandbox_path", None) or repo_path
        stem = Path(file_path).stem
        try:
            result = subprocess.run(
                ["grep", "-rl", "--include=*.py", f"import {stem}", str(search_dir)],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if line and "test_" not in line and "conftest" not in line:
                    try:
                        rel = str(Path(line).relative_to(search_dir))
                        if rel not in callers and rel != file_path:
                            callers.append(rel)
                    except ValueError:
                        pass
        except Exception:
            pass

    if not callers:
        return f"No callers found for {file_path}" + (f"::{function_name}" if function_name else "") + ". This file may be a leaf node or the graph may not be indexed."

    risk = "CRITICAL" if len(callers) > 5 else "HIGH" if len(callers) > 2 else "MEDIUM" if len(callers) > 0 else "LOW"
    result_lines = [f"BLAST RADIUS: {risk} ({len(callers)} caller(s) found for {file_path})"]
    for c in callers[:10]:
        result_lines.append(f"  - {c}")
    if len(callers) > 10:
        result_lines.append(f"  ... and {len(callers) - 10} more")
    result_lines.append("")
    result_lines.append("ACTION: Read these files to check if your changes break them.")
    result_lines.append("If you changed a function signature, return type, or removed something, update callers too.")
    return "\n".join(result_lines)


@tool
def get_blast_radius(file_path: str) -> str:
    """
    Quick blast radius check for a file you modified. Returns the risk level
    and list of dependent files. Call this after editing to decide if you need
    to update other files.

    Args:
        file_path: Relative path to the modified file
    """
    # Delegate to get_callers with no function filter
    return get_callers.invoke({"file_path": file_path, "function_name": ""})


# ---------------------------------------------------------------------------
# Tool collections
# ---------------------------------------------------------------------------

EDIT_TOOLS = [string_replace, check_syntax, create_file]
SANDBOX_TOOLS = [create_sandbox, run_tests]
MULTI_FILE_TOOLS = [get_callers, get_blast_radius]
COMPLETION_TOOLS = [record_localization, request_review, submit_fix, escalate]

# All react-specific tools (exploration tools are added from explore_tools.py)
REACT_TOOLS = EDIT_TOOLS + SANDBOX_TOOLS + MULTI_FILE_TOOLS + COMPLETION_TOOLS
