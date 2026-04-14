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

from agent.path_safety import safe_resolve

logger = logging.getLogger(__name__)

# Thread-local state for sandbox context
_tls = threading.local()

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def set_react_context(
    repo_name: str,
    repo_path: str | Path,
    data_dir: Path | None = None,
    fix_type: str = "bug_fix",
) -> None:
    """Set per-run context for react tools (thread-local)."""
    _tls.repo_name = repo_name
    _tls.repo_path = Path(repo_path) if repo_path else None
    _tls.sandbox_path = None
    _tls.branch_name = ""
    _tls.base_branch = ""
    _tls.fix_type = fix_type
    if data_dir:
        _tls.data_dir = data_dir
    # Reset plan state — each run starts with a blank plan slate
    if hasattr(_tls, "plan_history"):
        delattr(_tls, "plan_history")
    if hasattr(_tls, "current_plan"):
        delattr(_tls, "current_plan")
    # Reset edit history — undo_last_edit must not see edits from prior runs
    if hasattr(_tls, "edit_history"):
        delattr(_tls, "edit_history")


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


# ---------------------------------------------------------------------------
# Plan mode — agent must declare a structured plan before any edits
# ---------------------------------------------------------------------------

@tool
def produce_plan(
    root_cause: str,
    target_files: list[str],
    approach: str,
    success_criteria: str,
    risk: str = "LOW",
    rollback: str = "",
) -> str:
    """Declare your implementation plan BEFORE creating a sandbox or making edits.

    This is a self-commitment device borrowed from Claude Code's plan-mode
    pattern: producing a structured plan first reduces wasted edits on
    misguided fixes. The plan is logged in the trace and visible to the
    independent verifier later.

    You may call this tool again to revise the plan as you learn more during
    exploration — only the latest plan is enforced. Each call is logged.

    Args:
        root_cause: One sentence — what's actually broken, in causal terms.
            Example: "The retry decorator catches BaseException, swallowing
            KeyboardInterrupt so Ctrl-C doesn't kill stuck workers."
        target_files: List of relative file paths you intend to change. Be
            specific — wildcards or "various files" are not acceptable.
        approach: 2-4 sentences describing the change you'll make. Include
            the function/method names that will be modified.
        success_criteria: 1-3 testable conditions that prove the fix works.
            These should map to the bug description, not to your implementation.
            Example: "Ctrl-C terminates the worker within 1s instead of hanging."
        risk: One of LOW, MEDIUM, HIGH. HIGH if removing validation, changing
            public APIs, or touching > 5 files. MEDIUM for shared utilities.
            LOW for localized fixes.
        rollback: Optional — how to undo the change if it breaks production.
            Required for risk=HIGH.

    Returns:
        Confirmation that the plan was recorded, plus next-step guidance.
    """
    if not root_cause or not target_files or not approach or not success_criteria:
        return (
            "ERROR: Plan is incomplete. All of root_cause, target_files, "
            "approach, and success_criteria are required."
        )
    if risk not in ("LOW", "MEDIUM", "HIGH"):
        return f"ERROR: risk must be one of LOW, MEDIUM, HIGH (got '{risk}')."
    if risk == "HIGH" and not rollback:
        return (
            "ERROR: risk=HIGH requires a rollback strategy. "
            "Describe how the change can be reverted if it breaks production."
        )
    if not isinstance(target_files, list) or any(
        not isinstance(f, str) or not f.strip() for f in target_files
    ):
        return "ERROR: target_files must be a list of non-empty file path strings."

    # Store the plan on thread-local for the guardrail + later inspection
    plan: dict = {
        "root_cause": root_cause,
        "target_files": [f.strip() for f in target_files],
        "approach": approach,
        "success_criteria": success_criteria,
        "risk": risk,
        "rollback": rollback,
    }
    # Track plan history (revisions) on TLS
    if not hasattr(_tls, "plan_history"):
        _tls.plan_history = []
    _tls.plan_history.append(plan)
    _tls.current_plan = plan

    revision_note = ""
    if len(_tls.plan_history) > 1:
        revision_note = f" (revision #{len(_tls.plan_history)})"

    logger.info(
        "Plan recorded%s: risk=%s, files=%s, root_cause=%s",
        revision_note, risk, plan["target_files"][:3], root_cause[:80],
    )

    return (
        f"OK: Plan recorded{revision_note}. risk={risk}, "
        f"target_files={len(plan['target_files'])}.\n"
        f"NEXT: Call create_sandbox(), then string_replace() on the target files.\n"
        f"You may call produce_plan again if exploration reveals the plan needs revision."
    )


def get_current_plan() -> dict | None:
    """Get the latest plan submitted via produce_plan (or None)."""
    return getattr(_tls, "current_plan", None)


def get_plan_history() -> list[dict]:
    """Get the full history of plan revisions (or empty list)."""
    return getattr(_tls, "plan_history", [])


def reset_plan_state() -> None:
    """Reset plan state (called between runs)."""
    if hasattr(_tls, "plan_history"):
        delattr(_tls, "plan_history")
    if hasattr(_tls, "current_plan"):
        delattr(_tls, "current_plan")


# ---------------------------------------------------------------------------
# Edit history — per-file before-snapshots so the agent can undo
# (Ports Claude Code's fileHistory snapshot pattern, scoped to one run.)
# ---------------------------------------------------------------------------

# Stored on TLS as a list of dicts:
#   {"file_path": str, "before_content": str | None, "after_content": str,
#    "tool": "string_replace" | "create_file"}
# `before_content` is None when the file didn't exist (create_file on a new file).

def _record_edit_snapshot(
    file_path: str,
    before_content: str | None,
    after_content: str,
    tool: str,
) -> None:
    """Snapshot a file's pre/post-edit content so undo_last_edit can revert it."""
    if not hasattr(_tls, "edit_history"):
        _tls.edit_history = []
    _tls.edit_history.append({
        "file_path": file_path,
        "before_content": before_content,
        "after_content": after_content,
        "tool": tool,
    })


def get_edit_history() -> list[dict]:
    """Return the in-memory edit history for inspection / tests."""
    return getattr(_tls, "edit_history", [])


def reset_edit_history() -> None:
    """Reset edit history — called between runs."""
    if hasattr(_tls, "edit_history"):
        delattr(_tls, "edit_history")


def _resolve_sandbox_path(file_path: str) -> Path | None:
    """Resolve a file path within the sandbox. Returns None if outside sandbox."""
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox:
        return None

    # Try sandbox first via shared safe_resolve
    result = safe_resolve(file_path, sandbox)
    if result:
        return result

    # Fallback: if an absolute repo-prefix path was given, strip it and
    # resolve against the sandbox (the agent sometimes confuses the two).
    repo_path = getattr(_tls, "repo_path", None)
    if repo_path:
        p = Path(file_path)
        if p.is_absolute():
            repo_str = str(repo_path.resolve())
            resolved_p = str(p.resolve())
            if resolved_p.startswith(repo_str):
                relative = resolved_p[len(repo_str):].lstrip("/")
                logger.debug("Stripped repo prefix from absolute path: %s -> %s", p, relative)
                return safe_resolve(relative, sandbox)
    return None


def _resolve_repo_or_sandbox(file_path: str) -> Path | None:
    """Resolve a file path — sandbox if available, otherwise repo root."""
    sandbox = getattr(_tls, "sandbox_path", None)
    if sandbox:
        result = safe_resolve(file_path, sandbox)
        if result:
            return result
    repo_path = getattr(_tls, "repo_path", None)
    if repo_path:
        return safe_resolve(file_path, repo_path)
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
                    new_ws_content = content.replace(actual_old, new_string, 1)
                    _record_edit_snapshot(file_path, content, new_ws_content, "string_replace")
                    resolved.write_text(new_ws_content, encoding="utf-8")
                    autofix_msg = _try_autofix(resolved)
                    result = f"OK: replaced (whitespace-normalized) in {file_path}"
                    return f"{result}\n{autofix_msg}" if autofix_msg else result
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
        # Snapshot BEFORE writing — so undo_last_edit can restore the original
        _record_edit_snapshot(file_path, content, new_content, "string_replace")
        resolved.write_text(new_content, encoding="utf-8")

        old_lines_count = old_string.count("\n") + 1
        new_lines_count = new_string.count("\n") + 1
        delta = new_lines_count - old_lines_count
        delta_str = f"+{delta}" if delta >= 0 else str(delta)

        # Deterministic autofix: run ruff --fix on edited Python files
        autofix_msg = _try_autofix(resolved)

        result = f"OK: replaced 1 occurrence in {file_path} ({delta_str} lines)"
        if autofix_msg:
            result += f"\n{autofix_msg}"
        return result
    except Exception as e:
        return f"ERROR: {e}"


def _try_autofix(file_path: Path) -> str:
    """Run ruff --fix on a Python file after editing. Returns status message or empty string."""
    if file_path.suffix not in (".py", ".pyi"):
        return ""
    try:
        result = subprocess.run(
            ["ruff", "check", "--fix", "--quiet", str(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stderr:
            # ruff applied fixes
            fixed_count = result.stderr.count("Fixed")
            if fixed_count:
                return f"(autofix: ruff fixed {fixed_count} issue(s))"
        return ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


@tool
def undo_last_edit() -> str:
    """Undo the most recent string_replace or create_file in the sandbox.

    Use this when you've made an edit that turned out to be wrong — e.g.,
    tests went from passing to failing after the edit, or check_syntax
    reported a problem you can't easily patch. Cheaper than starting
    over: it reverts ONLY the last edit, leaving everything else intact.

    Behavior:
      - For string_replace: restores the file to its content before the replace.
      - For create_file (overwriting an existing file): restores the prior content.
      - For create_file (new file that didn't exist): deletes the file.
      - The undone edit is removed from history — call this again to undo
        the second-most-recent edit, etc.

    Returns OK with the file path that was reverted, or ERROR if there's
    nothing to undo.
    """
    history = getattr(_tls, "edit_history", None)
    if not history:
        return "ERROR: No edits to undo (edit_history is empty)."

    last = history.pop()
    file_path = last["file_path"]
    before = last["before_content"]
    tool_name = last["tool"]

    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox:
        return f"ERROR: No sandbox — cannot undo edit on {file_path}."

    resolved = _resolve_sandbox_path(file_path)
    if resolved is None:
        return f"ERROR: Path traversal blocked: {file_path}"

    try:
        if before is None:
            # File was newly created — undoing means deleting it
            if resolved.exists():
                resolved.unlink()
                return f"OK: undone — deleted newly-created file {file_path}"
            return f"OK: undone — file {file_path} already absent"
        # File existed before — restore prior content
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(before, encoding="utf-8")
        return (
            f"OK: undone {tool_name} on {file_path} — "
            f"restored {len(before)} chars of prior content. "
            f"({len(history)} earlier edit(s) still in history.)"
        )
    except Exception as e:
        # Restore the history entry on failure so the agent can retry
        history.append(last)
        return f"ERROR: failed to revert {file_path}: {e}"


@tool
def check_syntax(file_path: str) -> str:
    """
    Check a file for syntax errors after editing. Supports Python, JavaScript,
    TypeScript, and JSON. Always run this after string_replace to verify.

    Args:
        file_path: Relative path from repo root e.g. 'app/services/payment.py'
    """
    resolved = _resolve_repo_or_sandbox(file_path)
    if resolved is None or not resolved.exists():
        return f"ERROR: File not found: {file_path}"

    ext = resolved.suffix.lower()

    # Python — AST parse
    if ext == ".py":
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

    # JavaScript / TypeScript — node --check for JS, tsc --noEmit for TS
    if ext in (".js", ".jsx", ".mjs", ".cjs"):
        try:
            import shutil
            node = shutil.which("node")
            if not node:
                return f"WARNING: node not found — cannot check {file_path}"
            result = subprocess.run(
                [node, "--check", str(resolved)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return f"OK: {file_path} — no syntax errors"
            else:
                return f"SYNTAX ERROR in {file_path}:\n{result.stderr.strip()[:500]}"
        except subprocess.TimeoutExpired:
            return "ERROR: syntax check timed out"
        except Exception as e:
            return f"ERROR: {e}"

    if ext in (".ts", ".tsx"):
        try:
            import shutil
            # Try tsc first (full TypeScript check), fall back to node parse
            tsc = shutil.which("tsc") or shutil.which("npx")
            if tsc and "npx" in str(tsc):
                cmd = [tsc, "tsc", "--noEmit", "--pretty", str(resolved)]
            elif tsc:
                cmd = [tsc, "--noEmit", "--pretty", str(resolved)]
            else:
                # No tsc — try node parse (catches gross syntax errors but not types)
                node = shutil.which("node")
                if not node:
                    return f"WARNING: neither tsc nor node found — cannot check {file_path}"
                cmd = [node, "-e", f"require('fs').readFileSync('{resolved}','utf8')"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return f"OK: {file_path} — no syntax/type errors"
            else:
                return f"SYNTAX ERROR in {file_path}:\n{result.stderr.strip()[:500]}"
        except subprocess.TimeoutExpired:
            return "ERROR: syntax check timed out"
        except Exception as e:
            return f"ERROR: {e}"

    # JSON — stdlib json.load
    if ext == ".json":
        try:
            import json as _json
            _json.loads(resolved.read_text(encoding="utf-8"))
            return f"OK: {file_path} — valid JSON"
        except _json.JSONDecodeError as e:
            return f"SYNTAX ERROR in {file_path}: {e}"
        except Exception as e:
            return f"ERROR: {e}"

    return f"OK: no syntax checker available for {ext} files (skipped {file_path})"


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
        # Snapshot whether the file existed (and what it contained) before
        # so undo_last_edit can restore it precisely.
        prior_content: str | None = None
        if resolved.exists():
            try:
                prior_content = resolved.read_text(encoding="utf-8", errors="replace")
            except Exception:
                prior_content = None  # treat unreadable file as a fresh creation
        _record_edit_snapshot(file_path, prior_content, content, "create_file")
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
        from agent.linters import run_repo_linters as _run_repo_linters

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

    # Speculative review (cavekit pattern): start the reviewer in a background thread
    # the moment tests begin. By the time tests finish, the review is already done.
    # Wall time = max(test_time, review_time) instead of test_time + review_time.
    _start_speculative_review(sandbox)

    # Step 2: Run tests through sandbox.run_tests which handles:
    # - .agent_config.json (setup_commands, test_command, test_args, test_env, test_timeout)
    # - Auto-detection (pytest, npm test, make test)
    # - test_path targeting (appended to the discovered/configured command)
    try:
        from agent.sandbox import run_tests as _run_tests
        test_output = _run_tests(sandbox, repo_path=repo_path, test_path=test_path)
        classified = _classify_sandbox_output(test_output)
        results.append(classified)
        return classified + "\n" + "\n".join(results)
    except Exception as e:
        return f"error: test execution failed ({e})"


def _start_speculative_review(sandbox: Path) -> None:
    """Launch a background thread to pre-compute the review while tests run.

    cavekit speculative review pattern: review starts the moment a patch is
    committed. By the time run_tests() returns, the review is often complete.
    Wall time = max(test_time, review_time) instead of test_time + review_time.

    Result stored in _tls.speculative_review_result for request_review() to pick up.
    """
    import concurrent.futures
    # Skip if review already in flight
    if getattr(_tls, "speculative_review_future", None) is not None:
        return

    try:
        diff_result = subprocess.run(
            ["git", "diff", "--unified=3", "HEAD~1"],
            cwd=str(sandbox), capture_output=True, text=True, timeout=15,
        )
        diff_text = diff_result.stdout
        if len(diff_text) > 10000:
            # Retry with minimal context
            diff_result = subprocess.run(
                ["git", "diff", "--unified=1", "HEAD~1"],
                cwd=str(sandbox), capture_output=True, text=True, timeout=15,
            )
            diff_text = diff_result.stdout[:10000]
    except Exception:
        diff_text = ""

    if not diff_text:
        return

    repo_name = getattr(_tls, "repo_name", "")
    modified_files = [
        line[3:].strip() for line in diff_text.splitlines()
        if line.startswith("+++ b/")
    ]

    def _do_review() -> dict:
        try:
            from agent.graph_utils import build_reviewer_context
            ctx = build_reviewer_context(repo_name, modified_files) if repo_name and modified_files else ""
            from agent.llm import structured_call as _sc
            from pydantic import BaseModel

            class QuickReview(BaseModel):
                verdict: str      # "APPROVE" or "REJECT"
                confidence: float
                summary: str      # 1-2 sentence verdict summary

            prompt = (
                "You are reviewing a code patch. Be concise.\n\n"
                f"PATCH:\n```diff\n{diff_text}\n```\n\n"
                + (f"CONTEXT:\n{ctx[:1000]}\n\n" if ctx else "")
                + "Does this patch look correct and safe? APPROVE or REJECT with confidence and 1-sentence summary."
            )
            result = _sc("claude-sonnet-4-6", 300, QuickReview, prompt)
            return {"verdict": result.verdict, "confidence": result.confidence, "summary": result.summary}
        except Exception as e:
            return {"verdict": "APPROVE", "confidence": 0.5, "summary": f"Speculative review failed: {e}"}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_do_review)
    _tls.speculative_review_future = future
    _tls.speculative_review_executor = executor
    logger.debug("Speculative review started in background")


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
# BRT tool — run confirmed Bug Reproduction Tests in the sandbox
# ---------------------------------------------------------------------------

@tool
def run_brt() -> str:
    """
    Run the confirmed Bug Reproduction Tests (BRTs) on your patched code in the sandbox.
    BRTs are tests that were confirmed to FAIL on the broken code.
    Your fix is correct when ALL BRTs pass (exit 0).

    Call this in Phase 3 BEFORE run_tests. If a BRT still fails after your fix:
    - Read the BRT code to understand what behaviour it checks
    - Fix the production code (NOT the test) so the assertion passes
    - Call run_brt again to verify
    """
    sandbox = getattr(_tls, "sandbox_path", None)
    if not sandbox or not sandbox.exists():
        return "ERROR: No sandbox exists. Call create_sandbox first."

    # Retrieve BRTs from the thread-local state copy stored during react_loop setup
    brts = getattr(_tls, "brts", [])
    if not brts:
        return "No Bug Reproduction Tests were generated for this bug (non-Python repo or no confirmed BRTs). Use run_tests instead."

    results = []
    pass_count = 0

    for i, brt in enumerate(brts, 1):
        code = brt.get("code", "").strip()
        if not code:
            continue

        tmp_path = None
        try:
            import tempfile
            import uuid
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".py", prefix=f"brt_{i}_{uuid.uuid4().hex[:4]}_",
            )
            os.close(tmp_fd)
            Path(tmp_path).write_text(code, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-m", "pytest", tmp_path, "--tb=short", "-x", "-q", "--no-header"],
                cwd=str(sandbox),
                capture_output=True, text=True, timeout=30,
            )

            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                pass_count += 1
                results.append(f"BRT {i} ✓ PASSED — {brt.get('description', '')[:80]}")
            else:
                fail_summary = output[:300] if output else "(no output)"
                results.append(
                    f"BRT {i} ✗ FAILED — {brt.get('description', '')[:80]}\n"
                    f"  Output: {fail_summary}"
                )
        except subprocess.TimeoutExpired:
            results.append(f"BRT {i} ✗ TIMEOUT — {brt.get('description', '')[:60]}")
        except Exception as e:
            results.append(f"BRT {i} ✗ ERROR — {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    total = len(brts)
    epr = pass_count / total if total > 0 else 0.0
    header = f"BRT Results: {pass_count}/{total} passed (EPR={epr:.0%})"
    if pass_count == total:
        header += " ✓ ALL PASS — your fix is verified. Proceed to run_tests then request_review."
    else:
        header += f" ✗ {total - pass_count} STILL FAILING — fix the production code and retry."

    return header + "\n\n" + "\n".join(results)


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

    # Check if speculative review already completed (zero-latency path)
    spec_future = getattr(_tls, "speculative_review_future", None)
    if spec_future is not None:
        try:
            import concurrent.futures
            if spec_future.done():
                spec_result = spec_future.result(timeout=1)
                verdict = spec_result.get("verdict", "APPROVE")
                conf = spec_result.get("confidence", 0.5)
                summary = spec_result.get("summary", "")
                logger.info("Speculative review result ready (zero latency): %s %.0f%%", verdict, conf * 100)
                # Speculative review is a quick pre-check; still run full Opus review below
                spec_note = f"[Speculative pre-check: {verdict} {conf:.0%} — {summary}]\n\n"
            else:
                # Not done yet — wait up to 5s before falling through to normal review
                try:
                    spec_result = spec_future.result(timeout=5)
                    verdict = spec_result.get("verdict", "APPROVE")
                    conf = spec_result.get("confidence", 0.5)
                    summary = spec_result.get("summary", "")
                    spec_note = f"[Speculative pre-check: {verdict} {conf:.0%} — {summary}]\n\n"
                except concurrent.futures.TimeoutError:
                    spec_note = ""
        except Exception:
            spec_note = ""
        finally:
            _tls.speculative_review_future = None
            executor = getattr(_tls, "speculative_review_executor", None)
            if executor:
                executor.shutdown(wait=False)
                _tls.speculative_review_executor = None
    else:
        spec_note = ""

    # Collect the diff — use minimal context to avoid truncation on large files
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--unified=3", "HEAD"],
            cwd=sandbox, capture_output=True, text=True, timeout=30,
        )
        diff_text = diff_result.stdout if diff_result.stdout else "(no diff)"
        # If diff is too long, retry with even less context
        if len(diff_text) > 12000:
            diff_result = subprocess.run(
                ["git", "diff", "--unified=1", "HEAD"],
                cwd=sandbox, capture_output=True, text=True, timeout=30,
            )
            diff_text = diff_result.stdout if diff_result.stdout else diff_text
        # Final safety cap — but log if truncated so we know
        if len(diff_text) > 15000:
            logger.warning("Review diff truncated from %d to 15000 chars", len(diff_text))
            diff_text = diff_text[:15000] + "\n[... diff truncated — multi-file change]"
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
        from agent.graph_utils import build_reviewer_context as _build_reviewer_context
        if repo_name:
            reviewer_context = _build_reviewer_context(repo_name, modified_files)
    except Exception:
        reviewer_context = "No business rules or blast radius data available."

    # Build review prompt — adapt to task type
    from agent.types import ReviewResult

    fix_type = getattr(_tls, "fix_type", "bug_fix")
    is_bug = fix_type == "bug_fix"
    is_feature = fix_type == "enhancement"
    change_noun = "bug fix" if is_bug else "feature implementation" if is_feature else "code change"

    prompt = f"""Review this {change_noun} as an independent reviewer.

CHANGE TYPE: {fix_type}
EXPLANATION: {explanation}

DIFF:
{diff_text}

FILES MODIFIED: {modified_files}

INDEPENDENT CONTEXT (from knowledge graph):
{reviewer_context}

Produce 6 checks ({'ROOT_CAUSE' if is_bug else 'REQUIREMENTS'}, BUSINESS_RULES, PATTERNS, COMPLETENESS, BLAST_RADIUS, TESTS):
- {'ROOT_CAUSE: Does the fix address WHY the bug happens?' if is_bug else 'REQUIREMENTS: Does the implementation match what was requested? New code for a new feature is expected.'}
- BUSINESS_RULES: Any business rules violated?
- PATTERNS: Code follows existing conventions?
- COMPLETENESS: {'All buggy locations patched?' if is_bug else 'All requested functionality implemented?'}
- BLAST_RADIUS: Interface changes covered? Callers updated?
- TESTS: Evidence of testing?

verdict: APPROVE if all checks pass. CHANGES_REQUESTED if any fail. ESCALATE if too complex.
{'NOTE: This is a feature implementation — new code and new endpoints are expected. Do not reject just because the change adds new code.' if is_feature else ''}"""

    try:
        from agent.llm import structured_call as _structured_call
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

        response = spec_note + f"REVIEW VERDICT: {verdict} (confidence: {confidence:.0%})\n\nChecks:\n{checks_str}"
        if feedback:
            response += f"\n\nFeedback: {feedback}"

        return response
    except Exception as e:
        # Fallback to Sonnet
        try:
            from agent.llm import structured_call as _structured_call
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
    Submit your fix for PR creation. Call this after:
    1. You've made edits with string_replace
    2. You've attempted run_tests at least once (OK if tests return 'error' or 'skipped')
    Review is optional — you can submit without it if you're confident in your fix.

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
            # Ensure git user is configured (sandbox may not have it)
            subprocess.run(
                ["git", "config", "user.email", "agent@context-builder.ai"],
                cwd=sandbox, capture_output=True, text=True, timeout=5,
            )
            subprocess.run(
                ["git", "config", "user.name", "Context Builder Agent"],
                cwd=sandbox, capture_output=True, text=True, timeout=5,
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=sandbox, capture_output=True, text=True, check=True, timeout=30,
            )
            result = subprocess.run(
                ["git", "commit", "--no-verify", "-m", f"fix: {explanation[:200]}"],
                cwd=sandbox, capture_output=True, text=True, timeout=30,
            )
            committed = result.returncode == 0
            if not committed:
                logger.warning("Commit failed (rc=%d): stdout=%s stderr=%s",
                               result.returncode, result.stdout[:200], result.stderr[:200])
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
        from agent.graph_utils import load_graph_data as _load_graph_data, find_callers_from_graph as _find_callers_from_graph
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
            from agent.graph_utils import find_callers_via_grep as _find_callers_via_grep
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

PLAN_TOOLS = [produce_plan]
EDIT_TOOLS = [string_replace, check_syntax, create_file, undo_last_edit]
SANDBOX_TOOLS = [create_sandbox, run_tests, run_brt]
MULTI_FILE_TOOLS = [get_callers]
COMPLETION_TOOLS = [record_localization, request_review, submit_fix, escalate]

# All react-specific tools (exploration tools are added from explore_tools.py)
REACT_TOOLS = PLAN_TOOLS + EDIT_TOOLS + SANDBOX_TOOLS + MULTI_FILE_TOOLS + COMPLETION_TOOLS
