"""
sandbox.py — Git worktree sandbox, test execution, and cleanup.

Extracted from pipeline.py to keep each module focused on a single concern.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .agent_config import load_agent_config, AgentConfig

logger = logging.getLogger(__name__)


def run_tests(
    worktree_path: Path,
    repo_path: "str | Path | None" = None,
    agent_config: "AgentConfig | None" = None,
) -> str:
    """Auto-detect test runner and execute tests.

    Searches both the worktree root and one level of subdirectories so that
    projects with a backend/ or src/ layout are found correctly.

    If a .agent_config.json is present in repo_path (or agent_config is
    supplied directly), its test_command, test_timeout, test_args, test_env,
    setup_commands, and test_pattern take precedence over auto-detection.

    Args:
        worktree_path: Path to the sandbox worktree where tests run.
        repo_path: Original repo root used to load .agent_config.json when
            agent_config is not provided explicitly.
        agent_config: Pre-loaded AgentConfig; if None and repo_path is given,
            the config is loaded automatically.

    Returns:
        A human-readable string starting with "passed", "failed", "skipped",
        or "error".
    """
    worktree_path = Path(worktree_path)

    # Load per-repo config if not provided
    if agent_config is None and repo_path:
        agent_config = load_agent_config(repo_path)

    # ------------------------------------------------------------------ #
    # Path A — config-driven: .agent_config.json present and non-default  #
    # ------------------------------------------------------------------ #
    if agent_config is not None and agent_config._cfg:
        timeout = agent_config.test_timeout
        test_cmd_base = agent_config.test_command
        test_args = agent_config.test_args
        extra_env = agent_config.test_env
        setup_commands = agent_config.setup_commands
        test_pattern = agent_config.test_pattern

        env = {**os.environ, **extra_env}

        # Run setup commands first (e.g., pip install, db migrate)
        for cmd in setup_commands:
            logger.info("Running setup command: %s", cmd)
            try:
                subprocess.run(
                    shlex.split(cmd), cwd=str(worktree_path),
                    env=env, timeout=120, capture_output=True,
                )
            except Exception as e:
                logger.warning("Setup command failed: %s — %s", cmd, e)

        # Build test command
        if test_cmd_base in ("pytest", "python -m pytest"):
            cmd_parts = [sys.executable, "-m", "pytest"] + test_args
        else:
            cmd_parts = test_cmd_base.split()

        if test_pattern:
            cmd_parts.append(test_pattern)

        logger.info("Running tests (config-driven) in %s: %s", worktree_path, " ".join(cmd_parts))
        try:
            result = subprocess.run(
                cmd_parts,
                cwd=str(worktree_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw_output = (result.stdout + "\n" + result.stderr).strip()
            return _format_test_output(result.returncode, raw_output, timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Tests timed out after %ds", timeout)
            return f"failed: timed out after {timeout}s"
        except Exception as e:
            logger.warning("Test execution error: %s", e)
            return f"error: {e}"

    # ------------------------------------------------------------------ #
    # Path B — auto-detection (original behaviour, no config file)        #
    # ------------------------------------------------------------------ #
    pytest_markers = ("pytest.ini", "pyproject.toml", "setup.py", "setup.cfg")

    test_cwd = worktree_path
    cmd = None
    for search_dir in [worktree_path] + sorted(worktree_path.iterdir()):
        if not search_dir.is_dir():
            continue
        if any((search_dir / m).exists() for m in pytest_markers):
            cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"]
            test_cwd = search_dir
            break
        if (search_dir / "package.json").exists() and cmd is None:
            cmd = ["npm", "test"]
            test_cwd = search_dir
        if (search_dir / "Makefile").exists() and cmd is None:
            cmd = ["make", "test"]
            test_cwd = search_dir

    if cmd is None:
        logger.info("No test runner detected — skipping tests")
        return "skipped: no test runner found"

    logger.info("Running tests in %s: %s", test_cwd, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=test_cwd, capture_output=True, text=True, timeout=300,
        )
        raw_output = (result.stdout + "\n" + result.stderr).strip()
        return _format_test_output(result.returncode, raw_output, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("Tests timed out after 5 minutes")
        return "failed: timed out after 5 minutes"
    except Exception as e:
        logger.warning("Test execution error: %s", e)
        return f"error: {e}"


def _format_test_output(returncode: int, raw_output: str, timeout: int) -> str:
    """Format raw subprocess output into a concise test result string.

    Pytest exit codes:
      0 = all tests passed
      1 = some tests failed
      2 = interrupted
      3 = internal error
      4 = no tests collected (not a failure — means test discovery found nothing)
      5 = no tests ran (not a failure — no matching tests)
    """
    if returncode == 0:
        logger.info("Tests passed")
        summary_lines = [
            line for line in raw_output.splitlines()
            if "passed" in line.lower() or "ok" in line.lower()
        ]
        return "passed\n" + ("\n".join(summary_lines[-5:]) if summary_lines else raw_output[:500])

    # Pytest exit code 4 = no tests collected, 5 = no tests ran
    # These are NOT failures — they mean test discovery didn't find anything
    if returncode in (4, 5):
        logger.info("No tests collected/ran (exit code %d) — treating as skipped", returncode)
        return (
            f"skipped: no tests collected (pytest exit code {returncode}). "
            "This usually means the test path didn't match any tests, or the repo "
            "needs special setup. Try a more specific test_path.\n"
            + raw_output[-500:]
        )

    logger.warning("Tests failed (exit code %d)", returncode)
    error_lines = []
    for line in raw_output.splitlines():
        line_lower = line.lower()
        if any(kw in line_lower for kw in (
            "error", "fail", "assert", "exception", "traceback",
            "syntaxerror", "nameerror", "import",
        )):
            error_lines.append(line)
        elif line.startswith("E ") or line.startswith("> "):
            error_lines.append(line)
        elif line.startswith("FAILED "):
            error_lines.append(line)
    parsed = "\n".join(error_lines[:40]) if error_lines else raw_output[:3000]
    return f"failed (exit code {returncode})\n{parsed}"


def create_agent_config_template(repo_path: "str | Path") -> None:
    """Write a .agent_config.json.example template to the repo root."""
    template_path = Path(repo_path) / ".agent_config.json.example"
    if template_path.exists():
        return
    template = {
        "_comment": "Rename to .agent_config.json to activate. All fields are optional.",
        "test_command": "pytest",
        "test_timeout": 300,
        "test_args": ["--tb=short", "-q"],
        "test_pattern": None,
        "setup_commands": [],
        "env": {
            "DATABASE_URL": "sqlite:///test.db"
        },
        "max_tool_calls": 50,
        "skip_bug_categories": [],
        "min_confidence": None,
    }
    try:
        template_path.write_text(json.dumps(template, indent=2))
        logger.info("Wrote .agent_config.json.example to %s", repo_path)
    except Exception:
        pass


def cleanup_worktree(repo_path: Path | None, sandbox_path: str) -> None:
    """Clean up git worktree and its directory. Best-effort, never throws."""
    if not sandbox_path or not repo_path:
        return
    sandbox_path = Path(sandbox_path)
    repo_path = Path(repo_path)

    # Step 1: Try git worktree remove --force
    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(sandbox_path)],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("git worktree remove failed: %s", result.stderr)
    except Exception as e:
        logger.warning("git worktree remove exception: %s", e)

    # Step 2: Always try to remove directory even if git command failed
    if sandbox_path.exists():
        try:
            shutil.rmtree(sandbox_path, ignore_errors=True)
            logger.info("Removed sandbox directory: %s", sandbox_path)
        except Exception as e:
            logger.warning("shutil.rmtree failed: %s", e)

    # Step 3: Prune stale worktree references
    try:
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(repo_path),
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def cleanup_stale_worktrees(repo_path) -> None:
    """Periodic cleanup of stale sandbox directories and orphaned branches."""
    import glob
    import time

    cutoff = time.time() - 7200  # 2 hours
    for sandbox_dir in glob.glob("/tmp/agent_sandbox_*"):
        try:
            if os.path.getmtime(sandbox_dir) < cutoff:
                cleanup_worktree(repo_path, sandbox_dir)
                logger.info("Cleaned stale sandbox: %s", sandbox_dir)
        except Exception as e:
            logger.warning("Failed to clean stale sandbox %s: %s", sandbox_dir, e)

    # Delete stale fix/* branches from the repo
    try:
        result = subprocess.run(
            ["git", "branch", "--list", "fix/*"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            for branch in result.stdout.splitlines():
                branch = branch.strip().lstrip("* ")
                if not branch:
                    continue
                try:
                    subprocess.run(
                        ["git", "branch", "-D", branch],
                        cwd=str(repo_path),
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    logger.info("Deleted stale branch: %s", branch)
                except Exception as e:
                    logger.warning("Failed to delete branch %s: %s", branch, e)
    except Exception as e:
        logger.warning("Failed to list fix/* branches: %s", e)


def append_test_business_context(state: dict, work_order: dict) -> None:
    """If tests failed, look up business-intent enrichments and append context."""
    test_result = state.get("test_result", "")
    if not test_result or "fail" not in test_result.lower():
        return

    repo = work_order.get("repo", work_order.get("repo_name", ""))
    if not repo:
        return

    try:
        from enricher.test_enricher import lookup_failed_tests, format_failure_context

        failed_names: list[str] = []
        for line in test_result.splitlines():
            if "FAILED" in line:
                match = re.search(r"FAILED\s+\S+::(\w+)", line)
                if match:
                    failed_names.append(match.group(1))

        if not failed_names:
            return

        enrichments = lookup_failed_tests(repo, failed_names)
        if enrichments:
            context_str = format_failure_context(enrichments)
            state["test_result"] = test_result + context_str
            logger.info("Appended business context for %d failed tests", len(enrichments))
    except Exception as e:
        logger.debug("Could not append test business context: %s", e)
