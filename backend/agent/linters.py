"""
linters.py — Repo-aware linter execution extracted from pipeline.py.

Discovers and runs pre-commit, ruff, or flake8 on patched files.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_repo_linters(worktree_path: Path, patches: list[dict]) -> str:
    """Discover and run the target repo's linters on patched files.

    Checks for pre-commit config, ruff, flake8 — in that order.
    Returns error string if linting fails, empty string if OK.
    """
    patched_files = []
    for p in patches:
        fpath = p.get("file_path", "")
        if fpath and (worktree_path / fpath).exists():
            patched_files.append(fpath)
    if not patched_files:
        return ""

    # Strategy 1: pre-commit
    precommit_cfg = worktree_path / ".pre-commit-config.yaml"
    if precommit_cfg.exists():
        try:
            subprocess.run(
                ["git", "add"] + patched_files,
                cwd=worktree_path, capture_output=True, text=True, timeout=30,
            )
            result = subprocess.run(
                ["pre-commit", "run", "--files"] + patched_files,
                cwd=worktree_path, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                output = result.stdout + result.stderr
                error_lines = [
                    line for line in output.splitlines()
                    if line.strip() and "Passed" not in line and "Skipped" not in line
                ]
                if error_lines:
                    return "\n".join(error_lines[-20:])
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    py_files = [f for f in patched_files if f.endswith(".py")]
    if not py_files:
        return ""

    # Strategy 2: ruff
    try:
        result = subprocess.run(
            ["ruff", "check", "--no-fix"] + py_files,
            cwd=worktree_path, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            errors = result.stdout.strip()
            if errors:
                return errors[:500]
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Strategy 3: flake8
    try:
        result = subprocess.run(
            ["flake8", "--max-line-length=120"] + py_files,
            cwd=worktree_path, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            errors = result.stdout.strip()
            critical = [
                l for l in errors.splitlines()
                if len(l.split(":")) > 3 and ("E999" in l or "F" in l.split(":")[3])
            ]
            if critical:
                return "\n".join(critical[:10])
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return ""
