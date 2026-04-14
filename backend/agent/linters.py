"""
linters.py — Repo-aware linter execution extracted from pipeline.py.

Discovers and runs linters on patched files. Supports:
  - Pre-commit (any language — runs whatever hooks are configured)
  - Python: ruff, flake8
  - JavaScript/TypeScript: eslint, biome
  - Go: golangci-lint (if installed)

Linters are discovered by checking for config files in the worktree root
and tool availability on PATH. First match wins within each language group.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_repo_linters(worktree_path: Path, patches: list[dict]) -> str:
    """Discover and run the target repo's linters on patched files.

    Checks for pre-commit config first (language-agnostic), then
    language-specific linters based on file extensions.

    Returns error string if linting fails, empty string if OK.
    """
    patched_files = []
    for p in patches:
        fpath = p.get("file_path", "")
        if fpath and (worktree_path / fpath).exists():
            patched_files.append(fpath)
    if not patched_files:
        return ""

    # Strategy 1: pre-commit (language-agnostic — the best option when available)
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

    # Split files by language for targeted linting
    py_files = [f for f in patched_files if f.endswith(".py")]
    js_ts_files = [
        f for f in patched_files
        if any(f.endswith(ext) for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))
    ]
    go_files = [f for f in patched_files if f.endswith(".go")]

    errors: list[str] = []

    # ---- Python linters ----
    if py_files:
        py_err = _lint_python(worktree_path, py_files)
        if py_err:
            errors.append(py_err)

    # ---- JavaScript / TypeScript linters ----
    if js_ts_files:
        js_err = _lint_js_ts(worktree_path, js_ts_files)
        if js_err:
            errors.append(js_err)

    # ---- Go linters ----
    if go_files:
        go_err = _lint_go(worktree_path, go_files)
        if go_err:
            errors.append(go_err)

    return "\n".join(errors) if errors else ""


# ---------------------------------------------------------------------------
# Language-specific linter runners
# ---------------------------------------------------------------------------

def _lint_python(worktree_path: Path, files: list[str]) -> str:
    """Run ruff or flake8 on Python files."""
    # ruff (preferred — faster, more modern)
    if shutil.which("ruff"):
        try:
            result = subprocess.run(
                ["ruff", "check", "--no-fix"] + files,
                cwd=worktree_path, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                errors = result.stdout.strip()
                if errors:
                    return f"[ruff] {errors[:500]}"
        except (subprocess.TimeoutExpired, Exception):
            pass

    # flake8 (fallback)
    if shutil.which("flake8"):
        try:
            result = subprocess.run(
                ["flake8", "--max-line-length=120"] + files,
                cwd=worktree_path, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                errors = result.stdout.strip()
                critical = [
                    l for l in errors.splitlines()
                    if len(l.split(":")) > 3 and ("E999" in l or "F" in l.split(":")[3])
                ]
                if critical:
                    return f"[flake8] {chr(10).join(critical[:10])}"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

    return ""


def _lint_js_ts(worktree_path: Path, files: list[str]) -> str:
    """Run eslint or biome on JavaScript/TypeScript files."""
    # eslint
    eslint_configs = [
        ".eslintrc", ".eslintrc.js", ".eslintrc.json", ".eslintrc.cjs",
        ".eslintrc.yaml", ".eslintrc.yml",
    ]
    has_eslint = any((worktree_path / cfg).exists() for cfg in eslint_configs) or \
                 (worktree_path / "eslint.config.js").exists() or \
                 (worktree_path / "eslint.config.mjs").exists()

    if has_eslint:
        # Try npx eslint (works even without global install)
        npx = shutil.which("npx")
        if npx:
            try:
                result = subprocess.run(
                    [npx, "eslint", "--no-fix", "--format", "compact"] + files,
                    cwd=worktree_path, capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    errors = result.stdout.strip()
                    if errors:
                        # Filter to error-level only (skip warnings)
                        err_lines = [l for l in errors.splitlines() if "Error" in l or "error" in l.lower()]
                        if err_lines:
                            return f"[eslint] {chr(10).join(err_lines[:10])}"
            except (subprocess.TimeoutExpired, Exception):
                pass

    # biome
    biome_config = worktree_path / "biome.json"
    biome_config_c = worktree_path / "biome.jsonc"
    if biome_config.exists() or biome_config_c.exists():
        npx = shutil.which("npx")
        if npx:
            try:
                result = subprocess.run(
                    [npx, "@biomejs/biome", "check"] + files,
                    cwd=worktree_path, capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    errors = (result.stdout + result.stderr).strip()
                    if errors:
                        return f"[biome] {errors[:500]}"
            except (subprocess.TimeoutExpired, Exception):
                pass

    return ""


def _lint_go(worktree_path: Path, files: list[str]) -> str:
    """Run golangci-lint on Go files."""
    if not shutil.which("golangci-lint"):
        return ""

    # golangci-lint runs on packages, not individual files.
    # Deduplicate to unique directories.
    dirs = sorted(set(str(Path(f).parent) for f in files))
    pkg_args = [f"./{d}/..." if d != "." else "./..." for d in dirs]

    try:
        result = subprocess.run(
            ["golangci-lint", "run", "--new-from-rev=HEAD~1"] + pkg_args,
            cwd=worktree_path, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            errors = result.stdout.strip()
            if errors:
                return f"[golangci-lint] {errors[:500]}"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return ""
