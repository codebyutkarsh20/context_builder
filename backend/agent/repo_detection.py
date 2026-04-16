"""
repo_detection.py — Auto-detect project type, test runner, and language from project files.

When the agent encounters a NEW repo it's never seen before, this module
inspects the file tree to determine:
  - Primary language (python, javascript, typescript, go, rust, etc.)
  - Package manager (pip, npm, yarn, pnpm, go modules, cargo)
  - Test runner (pytest, jest, vitest, mocha, go test, cargo test)
  - Lint tool (ruff, eslint, prettier, golangci-lint, clippy)
  - Build command (if needed)

The result is a dict that can be written to `.agent_config.json` so the
sandbox, linters, and system prompt all adapt to the repo's actual stack.

Supports: Python, JavaScript, TypeScript (current focus per user).
Extensible to Go, Rust, Java, Ruby via the same pattern.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection result schema
# ---------------------------------------------------------------------------

def detect_project(repo_dir: str | Path) -> dict[str, Any]:
    """Auto-detect project type and return a config dict.

    The returned dict is compatible with `.agent_config.json` and can be
    written directly or merged with an existing config. Keys:

        language        : str — "python", "javascript", "typescript", "mixed", "unknown"
        package_manager : str — "pip", "npm", "yarn", "pnpm", "unknown"
        test_command    : str — full command to run tests
        test_args       : list[str] — default args for the test runner
        lint_command    : str — full lint command (or "")
        setup_commands  : list[str] — commands to run before tests
        env             : dict[str, str] — env vars to inject
        has_monorepo    : bool — True if frontend/ + backend/ detected
        detected_files  : list[str] — project files that drove the detection

    Parameters
    ----------
    repo_dir : str or Path
        Root of the repository to inspect.

    Returns
    -------
    dict
        Agent config dict. Always returns a valid dict — never raises.
    """
    repo = Path(repo_dir).resolve()
    result: dict[str, Any] = {
        "language": "unknown",
        "package_manager": "unknown",
        "test_command": "",
        "test_args": [],
        "lint_command": "",
        "setup_commands": [],
        "env": {},
        "has_monorepo": False,
        "detected_files": [],
    }

    if not repo.is_dir():
        logger.warning("detect_project: %s is not a directory", repo)
        return result

    # Gather project markers
    files = set()
    for entry in repo.iterdir():
        if entry.is_file():
            files.add(entry.name)
        elif entry.is_dir() and entry.name in ("src", "lib", "backend", "frontend", "packages"):
            files.add(entry.name + "/")
            # Look one level deeper for key files
            for sub in entry.iterdir():
                if sub.is_file():
                    files.add(f"{entry.name}/{sub.name}")

    result["detected_files"] = sorted(files)[:30]  # cap for logging

    # Monorepo detection
    if "frontend/" in files and "backend/" in files:
        result["has_monorepo"] = True
    if "packages/" in files:
        result["has_monorepo"] = True

    # --------------- Language detection ---------------
    # Check both root AND nested (backend/, src/) for project markers.
    # Monorepos like aria-ats have backend/requirements.txt, not root-level.
    _python_markers = {"pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "requirements.txt"}
    _js_markers = {"package.json", "tsconfig.json", "jsconfig.json"}

    has_python = any(
        f in _python_markers or f.split("/", 1)[-1] in _python_markers
        for f in files
    )
    has_js_ts = any(
        f in _js_markers or f.split("/", 1)[-1] in _js_markers
        for f in files
    )
    has_go = "go.mod" in files
    has_rust = "Cargo.toml" in files

    if has_python and has_js_ts:
        result["language"] = "mixed"
    elif has_python:
        result["language"] = "python"
    elif has_js_ts:
        # Check for TypeScript specifically
        if "tsconfig.json" in files or any(
            f.endswith(".ts") or f.endswith(".tsx") for f in files
        ):
            result["language"] = "typescript"
        else:
            result["language"] = "javascript"
    elif has_go:
        result["language"] = "go"
    elif has_rust:
        result["language"] = "rust"

    # --------------- Package manager detection ---------------

    if "yarn.lock" in files:
        result["package_manager"] = "yarn"
    elif "pnpm-lock.yaml" in files:
        result["package_manager"] = "pnpm"
    elif "bun.lockb" in files or "bun.lock" in files:
        result["package_manager"] = "bun"
    elif "package-lock.json" in files or "package.json" in files:
        result["package_manager"] = "npm"
    elif any(f in files for f in ("Pipfile.lock", "Pipfile")):
        result["package_manager"] = "pipenv"
    elif has_python:
        result["package_manager"] = "pip"
    elif has_go:
        result["package_manager"] = "go"
    elif has_rust:
        result["package_manager"] = "cargo"

    # --------------- Test runner detection ---------------

    _detect_test_runner(repo, files, result)

    # --------------- Lint detection ---------------

    _detect_linter(repo, files, result)

    # --------------- Setup commands ---------------

    _detect_setup_commands(repo, files, result)

    logger.info(
        "detect_project: language=%s, pkg=%s, test=%s, lint=%s",
        result["language"], result["package_manager"],
        result["test_command"][:40] or "(none)",
        result["lint_command"][:40] or "(none)",
    )
    return result


# ---------------------------------------------------------------------------
# Test runner detection
# ---------------------------------------------------------------------------

def _detect_test_runner(repo: Path, files: set[str], result: dict) -> None:
    """Detect the test runner and set test_command + test_args."""

    lang = result["language"]

    # ---- Python ----
    if lang in ("python", "mixed"):
        # Check for pytest config
        if "pytest.ini" in files or _file_contains(repo / "pyproject.toml", "[tool.pytest"):
            result["test_command"] = "pytest"
            result["test_args"] = ["-x", "--tb=short", "-q"]
        elif "setup.py" in files or "setup.cfg" in files:
            result["test_command"] = "pytest"
            result["test_args"] = ["-x", "--tb=short", "-q"]
        elif "tox.ini" in files:
            result["test_command"] = "tox"
        else:
            result["test_command"] = "pytest"
            result["test_args"] = ["-x", "--tb=short", "-q"]

    # ---- JavaScript / TypeScript ----
    if lang in ("javascript", "typescript", "mixed"):
        pkg_json = _read_package_json(repo)

        # Check package.json scripts for test runner hints
        scripts = pkg_json.get("scripts", {})
        test_script = scripts.get("test", "")
        dev_deps = {**pkg_json.get("devDependencies", {}), **pkg_json.get("dependencies", {})}

        if "vitest" in dev_deps or "vitest" in test_script:
            pm = result["package_manager"]
            runner = "npx" if pm in ("npm", "unknown") else pm
            result["test_command"] = f"{runner} vitest run"
            result["test_args"] = []
        elif "jest" in dev_deps or "jest" in test_script:
            pm = result["package_manager"]
            runner = "npx" if pm in ("npm", "unknown") else pm
            result["test_command"] = f"{runner} jest"
            result["test_args"] = ["--passWithNoTests"]
        elif "mocha" in dev_deps or "mocha" in test_script:
            pm = result["package_manager"]
            runner = "npx" if pm in ("npm", "unknown") else pm
            result["test_command"] = f"{runner} mocha"
        elif test_script:
            # Has a test script but unknown runner — use npm test
            pm = result["package_manager"]
            result["test_command"] = f"{pm} test" if pm != "unknown" else "npm test"
        elif lang == "mixed" and not result["test_command"]:
            # Mixed repo, no JS test detected, keep Python default
            pass

        # For mixed repos, prefer the Python test command unless ONLY JS tests exist
        if lang == "mixed" and not result["test_command"]:
            result["test_command"] = "pytest"
            result["test_args"] = ["-x", "--tb=short", "-q"]

    # ---- Go ----
    if lang == "go":
        result["test_command"] = "go test"
        result["test_args"] = ["./..."]

    # ---- Rust ----
    if lang == "rust":
        result["test_command"] = "cargo test"

    # ---- Fallback: Makefile ----
    if not result["test_command"] and "Makefile" in files:
        makefile_content = _read_file_safe(repo / "Makefile")
        if "test:" in makefile_content or "test :" in makefile_content:
            result["test_command"] = "make test"


# ---------------------------------------------------------------------------
# Lint detection
# ---------------------------------------------------------------------------

def _detect_linter(repo: Path, files: set[str], result: dict) -> None:
    """Detect the lint tool and set lint_command."""

    lang = result["language"]

    # Python
    if lang in ("python", "mixed"):
        if _file_contains(repo / "pyproject.toml", "[tool.ruff"):
            result["lint_command"] = "ruff check --fix"
        elif ".flake8" in files or _file_contains(repo / "setup.cfg", "[flake8]"):
            result["lint_command"] = "flake8"

    # JS/TS
    if lang in ("javascript", "typescript", "mixed"):
        if ".eslintrc" in files or ".eslintrc.js" in files or \
           ".eslintrc.json" in files or ".eslintrc.cjs" in files or \
           _file_contains(repo / "package.json", '"eslint"'):
            pm = result["package_manager"]
            runner = "npx" if pm in ("npm", "unknown") else pm
            result["lint_command"] = f"{runner} eslint --fix ."
        elif "biome.json" in files or "biome.jsonc" in files:
            result["lint_command"] = "npx @biomejs/biome check --write ."

    # Pre-commit (generic — works for any language)
    if ".pre-commit-config.yaml" in files and not result["lint_command"]:
        result["lint_command"] = "pre-commit run --files"

    # Go
    if lang == "go":
        result["lint_command"] = "golangci-lint run"


# ---------------------------------------------------------------------------
# Setup command detection
# ---------------------------------------------------------------------------

def _detect_setup_commands(repo: Path, files: set[str], result: dict) -> None:
    """Detect setup commands (install deps) based on project type."""

    lang = result["language"]
    pm = result["package_manager"]

    if lang in ("python", "mixed"):
        if "requirements.txt" in files:
            result["setup_commands"].append("pip install -r requirements.txt")
        elif any(f.endswith("/requirements.txt") for f in files):
            # Monorepo: backend/requirements.txt or src/requirements.txt
            req_file = next(f for f in files if f.endswith("/requirements.txt"))
            result["setup_commands"].append(f"pip install -r {req_file}")
        elif "pyproject.toml" in files:
            result["setup_commands"].append("pip install -e .")
        elif "setup.py" in files:
            result["setup_commands"].append("pip install -e .")

    if lang in ("javascript", "typescript", "mixed"):
        if pm == "yarn":
            result["setup_commands"].append("yarn install --frozen-lockfile")
        elif pm == "pnpm":
            result["setup_commands"].append("pnpm install --frozen-lockfile")
        elif pm == "bun":
            result["setup_commands"].append("bun install")
        elif "package.json" in files:
            result["setup_commands"].append("npm install")

    if lang == "go":
        result["setup_commands"].append("go mod download")

    if lang == "rust":
        result["setup_commands"].append("cargo build")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file_safe(path: Path, max_bytes: int = 50_000) -> str:
    """Read a file safely, returning "" on any error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception:
        return ""


def _file_contains(path: Path, substring: str) -> bool:
    """Check if a file contains a substring without reading the whole thing."""
    return substring in _read_file_safe(path)


def _read_package_json(repo: Path) -> dict:
    """Read and parse package.json, returning {} on any error."""
    try:
        return json.loads((repo / "package.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Write agent config from detection
# ---------------------------------------------------------------------------

def write_agent_config_from_detection(
    repo_dir: str | Path,
    overrides: dict | None = None,
) -> Path:
    """Auto-detect project type and write .agent_config.json.

    If a .agent_config.json already exists, its values take precedence
    (detection fills in gaps but doesn't overwrite explicit config).

    Parameters
    ----------
    repo_dir
        Repository root.
    overrides
        Optional dict to merge on top of detection (highest priority).

    Returns
    -------
    Path
        Path to the written .agent_config.json.
    """
    repo = Path(repo_dir).resolve()
    config_path = repo / ".agent_config.json"

    # Existing config takes priority
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    detected = detect_project(repo)

    # Merge: existing > overrides > detected.
    # Treat "unknown" and empty strings as unset so fresh detection can override
    # stale configs written by earlier (buggy) detection runs.
    def _is_set(val: Any) -> bool:
        if not val:
            return False
        if isinstance(val, str) and val.strip().lower() in ("", "unknown", "(none)"):
            return False
        return True

    merged: dict = {}
    for key in ("test_command", "test_args", "lint_command", "setup_commands", "env"):
        if _is_set(existing.get(key)):
            merged[key] = existing[key]
        elif overrides and _is_set(overrides.get(key)):
            merged[key] = overrides[key]
        elif _is_set(detected.get(key)):
            merged[key] = detected[key]

    # Always include language + package_manager for downstream consumers
    merged["language"] = (
        existing.get("language") if _is_set(existing.get("language"))
        else detected.get("language", "unknown")
    )
    merged["package_manager"] = (
        existing.get("package_manager") if _is_set(existing.get("package_manager"))
        else detected.get("package_manager", "unknown")
    )

    # Copy pytest_path if we detected pytest
    if merged.get("test_command", "").startswith("pytest"):
        merged.setdefault("pytest_path", "pytest")

    config_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    logger.info("Wrote .agent_config.json: %s", json.dumps(merged)[:200])
    return config_path
