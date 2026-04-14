"""
test_repo_detection.py — Tests for auto-detection of project type, test runner, and language.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.repo_detection import detect_project, write_agent_config_from_detection


# ---------------------------------------------------------------------------
# Python repos
# ---------------------------------------------------------------------------

class TestPythonDetection:
    def test_pytest_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = "-x"')
        (tmp_path / "src").mkdir()
        result = detect_project(tmp_path)
        assert result["language"] == "python"
        assert result["package_manager"] == "pip"
        assert "pytest" in result["test_command"]

    def test_pytest_from_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup; setup()")
        result = detect_project(tmp_path)
        assert result["language"] == "python"
        assert "pytest" in result["test_command"]

    def test_requirements_txt_setup(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask==2.0")
        result = detect_project(tmp_path)
        assert result["language"] == "python"
        assert "pip install" in result["setup_commands"][0]

    def test_ruff_lint_detected(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[tool.ruff]\nline-length = 100')
        result = detect_project(tmp_path)
        assert "ruff" in result["lint_command"]


# ---------------------------------------------------------------------------
# JavaScript / TypeScript repos
# ---------------------------------------------------------------------------

class TestJsDetection:
    def test_npm_with_jest(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "jest"},
            "devDependencies": {"jest": "^29.0"},
        }))
        result = detect_project(tmp_path)
        assert result["language"] == "javascript"
        assert result["package_manager"] == "npm"
        assert "jest" in result["test_command"]

    def test_typescript_detected(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"name": "app"}))
        (tmp_path / "tsconfig.json").write_text("{}")
        result = detect_project(tmp_path)
        assert result["language"] == "typescript"

    def test_vitest_detected(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "vitest run"},
            "devDependencies": {"vitest": "^1.0"},
        }))
        result = detect_project(tmp_path)
        assert "vitest" in result["test_command"]

    def test_yarn_lock_detected(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"name": "app"}))
        (tmp_path / "yarn.lock").write_text("")
        result = detect_project(tmp_path)
        assert result["package_manager"] == "yarn"

    def test_pnpm_lock_detected(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"name": "app"}))
        (tmp_path / "pnpm-lock.yaml").write_text("")
        result = detect_project(tmp_path)
        assert result["package_manager"] == "pnpm"

    def test_eslint_detected(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "devDependencies": {"eslint": "^8.0"},
        }))
        (tmp_path / ".eslintrc.json").write_text("{}")
        result = detect_project(tmp_path)
        assert "eslint" in result["lint_command"]

    def test_npm_test_fallback(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "node test.js"},
        }))
        result = detect_project(tmp_path)
        assert "npm test" in result["test_command"]

    def test_npm_install_in_setup(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"name": "app"}))
        result = detect_project(tmp_path)
        assert any("npm install" in cmd for cmd in result["setup_commands"])


# ---------------------------------------------------------------------------
# Mixed repos
# ---------------------------------------------------------------------------

class TestMixedDetection:
    def test_python_plus_js(self, tmp_path):
        (tmp_path / "setup.py").write_text("")
        (tmp_path / "package.json").write_text(json.dumps({"name": "x"}))
        result = detect_project(tmp_path)
        assert result["language"] == "mixed"

    def test_monorepo_detected(self, tmp_path):
        (tmp_path / "frontend").mkdir()
        (tmp_path / "backend").mkdir()
        (tmp_path / "package.json").write_text(json.dumps({"name": "x"}))
        result = detect_project(tmp_path)
        assert result["has_monorepo"] is True


# ---------------------------------------------------------------------------
# Go / Rust repos
# ---------------------------------------------------------------------------

class TestGoRustDetection:
    def test_go_mod_detected(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/foo")
        result = detect_project(tmp_path)
        assert result["language"] == "go"
        assert "go test" in result["test_command"]
        assert result["package_manager"] == "go"

    def test_cargo_detected(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'")
        result = detect_project(tmp_path)
        assert result["language"] == "rust"
        assert "cargo test" in result["test_command"]


# ---------------------------------------------------------------------------
# Makefile fallback
# ---------------------------------------------------------------------------

class TestMakefileDetection:
    def test_makefile_test_target(self, tmp_path):
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
        result = detect_project(tmp_path)
        assert result["test_command"] == "make test"


# ---------------------------------------------------------------------------
# Empty / unrecognized repos
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dir(self, tmp_path):
        result = detect_project(tmp_path)
        assert result["language"] == "unknown"
        assert result["test_command"] == ""

    def test_nonexistent_dir(self):
        result = detect_project("/nonexistent/path")
        assert result["language"] == "unknown"


# ---------------------------------------------------------------------------
# write_agent_config_from_detection
# ---------------------------------------------------------------------------

class TestWriteConfig:
    def test_writes_config_file(self, tmp_path):
        (tmp_path / "setup.py").write_text("")
        path = write_agent_config_from_detection(tmp_path)
        assert path.exists()
        config = json.loads(path.read_text())
        assert config["language"] == "python"
        assert "pytest" in config["test_command"]

    def test_existing_config_takes_priority(self, tmp_path):
        (tmp_path / "setup.py").write_text("")
        (tmp_path / ".agent_config.json").write_text(json.dumps({
            "test_command": "custom_runner --special",
        }))
        path = write_agent_config_from_detection(tmp_path)
        config = json.loads(path.read_text())
        assert config["test_command"] == "custom_runner --special"

    def test_overrides_fill_gaps(self, tmp_path):
        (tmp_path / "setup.py").write_text("")
        path = write_agent_config_from_detection(tmp_path, overrides={
            "env": {"DATABASE_URL": "sqlite:///test.db"},
        })
        config = json.loads(path.read_text())
        assert config["env"]["DATABASE_URL"] == "sqlite:///test.db"
        # Detection-based values still present
        assert "pytest" in config["test_command"]
