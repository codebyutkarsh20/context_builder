"""
agent_config.py — Per-repo agent configuration loader.

Reads .agent_config.json from the repository root to customize:
- Test command, timeout, and pattern
- Setup commands (e.g., installing deps, seeding data)
- Environment variables for test runs
- Max tool calls for exploration
- Bug categories to auto-skip
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentConfig:
    """Per-repo agent configuration. Loaded from .agent_config.json."""

    # Defaults
    DEFAULT_TEST_COMMAND = "pytest"
    DEFAULT_TEST_TIMEOUT = 300       # 5 minutes
    DEFAULT_TEST_ARGS = ["--tb=short", "-q"]
    DEFAULT_MAX_TOOL_CALLS = 50
    DEFAULT_SKIP_CATEGORIES: list[str] = []  # bug categories to skip (A/B/C)

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config

    # ---- Test configuration ----

    @property
    def test_command(self) -> str:
        return self._cfg.get("test_command", self.DEFAULT_TEST_COMMAND)

    @property
    def test_timeout(self) -> int:
        return int(self._cfg.get("test_timeout", self.DEFAULT_TEST_TIMEOUT))

    @property
    def test_args(self) -> list[str]:
        return self._cfg.get("test_args", self.DEFAULT_TEST_ARGS)

    @property
    def test_pattern(self) -> str | None:
        """Optional glob pattern for test files, e.g. 'tests/unit/test_*.py'."""
        return self._cfg.get("test_pattern")

    @property
    def test_env(self) -> dict[str, str]:
        """Extra environment variables for test runs."""
        return {str(k): str(v) for k, v in self._cfg.get("env", {}).items()}

    # ---- Setup commands ----

    @property
    def setup_commands(self) -> list[str]:
        """Shell commands to run before tests (e.g., 'pip install -r requirements.txt')."""
        return self._cfg.get("setup_commands", [])

    # ---- Exploration ----

    @property
    def max_tool_calls(self) -> int:
        return int(self._cfg.get("max_tool_calls", self.DEFAULT_MAX_TOOL_CALLS))

    # ---- Bug triage ----

    @property
    def skip_bug_categories(self) -> list[str]:
        """Bug categories to skip without running (e.g. ['C'] to skip race conditions)."""
        return self._cfg.get("skip_bug_categories", self.DEFAULT_SKIP_CATEGORIES)

    @property
    def min_confidence(self) -> float | None:
        """Override for MIN_CONFIDENCE_TO_REPAIR, or None to use global default."""
        val = self._cfg.get("min_confidence")
        return float(val) if val is not None else None

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "test_command": self.test_command,
            "test_timeout": self.test_timeout,
            "test_args": self.test_args,
            "test_pattern": self.test_pattern,
            "test_env": self.test_env,
            "setup_commands": self.setup_commands,
            "max_tool_calls": self.max_tool_calls,
            "skip_bug_categories": self.skip_bug_categories,
            "min_confidence": self.min_confidence,
        }

    def __repr__(self) -> str:
        return f"AgentConfig({self._cfg})"


_DEFAULT_CONFIG = AgentConfig({})


def load_agent_config(repo_path: str | Path) -> AgentConfig:
    """
    Load .agent_config.json from the repo root.

    Returns the default AgentConfig if the file doesn't exist or is invalid.

    Example .agent_config.json:
    {
        "test_command": "python -m pytest",
        "test_timeout": 600,
        "test_args": ["--tb=short", "-q", "-x"],
        "test_pattern": "tests/unit/",
        "setup_commands": [
            "pip install -r requirements-test.txt"
        ],
        "env": {
            "DATABASE_URL": "sqlite:///test.db",
            "DJANGO_SETTINGS_MODULE": "myapp.settings.test"
        },
        "max_tool_calls": 60,
        "skip_bug_categories": ["C"],
        "min_confidence": 0.7
    }
    """
    config_path = Path(repo_path) / ".agent_config.json"
    if not config_path.exists():
        logger.debug("No .agent_config.json found at %s — using defaults", config_path)
        return _DEFAULT_CONFIG

    try:
        raw = json.loads(config_path.read_text())
        if not isinstance(raw, dict):
            logger.warning(".agent_config.json must be a JSON object — using defaults")
            return _DEFAULT_CONFIG
        config = AgentConfig(raw)
        logger.info(
            "Loaded .agent_config.json from %s: timeout=%ds, command=%s",
            repo_path,
            config.test_timeout,
            config.test_command,
        )
        return config
    except json.JSONDecodeError as e:
        logger.warning("Invalid .agent_config.json at %s: %s — using defaults", config_path, e)
        return _DEFAULT_CONFIG
    except Exception as e:
        logger.warning("Failed to load .agent_config.json: %s — using defaults", e)
        return _DEFAULT_CONFIG
