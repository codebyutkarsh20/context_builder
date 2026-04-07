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

    # Pipeline Tuning defaults
    DEFAULT_BRT_MIN_CONFIRMED = 3            # BRT confirmed minimum threshold
    DEFAULT_BRT_SOURCE_CHAR_LIMIT = 3000     # max chars per BRT source snippet
    DEFAULT_BRT_TOKEN_BUDGET = 3500          # token budget for BRT context
    DEFAULT_VERIFIER_CONFIDENCE_THRESHOLD = 0.8  # verifier must exceed this
    DEFAULT_BEST_OF_N_MAX = 5                # best-of-N patch sampling cap
    DEFAULT_GRAPH_SUMMARY_CHAR_LIMIT = 1200  # graph summary truncation limit
    DEFAULT_MAX_GREP_ATTEMPTS = 5            # anti-pattern: max grep calls
    DEFAULT_MAX_FILE_REREADS = 3             # anti-pattern: max re-reads of same file
    DEFAULT_MAX_TEST_RETRIES = 2             # anti-pattern: max test retry loops
    DEFAULT_SCOUT_TIMEOUT_S = 30             # scout fault-localization timeout (seconds)

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

    @property
    def test_cwd(self) -> str:
        """Subdirectory (relative to repo root) to run tests from. Empty = repo root."""
        return self._cfg.get("test_cwd", "")

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

    # ---- Pipeline Tuning ----

    @property
    def brt_min_confirmed(self) -> int:
        """Minimum BRT confirmed locations before repair proceeds."""
        return int(self._cfg.get("brt_min_confirmed", self.DEFAULT_BRT_MIN_CONFIRMED))

    @property
    def brt_source_char_limit(self) -> int:
        """Max characters per BRT source snippet included in context."""
        return int(self._cfg.get("brt_source_char_limit", self.DEFAULT_BRT_SOURCE_CHAR_LIMIT))

    @property
    def brt_token_budget(self) -> int:
        """Token budget allocated for BRT context assembly."""
        return int(self._cfg.get("brt_token_budget", self.DEFAULT_BRT_TOKEN_BUDGET))

    @property
    def verifier_confidence_threshold(self) -> float:
        """Confidence threshold the verifier must exceed to accept a patch."""
        return float(self._cfg.get("verifier_confidence_threshold", self.DEFAULT_VERIFIER_CONFIDENCE_THRESHOLD))

    @property
    def best_of_n_max(self) -> int:
        """Maximum number of candidate patches in best-of-N sampling."""
        return int(self._cfg.get("best_of_n_max", self.DEFAULT_BEST_OF_N_MAX))

    @property
    def graph_summary_char_limit(self) -> int:
        """Character limit for graph summary truncation in scout FL."""
        return int(self._cfg.get("graph_summary_char_limit", self.DEFAULT_GRAPH_SUMMARY_CHAR_LIMIT))

    @property
    def max_grep_attempts(self) -> int:
        """Anti-pattern threshold: max grep calls before flagging."""
        return int(self._cfg.get("max_grep_attempts", self.DEFAULT_MAX_GREP_ATTEMPTS))

    @property
    def max_file_rereads(self) -> int:
        """Anti-pattern threshold: max re-reads of the same file."""
        return int(self._cfg.get("max_file_rereads", self.DEFAULT_MAX_FILE_REREADS))

    @property
    def max_test_retries(self) -> int:
        """Anti-pattern threshold: max test retry loops before aborting."""
        return int(self._cfg.get("max_test_retries", self.DEFAULT_MAX_TEST_RETRIES))

    @property
    def scout_timeout_s(self) -> int:
        """Timeout in seconds for scout fault-localization."""
        return int(self._cfg.get("scout_timeout_s", self.DEFAULT_SCOUT_TIMEOUT_S))

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
            # Pipeline tuning
            "brt_min_confirmed": self.brt_min_confirmed,
            "brt_source_char_limit": self.brt_source_char_limit,
            "brt_token_budget": self.brt_token_budget,
            "verifier_confidence_threshold": self.verifier_confidence_threshold,
            "best_of_n_max": self.best_of_n_max,
            "graph_summary_char_limit": self.graph_summary_char_limit,
            "max_grep_attempts": self.max_grep_attempts,
            "max_file_rereads": self.max_file_rereads,
            "max_test_retries": self.max_test_retries,
            "scout_timeout_s": self.scout_timeout_s,
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
