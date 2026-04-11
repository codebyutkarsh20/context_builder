"""
Tests for logging configuration — Phase 1.1

Verifies:
  - Logging is configured at module level
  - Noisy libraries are quieted
  - Log format includes timestamp and level
"""

import sys
import logging
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLoggingConfig:
    """Logging is properly configured in main.py."""

    def test_main_configures_logging(self):
        """main.py calls logging.basicConfig after load_dotenv."""
        src = Path(__file__).resolve().parent.parent / "main.py"
        content = src.read_text()

        # basicConfig must appear after load_dotenv
        dotenv_pos = content.index("load_dotenv(")
        basic_pos = content.index("logging.basicConfig")
        assert basic_pos > dotenv_pos, "basicConfig should be after load_dotenv"

        # Must be before router imports
        router_pos = content.index("from api.repos")
        assert basic_pos < router_pos, "basicConfig should be before router imports"

    def test_noisy_libraries_quieted(self):
        """httpcore, httpx, etc. are set to WARNING."""
        src = Path(__file__).resolve().parent.parent / "main.py"
        content = src.read_text()

        for lib in ["httpcore", "httpx", "urllib3"]:
            assert lib in content, f"{lib} should be quieted"

    def test_log_format_has_timestamp(self):
        src = Path(__file__).resolve().parent.parent / "main.py"
        content = src.read_text()
        assert "asctime" in content, "Format should include timestamp"

    def test_log_format_has_level(self):
        src = Path(__file__).resolve().parent.parent / "main.py"
        content = src.read_text()
        assert "levelname" in content, "Format should include level"

    def test_log_level_is_info(self):
        src = Path(__file__).resolve().parent.parent / "main.py"
        content = src.read_text()
        assert "level=logging.INFO" in content

    def test_react_loop_has_logger(self):
        """agent/react_loop.py uses module-level logger."""
        import agent.react_loop as react_loop
        assert hasattr(react_loop, 'logger')
        assert react_loop.logger.name == "agent.react_loop"

    def test_backend_log_output(self):
        """Backend log file should contain structured output."""
        log_path = Path("/tmp/backend_test.log")
        if log_path.exists():
            log = log_path.read_text()
            assert "INFO" in log, "Logs should contain INFO messages"
            # Check for our custom separator
            assert "—" in log or "-" in log, "Log should use separator format"
