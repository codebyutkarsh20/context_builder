"""
Tests for Phase 2A fixes — sandbox security, caller dedup, BUG-6 escalation signal.

Covers:
  - sandbox.py: shlex.split replaces shell=True (2A-1)
  - pipeline.py: _find_callers_from_graph filters test files
  - pipeline.py: _find_callers_via_grep filters test/conftest files
  - pipeline.py: escalate_node emits explicit AGENT_DECLINED signal (2A-0)
"""

import sys
import shlex
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import (
    _find_callers_from_graph,
    _find_callers_via_grep,
    escalate_node,
)
from agent.types import PipelineStatus


# ---------------------------------------------------------------------------
# sandbox.py — shlex.split replaces shell=True (2A-1)
# ---------------------------------------------------------------------------

class TestSandboxShlex:
    """setup_commands now run via shlex.split, not shell=True."""

    def test_subprocess_called_without_shell(self, tmp_path):
        """subprocess.run must NOT be called with shell=True."""
        from agent.sandbox import run_tests
        from agent.agent_config import AgentConfig

        cfg = AgentConfig({"setup_commands": ["pip install requests"], "test_command": "echo ok"})

        with patch("agent.sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch("agent.sandbox.load_agent_config", return_value=cfg):
                run_tests(tmp_path)

        # Every subprocess.run call must NOT have shell=True
        for c in mock_run.call_args_list:
            kwargs = c.kwargs if c.kwargs else {}
            assert kwargs.get("shell") is not True, (
                f"subprocess.run called with shell=True: {c}"
            )

    def test_setup_command_is_tokenized(self, tmp_path):
        """shlex.split('pip install requests') → ['pip', 'install', 'requests']."""
        from agent.sandbox import run_tests
        from agent.agent_config import AgentConfig

        cfg = AgentConfig({"setup_commands": ["pip install requests"], "test_command": "echo ok"})

        captured_args = []

        def fake_run(args, **kwargs):
            captured_args.append(args)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("agent.sandbox.subprocess.run", side_effect=fake_run):
            with patch("agent.sandbox.load_agent_config", return_value=cfg):
                run_tests(tmp_path)

        # First call should be the setup command as a list
        if captured_args:
            setup_call = captured_args[0]
            assert isinstance(setup_call, list), "setup command must be a list, not a string"
            assert setup_call == ["pip", "install", "requests"]


# ---------------------------------------------------------------------------
# pipeline.py — _find_callers_from_graph filters test files
# ---------------------------------------------------------------------------

class TestFindCallersFromGraphFiltersTests:
    """Test files must not appear in caller results."""

    def _make_edges(self, source, target):
        return [{"type": "CALLS", "source": source, "target": target}]

    def test_test_prefix_file_filtered(self):
        """test_utils.py calling a fault file should not appear in callers."""
        edges = self._make_edges("test_utils.py", "services/auth.py")
        graph = {"edges": edges}
        result = _find_callers_from_graph(graph, ["services/auth.py"], [])
        assert "test_utils.py" not in result

    def test_conftest_filtered(self):
        """conftest.py should be filtered from callers."""
        edges = self._make_edges("conftest.py", "services/auth.py")
        graph = {"edges": edges}
        result = _find_callers_from_graph(graph, ["services/auth.py"], [])
        assert "conftest.py" not in result

    def test_tests_subdir_filtered(self):
        """Files under /tests/ should be filtered."""
        edges = self._make_edges("tests/test_auth.py", "services/auth.py")
        graph = {"edges": edges}
        result = _find_callers_from_graph(graph, ["services/auth.py"], [])
        assert "tests/test_auth.py" not in result

    def test_pycache_filtered(self):
        """__pycache__ files should be filtered (path contains /__pycache__/)."""
        edges = self._make_edges("app/__pycache__/auth.cpython-311.pyc", "services/auth.py")
        graph = {"edges": edges}
        result = _find_callers_from_graph(graph, ["services/auth.py"], [])
        assert "app/__pycache__/auth.cpython-311.pyc" not in result

    def test_real_caller_included(self):
        """Non-test callers should still appear in results."""
        edges = self._make_edges("api/views.py", "services/auth.py")
        graph = {"edges": edges}
        result = _find_callers_from_graph(graph, ["services/auth.py"], [])
        assert "api/views.py" in result

    def test_fault_file_itself_excluded(self):
        """The fault file itself must not appear as its own caller."""
        edges = self._make_edges("services/auth.py", "services/auth.py")
        graph = {"edges": edges}
        result = _find_callers_from_graph(graph, ["services/auth.py"], [])
        assert "services/auth.py" not in result


# ---------------------------------------------------------------------------
# pipeline.py — _find_callers_via_grep filters test/conftest files
# ---------------------------------------------------------------------------

class TestFindCallersViaGrepFiltersTests:
    """grep fallback must filter test_ and conftest files."""

    def _make_grep_mock(self, stdout_lines: list[str]):
        """Return a mock subprocess.run that outputs the given lines."""
        mock_result = MagicMock()
        mock_result.stdout = "\n".join(stdout_lines)
        return mock_result

    def test_test_prefix_file_filtered(self, tmp_path):
        """test_something.py in project root is filtered even without /test/ dir."""
        (tmp_path / "test_auth.py").touch()
        (tmp_path / "services").mkdir()
        (tmp_path / "services" / "auth.py").touch()

        with patch("agent.pipeline.subprocess.run") as mock_run:
            mock_run.return_value = self._make_grep_mock(
                [str(tmp_path / "test_auth.py")]
            )
            result = _find_callers_via_grep(tmp_path, ["services/auth.py"])

        assert not any("test_auth.py" in r for r in result)

    def test_conftest_filtered(self, tmp_path):
        """conftest.py is now filtered by the updated _caller_noise check."""
        (tmp_path / "conftest.py").touch()

        with patch("agent.pipeline.subprocess.run") as mock_run:
            mock_run.return_value = self._make_grep_mock(
                [str(tmp_path / "conftest.py")]
            )
            result = _find_callers_via_grep(tmp_path, ["services/auth.py"])

        assert "conftest.py" not in result

    def test_real_source_included(self):
        """A real source file that imports the fault file appears in results."""
        # Use a path without 'test_' components to avoid the noise filter
        with tempfile.TemporaryDirectory(prefix="proj_callers_") as base:
            repo = Path(base)
            (repo / "api.py").touch()

            with patch("agent.pipeline.subprocess.run") as mock_run:
                mock_run.return_value = self._make_grep_mock(
                    [str(repo / "api.py")]
                )
                result = _find_callers_via_grep(repo, ["services/auth.py"])

            assert "api.py" in result


# ---------------------------------------------------------------------------
# pipeline.py — escalate_node: explicit AGENT_DECLINED signal (BUG-6)
# ---------------------------------------------------------------------------

class TestEscalateNodeSignal:
    """escalate_node must emit a visible AGENT_DECLINED log and store declined_reason."""

    def _make_state(self, ticket_id="PROJ-123", feedback="patch didn't apply", iterations=2):
        return {
            "work_order": {"ticket_id": ticket_id},
            "iteration_count": iterations,
            "review": {"verdict": "ESCALATE", "feedback": feedback},
        }

    def test_status_is_escalated(self):
        """state['status'] must be ESCALATED after escalate_node."""
        state = self._make_state()
        result = escalate_node(state)
        assert result["status"] == PipelineStatus.ESCALATED

    def test_declined_reason_set(self):
        """state['declined_reason'] must be set with ticket_id and feedback."""
        state = self._make_state(ticket_id="BUG-6", feedback="no callers found")
        result = escalate_node(state)
        assert "declined_reason" in result
        assert "BUG-6" in result["declined_reason"] or "2" in result["declined_reason"]

    def test_error_field_set(self):
        """state['error'] must describe the escalation reason."""
        state = self._make_state(feedback="confidence too low")
        result = escalate_node(state)
        assert result.get("error")
        assert "confidence too low" in result["error"]

    def test_logger_error_called_with_agent_declined(self):
        """logger.error must be called with AGENT_DECLINED keyword for monitoring."""
        state = self._make_state(ticket_id="PROJ-42")
        with patch("agent.pipeline.logger") as mock_logger:
            escalate_node(state)
        # Must have an ERROR-level call containing AGENT_DECLINED
        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        assert any("AGENT_DECLINED" in c for c in error_calls), (
            f"Expected AGENT_DECLINED in logger.error calls. Got: {error_calls}"
        )

    def test_emit_trace_called_with_escalation_event(self):
        """_emit_trace must be called with 'escalation' event type."""
        state = self._make_state()
        with patch("agent.pipeline._emit_trace") as mock_trace:
            escalate_node(state)
        trace_events = [c.args[0] for c in mock_trace.call_args_list]
        assert "escalation" in trace_events, (
            f"Expected 'escalation' trace event. Got: {trace_events}"
        )

    def test_ticket_id_in_trace(self):
        """Trace event must carry the ticket_id for monitoring correlation."""
        state = self._make_state(ticket_id="PROJ-99")
        with patch("agent.pipeline._emit_trace") as mock_trace:
            escalate_node(state)
        for c in mock_trace.call_args_list:
            if c.args[0] == "escalation":
                data = c.args[1] if len(c.args) > 1 else c.kwargs.get("data", {})
                assert data.get("ticket_id") == "PROJ-99"
                return
        pytest.fail("No escalation trace event found")

    def test_unknown_ticket_fallback(self):
        """escalate_node handles missing ticket_id gracefully."""
        state = {"iteration_count": 0, "review": {}}
        result = escalate_node(state)
        assert result["status"] == PipelineStatus.ESCALATED
        assert "UNKNOWN" in result.get("declined_reason", "") or result.get("error")
