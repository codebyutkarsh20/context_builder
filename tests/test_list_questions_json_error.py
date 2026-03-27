"""Tests for graceful error handling in the list_questions endpoint."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_graph() -> str:
    return json.dumps({
        "decision_points": [
            {"id": "dp1", "question_for_human": "Is this a test?", "node": "A", "options": ["Yes", "No"]}
        ]
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestListQuestionsJsonError:
    """Tests that list_questions handles invalid/corrupted graph.json gracefully."""

    def _get_client(self):
        """Import and build a TestClient lazily so path patching can work."""
        try:
            from api.knowledge import router
        except ImportError:
            from backend.api.knowledge import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    def test_corrupted_graph_json_returns_422(self, tmp_path):
        """A corrupted graph.json must produce a 422 response, not a 500 crash."""
        repo = "testrepo"
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True)
        graph_file = repo_dir / "graph.json"
        graph_file.write_text("{this is not valid json!!!")

        try:
            import api.knowledge as knowledge_mod
        except ImportError:
            import backend.api.knowledge as knowledge_mod

        original_data_dir = knowledge_mod._DATA_DIR
        knowledge_mod._DATA_DIR = tmp_path
        try:
            client = self._get_client()
            response = client.get(f"/knowledge/{repo}/questions")
            assert response.status_code in (422, 400), (
                f"Expected 422 or 400 for corrupted JSON, got {response.status_code}. "
                f"Body: {response.text}"
            )
            body = response.json()
            # Should be a well-formed error object, not a traceback
            assert "detail" in body or "message" in body, (
                f"Response should contain 'detail' or 'message'. Got: {body}"
            )
        finally:
            knowledge_mod._DATA_DIR = original_data_dir

    def test_missing_graph_json_returns_404(self, tmp_path):
        """A missing graph.json must still produce a 404 (regression guard)."""
        repo = "missingrepo"

        try:
            import api.knowledge as knowledge_mod
        except ImportError:
            import backend.api.knowledge as knowledge_mod

        original_data_dir = knowledge_mod._DATA_DIR
        knowledge_mod._DATA_DIR = tmp_path
        try:
            client = self._get_client()
            response = client.get(f"/knowledge/{repo}/questions")
            assert response.status_code == 404, (
                f"Expected 404 for missing repo, got {response.status_code}"
            )
        finally:
            knowledge_mod._DATA_DIR = original_data_dir

    def test_valid_graph_json_returns_200(self, tmp_path):
        """A valid graph.json must continue to be parsed and returned (happy path)."""
        repo = "validrepo"
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True)
        graph_file = repo_dir / "graph.json"
        graph_file.write_text(_make_valid_graph())

        try:
            import api.knowledge as knowledge_mod
        except ImportError:
            import backend.api.knowledge as knowledge_mod

        original_data_dir = knowledge_mod._DATA_DIR
        knowledge_mod._DATA_DIR = tmp_path
        try:
            client = self._get_client()
            response = client.get(f"/knowledge/{repo}/questions")
            assert response.status_code == 200, (
                f"Expected 200 for valid graph.json, got {response.status_code}. "
                f"Body: {response.text}"
            )
            # Response must be JSON-parseable (well-formed)
            body = response.json()
            assert isinstance(body, (dict, list)), (
                f"Expected a JSON object or array in 200 response, got: {type(body)}"
            )
        finally:
            knowledge_mod._DATA_DIR = original_data_dir

    def test_corrupted_graph_json_does_not_raise_unhandled_exception(self, tmp_path):
        """
        Verifies that a JSONDecodeError does NOT propagate unhandled out of the endpoint.
        The endpoint must catch it and return a structured HTTP error.
        """
        repo = "crashrepo"
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True)
        (repo_dir / "graph.json").write_text("TRUNCATED{{{")

        try:
            import api.knowledge as knowledge_mod
        except ImportError:
            import backend.api.knowledge as knowledge_mod

        original_data_dir = knowledge_mod._DATA_DIR
        knowledge_mod._DATA_DIR = tmp_path
        try:
            client = self._get_client()
            # raise_server_exceptions=False means we get the HTTP response,
            # not a re-raised Python exception in the test process.
            response = client.get(f"/knowledge/{repo}/questions")
            # Any 4xx is acceptable; 5xx means the exception propagated unhandled.
            assert response.status_code < 500, (
                f"Endpoint must not return 5xx for corrupted JSON. Got {response.status_code}. "
                f"Body: {response.text}"
            )
        finally:
            knowledge_mod._DATA_DIR = original_data_dir
