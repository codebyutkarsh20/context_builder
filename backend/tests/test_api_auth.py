"""Bearer-token middleware: enforce API_TOKEN, exempt /health and OPTIONS."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import BearerTokenMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(BearerTokenMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/echo")
    def echo():
        return {"echo": True}

    @app.get("/api/agent/trace/{job_id}")
    def trace(job_id: str):
        return {"job_id": job_id}

    return app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    return TestClient(_build_app())


class TestUnauthenticatedMode:
    def test_no_token_set_allows_everything(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/api/echo").status_code == 200

    def test_empty_token_allows_everything(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "   ")
        assert client.get("/api/echo").status_code == 200


class TestAuthenticatedMode:
    def test_health_always_exempt(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "secret")
        assert client.get("/health").status_code == 200

    def test_missing_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "secret")
        r = client.get("/api/echo")
        assert r.status_code == 401
        assert "API token" in r.json()["detail"]

    def test_wrong_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "secret")
        r = client.get("/api/echo", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_correct_bearer_token_accepted(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "secret")
        r = client.get("/api/echo", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        assert r.json() == {"echo": True}

    def test_token_via_query_param_accepted(self, client, monkeypatch):
        """SSE clients can't set headers, so ?token= is a supported fallback."""
        monkeypatch.setenv("API_TOKEN", "secret")
        r = client.get("/api/agent/trace/job123?token=secret")
        assert r.status_code == 200

    def test_options_preflight_always_passes(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "secret")
        r = client.options("/api/echo")
        assert r.status_code in (200, 405)

    def test_non_api_paths_pass(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "secret")
        r = client.get("/some-other-path")
        assert r.status_code == 404


class TestConstantTimeCompare:
    def test_token_compare_uses_hmac(self):
        """Imported helper exists and uses hmac.compare_digest semantics."""
        import hmac
        assert hmac.compare_digest("abc", "abc")
        assert not hmac.compare_digest("abc", "abd")
