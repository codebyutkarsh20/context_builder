"""
Tests for the Repos API — Phase 2.1

Verifies:
  - GET /api/repos returns repo_path
  - repo_path sourced from graph.json stats
  - Health endpoint
"""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://localhost:8000"


def api(method, path, **kwargs):
    return getattr(requests, method)(f"{BASE}{path}", timeout=10, **kwargs)


class TestHealthEndpoint:
    """GET /health returns basic status."""

    def test_health_ok(self):
        resp = api("get", "/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "neo4j" in data


class TestListRepos:
    """GET /api/repos lists analyzed repositories."""

    def test_returns_list(self):
        resp = api("get", "/api/repos")
        assert resp.status_code == 200
        repos = resp.json()
        assert isinstance(repos, list)

    def test_repos_have_name(self):
        repos = api("get", "/api/repos").json()
        for repo in repos:
            assert "name" in repo
            assert isinstance(repo["name"], str)
            assert len(repo["name"]) > 0

    def test_repos_have_context_flags(self):
        repos = api("get", "/api/repos").json()
        for repo in repos:
            assert "has_context" in repo
            assert "has_summary" in repo

    def test_context_builder_has_repo_path(self):
        """context_builder (this repo) should have a valid repo_path.
        Accepts any name that starts with 'context_builder' (e.g. context_builder_test).
        """
        repos = api("get", "/api/repos").json()
        cb = next((r for r in repos if r["name"].startswith("context_builder")), None)
        assert cb is not None, "context_builder repo should be in repos (found: " + str([r["name"] for r in repos]) + ")"
        assert "repo_path" in cb
        assert cb["repo_path"] != ""
        assert Path(cb["repo_path"]).exists(), f"repo_path should exist: {cb['repo_path']}"

    def test_repo_path_is_absolute(self):
        """repo_path should be an absolute filesystem path."""
        repos = api("get", "/api/repos").json()
        for repo in repos:
            if "repo_path" in repo and repo["repo_path"]:
                assert repo["repo_path"].startswith("/"), f"Should be absolute: {repo['repo_path']}"


class TestAnalyzeEndpoint:
    """POST /api/analyze validates inputs."""

    def test_rejects_nonexistent_path(self):
        resp = api("post", "/api/analyze", json={
            "repo_path": "/nonexistent/path/to/repo",
        })
        assert resp.status_code == 400

    def test_rejects_file_path(self):
        resp = api("post", "/api/analyze", json={
            "repo_path": "/etc/hosts",  # file, not directory
        })
        assert resp.status_code == 400
