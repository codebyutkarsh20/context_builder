"""Regression tests for non-persistent GitHub authentication."""

import os
import stat
from pathlib import Path
from unittest.mock import patch

from agent.git_auth import git_push_environment


def test_token_uses_ephemeral_askpass_without_embedding_secret():
    token = "ghp_test_token_that_must_not_touch_disk"

    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        with git_push_environment(token) as env:
            askpass = Path(env["GIT_ASKPASS"])
            assert askpass.exists()
            assert stat.S_IMODE(askpass.stat().st_mode) == 0o700
            assert token not in askpass.read_text()
            assert env["GIT_PASSWORD"] == token
            assert env["GIT_TERMINAL_PROMPT"] == "0"
            assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
            assert env["GIT_CONFIG_VALUE_0"] == ""

        assert not askpass.exists()


def test_missing_token_does_not_install_askpass():
    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
        with git_push_environment("") as env:
            assert env["GIT_TERMINAL_PROMPT"] == "0"
            assert "GIT_ASKPASS" not in env
            assert "GIT_PASSWORD" not in env


def test_pipeline_sources_never_embed_tokens_in_git_configuration():
    agent_dir = Path(__file__).resolve().parents[1] / "agent"
    source = (agent_dir / "react_pipeline.py").read_text()
    assert "x-access-token:{gh_token}" not in source
    assert "auth_url" not in source
    assert "insteadOf" not in source
    assert "git_push_environment" in source
