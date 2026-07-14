"""Ephemeral Git authentication helpers.

GitHub tokens must never be written into remotes or repository configuration.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_ASKPASS_SCRIPT = """#!/bin/sh
case "$1" in
    *Username*) printf '%s\\n' "${GIT_USERNAME:-x-access-token}" ;;
    *Password*) printf '%s\\n' "$GIT_PASSWORD" ;;
    *) exit 1 ;;
esac
"""


@contextmanager
def git_push_environment(token: str) -> Iterator[dict[str, str]]:
    """Yield a push environment that authenticates without persisting a token."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if not token:
        yield env
        return

    fd, askpass_path = tempfile.mkstemp(prefix="context-builder-git-askpass-")
    path = Path(askpass_path)
    try:
        with os.fdopen(fd, "w") as script:
            script.write(_ASKPASS_SCRIPT)
        path.chmod(0o700)

        env.update({
            "GIT_ASKPASS": str(path),
            "GIT_USERNAME": "x-access-token",
            "GIT_PASSWORD": token,
            # Override global helpers for this process only so Git reaches askpass.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "",
        })
        yield env
    finally:
        path.unlink(missing_ok=True)
