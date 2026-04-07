"""
path_safety.py -- Shared path traversal protection.

Provides safe_resolve() used by both explore_tools.py and react_tools.py
to validate that agent-provided paths stay inside an allowed root directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_resolve(file_path: str, root: Path) -> Path | None:
    """Resolve *file_path* relative to *root*, rejecting path traversal.

    If the agent passes an absolute path that starts with *root* or a known
    sandbox prefix, auto-strip the prefix so the call succeeds instead of
    triggering a confusing 'Path traversal blocked' error.

    Returns the resolved ``Path`` on success, or ``None`` if the path escapes
    the root.
    """
    try:
        p = Path(file_path)
        if p.is_absolute():
            resolved_root = str(root.resolve())
            resolved_p = str(p.resolve())
            if resolved_p.startswith(resolved_root):
                file_path = resolved_p[len(resolved_root):].lstrip("/")
            elif "/agent_sandbox_" in str(p):
                # Heuristic: strip everything up to and including the sandbox dir
                # e.g. /tmp/agent_sandbox_flask_abc123/flask/app.py -> flask/app.py
                parts = str(p).split("/")
                sandbox_idx = next(
                    (i for i, part in enumerate(parts) if "agent_sandbox_" in part),
                    None,
                )
                if sandbox_idx is not None:
                    file_path = "/".join(parts[sandbox_idx + 1:])
                else:
                    logger.warning("Path traversal attempt blocked: %s", file_path)
                    return None
            else:
                logger.warning("Path traversal attempt blocked: %s", file_path)
                return None

        resolved = (root / file_path).resolve()
        if not str(resolved).startswith(str(root.resolve())):
            logger.warning("Path traversal attempt blocked: %s", file_path)
            return None
        return resolved
    except (OSError, ValueError) as e:
        logger.debug("Path resolution failed for %s: %s", file_path, e)
        return None


def safe_resolve_rglob(match: Path, root: Path) -> Path | None:
    """Validate that an rglob result is inside *root*."""
    try:
        resolved = match.resolve()
        if not str(resolved).startswith(str(root.resolve())):
            logger.warning("Path traversal (rglob) blocked: %s", match)
            return None
        return resolved
    except (OSError, ValueError) as e:
        logger.debug("Rglob path resolution failed for %s: %s", match, e)
        return None
