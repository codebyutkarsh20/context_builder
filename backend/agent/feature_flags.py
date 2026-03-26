"""
feature_flags.py — Lightweight JSON-based feature flag system for AI Deploy Agent.

Step 19: Feature Flag Integration.
Each agent-generated PR gets a feature flag so changes can be toggled
without a redeploy.  Flags are stored per-repo in DATA_DIR/{repo}/feature_flags.json.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))


def _flags_path(repo_name: str) -> Path:
    return DATA_DIR / repo_name / "feature_flags.json"


def _load_flags(repo_name: str) -> list[dict[str, Any]]:
    path = _flags_path(repo_name)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read feature flags for %s: %s", repo_name, exc)
        return []


def _save_flags(repo_name: str, flags: list[dict[str, Any]]) -> None:
    path = _flags_path(repo_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(flags, indent=2))


def _slugify(text: str) -> str:
    """Turn arbitrary text into a safe flag-name slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80] if slug else "flag"


# ── Public API ────────────────────────────────────────────────────────────


def create_flag(
    repo_name: str,
    ticket_id: str,
    description: str,
    files_changed: list[str],
) -> str:
    """Create a new feature flag for an agent-generated change.

    Returns the generated flag name.
    """
    flag_name = f"fix_{_slugify(ticket_id)}_{_slugify(description)}"
    flags = _load_flags(repo_name)

    # Avoid duplicates — if a flag with the same name exists, return it.
    for f in flags:
        if f["name"] == flag_name:
            logger.info("Flag %s already exists for %s", flag_name, repo_name)
            return flag_name

    entry: dict[str, Any] = {
        "name": flag_name,
        "ticket_id": ticket_id,
        "description": description,
        "files_changed": files_changed,
        "enabled": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pr_url": "",
    }
    flags.append(entry)
    _save_flags(repo_name, flags)
    logger.info("Created feature flag %s for repo %s", flag_name, repo_name)
    return flag_name


def list_flags(repo_name: str) -> list[dict[str, Any]]:
    """Return all feature flags for a repo."""
    return _load_flags(repo_name)


def toggle_flag(repo_name: str, flag_name: str, enabled: bool) -> dict[str, Any] | None:
    """Enable or disable a feature flag. Returns updated flag or None if not found."""
    flags = _load_flags(repo_name)
    for f in flags:
        if f["name"] == flag_name:
            f["enabled"] = enabled
            _save_flags(repo_name, flags)
            logger.info("Toggled flag %s → %s (repo %s)", flag_name, enabled, repo_name)
            return f
    logger.warning("Flag %s not found for repo %s", flag_name, repo_name)
    return None


def get_flag(repo_name: str, flag_name: str) -> dict[str, Any] | None:
    """Get details for a single feature flag."""
    for f in _load_flags(repo_name):
        if f["name"] == flag_name:
            return f
    return None


def set_pr_url(repo_name: str, flag_name: str, pr_url: str) -> None:
    """Update the pr_url field after PR creation succeeds."""
    flags = _load_flags(repo_name)
    for f in flags:
        if f["name"] == flag_name:
            f["pr_url"] = pr_url
            _save_flags(repo_name, flags)
            return
    logger.warning("Flag %s not found for repo %s", flag_name, repo_name)
