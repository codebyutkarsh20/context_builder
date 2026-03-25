import re
from fastapi import HTTPException

_SAFE_REPO_NAME = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def validate_repo_name(repo: str) -> str:
    """Sanitize repo name to prevent path traversal."""
    if not _SAFE_REPO_NAME.match(repo) or ".." in repo:
        raise HTTPException(status_code=400, detail="Invalid repository name")
    return repo
