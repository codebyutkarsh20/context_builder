"""
API endpoints for feature flag management (Step 19).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.feature_flags import get_flag, list_flags, toggle_flag

router = APIRouter(tags=["flags"])


class ToggleRequest(BaseModel):
    enabled: bool


@router.get("/flags/{repo}")
def api_list_flags(repo: str):
    """List all feature flags for a repo."""
    return {"repo": repo, "flags": list_flags(repo)}


@router.post("/flags/{repo}/{flag_name}/toggle")
def api_toggle_flag(repo: str, flag_name: str, body: ToggleRequest):
    """Toggle a feature flag on or off."""
    updated = toggle_flag(repo, flag_name, body.enabled)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_name}' not found in repo '{repo}'")
    return {"repo": repo, "flag": updated}


@router.get("/flags/{repo}/{flag_name}")
def api_get_flag(repo: str, flag_name: str):
    """Get details for a single feature flag."""
    flag = get_flag(repo, flag_name)
    if flag is None:
        raise HTTPException(status_code=404, detail=f"Flag '{flag_name}' not found in repo '{repo}'")
    return {"repo": repo, "flag": flag}
