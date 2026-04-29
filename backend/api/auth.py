"""
Optional bearer-token auth for the public HTTP API.

Behavior:
  - If env var `API_TOKEN` is unset or empty, every request is allowed (back-compat
    with existing local-dev installs and pre-auth deployments).
  - If `API_TOKEN` is set, every request under `/api/*` must present a matching token
    via either the `Authorization: Bearer <token>` header OR a `?token=<token>` query
    parameter (the query form exists because `EventSource`-based SSE clients cannot
    set custom headers).
  - `/health`, `OPTIONS` preflights, and OpenAPI/docs paths are always exempt.

Why a single static token rather than full OAuth: the threat model is "stop random
internet scanners from running an agent on the host", not multi-tenant identity.
A single shared token is the smallest thing that achieves that and ships today.
For multi-user or multi-tenant deployments, replace this middleware with a real
auth layer (see SECURITY.md).
"""

from __future__ import annotations

import hmac
import os
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PROTECTED_PREFIX = "/api"
EXEMPT_PATHS: tuple[str, ...] = ("/health", "/openapi.json", "/docs", "/redoc")


def _current_token() -> str:
    return os.environ.get("API_TOKEN", "").strip()


def _extract_presented_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    qp = request.query_params.get("token")
    return qp.strip() if qp else None


def _is_exempt(path: str, exempt: Iterable[str]) -> bool:
    if not path.startswith(PROTECTED_PREFIX):
        return True
    for prefix in exempt:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Enforce `API_TOKEN` on `/api/*` endpoints when configured.

    No-op when `API_TOKEN` is unset/empty so that local-dev and existing
    `docker-compose up` workflows keep working without configuration."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        expected = _current_token()
        if not expected:
            return await call_next(request)

        if _is_exempt(request.url.path, EXEMPT_PATHS):
            return await call_next(request)

        presented = _extract_presented_token(request)
        if not presented or not hmac.compare_digest(presented, expected):
            return JSONResponse(
                {"detail": "Invalid or missing API token. Set `Authorization: Bearer <token>` or `?token=...`."},
                status_code=401,
            )

        return await call_next(request)
