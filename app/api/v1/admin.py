# admin router
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.auth.rbac import Principal, authorize
from app.rate_limit.limiter import limiter
from app.schemas.chat import AdminStatsResponse

router = APIRouter(tags=["Admin"])


async def get_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    from app.auth.rbac import AuthError, authenticate
    try:
        return authenticate(x_api_key)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@router.get("/admin/stats", response_model=AdminStatsResponse)
async def admin_stats(principal: Principal = Depends(get_principal)):
    """Retrieve gateway statistics and remaining rate limit tokens for principal."""
    try:
        authorize(principal, "admin:view_stats")
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    return AdminStatsResponse(
        principal_id=principal.id,
        tokens_remaining=limiter.remaining(principal.id),
    )
