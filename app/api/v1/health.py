# health router
from __future__ import annotations

from fastapi import APIRouter
from app.schemas.chat import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Health check endpoint for liveness and readiness probes."""
    return HealthResponse(status="ok", version="0.2.0")
