# chat router
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.auth.rbac import AuthError, Principal, authenticate
from app.core.context import SecurityContext
from app.core.pipeline import SecurityPipeline
from app.llm.client import LLMError, llm_client
from app.logging.logger import log_event
from app.schemas.chat import ChatRequest, ChatResponse
from app.stages import create_default_pipeline

router = APIRouter(tags=["Chat"])

# Default pipeline singleton
pipeline: SecurityPipeline = create_default_pipeline()


async def get_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    """Dependency to authenticate API key from request headers."""
    try:
        return authenticate(x_api_key)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    principal: Principal = Depends(get_principal),
):
    """Enforce defense-in-depth pipeline and proxy completion to upstream LLM."""
    ctx = SecurityContext(prompt=body.prompt, principal=principal)

    # 1. Execute outbound defensive stages
    ctx = await pipeline.execute_request(ctx)
    if ctx.is_blocked:
        raise HTTPException(
            status_code=ctx.block_status_code,
            detail=ctx.block_detail or "Request rejected by security policy",
        )

    # 2. Invoke upstream LLM
    try:
        llm_response = await llm_client.complete(ctx.prompt)
    except LLMError as e:
        log_event("upstream_error", principal_id=principal.id, error=str(e))
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}") from e

    ctx.response_text = llm_response.text

    # 3. Execute inbound defensive stages
    ctx = await pipeline.execute_response(ctx)

    return ChatResponse(
        response=ctx.response_text or "",
        pii_redacted=ctx.pii_redacted,
    )
