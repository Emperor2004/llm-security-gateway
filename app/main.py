# main
"""
LLM Security Gateway — FastAPI entrypoint.

Request pipeline for POST /v1/chat, in order:
  1. authenticate (API key -> Principal) + authorize ("chat" permission)
  2. rate limit (token bucket) — cheapest check, runs first
  3. anomaly / extraction-signal detection (burst + prompt diversity)
  4. jailbreak classification — block if score >= threshold
  5. PII scrub the outbound prompt (redact before it leaves the gateway)
  6. call upstream LLM
  7. PII scrub the inbound response (redact before it reaches the caller)
  8. structured log of the whole event

Every rejection returns a distinct HTTP status + reason so a caller
(or the dashboard) can tell *why* a request was blocked instead of a
generic 4xx.
"""
from __future__ import annotations

import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth.rbac import AuthError, Principal, authenticate, authorize
from app.config import get_settings
from app.jailbreak.classifier import classifier
from app.llm.client import LLMError, llm_client
from app.logging.logger import log_event
from app.pii.scrubber import scrub
from app.rate_limit.anomaly import anomaly_detector
from app.rate_limit.limiter import limiter

app = FastAPI(title="LLM Security Gateway", version="0.1.0")


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    pii_redacted: bool = False


async def get_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    try:
        return authenticate(x_api_key)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, principal: Principal = Depends(get_principal)):
    settings = get_settings()
    start = time.monotonic()

    try:
        authorize(principal, "chat")
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    if not limiter.allow(principal.id):
        log_event("request_blocked", reason="rate_limited", principal_id=principal.id)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    anomaly = anomaly_detector.record_and_check(principal.id, body.prompt)
    if anomaly.is_extraction_signal:
        log_event(
            "request_blocked",
            reason="extraction_signal",
            principal_id=principal.id,
            diversity_score=anomaly.diversity_score,
            request_count=anomaly.request_count_in_window,
        )
        raise HTTPException(status_code=429, detail="Traffic pattern flagged as potential extraction attempt")

    jb_result = classifier.classify(body.prompt)
    if jb_result.is_jailbreak:
        log_event(
            "request_blocked",
            reason="jailbreak",
            principal_id=principal.id,
            score=jb_result.score,
            method=jb_result.method,
            matched_patterns=jb_result.matched_patterns,
        )
        raise HTTPException(status_code=400, detail="Request rejected by jailbreak/prompt-injection filter")

    outbound_scrub = scrub(body.prompt)
    if settings.pii_block_on_detect and outbound_scrub.has_pii:
        log_event("request_blocked", reason="pii_detected", principal_id=principal.id)
        raise HTTPException(status_code=400, detail="Request contains sensitive data and PII_BLOCK_ON_DETECT is enabled")

    try:
        llm_response = await llm_client.complete(outbound_scrub.text)
    except LLMError as e:
        log_event("upstream_error", principal_id=principal.id, error=str(e))
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}") from e

    inbound_scrub = scrub(llm_response.text)

    log_event(
        "request_completed",
        principal_id=principal.id,
        prompt=body.prompt,
        response=llm_response.text,
        pii_redacted=outbound_scrub.has_pii or inbound_scrub.has_pii,
        jailbreak_score=jb_result.score,
        latency_ms=round((time.monotonic() - start) * 1000, 2),
    )

    return ChatResponse(response=inbound_scrub.text, pii_redacted=outbound_scrub.has_pii or inbound_scrub.has_pii)


@app.get("/admin/stats")
async def admin_stats(principal: Principal = Depends(get_principal)):
    try:
        authorize(principal, "admin:view_stats")
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    return {
        "principal_id": principal.id,
        "tokens_remaining": limiter.remaining(principal.id),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
