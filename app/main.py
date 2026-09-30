# main
"""
LLM Security Gateway — Modular FastAPI Entrypoint.

Architectural Request Pipeline for POST /v1/chat:
  Client Request
    │
    ▼
  [AuthStage]            -> Validates X-API-Key & "chat" permission scope
    │
    ▼
  [RateLimitStage]       -> Token bucket limiter (cheapest check first)
    │
    ▼
  [AnomalyStage]         -> Burst rate & lexical Jaccard diversity extraction signal
    │
    ▼
  [JailbreakStage]       -> Heuristic regex + lazy DistilBERT classification
    │
    ▼
  [PIIStage (outbound)]  -> Redacts sensitive PII before forwarding to upstream
    │
    ▼
  [Upstream LLM]         -> Mock / Ollama / OpenAI-compatible endpoint
    │
    ▼
  [PIIStage (inbound)]   -> Redacts sensitive PII from model completion
    │
    ▼
  [AuditStage]           -> Writes structured JSONL event with redacted text
    │
    ▼
  Client Response
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.api.v1.admin import admin_stats
from app.api.v1.chat import chat, get_principal, pipeline
from app.api.v1.health import healthz
from app.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse

settings = get_settings()

app = FastAPI(
    title="LLM Security Gateway",
    version="0.2.0",
    description="Enterprise-grade defense-in-depth security gateway for LLMs",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modular API routers
app.include_router(api_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure all exceptions follow standard JSON error schema."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


__all__ = [
    "app",
    "chat",
    "healthz",
    "admin_stats",
    "get_principal",
    "pipeline",
    "ChatRequest",
    "ChatResponse",
]
