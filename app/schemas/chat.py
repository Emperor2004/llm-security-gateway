# chat schemas
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Client request payload for chat completion."""
    prompt: str = Field(..., description="User input prompt text", min_length=1)


class ChatResponse(BaseModel):
    """Client response payload containing completion text and security flags."""
    response: str = Field(..., description="Processed completion text from upstream LLM")
    pii_redacted: bool = Field(default=False, description="Whether sensitive data was scrubbed")


class HealthResponse(BaseModel):
    """Service health check response."""
    status: str = Field(default="ok", description="Gateway operational status")
    version: str = Field(default="0.2.0", description="Gateway version")


class AdminStatsResponse(BaseModel):
    """Admin statistics response for authenticated principal."""
    principal_id: str
    tokens_remaining: float
