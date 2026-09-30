# security context
"""
SecurityContext: Request/Response state passed across pipeline stages.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.auth.rbac import Principal


@dataclass
class SecurityContext:
    """Encapsulates the state of a single request passing through the gateway pipeline."""

    prompt: str
    original_prompt: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    principal: Principal | None = None
    response_text: str | None = None

    # Pipeline block verdict
    is_blocked: bool = False
    block_reason: str | None = None
    block_status_code: int = 400
    block_detail: str | None = None

    # Flags & Telemetry
    pii_redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if not self.original_prompt:
            self.original_prompt = self.prompt

    @property
    def latency_ms(self) -> float:
        return round((time.monotonic() - self.start_time) * 1000, 2)

    def block(self, reason: str, status_code: int, detail: str, **metadata: Any) -> None:
        """Mark context as rejected and short-circuit subsequent stages."""
        self.is_blocked = True
        self.block_reason = reason
        self.block_status_code = status_code
        self.block_detail = detail
        if metadata:
            self.metadata.update(metadata)
