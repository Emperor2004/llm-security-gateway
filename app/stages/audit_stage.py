# audit stage
from __future__ import annotations

from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage
from app.logging.logger import log_event


class AuditStage(PipelineStage):
    """Structured audit logging of all requests, blocks, and responses."""

    name = "audit"

    def __init__(self, enabled: bool = True):
        super().__init__(name="audit", enabled=enabled, always_run=True)

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        # If blocked at request stage, log the block immediately
        if ctx.is_blocked:
            principal_id = ctx.principal.id if ctx.principal else "anonymous"
            metadata = dict(ctx.metadata)
            metadata.pop("principal_id", None)
            log_event(
                "request_blocked",
                reason=ctx.block_reason,
                principal_id=principal_id,
                **metadata,
            )
        return ctx

    async def process_response(self, ctx: SecurityContext) -> SecurityContext:
        principal_id = ctx.principal.id if ctx.principal else "anonymous"

        if ctx.is_blocked:
            metadata = dict(ctx.metadata)
            metadata.pop("principal_id", None)
            log_event(
                "request_blocked",
                reason=ctx.block_reason,
                principal_id=principal_id,
                **metadata,
            )
            return ctx

        log_event(
            "request_completed",
            principal_id=principal_id,
            prompt=ctx.original_prompt,
            response=ctx.response_text or "",
            pii_redacted=ctx.pii_redacted,
            jailbreak_score=ctx.metadata.get("jailbreak_score", 0.0),
            latency_ms=ctx.latency_ms,
        )
        return ctx
