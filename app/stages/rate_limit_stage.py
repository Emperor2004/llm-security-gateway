# rate limit stage
from __future__ import annotations

from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage
from app.rate_limit.limiter import TokenBucketLimiter, limiter as default_limiter


class RateLimitStage(PipelineStage):
    """Enforces token bucket rate limiting per principal."""

    name = "rate_limit"

    def __init__(self, limiter: TokenBucketLimiter | None = None, enabled: bool = True):
        super().__init__(name="rate_limit", enabled=enabled)
        self.limiter = limiter or default_limiter

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        principal_id = ctx.principal.id if ctx.principal else "anonymous"

        if not self.limiter.allow(principal_id):
            ctx.block(
                reason="rate_limited",
                status_code=429,
                detail="Rate limit exceeded",
                principal_id=principal_id,
            )
        return ctx
