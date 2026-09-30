# auth stage
from __future__ import annotations

from app.auth.rbac import authorize
from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage


class AuthStage(PipelineStage):
    """Enforces authorization scopes on the authenticated principal."""

    name = "auth"

    def __init__(self, required_permission: str = "chat", enabled: bool = True):
        super().__init__(name="auth", enabled=enabled)
        self.required_permission = required_permission

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        if not ctx.principal:
            ctx.block(
                reason="unauthorized",
                status_code=401,
                detail="Missing or unauthenticated principal",
            )
            return ctx

        try:
            authorize(ctx.principal, self.required_permission)
        except Exception as e:
            ctx.block(
                reason="forbidden",
                status_code=403,
                detail=str(e),
            )
        return ctx
