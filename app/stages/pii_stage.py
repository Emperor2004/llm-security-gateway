# pii stage
from __future__ import annotations

from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage
from app.pii.scrubber import scrub


class PIIStage(PipelineStage):
    """Scrubs sensitive data (emails, cards, SSNs, phones, IPs, API keys) on request & response."""

    name = "pii"

    def __init__(self, block_on_detect: bool = False, enabled: bool = True):
        super().__init__(name="pii", enabled=enabled)
        self.block_on_detect = block_on_detect

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        outbound_scrub = scrub(ctx.prompt)
        ctx.metadata["outbound_has_pii"] = outbound_scrub.has_pii
        ctx.metadata["outbound_pii_findings"] = outbound_scrub.findings

        if outbound_scrub.has_pii:
            ctx.pii_redacted = True
            if self.block_on_detect:
                principal_id = ctx.principal.id if ctx.principal else "anonymous"
                ctx.block(
                    reason="pii_detected",
                    status_code=400,
                    detail="Request contains sensitive data and PII_BLOCK_ON_DETECT is enabled",
                    principal_id=principal_id,
                )
                return ctx

        # Update prompt with sanitized text for upstream LLM call
        ctx.prompt = outbound_scrub.text
        return ctx

    async def process_response(self, ctx: SecurityContext) -> SecurityContext:
        if not ctx.response_text:
            return ctx

        inbound_scrub = scrub(ctx.response_text)
        ctx.metadata["inbound_has_pii"] = inbound_scrub.has_pii
        ctx.metadata["inbound_pii_findings"] = inbound_scrub.findings

        if inbound_scrub.has_pii:
            ctx.pii_redacted = True

        ctx.response_text = inbound_scrub.text
        return ctx
