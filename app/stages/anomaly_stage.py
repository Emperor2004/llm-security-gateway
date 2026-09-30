# anomaly stage
from __future__ import annotations

from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage
from app.rate_limit.anomaly import AnomalyDetector, anomaly_detector as default_detector


class AnomalyStage(PipelineStage):
    """Detects burst traffic and high-diversity model extraction probing."""

    name = "anomaly"

    def __init__(self, detector: AnomalyDetector | None = None, enabled: bool = True):
        super().__init__(name="anomaly", enabled=enabled)
        self.detector = detector or default_detector

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        principal_id = ctx.principal.id if ctx.principal else "anonymous"

        anomaly = self.detector.record_and_check(principal_id, ctx.prompt)
        ctx.metadata["anomaly_burst"] = anomaly.is_burst
        ctx.metadata["request_count_in_window"] = anomaly.request_count_in_window
        ctx.metadata["diversity_score"] = anomaly.diversity_score

        if anomaly.is_extraction_signal:
            ctx.block(
                reason="extraction_signal",
                status_code=429,
                detail="Traffic pattern flagged as potential extraction attempt",
                principal_id=principal_id,
                diversity_score=anomaly.diversity_score,
                request_count=anomaly.request_count_in_window,
            )
        return ctx
