# stages package
from __future__ import annotations

from app.config import Settings, get_settings
from app.core.pipeline import SecurityPipeline
from app.stages.anomaly_stage import AnomalyStage
from app.stages.audit_stage import AuditStage
from app.stages.auth_stage import AuthStage
from app.stages.jailbreak_stage import JailbreakStage
from app.stages.pii_stage import PIIStage
from app.stages.rate_limit_stage import RateLimitStage


def create_default_pipeline(settings: Settings | None = None) -> SecurityPipeline:
    """Instantiate and configure the default security pipeline based on active settings."""
    cfg = settings or get_settings()

    stages = [
        AuthStage(required_permission="chat", enabled=cfg.stage_auth_enabled),
        RateLimitStage(enabled=cfg.stage_rate_limit_enabled),
        AnomalyStage(enabled=cfg.stage_anomaly_enabled),
        JailbreakStage(enabled=cfg.stage_jailbreak_enabled),
        PIIStage(block_on_detect=cfg.pii_block_on_detect, enabled=cfg.stage_pii_enabled),
        AuditStage(enabled=cfg.stage_audit_enabled),
    ]

    return SecurityPipeline(stages)


__all__ = [
    "AuthStage",
    "RateLimitStage",
    "AnomalyStage",
    "JailbreakStage",
    "PIIStage",
    "AuditStage",
    "create_default_pipeline",
]
