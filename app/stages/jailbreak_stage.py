# jailbreak stage
from __future__ import annotations

from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage
from app.jailbreak.classifier import JailbreakClassifier, classifier as default_classifier


class JailbreakStage(PipelineStage):
    """Evaluates prompt against heuristic and ML jailbreak/prompt-injection models."""

    name = "jailbreak"

    def __init__(self, classifier: JailbreakClassifier | None = None, enabled: bool = True):
        super().__init__(name="jailbreak", enabled=enabled)
        self.classifier = classifier or default_classifier

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        jb_result = self.classifier.classify(ctx.prompt)
        ctx.metadata["jailbreak_score"] = jb_result.score
        ctx.metadata["jailbreak_method"] = jb_result.method
        ctx.metadata["matched_patterns"] = jb_result.matched_patterns

        if jb_result.is_jailbreak:
            principal_id = ctx.principal.id if ctx.principal else "anonymous"
            ctx.block(
                reason="jailbreak",
                status_code=400,
                detail="Request rejected by jailbreak/prompt-injection filter",
                principal_id=principal_id,
                score=jb_result.score,
                method=jb_result.method,
                matched_patterns=jb_result.matched_patterns,
            )
        return ctx
