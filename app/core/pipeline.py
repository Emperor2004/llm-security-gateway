# security pipeline
"""
Modular Pipeline Engine: Pluggable interceptor chain for LLM requests & responses.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from app.core.context import SecurityContext

logger = logging.getLogger("llm_gateway.pipeline")


class PipelineStage(ABC):
    """Abstract base class for all security pipeline stages."""

    name: str = "base_stage"
    enabled: bool = True
    always_run: bool = False

    def __init__(self, name: str | None = None, enabled: bool = True, always_run: bool = False):
        if name:
            self.name = name
        self.enabled = enabled
        self.always_run = always_run

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        """Inspect or mutate outbound prompt before LLM invocation.

        To block a request, call ctx.block(reason=..., status_code=..., detail=...).
        """
        return ctx

    async def process_response(self, ctx: SecurityContext) -> SecurityContext:
        """Inspect or mutate inbound completion from upstream LLM."""
        return ctx

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} enabled={self.enabled} always_run={self.always_run}>"


class SecurityPipeline:
    """Manages ordered execution of defensive stages."""

    def __init__(self, stages: Sequence[PipelineStage] | None = None):
        self._stages: list[PipelineStage] = list(stages or [])

    @property
    def stages(self) -> list[PipelineStage]:
        return list(self._stages)

    def add_stage(self, stage: PipelineStage, index: int | None = None) -> None:
        """Register a stage dynamically."""
        if index is None:
            self._stages.append(stage)
        else:
            self._stages.insert(index, stage)

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name."""
        initial_len = len(self._stages)
        self._stages = [s for s in self._stages if s.name != name]
        return len(self._stages) < initial_len

    def get_stage(self, name: str) -> PipelineStage | None:
        """Retrieve a registered stage by name."""
        for s in self._stages:
            if s.name == name:
                return s
        return None

    async def execute_request(self, ctx: SecurityContext) -> SecurityContext:
        """Run request through all enabled stages sequentially until completion or rejection.

        When a stage blocks the context, subsequent normal inspection stages are skipped,
        but stages marked with always_run=True (such as AuditStage) will still execute.
        """
        for stage in self._stages:
            if not stage.enabled:
                continue
            if ctx.is_blocked and not stage.always_run:
                continue
            try:
                ctx = await stage.process_request(ctx)
            except Exception as e:
                logger.exception("Error executing request stage %s: %s", stage.name, e)
                raise
        return ctx

    async def execute_response(self, ctx: SecurityContext) -> SecurityContext:
        """Run model completion through all enabled response stages."""
        for stage in self._stages:
            if not stage.enabled:
                continue
            try:
                ctx = await stage.process_response(ctx)
            except Exception as e:
                logger.exception("Error executing response stage %s: %s", stage.name, e)
                raise
        return ctx
