# test pipeline engine
import pytest
from app.auth.rbac import Principal
from app.core.context import SecurityContext
from app.core.pipeline import PipelineStage, SecurityPipeline
from app.stages.auth_stage import AuthStage
from app.stages.jailbreak_stage import JailbreakStage
from app.stages.pii_stage import PIIStage
from app.stages.rate_limit_stage import RateLimitStage


class MockCustomStage(PipelineStage):
    name = "mock_custom"

    async def process_request(self, ctx: SecurityContext) -> SecurityContext:
        ctx.metadata["custom_stage_executed"] = True
        if "evil" in ctx.prompt:
            ctx.block(reason="custom_blocked", status_code=400, detail="Custom block rule triggered")
        return ctx

    async def process_response(self, ctx: SecurityContext) -> SecurityContext:
        if ctx.response_text:
            ctx.response_text = f"PREFIX: {ctx.response_text}"
        return ctx


@pytest.mark.anyio
async def test_pipeline_executes_stages_sequentially():
    custom_stage = MockCustomStage()
    pipeline = SecurityPipeline([custom_stage])

    ctx = SecurityContext(prompt="hello world")
    result_ctx = await pipeline.execute_request(ctx)

    assert not result_ctx.is_blocked
    assert result_ctx.metadata.get("custom_stage_executed") is True


@pytest.mark.anyio
async def test_pipeline_short_circuits_on_block():
    custom_stage = MockCustomStage()
    pii_stage = PIIStage()

    pipeline = SecurityPipeline([custom_stage, pii_stage])

    ctx = SecurityContext(prompt="something evil with jane.doe@example.com")
    result_ctx = await pipeline.execute_request(ctx)

    assert result_ctx.is_blocked
    assert result_ctx.block_reason == "custom_blocked"
    # pii_stage should NOT have altered prompt because pipeline stopped
    assert "jane.doe@example.com" in result_ctx.prompt


@pytest.mark.anyio
async def test_pipeline_disabled_stage_is_skipped():
    custom_stage = MockCustomStage(enabled=False)
    pipeline = SecurityPipeline([custom_stage])

    ctx = SecurityContext(prompt="something evil")
    result_ctx = await pipeline.execute_request(ctx)

    # Evil prompt is not blocked because stage is disabled!
    assert not result_ctx.is_blocked
    assert "custom_stage_executed" not in result_ctx.metadata


@pytest.mark.anyio
async def test_pipeline_add_remove_stages():
    pipeline = SecurityPipeline()
    stage1 = MockCustomStage(name="stage1")
    stage2 = MockCustomStage(name="stage2")

    pipeline.add_stage(stage1)
    pipeline.add_stage(stage2)
    assert len(pipeline.stages) == 2
    assert pipeline.get_stage("stage1") is stage1

    pipeline.remove_stage("stage1")
    assert len(pipeline.stages) == 1
    assert pipeline.get_stage("stage1") is None
    assert pipeline.get_stage("stage2") is stage2


@pytest.mark.anyio
async def test_pipeline_response_processing():
    custom_stage = MockCustomStage()
    pipeline = SecurityPipeline([custom_stage])

    ctx = SecurityContext(prompt="hello")
    ctx.response_text = "original answer"
    result_ctx = await pipeline.execute_response(ctx)

    assert result_ctx.response_text == "PREFIX: original answer"


@pytest.mark.anyio
async def test_pipeline_always_run_stage_executes_on_block():
    custom_blocking_stage = MockCustomStage(name="blocking_stage")
    always_run_stage = MockCustomStage(name="telemetry_stage", always_run=True)
    normal_stage = MockCustomStage(name="normal_stage")

    pipeline = SecurityPipeline([custom_blocking_stage, normal_stage, always_run_stage])

    ctx = SecurityContext(prompt="something evil that triggers block")
    result_ctx = await pipeline.execute_request(ctx)

    assert result_ctx.is_blocked
    # Normal stage should NOT have executed
    assert "normal_stage" not in result_ctx.metadata
    # always_run stage MUST have executed
    assert result_ctx.metadata.get("custom_stage_executed") is True

