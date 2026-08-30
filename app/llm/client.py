# client
"""
Upstream LLM client. Speaks the OpenAI-compatible chat-completions
shape so it works against OpenAI, Azure OpenAI-compatible endpoints,
or any local/self-hosted server exposing the same API.

LLM_PROVIDER=mock (the default) returns a canned response with zero
network calls, so the gateway is runnable and testable without a real
upstream key.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import get_settings


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict | None = None


class LLMClient:
    def __init__(self):
        self._settings = get_settings()

    async def complete(self, prompt: str) -> LLMResponse:
        if self._settings.llm_provider == "mock":
            return self._mock_complete(prompt)
        return await self._http_complete(prompt)

    @property
    def _requires_api_key(self) -> bool:
        # Ollama's OpenAI-compatible endpoint ignores auth entirely.
        return self._settings.llm_provider != "ollama"

    def _mock_complete(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=f"[mock response] received {len(prompt)} chars. "
            "Set LLM_PROVIDER and LLM_BASE_URL/LLM_API_KEY to reach a real model.",
            model="mock",
        )

    async def _http_complete(self, prompt: str) -> LLMResponse:
        settings = self._settings
        if self._requires_api_key and not settings.llm_api_key:
            raise LLMError("LLM_API_KEY is not set but LLM_PROVIDER is not 'mock' or 'ollama'")

        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.llm_api_key or 'ollama'}"}
        payload = {
            "model": settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise LLMError(f"Upstream LLM request failed: {e}") from e

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected upstream response shape: {e}") from e

        return LLMResponse(text=text, model=settings.llm_model, raw=data)


llm_client = LLMClient()
