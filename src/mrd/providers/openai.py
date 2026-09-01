"""OpenAI provider.

The SDK is imported lazily inside `complete` so that the offline test tier runs
without `openai` installed and without an API key present.
"""

from __future__ import annotations

import os
import time
from typing import Any

from . import pricing
from .base import Provider, ProviderError, Request, Response, Usage


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - exercised by env, not tests
                raise ProviderError(
                    "openai package not installed. Install with: uv pip install -e '.[providers]'"
                ) from exc
            # KeyError on missing key is deliberate (ECC rules/python/security.md):
            # fail loudly at call time rather than sending an unauthenticated request.
            key = self._api_key or os.environ["OPENAI_API_KEY"]
            # max_retries=0 is load-bearing, not a preference. The SDK retries
            # internally with its own backoff, and that backoff lands inside the
            # perf_counter window below - so a rate-limited call was recorded as a
            # 16-second model response, and the gate treats latency drift as a
            # regression. mrd.retry is the one retry layer; the SDK must not add a
            # second one underneath it.
            self._client = AsyncOpenAI(api_key=key, max_retries=0)
        return self._client

    async def complete(self, request: Request) -> Response:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "strict": True,
                    "schema": request.json_schema,
                },
            }

        started = time.perf_counter()
        try:
            raw = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"openai call failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        return normalize(raw, request, latency_ms)


def normalize(raw: Any, request: Request, latency_ms: int) -> Response:
    """Map an OpenAI SDK completion onto the normalized Response.

    Split out from `complete` so the mapping can be contract-tested against a
    canned SDK payload with no network and no SDK installed.
    """
    text = raw.choices[0].message.content or ""
    usage = Usage(
        input_tokens=int(getattr(raw.usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw.usage, "completion_tokens", 0) or 0),
    )
    return Response(
        text=text,
        model=request.model,
        provider="openai",
        usage=usage,
        latency_ms=latency_ms,
        cost_usd=pricing.cost_for(request.model, usage),
        fingerprint=request.fingerprint(),
    )
