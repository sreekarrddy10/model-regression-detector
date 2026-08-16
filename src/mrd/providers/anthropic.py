"""Anthropic provider.

Structured output is enforced with a forced tool call - Anthropic's native
mechanism - and the tool input is re-serialized to JSON text so that both
providers hand the feature layer the same thing: a JSON string. One parse path,
one schema, regardless of provider.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from . import pricing
from .base import Provider, ProviderError, Request, Response, Usage

_TOOL_NAME = "emit_classification"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - exercised by env, not tests
                raise ProviderError(
                    "anthropic package not installed. "
                    "Install with: uv pip install -e '.[providers]'"
                ) from exc
            key = self._api_key or os.environ["ANTHROPIC_API_KEY"]
            self._client = AsyncAnthropic(api_key=key)
        return self._client

    async def complete(self, request: Request) -> Response:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        if request.json_schema is not None:
            kwargs["tools"] = [
                {
                    "name": _TOOL_NAME,
                    "description": "Return the classification result.",
                    "input_schema": request.json_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}

        started = time.perf_counter()
        try:
            raw = await client.messages.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"anthropic call failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        return normalize(raw, request, latency_ms)


def normalize(raw: Any, request: Request, latency_ms: int) -> Response:
    """Map an Anthropic SDK message onto the normalized Response.

    A forced tool call yields a `tool_use` block whose `.input` is already a
    dict; it is dumped back to JSON so the feature layer sees the same JSON text
    it would get from OpenAI.
    """
    text = ""
    for block in raw.content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            text = json.dumps(block.input, sort_keys=True)
            break
        if block_type == "text":
            text = block.text

    usage = Usage(
        input_tokens=int(getattr(raw.usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw.usage, "output_tokens", 0) or 0),
    )
    return Response(
        text=text,
        model=request.model,
        provider="anthropic",
        usage=usage,
        latency_ms=latency_ms,
        cost_usd=pricing.cost_for(request.model, usage),
        fingerprint=request.fingerprint(),
    )
