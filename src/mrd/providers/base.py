"""Provider-agnostic request/response contract.

Every provider normalizes to the same `Response` so the eval engine can compare
runs across OpenAI and Anthropic without special-casing either. Token counts,
latency and cost are captured here rather than in the feature layer, because the
gate treats cost and latency drift as regressions (see docs/SPEC.md 3.4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Raised when a provider call fails for a non-retryable reason."""


@dataclass(frozen=True, slots=True)
class Request:
    """A single normalized completion request.

    `json_schema`, when set, instructs the provider to enforce structured output
    using whatever native mechanism it offers (OpenAI json_schema response format,
    Anthropic forced tool use).
    """

    model: str
    system: str
    user: str
    temperature: float = 0.0
    max_tokens: int = 512
    json_schema: dict[str, Any] | None = None

    def fingerprint(self) -> str:
        """Content address for this request.

        Cassette replay is keyed on this, so any change to the prompt, model or
        schema is a cache miss by construction - a stale recording can never be
        silently replayed against a changed prompt.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "system": self.system,
                "user": self.user,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "json_schema": self.json_schema,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Response:
    """Normalized completion response.

    `cost_usd` is None when the model is absent from config/pricing.yaml. The
    gate reports missing cost as "unavailable" rather than assuming zero, so an
    unpriced model cannot masquerade as a free one.
    """

    text: str
    model: str
    provider: str
    usage: Usage
    latency_ms: int
    cost_usd: float | None
    fingerprint: str


@runtime_checkable
class Provider(Protocol):
    """Minimal surface every provider implements."""

    name: str

    async def complete(self, request: Request) -> Response: ...
