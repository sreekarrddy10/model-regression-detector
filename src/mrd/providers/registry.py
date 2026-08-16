"""Provider selection.

Routing is by model-name prefix, so the caller names a model and never a vendor.
The tier decides whether that model is reached live, replayed from a cassette, or
recorded on miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .anthropic import AnthropicProvider
from .base import Provider, ProviderError
from .cassette import CassetteProvider
from .openai import OpenAIProvider

Tier = Literal["unit", "record", "smoke", "full"]

_PREFIXES: tuple[tuple[tuple[str, ...], type[Provider]], ...] = (
    (("gpt-", "o1", "o3", "o4"), OpenAIProvider),
    (("claude-",), AnthropicProvider),
)


def resolve(model: str) -> Provider:
    """Return the live provider that serves `model`."""
    for prefixes, cls in _PREFIXES:
        if model.startswith(prefixes):
            return cls()
    known = ", ".join(p for prefixes, _ in _PREFIXES for p in prefixes)
    raise ProviderError(f"no provider registered for model {model!r}. Known prefixes: {known}")


def build(model: str, *, tier: Tier, cassette_dir: Path) -> Provider:
    """Return the provider appropriate to the run tier.

    unit    - replay only; a miss is an error, never a silent network call
    record  - replay, and record misses from the live provider
    smoke   - live provider
    full    - live provider
    """
    if tier == "unit":
        return CassetteProvider(cassette_dir)
    if tier == "record":
        return CassetteProvider(cassette_dir, record_with=resolve(model))
    return resolve(model)
