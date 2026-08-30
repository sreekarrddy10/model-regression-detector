"""One retry policy, shared by everything that talks to a provider.

Transient provider failures - 429s above all - are the normal weather of a live
eval, not an exceptional condition. The runner has always retried them. This
module exists so that every other caller retries them the same way, rather than
each path inventing its own tolerance or, worse, having none.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .providers.base import ProviderError

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5

T = TypeVar("T")


async def with_retry(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = MAX_ATTEMPTS,
    backoff_base: float = BACKOFF_BASE_SECONDS,
    label: str,
) -> T:
    """Retry transient provider failures with exponential backoff."""
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    for attempt in range(attempts):
        try:
            return await factory()
        except ProviderError as exc:
            if attempt + 1 >= attempts:
                raise
            delay = backoff_base * (2**attempt)
            logger.warning("%s failed (attempt %d/%d): %s", label, attempt + 1, attempts, exc)
            await asyncio.sleep(delay)

    raise ProviderError(f"{label}: retry loop exhausted")  # unreachable
