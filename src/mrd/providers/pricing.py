"""Token pricing, loaded from config rather than hardcoded in source.

Provider prices change without notice. Baking them into a module guarantees the
cost dimension of the gate silently goes wrong; keeping them in a versioned
config file makes a price update a reviewable diff.

Unknown models return None, which the eval engine records as "cost unavailable".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .base import Usage

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "pricing.yaml"


@dataclass(frozen=True, slots=True)
class Price:
    """USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


@lru_cache(maxsize=1)
def _load(path: str) -> dict[str, Price]:
    file = Path(path)
    if not file.exists():
        logger.warning("pricing file not found at %s; all costs will be unavailable", file)
        return {}
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    models = raw.get("models") or {}
    return {
        name: Price(
            input_per_mtok=float(entry["input_per_mtok"]),
            output_per_mtok=float(entry["output_per_mtok"]),
        )
        for name, entry in models.items()
    }


def lookup(model: str, *, path: Path | None = None) -> Price | None:
    prices = _load(str(path or _DEFAULT_PATH))
    price = prices.get(model)
    if price is None:
        logger.warning(
            "no price configured for model %r; cost will be recorded as unavailable", model
        )
    return price


def cost_for(model: str, usage: Usage, *, path: Path | None = None) -> float | None:
    price = lookup(model, path=path)
    return None if price is None else price.cost(usage)
