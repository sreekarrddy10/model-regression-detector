"""Output contract for the feature under test."""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["billing", "technical", "account", "general"]

CATEGORIES: tuple[str, ...] = get_args(Category)


class Classification(BaseModel):
    """What the classifier must return for every email."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Category
    summary: str = Field(min_length=1, max_length=400)


def strict_schema() -> dict[str, Any]:
    """JSON Schema in the strict dialect both providers accept.

    OpenAI's strict json_schema mode requires `additionalProperties: false` and
    every property listed in `required`; Anthropic's tool input_schema accepts
    the same shape. Emitting one schema for both keeps the request fingerprint
    identical across providers where the prompt is identical.
    """
    schema = Classification.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = sorted(schema.get("properties", {}))
    schema.pop("title", None)
    # Strict mode on both providers rejects $ref. Pydantic inlines Literal enums
    # today; test_schema_is_strict_dialect fails loudly if a future version stops.
    return schema
