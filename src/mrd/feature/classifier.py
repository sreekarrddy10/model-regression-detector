"""The LLM feature under test: support email -> {category, summary}.

`classify` never raises on a bad model output. A malformed response is data the
eval engine needs (schema validity is a blocking code grader, docs/SPEC.md 3.2),
so it is captured as `parse_error` and returned alongside the raw response rather
than thrown away as an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from ..prompts import PromptConfig
from ..providers.base import Provider, Request, Response
from .schema import Classification, strict_schema


@dataclass(frozen=True, slots=True)
class Outcome:
    """One classification attempt, successful or not."""

    classification: Classification | None
    parse_error: str | None
    response: Response

    @property
    def ok(self) -> bool:
        return self.classification is not None


def build_request(email: str, prompt: PromptConfig) -> Request:
    return Request(
        model=prompt.model,
        system=prompt.render_system(),
        user=email,
        temperature=prompt.temperature,
        max_tokens=prompt.max_tokens,
        json_schema=strict_schema(),
    )


def parse(text: str) -> tuple[Classification | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        return Classification.model_validate(payload), None
    except ValidationError as exc:
        return None, f"schema violation: {exc.errors(include_url=False)}"


async def classify(email: str, prompt: PromptConfig, provider: Provider) -> Outcome:
    response = await provider.complete(build_request(email, prompt))
    classification, parse_error = parse(response.text)
    return Outcome(classification=classification, parse_error=parse_error, response=response)
