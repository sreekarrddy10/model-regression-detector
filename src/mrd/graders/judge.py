"""LLM-as-judge for summary quality.

Runs through the same provider abstraction as the feature under test, so judge
calls are cassette-replayable offline and their tokens, latency and cost land in
the same accounting as everything else.

A judge verdict never blocks a merge on its own. It advises, and only blocks when
the same degradation reproduces across independent seeds - and only when the
judge has itself passed calibration against human scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..providers.base import Provider, Request, Response

SYSTEM_PROMPT = """\
You grade one-sentence summaries of customer support emails.

You are given the original email, a reference summary written by a human, and a
candidate summary produced by a model. Score the candidate 1-5.

5  Captures the customer's actual request as well as the reference. No invented
   detail, no omission that would change how someone triages it.
4  Accurate and useful, but slightly vaguer or wordier than the reference.
3  Broadly correct, yet omits something a responder would need, or includes a
   detail the email does not support.
2  Misleading: the main request is wrong, buried, or contradicted.
1  Unusable: hallucinated content, empty, or unrelated to the email.

Judge only the summary. Ignore the category label entirely. Do not reward
verbosity. A summary that differs in wording from the reference but conveys the
same request is a 5, not a 4.

Give a one-sentence rationale naming the specific evidence for your score.\
"""


class Verdict(BaseModel):
    """Structured judge output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, max_length=500)


def verdict_schema() -> dict[str, Any]:
    schema = Verdict.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = sorted(schema.get("properties", {}))
    schema.pop("title", None)
    return schema


@dataclass(frozen=True, slots=True)
class JudgeResult:
    verdict: Verdict | None
    parse_error: str | None
    response: Response


def build_prompt(email: str, reference: str, candidate: str) -> str:
    return (
        f"Email:\n{email}\n\n"
        f"Reference summary:\n{reference}\n\n"
        f"Candidate summary:\n{candidate}"
    )


def build_request(
    email: str, reference: str, candidate: str, *, model: str, seed_note: str = ""
) -> Request:
    """Build a judge request.

    `seed_note` varies the prompt across confirmation runs. Repeating an
    identical request would replay the same cassette and, at temperature 0,
    tends to reproduce the same verdict - which would make "confirmed across 3
    seeds" a measurement of nothing.
    """
    system = SYSTEM_PROMPT if not seed_note else f"{SYSTEM_PROMPT}\n\n{seed_note}"
    return Request(
        model=model,
        system=system,
        user=build_prompt(email, reference, candidate),
        temperature=0.0,
        max_tokens=256,
        json_schema=verdict_schema(),
    )


def parse(text: str) -> tuple[Verdict | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        return Verdict.model_validate(payload), None
    except ValidationError as exc:
        return None, f"schema violation: {exc.errors(include_url=False)}"


async def score_summary(
    email: str,
    reference: str,
    candidate: str,
    provider: Provider,
    *,
    model: str,
    seed_note: str = "",
) -> JudgeResult:
    request = build_request(email, reference, candidate, model=model, seed_note=seed_note)
    response = await provider.complete(request)
    verdict, parse_error = parse(response.text)
    return JudgeResult(verdict=verdict, parse_error=parse_error, response=response)
