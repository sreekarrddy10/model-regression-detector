#!/usr/bin/env python3
"""Seed the offline cassette tier with deterministic stub responses.

These stubs let `pytest -m unit` run with no API key and no network on a fresh
clone. They are plumbing fixtures, not model behavior - once keys are available,
`make record` overwrites them with genuine recorded provider responses, and the
fingerprint key guarantees a recording can only ever replay for the exact prompt
it was captured against.

Usage:
    python scripts/seed_cassettes.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mrd.feature.classifier import build_request  # noqa: E402
from mrd.prompts import PromptConfig  # noqa: E402
from mrd.providers.base import Provider, Request, Response, Usage  # noqa: E402
from mrd.providers.cassette import CassetteProvider  # noqa: E402
from tests.fixtures import FIXTURE_EMAILS  # noqa: E402

CASSETTE_DIR = ROOT / "tests" / "cassettes"


class StubProvider(Provider):
    """Returns a canned answer for each fixture email. Deterministic by design."""

    name = "stub"

    def __init__(self) -> None:
        self._answers = {f.email: f for f in FIXTURE_EMAILS}

    async def complete(self, request: Request) -> Response:
        fixture = self._answers[request.user]
        if fixture.malformed:
            text = "I'm not sure how to categorise this one, sorry!"
        else:
            text = json.dumps(
                {"category": fixture.category, "summary": fixture.summary}, sort_keys=True
            )
        usage = Usage(input_tokens=len(request.system) // 4, output_tokens=len(text) // 4)
        return Response(
            text=text,
            model=request.model,
            provider="stub",
            usage=usage,
            latency_ms=0,
            cost_usd=None,
            fingerprint=request.fingerprint(),
        )


async def main() -> int:
    prompt = PromptConfig.load("v001", root=ROOT / "prompts")
    provider = CassetteProvider(CASSETTE_DIR, record_with=StubProvider())

    for fixture in FIXTURE_EMAILS:
        request = build_request(fixture.email, prompt)
        path = CASSETTE_DIR / f"{request.fingerprint()}.json"
        if path.exists():
            path.unlink()
        await provider.complete(request)
        print(f"  {fixture.key:28s} -> {path.name}")

    print(f"\nSeeded {len(FIXTURE_EMAILS)} cassettes in {CASSETTE_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
