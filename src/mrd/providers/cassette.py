"""Cassette provider - record once, replay forever, zero network.

This is ECC's sandbox-mode pattern (skills/ai-regression-testing) applied to LLM
calls. It exists so the harness's own logic can be tested with no API spend and
no flake: only the smoke and full tiers touch a real provider.

A cassette is keyed on `Request.fingerprint()`, so editing a prompt invalidates
its recording by construction. A stale recording cannot be replayed against a
changed prompt - the lookup simply misses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .base import Provider, Request, Response, Usage

logger = logging.getLogger(__name__)


class CassetteMiss(RuntimeError):
    """No recording exists for this request."""

    def __init__(self, request: Request, path: Path) -> None:
        super().__init__(
            f"No cassette for fingerprint {request.fingerprint()} "
            f"(model={request.model}). Expected: {path}\n"
            f"Record it with:  make record"
        )
        self.request = request
        self.path = path


class CassetteProvider(Provider):
    """Replays recorded responses; optionally records misses from a live provider."""

    name = "cassette"

    def __init__(self, cassette_dir: Path, record_with: Provider | None = None) -> None:
        self._dir = cassette_dir
        self._record_with = record_with

    def _path(self, request: Request) -> Path:
        return self._dir / f"{request.fingerprint()}.json"

    async def complete(self, request: Request) -> Response:
        path = self._path(request)
        if path.exists():
            return self._load(path)

        if self._record_with is None:
            raise CassetteMiss(request, path)

        response = await self._record_with.complete(request)
        self._save(path, request, response)
        logger.info("recorded cassette %s", path.name)
        return response

    @staticmethod
    def _load(path: Path) -> Response:
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data["response"]
        return Response(
            text=payload["text"],
            model=payload["model"],
            provider=payload["provider"],
            usage=Usage(
                input_tokens=payload["usage"]["input_tokens"],
                output_tokens=payload["usage"]["output_tokens"],
            ),
            # Replay latency is meaningless and must not pollute the latency
            # dimension of the gate; the recorded value is kept for reference only.
            latency_ms=payload["latency_ms"],
            cost_usd=payload["cost_usd"],
            fingerprint=payload["fingerprint"],
        )

    @staticmethod
    def _save(path: Path, request: Request, response: Response) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The request is echoed into the file so a cassette diff is human-reviewable.
        document = {"request": asdict(request), "response": asdict(response)}
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
