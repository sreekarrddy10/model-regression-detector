#!/usr/bin/env python3
"""Container healthcheck.

Asserts rather than reports. A container that starts but cannot load its own
configuration is worse than one that fails, because it will run evals and record
every cost as "unavailable" without anyone noticing.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import mrd.cli  # noqa: F401
        from mrd.prompts import PromptConfig
        from mrd.providers import pricing
    except Exception as exc:  # pragma: no cover - exercised by the image, not tests
        print(f"unhealthy: package import failed: {exc}", file=sys.stderr)
        return 1

    path = pricing.default_path()
    if pricing.lookup("gpt-4o-mini", path=path) is None:
        print(f"unhealthy: no pricing for gpt-4o-mini at {path}", file=sys.stderr)
        return 1

    try:
        PromptConfig.latest()
    except Exception as exc:
        print(f"unhealthy: no loadable prompt version: {exc}", file=sys.stderr)
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
