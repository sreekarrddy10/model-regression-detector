"""Project resource discovery.

The package is imported from two very different places: a source checkout, where
everything sits beside `src/`, and an installed wheel inside a container, where
the code lives in `site-packages` and the data is mounted next to the working
directory. A path anchored only to `__file__` is correct in the first case and
silently wrong in the second - the container build caught exactly that, three
times, in pricing, prompts and the CLI defaults.

One resolver, one order of precedence, used everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/mrd/paths.py -> repo root
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def resolve(relative: str, *, env_var: str | None = None) -> Path:
    """Locate a project resource by relative path.

    Order of precedence:
      1. `env_var`, when set - the explicit override always wins
      2. the working directory, which is how both a source checkout run from the
         repo root and the container (WORKDIR holds the mounted data) find it
      3. the source tree, so imports work from anywhere in a checkout

    Returns the working-directory candidate when nothing exists yet, so error
    messages name the path a user would expect to create.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return Path(override)

    candidate = Path.cwd() / relative
    if candidate.exists():
        return candidate

    from_source = SOURCE_ROOT / relative
    if from_source.exists():
        return from_source

    return candidate
