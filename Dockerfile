# Multi-stage: the runtime image carries no build tooling and no dev
# dependencies, so a container that runs evals cannot also run a linter, a
# compiler, or anything else that widens the blast radius.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install '.[providers]'


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:${PATH}"

# Non-root. The container reads a dataset and posts to Slack; nothing it does
# needs root, so nothing it does gets root.
RUN useradd --create-home --uid 10001 mrd

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=mrd:mrd config ./config
COPY --chown=mrd:mrd prompts ./prompts
COPY --chown=mrd:mrd data ./data

USER mrd

# Proves the package imports and the pricing config parses. A container that
# starts but cannot load its own configuration is worse than one that fails.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import mrd, mrd.cli; from mrd.providers import pricing; \
        pricing.lookup('gpt-4o-mini'); print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "mrd.cli"]
CMD ["eval", "--tier", "smoke"]
