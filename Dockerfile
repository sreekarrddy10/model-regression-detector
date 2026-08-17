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

# Proves the package imports AND that pricing actually resolved. An earlier
# version called lookup() without checking the result - it printed "ok" while
# the config was unreachable, which is precisely the failure this is here to
# catch. A healthcheck that cannot fail is decoration.
COPY --chown=mrd:mrd docker/healthcheck.py /app/healthcheck.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

ENTRYPOINT ["python", "-m", "mrd.cli"]
CMD ["eval", "--tier", "smoke"]
