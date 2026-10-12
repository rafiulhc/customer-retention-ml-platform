# The pipeline image: training + pipeline dependencies, for the scheduled retrain and call-list tasks.
#   docker build -f infra/pipeline/Dockerfile -t retention-pipeline .
# Flows run without a Prefect server here (Prefect's ephemeral mode); their logs go to CloudWatch.

FROM ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 AS uv

# ---------- build: install into /app/.venv (the uv download cache stays in this stage) ----------
FROM python:3.12.15-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-group dev --no-install-project
COPY src ./src
# Editable install: the project root stays /app, so data/ and reports/ are written there
RUN uv sync --frozen --no-group dev

# ---------- runtime ----------
FROM python:3.12.15-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
# OpenMP runtime: LightGBM and XGBoost use it for multi-threaded training
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 10001 --create-home app
COPY --from=build --chown=app:app /app /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PREFECT_HOME=/home/app/.prefect
USER app
WORKDIR /app
CMD ["python", "-m", "retention.pipeline.flows"]
