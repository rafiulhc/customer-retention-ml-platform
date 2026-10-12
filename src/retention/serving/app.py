"""The online API: score one customer (or a small batch) with the champion model.

    uv run uvicorn retention.serving.app:app --port 8000      docs: http://localhost:8000/docs

The model is loaded once at startup. To serve a newly promoted champion, restart the service (a
rolling restart in production): every response says which model version produced it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, Response

from retention.models.champion import MODEL_NAME
from retention.serving.schemas import Batch, BatchPrediction, Customer, ModelInfo, Prediction, Scored
from retention.serving.service import ScoringService
from retention.tracking import connect

logger = logging.getLogger("retention.serving")


def _load() -> ScoringService:
    connect()
    return ScoringService.from_registry()


def create_app(load: Callable[[], ScoringService] = _load) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        # Fail fast: a server without a valid model must not accept traffic (the platform restarts it).
        app.state.service = load()
        logger.info(
            "Serving %s v%s (threshold %.2f)",
            MODEL_NAME,
            app.state.service.version,
            app.state.service.threshold,
        )
        yield

    app = FastAPI(
        title="Customer retention API",
        version="1.0",
        description="Churn risk, call decision and reasons for a customer, from the champion model.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        # Method, path, status and timing only: request bodies hold customer data and stay out of logs
        logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    def service(request: Request) -> ScoringService:
        svc: ScoringService = request.app.state.service
        return svc

    def score(svc: ScoringService, customers: list[Customer]) -> list[Scored]:  # type: ignore[valid-type]
        frame = pd.DataFrame([c.model_dump() for c in customers])  # type: ignore[attr-defined]
        scored = svc.score(frame)
        return [
            Scored(customer_id=cid, p_leaves=float(p), worth_calling=bool(call), reasons=list(why))
            for cid, p, call, why in zip(
                frame["customer_id"],
                scored["p_leaves"],
                scored["worth_calling"],
                scored["reasons"],
                strict=True,
            )
        ]

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "model_version": service(request).version}

    @app.get("/model", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        svc = service(request)
        return ModelInfo(
            name=MODEL_NAME,
            version=svc.version,
            threshold=svc.threshold,
            trained_on_commit=svc.tags.get("code.git_commit"),
            data_fingerprint=svc.tags.get("data.train_content_sha256"),
            evaluation_run_id=svc.tags.get("evaluation_run_id"),
            criteria_version=svc.tags.get("criteria_version"),
            passed_success_criteria=svc.tags.get("passed_success_criteria"),
            ship_decision=svc.tags.get("ship_decision"),
        )

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer, request: Request) -> Prediction:  # type: ignore[valid-type]
        svc = service(request)
        return Prediction(model_version=svc.version, **score(svc, [customer])[0].model_dump())

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(batch: Batch, request: Request) -> BatchPrediction:
        svc = service(request)
        return BatchPrediction(
            model_version=svc.version, threshold=svc.threshold, predictions=score(svc, batch.customers)
        )

    return app


app = create_app()
