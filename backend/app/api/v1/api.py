"""API v1 router aggregator."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1.endpoints import explain, health, models, monitor, predict

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Status"])
api_router.include_router(models.router, tags=["Models & Registry"])
api_router.include_router(predict.router, tags=["Inference & Prediction"])
api_router.include_router(monitor.router, tags=["Drift & Data Quality Monitoring"])
api_router.include_router(explain.router, tags=["Explainability & Attributions"])
