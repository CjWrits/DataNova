"""Health and System Status Endpoints."""

from __future__ import annotations

import os
import time
from fastapi import APIRouter, status

from backend.app.core.config import settings
from backend.app.schemas.flow import HealthResponse, SystemStatusResponse
from backend.app.services.model_service import ModelService
from backend.app.services.monitoring_service import MonitoringService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health and Readiness Check",
    description="Returns high-level service status and distinguishes general availability from model/reference readiness.",
)
def get_health() -> HealthResponse:
    model_service = ModelService.get_instance()
    monitoring_service = MonitoringService.get_instance()

    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        models_ready=model_service.is_loaded,
        reference_data_ready=monitoring_service.is_loaded,
    )


@router.get(
    "/system/status",
    response_model=SystemStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed System Status and Runtime Telemetry",
    description="Exposes detailed operational telemetry including uptime, memory footprint, and loaded models.",
)
def get_system_status() -> SystemStatusResponse:
    model_service = ModelService.get_instance()
    monitoring_service = MonitoringService.get_instance()

    uptime = time.time() - model_service.start_time

    # Measure memory usage if psutil is available
    memory_mb = 0.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        memory_mb = -1.0

    return SystemStatusResponse(
        status="operational" if model_service.is_loaded else "initializing",
        uptime_seconds=round(uptime, 2),
        models_loaded=list(model_service.models.keys()),
        reference_data_loaded=monitoring_service.is_loaded,
        reference_rows=monitoring_service.ref_row_count,
        environment=settings.ENVIRONMENT,
        memory_rss_mb=memory_mb,
        active_binary_threshold=settings.DEFAULT_BINARY_THRESHOLD,
    )
