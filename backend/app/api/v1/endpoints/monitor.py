"""Statistical Drift and Data Quality Telemetry Monitoring Endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.flow import DriftMonitoringRequest, DriftMonitoringResponse
from backend.app.services.model_service import ModelService
from backend.app.services.monitoring_service import MonitoringService

router = APIRouter()


@router.post(
    "/monitor/drift",
    response_model=DriftMonitoringResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Telemetry Batch for Feature Drift and Pipeline Quality",
    description=(
        "Computes Population Stability Index (PSI), Kolmogorov-Smirnov test divergence, "
        "missingness / non-finite rates, and novel uncataloged categories against the baseline "
        "training distribution (151,325 rows). Automatically generates rule-based operational "
        "guidance for SOC teams without triggering unverified retraining."
    ),
)
def monitor_drift(request: DriftMonitoringRequest) -> DriftMonitoringResponse:
    if len(request.flows) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Flow telemetry batch cannot be empty.",
        )

    monitoring_service = MonitoringService.get_instance()
    model_service = ModelService.get_instance()

    c8_bundle = model_service.models.get("c8", {}).get("bundle") if model_service.is_loaded else None
    c10_bundle = model_service.models.get("c10", {}).get("bundle") if model_service.is_loaded else None

    try:
        return monitoring_service.evaluate_telemetry_batch(
            raw_flows=request.flows,
            batch_id=request.batch_id or "api_batch_001",
            c8_bundle=c8_bundle,
            c10_bundle=c10_bundle,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift monitoring computation failure: {str(e)}",
        )
