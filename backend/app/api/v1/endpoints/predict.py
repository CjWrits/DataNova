"""Network Intrusion Inference Endpoints (Binary, Multiclass, and Model Comparison)."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.flow import (
    BinaryPredictionRequest,
    BinaryPredictionResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    MulticlassPredictionRequest,
    MulticlassPredictionResponse,
)
from backend.app.services.model_service import ModelService

router = APIRouter()
logger = logging.getLogger("backend.api.v1.endpoints.predict")


@router.post(
    "/predict/binary",
    response_model=BinaryPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Binary Intrusion Detection (Benign vs. Attack)",
    description=(
        "Classifies network flows as Normal (0) or Attack (1) using the Phase 4 calibrated model. "
        "Supports operating threshold presets: 'default' (0.50), 'f1_optimal', 'cost_optimal', "
        "'low_false_alarm', or 'high_recall'."
    ),
)
def predict_binary(request: BinaryPredictionRequest) -> BinaryPredictionResponse:
    if len(request.flows) > settings.MAX_PREDICT_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size ({len(request.flows)}) exceeds maximum permitted limit "
                f"({settings.MAX_PREDICT_BATCH_SIZE})."
            ),
        )

    model_service = ModelService.get_instance()
    preset = request.threshold_preset or "default"
    try:
        return model_service.predict_binary(request.flows, threshold_preset=preset)
    except Exception:
        logger.exception("Inference failure in predict_binary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during binary inference.",
        )


@router.post(
    "/predict/multiclass",
    response_model=MulticlassPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Multiclass Attack Category Classification",
    description=(
        "Classifies flows into 10 distinct attack families (Normal, Generic, Exploits, Fuzzers, "
        "DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms). "
        "Supports explicit model selection between 'c8' (Primary w/ STTL) and 'c10' (No-STTL). "
        "Models are NEVER switched silently."
    ),
)
def predict_multiclass(request: MulticlassPredictionRequest) -> MulticlassPredictionResponse:
    if len(request.flows) > settings.MAX_PREDICT_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size ({len(request.flows)}) exceeds maximum permitted limit "
                f"({settings.MAX_PREDICT_BATCH_SIZE})."
            ),
        )

    model_service = ModelService.get_instance()
    model_id = (request.model_id or "c8").lower().strip()
    if model_id not in ("c8", "c10"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_id '{request.model_id}'. Must be explicitly 'c8' or 'c10'.",
        )

    try:
        return model_service.predict_multiclass(request.flows, model_id=model_id)
    except Exception:
        logger.exception("Inference failure in predict_multiclass")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during multiclass prediction.",
        )


@router.post(
    "/predict/compare",
    response_model=ModelComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Side-by-Side Model Comparison (C8 Primary vs. C10 Robustness)",
    description=(
        "Evaluates the exact same batch of flows through Candidate 8 (with STTL) and Candidate 10 (No-STTL). "
        "Returns individual predictions, confidences, agreement rate, and L1 probability divergence. "
        "Note: Model disagreement indicates feature-sensitivity variance, not proof of an attack."
    ),
)
def compare_models(request: ModelComparisonRequest) -> ModelComparisonResponse:
    if len(request.flows) > settings.MAX_PREDICT_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size ({len(request.flows)}) exceeds maximum permitted limit "
                f"({settings.MAX_PREDICT_BATCH_SIZE})."
            ),
        )

    model_service = ModelService.get_instance()
    try:
        return model_service.compare_models(request.flows)
    except Exception:
        logger.exception("Comparison failure in compare_models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during model comparison.",
        )
