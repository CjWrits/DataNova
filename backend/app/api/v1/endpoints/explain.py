"""SHAP Explainability and Model Attribution Endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.flow import ExplainRequest, ExplainResponse
from backend.app.services.explain_service import ExplainService
from backend.app.services.model_service import ModelService

router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="SHAP TreeExplainer Local Feature Attributions",
    description=(
        "Computes exact tree-based feature attributions for a small bounded batch of flows (max 10). "
        "Attributions are computed in raw log-odds margin logit units (pre-softmax) and aggregated back "
        "to parent features. Supports models 'c8' (default) and 'c10'."
    ),
)
def explain_predictions(request: ExplainRequest) -> ExplainResponse:
    if len(request.flows) > settings.MAX_EXPLAIN_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size ({len(request.flows)}) exceeds maximum permitted explanation limit "
                f"({settings.MAX_EXPLAIN_BATCH_SIZE}) to protect server memory."
            ),
        )

    model_service = ModelService.get_instance()
    explain_service = ExplainService.get_instance()

    resolved_id = (request.model_id or "c8").lower().strip()
    if resolved_id not in ("c8", "c10"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_id '{request.model_id}'. Must be explicitly 'c8' or 'c10'.",
        )

    if not model_service.is_loaded:
        model_service.load_models()

    mdata = model_service.models[resolved_id]
    bundle = mdata["bundle"]
    schema = mdata["schema_name"]
    top_k = request.top_k or 10

    try:
        return explain_service.explain_flows(
            flows=request.flows,
            bundle=bundle,
            model_id=resolved_id,
            model_name=mdata["name"],
            schema=schema,
            top_k=top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failure: {str(e)}",
        )
