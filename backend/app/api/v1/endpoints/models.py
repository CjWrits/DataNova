"""Model Catalog and Inventory Endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from backend.app.schemas.flow import ModelsListResponse
from backend.app.services.model_service import ModelService

router = APIRouter()


@router.get(
    "/models",
    response_model=ModelsListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Loaded NIDS Models and Schemas",
    description="Returns metadata for all frozen models currently loaded in memory, including feature schemas and task types.",
)
def list_models() -> ModelsListResponse:
    model_service = ModelService.get_instance()
    if not model_service.is_loaded:
        model_service.load_models()
    return model_service.list_models()
