"""Main FastAPI Application Entry Point for Adaptive Threat Intelligence Backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.api import api_router
from backend.app.core.config import settings
from backend.app.schemas.flow import HealthResponse
from backend.app.services.model_service import ModelService
from backend.app.services.monitoring_service import MonitoringService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: preloads models and reference distributions on startup."""
    logger.info("Initializing application lifecycle: loading models and reference data...")
    try:
        model_service = ModelService.get_instance()
        model_service.load_models()

        monitoring_service = MonitoringService.get_instance()
        monitoring_service.initialize_reference()
        logger.info("All model bundles and reference distributions successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Critical startup failure during artifact loading: {e}", exc_info=True)
        raise e

    yield

    logger.info("Application lifecycle shutdown: releasing cached resources.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-oriented REST API for Network Intrusion Detection, Model Interpretability, "
        "and Statistical Drift Monitoring. Exposes frozen binary and multiclass XGBoost models, "
        "SHAP feature attributions, and non-destructive adaptive response recommendations.\n\n"
        "**Operational Scope Note:** This service accepts pre-computed tabular flow telemetry "
        "records (42 features). It is not an inline packet capture daemon and does not perform live network sniffing."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware (stateless REST API: no cookies/credentials required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root Service Index
@app.get(
    "/",
    tags=["Health & Status"],
    summary="Root Service Index",
    description="Returns high-level service status and links to documentation.",
)
def root_index() -> dict:
    model_service = ModelService.get_instance()
    monitoring_service = MonitoringService.get_instance()
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "models_ready": model_service.is_loaded,
        "reference_data_ready": monitoring_service.is_loaded,
        "docs_url": "/docs",
        "health_url": "/health",
    }


# Root Health Check
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health & Status"],
    summary="Root Health Check",
    description="Quick health and readiness check indicating if the service and models are operational.",
)
def root_health() -> HealthResponse:
    model_service = ModelService.get_instance()
    monitoring_service = MonitoringService.get_instance()
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        models_ready=model_service.is_loaded,
        reference_data_ready=monitoring_service.is_loaded,
    )


# Attach API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


# ============================================================================
# Exception Handlers (Clean, Secure Error Formats)
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Sanitize and structure Pydantic validation errors."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append({"field": field, "message": msg})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error_type": "ValidationError",
            "message": "Input validation failed. Please check field formats and types.",
            "detail": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Format HTTP exceptions into structured error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": "HTTPException",
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled server errors while masking sensitive internal traces."""
    logger.error(f"Unhandled server exception: {exc}", exc_info=True)
    message = "An internal server error occurred while processing the request."
    if not settings.HIDE_INTERNAL_ERRORS:
        message = str(exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "InternalServerError",
            "message": message,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
