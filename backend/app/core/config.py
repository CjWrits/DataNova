"""Backend configuration and environment settings for Adaptive Threat Intelligence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application configuration settings."""

    PROJECT_NAME: str = "Adaptive Threat Intelligence NIDS API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # CORS configuration
    CORS_ORIGINS: List[str] = ["*"]

    # Model Artifact Paths
    BINARY_MODEL_PATH: Path = (
        PROJECT_ROOT / "artifacts" / "models" / "phase4" / "binary_calibrated_winner.joblib"
    )
    C8_MULTICLASS_PATH: Path = (
        PROJECT_ROOT / "artifacts" / "models" / "phase4" / "multiclass_primary_winner.joblib"
    )
    C10_NOSTTL_PATH: Path = (
        PROJECT_ROOT / "artifacts" / "models" / "phase4" / "multiclass_nosttl_winner.joblib"
    )

    # Reference Dataset Path (Training Partition ONLY)
    REFERENCE_DATA_PATH: Path = (
        PROJECT_ROOT / "data" / "processed" / "unsw_nb15" / "multiclass" / "train.csv"
    )

    # Batch Size Safeguards
    MAX_PREDICT_BATCH_SIZE: int = 1000
    MAX_EXPLAIN_BATCH_SIZE: int = 10
    MIN_DRIFT_BATCH_SIZE: int = 30

    # Inference Presets
    DEFAULT_BINARY_THRESHOLD_PRESET: str = "default"
    DEFAULT_BINARY_THRESHOLD: float = 0.50

    # Safe error exposure: Hide raw file paths and system traces from responses
    HIDE_INTERNAL_ERRORS: bool = True

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()
