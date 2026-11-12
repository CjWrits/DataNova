"""Centralized Model Management and Inference Service.

Loads, manages, and executes predictions for:
1. Binary Intrusion Detection (Phase 4 Calibrated Winner)
2. Multiclass Attack Family Classifier (Candidate 8 Primary Winner)
3. Multiclass Robustness Classifier (Candidate 10 No-STTL Winner)

Enforces strict immutability: models are loaded once during startup and never retrained,
recalibrated, or mutated during runtime.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.flow import (
    BinaryPredictionItem,
    BinaryPredictionResponse,
    ModelComparisonItem,
    ModelComparisonResponse,
    ModelInfo,
    ModelsListResponse,
    MulticlassPredictionItem,
    MulticlassPredictionResponse,
    NetworkFlowRecord,
)
from ml.explainability.shap_explainer import extract_features_and_transform
from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    BINARY_LABEL_MAPPING,
    MULTICLASS_LABEL_MAPPING,
    PRIMARY_FEATURE_COLUMNS,
)

logger = logging.getLogger("backend.services.model_service")


class ModelService:
    """Singleton service for loading model bundles and executing inference."""

    _instance: Optional[ModelService] = None

    def __init__(self) -> None:
        self.is_loaded: bool = False
        self.models: Dict[str, Dict[str, Any]] = {}
        self.classes: List[str] = list(MULTICLASS_LABEL_MAPPING.keys())
        self.start_time: float = time.time()

    @classmethod
    def get_instance(cls) -> ModelService:
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load_models(self) -> None:
        """Load and validate all model artifacts into memory."""
        logger.info("Initializing ModelService: loading serialized model artifacts...")

        # 1. Binary Model (Phase 4 Calibrated Winner)
        bin_path = settings.BINARY_MODEL_PATH
        if not bin_path.exists():
            # Fallback to Phase 3 primary binary if phase4 artifact not present
            fallback = bin_path.parent.parent / "binary_primary_xgboost.joblib"
            logger.warning(f"Binary model artifact {bin_path} not found. Attempting fallback: {fallback}")
            bin_path = fallback

        if not bin_path.exists():
            raise FileNotFoundError(f"Binary model artifact not found at {bin_path}")

        bin_bundle = joblib.load(bin_path)
        self.models["binary"] = {
            "id": "binary_winner",
            "name": bin_bundle.get("name", "Binary Calibrated XGBoost Winner"),
            "version": "4.0.0",
            "task_type": "binary_detection",
            "schema_name": "primary",
            "feature_count": 42,
            "bundle": bin_bundle,
            "threshold_points": bin_bundle.get("opt_points", {}),
            "description": "Cost-sensitive calibrated XGBoost for high-recall benign vs. attack discrimination.",
            "is_default": True,
        }

        # 2. Multiclass Primary Winner (Candidate 8)
        c8_path = settings.C8_MULTICLASS_PATH
        if not c8_path.exists():
            raise FileNotFoundError(f"Candidate 8 model artifact not found at {c8_path}")

        c8_bundle = joblib.load(c8_path)
        self.models["c8"] = {
            "id": "c8",
            "name": c8_bundle.get("name", "Candidate 8: FE + Cost-Sensitive Sqrt"),
            "version": "4.0.0",
            "task_type": "multiclass_classification",
            "schema_name": "primary",
            "feature_count": 42,
            "bundle": c8_bundle,
            "description": "Primary multiclass winner with domain feature engineering and square-root class weighting.",
            "is_default": True,
        }

        # 3. Multiclass No-STTL Alternative Winner (Candidate 10)
        c10_path = settings.C10_NOSTTL_PATH
        if not c10_path.exists():
            raise FileNotFoundError(f"Candidate 10 model artifact not found at {c10_path}")

        c10_bundle = joblib.load(c10_path)
        self.models["c10"] = {
            "id": "c10",
            "name": c10_bundle.get("name", "Candidate 10: Robustness Track No-STTL"),
            "version": "4.0.0",
            "task_type": "multiclass_classification",
            "schema_name": "alternative_no_sttl",
            "feature_count": 41,
            "bundle": c10_bundle,
            "description": "Robustness-oriented multiclass winner excluding synthetic STTL artifact.",
            "is_default": False,
        }

        self.is_loaded = True
        logger.info(f"ModelService successfully loaded {len(self.models)} models: {list(self.models.keys())}")

    def list_models(self) -> ModelsListResponse:
        """Expose catalog of available loaded models."""
        infos: List[ModelInfo] = []
        for mid, mdata in self.models.items():
            threshold_info = None
            if mid == "binary":
                opt_points = mdata.get("threshold_points", {})
                threshold_info = {
                    name: round(float(pt.threshold), 4) for name, pt in opt_points.items()
                } if isinstance(opt_points, dict) else {"default": 0.50}

            infos.append(
                ModelInfo(
                    id=mdata["id"],
                    name=mdata["name"],
                    version=mdata["version"],
                    task_type=mdata["task_type"],
                    schema_name=mdata["schema_name"],
                    feature_count=mdata["feature_count"],
                    description=mdata["description"],
                    is_default=mdata["is_default"],
                    threshold_info=threshold_info,
                )
            )
        return ModelsListResponse(models=infos, count=len(infos))

    def _flows_to_dataframe(self, flows: List[NetworkFlowRecord]) -> pd.DataFrame:
        """Convert validated Pydantic flow records to a pandas DataFrame."""
        records = [f.model_dump() for f in flows]
        return pd.DataFrame(records)

    def predict_binary(
        self,
        flows: List[NetworkFlowRecord],
        threshold_preset: str = "default",
    ) -> BinaryPredictionResponse:
        """Execute binary intrusion detection with configurable threshold preset."""
        t0 = time.perf_counter()
        if not self.is_loaded:
            self.load_models()

        mdata = self.models["binary"]
        bundle = mdata["bundle"]
        opt_points = mdata.get("threshold_points", {})

        # Resolve threshold from preset
        resolved_threshold = 0.50
        if isinstance(opt_points, dict) and threshold_preset in opt_points:
            pt = opt_points[threshold_preset]
            resolved_threshold = float(getattr(pt, "threshold", 0.50))
        elif threshold_preset == "default":
            resolved_threshold = 0.50

        df = self._flows_to_dataframe(flows)
        X_proc, _, _, _ = extract_features_and_transform(bundle, df, schema="primary")

        # Inference
        probs = bundle["model"].predict_proba(X_proc)
        attack_probs = probs[:, 1]
        preds = (attack_probs >= resolved_threshold).astype(int)

        items: List[BinaryPredictionItem] = []
        for i, (p, prob) in enumerate(zip(preds, attack_probs)):
            items.append(
                BinaryPredictionItem(
                    index=i,
                    label=int(p),
                    label_name=BINARY_LABEL_MAPPING.get(int(p), "Unknown"),
                    attack_probability=round(float(prob), 4),
                    threshold_applied=round(resolved_threshold, 4),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BinaryPredictionResponse(
            model_id=mdata["id"],
            model_name=mdata["name"],
            threshold_preset=threshold_preset,
            threshold_applied=round(resolved_threshold, 4),
            predictions=items,
            flow_count=len(items),
            processing_time_ms=round(elapsed_ms, 2),
        )

    def predict_multiclass(
        self,
        flows: List[NetworkFlowRecord],
        model_id: str = "c8",
    ) -> MulticlassPredictionResponse:
        """Execute multiclass attack category prediction on C8 or C10."""
        t0 = time.perf_counter()
        if not self.is_loaded:
            self.load_models()

        resolved_id = model_id.lower().strip()
        if resolved_id not in ("c8", "c10"):
            raise ValueError(f"Invalid model_id '{model_id}'. Must be explicitly 'c8' or 'c10'.")

        mdata = self.models[resolved_id]
        bundle = mdata["bundle"]
        schema = mdata["schema_name"]

        df = self._flows_to_dataframe(flows)
        X_proc, _, _, _ = extract_features_and_transform(bundle, df, schema=schema)

        preds = bundle["model"].predict(X_proc)
        probs = bundle["model"].predict_proba(X_proc)

        items: List[MulticlassPredictionItem] = []
        for i, (pred_idx, row_probs) in enumerate(zip(preds, probs)):
            class_name = self.classes[int(pred_idx)] if int(pred_idx) < len(self.classes) else "Unknown"
            conf = float(np.max(row_probs))
            prob_dict = {
                self.classes[c]: round(float(row_probs[c]), 4)
                for c in range(min(len(self.classes), len(row_probs)))
            }
            items.append(
                MulticlassPredictionItem(
                    index=i,
                    predicted_class_id=int(pred_idx),
                    predicted_class_name=class_name,
                    confidence=round(conf, 4),
                    probabilities=prob_dict,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return MulticlassPredictionResponse(
            model_id=mdata["id"],
            model_name=mdata["name"],
            schema_used=schema,
            predictions=items,
            flow_count=len(items),
            processing_time_ms=round(elapsed_ms, 2),
        )

    def compare_models(self, flows: List[NetworkFlowRecord]) -> ModelComparisonResponse:
        """Run flows simultaneously through C8 and C10 to evaluate agreement and divergence."""
        t0 = time.perf_counter()
        if not self.is_loaded:
            self.load_models()

        df = self._flows_to_dataframe(flows)

        # C8 Inference
        b8 = self.models["c8"]["bundle"]
        X8, _, _, _ = extract_features_and_transform(b8, df, schema="primary")
        preds_c8 = b8["model"].predict(X8)
        probs_c8 = b8["model"].predict_proba(X8)

        # C10 Inference
        b10 = self.models["c10"]["bundle"]
        X10, _, _, _ = extract_features_and_transform(b10, df, schema="alternative_no_sttl")
        preds_c10 = b10["model"].predict(X10)
        probs_c10 = b10["model"].predict_proba(X10)

        items: List[ModelComparisonItem] = []
        agree_count = 0

        for i in range(len(flows)):
            c8_class = self.classes[int(preds_c8[i])]
            c10_class = self.classes[int(preds_c10[i])]
            c8_conf = float(np.max(probs_c8[i]))
            c10_conf = float(np.max(probs_c10[i]))
            agreement = bool(preds_c8[i] == preds_c10[i])
            if agreement:
                agree_count += 1

            # L1 divergence across probability vectors
            l1_dist = float(np.sum(np.abs(probs_c8[i] - probs_c10[i])))

            items.append(
                ModelComparisonItem(
                    index=i,
                    c8_predicted_class=c8_class,
                    c8_confidence=round(c8_conf, 4),
                    c10_predicted_class=c10_class,
                    c10_confidence=round(c10_conf, 4),
                    agreement=agreement,
                    l1_distribution_divergence=round(l1_dist, 4),
                )
            )

        n = len(flows)
        disagree_count = n - agree_count
        agree_rate = float(agree_count / n) if n > 0 else 1.0
        disagree_rate = float(disagree_count / n) if n > 0 else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ModelComparisonResponse(
            total_flows=n,
            agreement_count=agree_count,
            disagreement_count=disagree_count,
            agreement_rate=round(agree_rate, 4),
            disagreement_rate=round(disagree_rate, 4),
            comparison_items=items,
            c8_model_name=self.models["c8"]["name"],
            c10_model_name=self.models["c10"]["name"],
            processing_time_ms=round(elapsed_ms, 2),
        )
