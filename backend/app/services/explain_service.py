"""Centralized SHAP Feature Attribution Service.

Computes exact tree-based feature attributions for bounded batches of network flows
using shap.TreeExplainer on frozen XGBoost tree ensembles.

Enforces:
1. Memory Bounds: Maximum batch size of 10 samples per request to prevent OOM.
2. Raw Log-Odds Margins: Explanations are explicitly computed in logit margin space.
3. Feature Aggregation: One-hot encoded dummy variables are aggregated back to their parent features.
4. Model Immutability: Models are never modified or retrained during explanation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from backend.app.core.config import settings
from backend.app.schemas.flow import (
    ExplainResponse,
    FeatureAttributionItem,
    FlowExplanationItem,
    NetworkFlowRecord,
)
from ml.explainability.shap_explainer import extract_features_and_transform
from ml.preprocessing.schema import MULTICLASS_LABEL_MAPPING

logger = logging.getLogger("backend.services.explain_service")


class ExplainService:
    """Singleton service for generating feature attributions."""

    _instance: Optional[ExplainService] = None

    def __init__(self) -> None:
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self.classes: List[str] = list(MULTICLASS_LABEL_MAPPING.keys())

    @classmethod
    def get_instance(cls) -> ExplainService:
        if cls._instance is None:
            cls._instance = ExplainService()
        return cls._instance

    def _get_explainer(self, model_id: str, bundle: Dict[str, Any]) -> shap.TreeExplainer:
        """Get or initialize cached TreeExplainer for a model."""
        if model_id not in self.explainers:
            logger.info(f"Initializing SHAP TreeExplainer for model '{model_id}'...")
            xgb_model = bundle["model"]
            explainer = shap.TreeExplainer(xgb_model)
            self.explainers[model_id] = explainer
        return self.explainers[model_id]

    def explain_flows(
        self,
        flows: List[NetworkFlowRecord],
        bundle: Dict[str, Any],
        model_id: str = "c8",
        model_name: str = "Candidate 8",
        schema: str = "primary",
        top_k: int = 10,
    ) -> ExplainResponse:
        """Generate SHAP feature attributions for a small bounded batch of flows."""
        t0 = time.perf_counter()

        if len(flows) > settings.MAX_EXPLAIN_BATCH_SIZE:
            raise ValueError(
                f"Batch size ({len(flows)}) exceeds maximum permitted explanation limit "
                f"({settings.MAX_EXPLAIN_BATCH_SIZE}). Please submit smaller batches."
            )

        df = pd.DataFrame([f.model_dump() for f in flows])
        X_proc, trans_names, feat_map, unique_parents = extract_features_and_transform(
            bundle, df, schema=schema
        )

        # Get explainer
        explainer = self._get_explainer(model_id, bundle)
        shap_values = explainer.shap_values(X_proc)

        # Base values (expected margins)
        expected_values = explainer.expected_value
        if isinstance(expected_values, (list, np.ndarray)):
            base_values = np.array(expected_values)
        else:
            base_values = np.array([expected_values] * len(self.classes))

        # Model predictions
        preds = bundle["model"].predict(X_proc)

        explanations: List[FlowExplanationItem] = []

        # For each flow
        for i in range(len(flows)):
            pred_class_idx = int(preds[i])
            pred_class_name = self.classes[pred_class_idx] if pred_class_idx < len(self.classes) else "Unknown"
            flow_base = float(base_values[pred_class_idx]) if pred_class_idx < len(base_values) else 0.0

            # SHAP attribution array for predicted class
            # Shape handling: shap_values can be (N, num_features, num_classes) or list of (N, num_features)
            if isinstance(shap_values, list):
                class_shap = shap_values[pred_class_idx][i]
            elif shap_values.ndim == 3:
                # Shape is (N, num_features, num_classes) or (N, num_classes, num_features)
                if shap_values.shape[2] == len(self.classes):
                    class_shap = shap_values[i, :, pred_class_idx]
                else:
                    class_shap = shap_values[i, pred_class_idx, :]
            else:
                class_shap = shap_values[i]

            # Aggregate to parent features
            parent_attributions: Dict[str, float] = {p: 0.0 for p in unique_parents}
            for trans_col_idx, trans_name in enumerate(trans_names):
                parent_col = feat_map.get(trans_name, trans_name)
                parent_attributions[parent_col] = parent_attributions.get(parent_col, 0.0) + float(class_shap[trans_col_idx])

            # Rank by absolute magnitude
            sorted_parents = sorted(
                parent_attributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:top_k]

            attr_items: List[FeatureAttributionItem] = []
            for feat_name, attr_val in sorted_parents:
                direction = "push_towards_class" if attr_val >= 0 else "push_away_from_class"
                attr_items.append(
                    FeatureAttributionItem(
                        feature_name=feat_name,
                        attribution_margin=round(float(attr_val), 4),
                        direction=direction,
                    )
                )

            explanations.append(
                FlowExplanationItem(
                    index=i,
                    predicted_class=pred_class_name,
                    output_space="raw log-odds margin logit units",
                    base_margin=round(flow_base, 4),
                    top_attributions=attr_items,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ExplainResponse(
            model_id=model_id,
            model_name=model_name,
            output_space_description="Attributions represent additive shifts in raw pre-softmax log-odds margins.",
            explanations=explanations,
            flow_count=len(explanations),
            processing_time_ms=round(elapsed_ms, 2),
        )
