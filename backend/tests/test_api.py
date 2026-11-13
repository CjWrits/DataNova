"""Comprehensive Test Suite for the FastAPI Backend Inference and Monitoring Service."""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.flow import NetworkFlowRecord
from backend.app.services.model_service import ModelService
from backend.app.services.monitoring_service import MonitoringService
from ml.explainability.shap_explainer import extract_features_and_transform


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient with triggered startup and shutdown lifespan events."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_valid_flow_dict() -> dict:
    """Returns a valid dictionary representing a single benign network flow."""
    return {
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "dur": 0.1214,
        "spkts": 10.0,
        "dpkts": 8.0,
        "sbytes": 2520.0,
        "dbytes": 1040.0,
        "rate": 140.0,
        "sttl": 62.0,
        "dttl": 252.0,
        "sload": 150000.0,
        "dload": 65000.0,
        "sloss": 0.0,
        "dloss": 0.0,
        "sinpkt": 12.14,
        "dinpkt": 14.28,
        "sjit": 1.5,
        "djit": 1.2,
        "swin": 255.0,
        "stcpb": 624100.0,
        "dtcpb": 812400.0,
        "dwin": 255.0,
        "tcprtt": 0.0012,
        "synack": 0.0006,
        "ackdat": 0.0006,
        "smean": 252.0,
        "dmean": 130.0,
        "trans_depth": 1.0,
        "response_body_len": 512.0,
        "ct_srv_src": 2.0,
        "ct_state_ttl": 0.0,
        "ct_dst_ltm": 3.0,
        "ct_src_dport_ltm": 1.0,
        "ct_dst_sport_ltm": 1.0,
        "ct_dst_src_ltm": 2.0,
        "is_ftp_login": 0.0,
        "ct_ftp_cmd": 0.0,
        "ct_flw_http_mthd": 1.0,
        "ct_src_ltm": 2.0,
        "ct_srv_dst": 2.0,
        "is_sm_ips_ports": 0.0,
    }


# ============================================================================
# 1. Health and System Status Tests
# ============================================================================

class TestHealthAndStatus:
    def test_root_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_ready"] is True
        assert data["reference_data_ready"] is True
        assert data["version"] == settings.VERSION

    def test_api_v1_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_ready"] is True

    def test_system_status_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert data["uptime_seconds"] > 0
        assert "binary" in data["models_loaded"]
        assert "c8" in data["models_loaded"]
        assert "c10" in data["models_loaded"]
        assert data["reference_data_loaded"] is True
        assert data["reference_rows"] > 100000


# ============================================================================
# 2. Model Catalog Tests
# ============================================================================

class TestModelCatalog:
    def test_list_models_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 3
        model_ids = [m["id"] for m in data["models"]]
        assert "binary_winner" in model_ids
        assert "c8" in model_ids
        assert "c10" in model_ids

        # Check C8 schema vs C10 schema
        c8_info = next(m for m in data["models"] if m["id"] == "c8")
        c10_info = next(m for m in data["models"] if m["id"] == "c10")
        assert c8_info["schema_name"] == "primary"
        assert c8_info["feature_count"] == 42
        assert c10_info["schema_name"] == "alternative_no_sttl"
        assert c10_info["feature_count"] == 41


# ============================================================================
# 3. Binary Inference Tests
# ============================================================================

class TestBinaryInference:
    def test_binary_prediction_default_threshold(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        model_service = ModelService.get_instance()
        expected_default = float(
            model_service.models["binary"]["threshold_points"]["default"].threshold
        ) if model_service.is_loaded and "default" in model_service.models["binary"]["threshold_points"] else 0.50

        payload = {"flows": [sample_valid_flow_dict], "threshold_preset": "default"}
        response = client.post("/api/v1/predict/binary", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["flow_count"] == 1
        assert data["threshold_applied"] == pytest.approx(expected_default, abs=1e-4)
        pred = data["predictions"][0]
        assert pred["label"] in (0, 1)
        assert pred["label_name"] in ("Normal", "Attack")
        assert 0.0 <= pred["attack_probability"] <= 1.0

    def test_binary_prediction_threshold_presets(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        for preset in ["f1_optimal", "cost_optimal", "low_false_alarm", "high_recall"]:
            payload = {"flows": [sample_valid_flow_dict], "threshold_preset": preset}
            response = client.post("/api/v1/predict/binary", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["threshold_preset"] == preset
            assert data["threshold_applied"] > 0.0


# ============================================================================
# 4. Multiclass Inference Tests
# ============================================================================

class TestMulticlassInference:
    def test_multiclass_prediction_c8_default(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        payload = {"flows": [sample_valid_flow_dict], "model_id": "c8"}
        response = client.post("/api/v1/predict/multiclass", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "c8"
        assert data["schema_used"] == "primary"
        assert data["flow_count"] == 1
        pred = data["predictions"][0]
        assert 0 <= pred["predicted_class_id"] <= 9
        assert pred["predicted_class_name"] in [
            "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
            "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"
        ]
        assert len(pred["probabilities"]) == 10
        assert sum(pred["probabilities"].values()) == pytest.approx(1.0, abs=0.01)

    def test_multiclass_prediction_c10_explicit(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        payload = {"flows": [sample_valid_flow_dict], "model_id": "c10"}
        response = client.post("/api/v1/predict/multiclass", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "c10"
        assert data["schema_used"] == "alternative_no_sttl"
        assert data["flow_count"] == 1

    def test_multiclass_prediction_invalid_model_id(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        payload = {"flows": [sample_valid_flow_dict], "model_id": "invalid_model_xyz"}
        response = client.post("/api/v1/predict/multiclass", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "Invalid model_id" in data["message"]


# ============================================================================
# 5. Model Comparison Tests (C8 vs C10)
# ============================================================================

class TestModelComparison:
    def test_model_comparison_endpoint(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        flows = [sample_valid_flow_dict, sample_valid_flow_dict]
        payload = {"flows": flows}
        response = client.post("/api/v1/predict/compare", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_flows"] == 2
        assert len(data["comparison_items"]) == 2
        assert 0.0 <= data["agreement_rate"] <= 1.0
        assert 0.0 <= data["disagreement_rate"] <= 1.0
        assert data["agreement_rate"] + data["disagreement_rate"] == pytest.approx(1.0, abs=1e-4)

        item = data["comparison_items"][0]
        assert "c8_predicted_class" in item
        assert "c10_predicted_class" in item
        assert isinstance(item["agreement"], bool)
        assert item["l1_distribution_divergence"] >= 0.0

    def test_model_comparison_class_mismatch_raises_value_error(
        self, sample_valid_flow_dict: dict
    ) -> None:
        """Verify compare_models raises an explicit ValueError when C8 and C10 classes mismatch."""
        from unittest.mock import MagicMock

        service = ModelService.get_instance()
        if not service.is_loaded:
            service.load_models()

        flow = NetworkFlowRecord(**sample_valid_flow_dict)
        original_model = service.models["c10"]["bundle"]["model"]
        mock_model = MagicMock()
        mock_model.classes_ = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.zeros((1, 10))

        try:
            # Temporarily simulate mismatched class ordering
            service.models["c10"]["bundle"]["model"] = mock_model
            with pytest.raises(ValueError, match="Class ordering mismatch between C8"):
                service.compare_models([flow])
        finally:
            service.models["c10"]["bundle"]["model"] = original_model


# ============================================================================
# 6. Drift Monitoring Integration Tests
# ============================================================================

class TestDriftMonitoring:
    def test_drift_monitoring_stable_batch(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        # Create a batch of 35 identical valid flows (>= min safeguard 30)
        batch = [copy.deepcopy(sample_valid_flow_dict) for _ in range(35)]
        payload = {"flows": batch, "batch_id": "test_stable_batch"}
        response = client.post("/api/v1/monitor/drift", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["batch_size"] == 35
        assert data["is_valid_size"] is True
        assert data["overall_drift_status"] in ("STABLE", "LOW_DRIFT", "MODERATE_DRIFT", "CRITICAL_DRIFT")
        assert "primary_recommendation" in data
        assert len(data["decisions"]) >= 1

    def test_drift_monitoring_small_batch_safeguard(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        # Batch below safeguard threshold (< 30)
        batch = [copy.deepcopy(sample_valid_flow_dict) for _ in range(10)]
        payload = {"flows": batch, "batch_id": "test_small_batch"}
        response = client.post("/api/v1/monitor/drift", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid_size"] is False
        assert len(data["warnings"]) >= 1
        assert "below minimum reliable safeguard" in data["warnings"][0]
        assert data["primary_recommendation"] == "CONTINUE_MONITORING"

    def test_drift_monitoring_empty_batch_rejected(self, client: TestClient) -> None:
        payload = {"flows": []}
        response = client.post("/api/v1/monitor/drift", json=payload)
        # Rejection via Pydantic or endpoint check
        assert response.status_code in (400, 422)


# ============================================================================
# 7. Explainability Integration Tests
# ============================================================================

class TestExplainability:
    def test_explain_c8_flows(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        payload = {"flows": [sample_valid_flow_dict], "model_id": "c8", "top_k": 5}
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "c8"
        assert data["flow_count"] == 1
        assert "pre-softmax log-odds" in data["output_space_description"]
        explanation = data["explanations"][0]
        assert explanation["predicted_class"] in [
            "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
            "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"
        ]
        assert explanation["output_space"] == "raw log-odds margin logit units"
        assert isinstance(explanation["base_margin"], float)
        assert len(explanation["top_attributions"]) <= 5
        top_attr = explanation["top_attributions"][0]
        assert "feature_name" in top_attr
        assert "attribution_margin" in top_attr
        assert top_attr["direction"] in ("push_towards_class", "push_away_from_class")

    def test_explain_c10_flows(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        payload = {"flows": [sample_valid_flow_dict], "model_id": "c10", "top_k": 5}
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "c10"

    def test_explain_batch_size_limit_exceeded(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        # Request with 11 flows (> max 10)
        flows = [sample_valid_flow_dict for _ in range(11)]
        payload = {"flows": flows, "model_id": "c8"}
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code in (400, 422)


# ============================================================================
# 8. Input Validation & Error Handling Tests
# ============================================================================

class TestInputValidation:
    def test_reject_non_finite_float(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        # 1. Direct model validator rejects NaN and Inf
        corrupted = copy.deepcopy(sample_valid_flow_dict)
        corrupted["dur"] = float("nan")
        with pytest.raises(ValueError, match="finite numeric value"):
            NetworkFlowRecord(**corrupted)

        corrupted["dur"] = float("inf")
        with pytest.raises(ValueError, match="finite numeric value"):
            NetworkFlowRecord(**corrupted)

        # 2. HTTP endpoint rejects non-numeric or malformed payload with 422
        corrupted_http = copy.deepcopy(sample_valid_flow_dict)
        corrupted_http["dur"] = "non_numeric_nan"
        payload = {"flows": [corrupted_http]}
        response = client.post("/api/v1/predict/binary", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error_type"] == "ValidationError"

    def test_reject_out_of_bound_fields(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Verify out-of-domain fields are rejected with HTTP 422 by backend validation."""
        # sttl > 255
        bad = copy.deepcopy(sample_valid_flow_dict)
        bad["sttl"] = 256.0
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        # dttl > 255
        bad = copy.deepcopy(sample_valid_flow_dict)
        bad["dttl"] = 256.0
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        # swin > 255
        bad = copy.deepcopy(sample_valid_flow_dict)
        bad["swin"] = 256.0
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        # dwin > 255
        bad = copy.deepcopy(sample_valid_flow_dict)
        bad["dwin"] = 256.0
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        # is_sm_ips_ports not in (0, 1)
        bad = copy.deepcopy(sample_valid_flow_dict)
        bad["is_sm_ips_ports"] = 0.5
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        bad["is_sm_ips_ports"] = 2.0
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        # is_ftp_login non-integer or negative
        bad = copy.deepcopy(sample_valid_flow_dict)
        bad["is_ftp_login"] = 1.5
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

        bad["is_ftp_login"] = -1.0
        assert client.post("/api/v1/predict/binary", json={"flows": [bad]}).status_code == 422

    def test_accept_valid_boundary_fields(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Verify valid boundary values (sttl/dttl/swin/dwin in [0, 255], is_ftp_login in [0, 1, 2, 4]) are accepted with HTTP 200."""
        # Boundary 0
        flow_b0 = copy.deepcopy(sample_valid_flow_dict)
        flow_b0["sttl"] = 0.0
        flow_b0["dttl"] = 0.0
        flow_b0["swin"] = 0.0
        flow_b0["dwin"] = 0.0
        flow_b0["is_sm_ips_ports"] = 0.0
        flow_b0["is_ftp_login"] = 0.0
        assert client.post("/api/v1/predict/binary", json={"flows": [flow_b0]}).status_code == 200

        # Boundary 255
        flow_b255 = copy.deepcopy(sample_valid_flow_dict)
        flow_b255["sttl"] = 255.0
        flow_b255["dttl"] = 255.0
        flow_b255["swin"] = 255.0
        flow_b255["dwin"] = 255.0
        flow_b255["is_sm_ips_ports"] = 1.0
        flow_b255["is_ftp_login"] = 1.0
        assert client.post("/api/v1/predict/binary", json={"flows": [flow_b255]}).status_code == 200

        # Multi-session FTP login values (observed 2 and 4 in dataset)
        for ftp_val in [2.0, 4.0]:
            flow_ftp = copy.deepcopy(sample_valid_flow_dict)
            flow_ftp["is_ftp_login"] = ftp_val
            assert client.post("/api/v1/predict/binary", json={"flows": [flow_ftp]}).status_code == 200

    def test_reject_missing_required_field(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        incomplete = copy.deepcopy(sample_valid_flow_dict)
        del incomplete["dur"]
        payload = {"flows": [incomplete]}
        response = client.post("/api/v1/predict/binary", json=payload)
        assert response.status_code == 422

    def test_reject_oversized_prediction_batch(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        flows = [sample_valid_flow_dict for _ in range(1001)]
        payload = {"flows": flows}
        response = client.post("/api/v1/predict/binary", json=payload)
        assert response.status_code in (400, 422)


# ============================================================================
# 9. Offline Consistency & Model Immutability Tests
# ============================================================================

class TestConsistencyAndImmutability:
    def test_api_predictions_match_direct_pipeline_inference(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Verify API prediction output is identical to direct model invocation."""
        model_service = ModelService.get_instance()
        bundle_c8 = model_service.models["c8"]["bundle"]

        df_flow = pd.DataFrame([sample_valid_flow_dict])
        X_proc, _, _, _ = extract_features_and_transform(bundle_c8, df_flow, schema="primary")

        direct_pred = int(bundle_c8["model"].predict(X_proc)[0])
        direct_probs = bundle_c8["model"].predict_proba(X_proc)[0]

        # Call via API
        response = client.post(
            "/api/v1/predict/multiclass",
            json={"flows": [sample_valid_flow_dict], "model_id": "c8"},
        )
        assert response.status_code == 200
        api_data = response.json()
        api_pred_item = api_data["predictions"][0]

        assert api_pred_item["predicted_class_id"] == direct_pred
        for c_idx, c_name in enumerate(model_service.classes):
            assert api_pred_item["probabilities"][c_name] == pytest.approx(direct_probs[c_idx], abs=1e-3)

    def test_models_remain_immutable_after_requests(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Confirm model objects and weights are not modified during inference calls."""
        model_service = ModelService.get_instance()
        b_c8 = model_service.models["c8"]["bundle"]["model"]
        initial_booster = b_c8.get_booster()
        trees_dump_before = initial_booster.get_dump()

        # Run several inference and explanation requests
        _ = client.post("/api/v1/predict/binary", json={"flows": [sample_valid_flow_dict]})
        _ = client.post("/api/v1/predict/multiclass", json={"flows": [sample_valid_flow_dict]})
        _ = client.post("/api/v1/explain", json={"flows": [sample_valid_flow_dict]})

        trees_dump_after = b_c8.get_booster().get_dump()
        assert trees_dump_before == trees_dump_after

    def test_drift_monitoring_does_not_mutate_reference_or_models(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Verify drift monitoring requests leave reference row counts and models unchanged."""
        monitoring_service = MonitoringService.get_instance()
        initial_ref_count = monitoring_service.ref_row_count

        # Send drift monitoring request
        batch = [copy.deepcopy(sample_valid_flow_dict) for _ in range(35)]
        response = client.post("/api/v1/monitor/drift", json={"flows": batch, "batch_id": "immutability_check"})
        assert response.status_code == 200

        # Reference count and detector reference dataframe must remain unchanged
        assert monitoring_service.ref_row_count == initial_ref_count
        assert len(monitoring_service.detector.ref_df) == initial_ref_count


class TestCorsConfiguration:
    """Verifies CORS preflight and cross-origin POST behavior under allow_credentials=False."""

    def test_cors_options_preflight_for_binary_prediction(self, client: TestClient) -> None:
        """Verify OPTIONS preflight for POST /api/v1/predict/binary succeeds with correct headers."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,accept",
        }
        response = client.options("/api/v1/predict/binary", headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        assert "POST" in response.headers.get("access-control-allow-methods", "")
        # allow_credentials=False must not return Access-Control-Allow-Credentials: true
        assert response.headers.get("access-control-allow-credentials") is None

    def test_cors_actual_post_for_binary_prediction(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Verify actual cross-origin POST /api/v1/predict/binary succeeds with correct headers and payload."""
        headers = {
            "Origin": "http://localhost:3000",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "flows": [sample_valid_flow_dict],
            "threshold_preset": "default",
        }
        response = client.post("/api/v1/predict/binary", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        assert response.headers.get("access-control-allow-credentials") is None
        data = response.json()
        assert data["model_id"] == "binary_winner"
        assert len(data["predictions"]) == 1
        assert "attack_probability" in data["predictions"][0]

    def test_cors_actual_post_for_multiclass_prediction(
        self, client: TestClient, sample_valid_flow_dict: dict
    ) -> None:
        """Verify actual cross-origin POST /api/v1/predict/multiclass succeeds with correct headers."""
        headers = {
            "Origin": "http://localhost:3000",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "flows": [sample_valid_flow_dict],
            "model_id": "c8",
        }
        response = client.post("/api/v1/predict/multiclass", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        assert response.headers.get("access-control-allow-credentials") is None
        data = response.json()
        assert data["model_id"] == "c8"
        assert len(data["predictions"]) == 1

