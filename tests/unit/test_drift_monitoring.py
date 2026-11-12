"""Unit test suite for Phase 6 Drift Detection, Monitoring & Adaptive Response Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.monitoring.adaptive_engine import (
    AdaptiveActionType,
    AdaptiveDecision,
    AdaptiveResponseEngine,
    ComprehensiveAdaptivePolicyResult,
)
from ml.monitoring.drift_detector import (
    BatchMonitoringSummary,
    CategoricalDriftResult,
    FeatureDriftDetector,
    ModelPredictionMonitor,
    NumericalDriftResult,
    PredictionDriftResult,
)
from ml.monitoring.simulation import (
    SimulationScenarioMetadata,
    generate_categorical_frequency_shift,
    generate_combined_multivariate_shift,
    generate_corrupted_missing_and_invalid_values,
    generate_novel_categories_injection,
    generate_numerical_mean_scale_shift,
    generate_stable_batch,
    generate_sttl_artifact_perturbation,
)


@pytest.fixture
def synthetic_ref_and_batch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic reference and batch dataframes for testing."""
    np.random.seed(42)
    n = 200

    ref_data = {
        "dur": np.random.exponential(1.0, size=n),
        "sbytes": np.random.lognormal(5.0, 1.0, size=n),
        "sttl": np.random.choice([31, 62, 254], size=n, p=[0.2, 0.5, 0.3]),
        "sload": np.random.uniform(100.0, 1000.0, size=n),
        "proto": np.random.choice(["tcp", "udp", "unas"], size=n, p=[0.7, 0.25, 0.05]),
        "service": np.random.choice(["-", "http", "dns", "ftp"], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "state": np.random.choice(["FIN", "CON", "INT"], size=n, p=[0.6, 0.3, 0.1]),
    }
    df_ref = pd.DataFrame(ref_data)
    df_batch = df_ref.sample(n=100, random_state=123).copy().reset_index(drop=True)

    return df_ref, df_batch


class TestFeatureDriftDetector:
    """Tests for statistical drift detection metrics."""

    def test_detector_initialization(self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        df_ref, _ = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        assert len(detector._ref_quantiles) > 0
        assert "sbytes" in detector._ref_quantiles
        assert len(detector._ref_cat_freqs) > 0
        assert "proto" in detector._ref_cat_freqs

    def test_numerical_psi_identical_distributions(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        res = detector.evaluate_numerical_feature("sbytes", df_batch["sbytes"])
        assert isinstance(res, NumericalDriftResult)
        assert res.psi < 0.10
        assert res.drift_level == "STABLE"
        assert res.missing_rate == 0.0
        assert res.non_finite_rate == 0.0

    def test_numerical_psi_heavily_shifted_distribution(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        # Apply extreme shift
        shifted_series = df_batch["sbytes"] * 10.0 + 50000.0
        res = detector.evaluate_numerical_feature("sbytes", shifted_series)

        assert res.psi >= 0.25
        assert res.drift_level == "CRITICAL_DRIFT"
        assert res.ks_statistic > 0.50
        assert res.ks_pvalue < 0.001

    def test_categorical_frequency_shift_and_novel_category(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        # Inject novel unseen categories
        cat_batch = df_batch["proto"].copy()
        cat_batch.iloc[:10] = "wireguard"
        cat_batch.iloc[10:20] = "quic"

        res = detector.evaluate_categorical_feature("proto", cat_batch)
        assert isinstance(res, CategoricalDriftResult)
        assert res.unknown_category_count == 20
        assert res.unknown_category_rate == 0.20
        assert "wireguard" in res.novel_categories
        assert "quic" in res.novel_categories
        assert res.drift_level == "CRITICAL_DRIFT"

    def test_missing_and_non_finite_tracking(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        corrupted_series = df_batch["dur"].copy()
        corrupted_series.iloc[:10] = np.nan
        corrupted_series.iloc[10:15] = np.inf
        corrupted_series.iloc[15:20] = -np.inf

        res = detector.evaluate_numerical_feature("dur", corrupted_series)
        assert res.missing_rate == pytest.approx(0.10, abs=1e-3)
        assert res.non_finite_rate == pytest.approx(0.20, abs=1e-3)

    def test_small_batch_safeguard(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref, min_batch_size=30)

        # Sub-threshold batch
        small_batch = df_batch.head(15)
        summary = detector.evaluate_batch(small_batch, batch_id="small_001")

        assert not summary.is_valid_size
        assert summary.batch_size == 15
        assert len(summary.warnings) > 0
        assert "below minimum reliable safeguard" in summary.warnings[0]


class TestModelPredictionMonitor:
    """Tests for model output probability and prediction drift monitoring."""

    def test_prediction_drift_stable(self) -> None:
        classes = ["Normal", "Generic", "Exploits"]
        ref_dist = {"Normal": 0.50, "Generic": 0.30, "Exploits": 0.20}
        monitor = ModelPredictionMonitor(class_names=classes, ref_class_distribution=ref_dist)

        # Batch matching distribution
        y_pred = np.array([0] * 50 + [1] * 30 + [2] * 20)
        y_prob = np.zeros((100, 3))
        for i, c in enumerate(y_pred):
            y_prob[i, c] = 0.90
            for oc in range(3):
                if oc != c:
                    y_prob[i, oc] = 0.05

        res = monitor.evaluate_predictions(y_pred, y_prob)
        assert isinstance(res, PredictionDriftResult)
        assert res.prediction_psi < 0.05
        assert res.drift_level == "STABLE"
        assert res.mean_confidence == pytest.approx(0.90, abs=0.01)
        assert res.low_confidence_rate == 0.0

    def test_confidence_collapse_and_entropy_spike(self) -> None:
        classes = ["Normal", "Generic", "Exploits"]
        monitor = ModelPredictionMonitor(class_names=classes)

        # Uniform near-random predictions (high entropy, low confidence)
        y_pred = np.random.choice([0, 1, 2], size=100)
        y_prob = np.ones((100, 3)) / 3.0  # max prob = 0.333 < 0.60

        res = monitor.evaluate_predictions(y_pred, y_prob)
        assert res.mean_confidence == pytest.approx(0.333, abs=0.01)
        assert res.low_confidence_rate == 1.0
        assert res.drift_level == "CRITICAL_DRIFT"
        assert res.mean_entropy == pytest.approx(-np.log(1.0 / 3.0), abs=0.01)


class TestSimulationScenarios:
    """Tests verifying simulation reproducibility and original dataset immutability."""

    def test_simulation_reproducibility_and_metadata(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_val = synthetic_ref_and_batch_data

        b1, m1 = generate_sttl_artifact_perturbation(df_val, sample_size=50, random_state=42)
        b2, m2 = generate_sttl_artifact_perturbation(df_val, sample_size=50, random_state=42)

        assert isinstance(m1, SimulationScenarioMetadata)
        assert m1.scenario_id == "scenario_5_sttl_artifact_perturbation"
        pd.testing.assert_frame_equal(b1, b2)

    def test_original_dataset_invariance(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        _, df_val = synthetic_ref_and_batch_data
        original_sttl = df_val["sttl"].copy()

        _ = generate_sttl_artifact_perturbation(df_val, sample_size=50, random_state=42)
        pd.testing.assert_series_equal(df_val["sttl"], original_sttl)


class TestAdaptiveResponseEngine:
    """Tests rule-based operational guidance decisions."""

    def test_stable_batch_decision(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)
        summary = detector.evaluate_batch(df_batch, batch_id="stable_batch")

        engine = AdaptiveResponseEngine()
        policy = engine.evaluate_batch_signals(summary)

        assert isinstance(policy, ComprehensiveAdaptivePolicyResult)
        assert policy.primary_recommendation == AdaptiveActionType.CONTINUE_MONITORING
        assert not policy.requires_soc_escalation
        assert not policy.requires_pipeline_investigation

    def test_sttl_divergence_triggers_no_sttl_fallback(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        # Mutate STTL to uniform random
        corrupted_sttl = df_batch.copy()
        corrupted_sttl["sttl"] = np.random.randint(1, 255, size=len(corrupted_sttl))
        summary = detector.evaluate_batch(corrupted_sttl, batch_id="sttl_shift")

        engine = AdaptiveResponseEngine()
        policy = engine.evaluate_batch_signals(summary, c8_vs_c10_disagreement_rate=0.18)

        actions = [d.action for d in policy.decisions]
        assert AdaptiveActionType.RECOMMEND_NO_STTL_FALLBACK in actions
        assert policy.requires_soc_escalation

    def test_data_corruption_triggers_feature_flagging(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        corrupted = df_batch.copy()
        corrupted.loc[:10, "sbytes"] = np.nan
        summary = detector.evaluate_batch(corrupted, batch_id="corrupt_batch")

        engine = AdaptiveResponseEngine()
        policy = engine.evaluate_batch_signals(summary)

        actions = [d.action for d in policy.decisions]
        assert AdaptiveActionType.FLAG_AFFECTED_FEATURES in actions
        assert policy.requires_pipeline_investigation

    def test_novel_category_triggers_data_collection(
        self, synthetic_ref_and_batch_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        df_ref, df_batch = synthetic_ref_and_batch_data
        detector = FeatureDriftDetector(df_ref)

        novel = df_batch.copy()
        novel.loc[:15, "proto"] = "wireguard"
        summary = detector.evaluate_batch(novel, batch_id="novel_batch")

        engine = AdaptiveResponseEngine()
        policy = engine.evaluate_batch_signals(summary)

        actions = [d.action for d in policy.decisions]
        assert AdaptiveActionType.RECOMMEND_DATA_COLLECTION in actions
        assert policy.requires_pipeline_investigation
