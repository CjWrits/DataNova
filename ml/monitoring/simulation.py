"""Controlled, Reproducible Drift Simulation Framework for Network Telemetry.

Generates labeled, controlled distribution shifts on validation data partitions
to rigorously evaluate monitoring sensitivity and adaptive response behaviors.

Strict Constraints:
- Applied exclusively to validation data copies; original datasets are never altered.
- Controlled synthetic perturbations are explicitly labeled as simulations, never as real traffic drift.
- Scenarios cover numerical shifts, categorical shifts, novel categories, data corruption, and STTL divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulationScenarioMetadata:
    scenario_id: str
    scenario_name: str
    description: str
    target_features: List[str]
    perturbation_type: str
    perturbation_magnitude: str
    affected_sample_count: int
    expected_drift_level: str  # 'STABLE', 'LOW_DRIFT', 'MODERATE_DRIFT', 'CRITICAL_DRIFT'
    expected_monitoring_behavior: str


def generate_stable_batch(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 0: Stable, unperturbed reference-like validation batch (Control)."""
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()
    meta = SimulationScenarioMetadata(
        scenario_id="scenario_0_stable_control",
        scenario_name="Stable Reference Control Batch",
        description="Unperturbed validation batch representing standard in-distribution network traffic.",
        target_features=[],
        perturbation_type="none",
        perturbation_magnitude="0.0",
        affected_sample_count=0,
        expected_drift_level="STABLE",
        expected_monitoring_behavior="All feature PSIs < 0.10; system status stays STABLE; no alerts triggered.",
    )
    return batch, meta


def generate_numerical_mean_scale_shift(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 1: Controlled distribution shift on transport volume features (dur, sbytes, sload)."""
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()

    # Shift sbytes by multiplying by 3.5 and adding noise (e.g. data-heavy protocol shift)
    # Shift dur by multiplying by 4.0 (sustained streaming session elongation)
    # Shift sload by multiplying by 5.0 (elevated bandwidth demand)
    np.random.seed(random_state)
    batch["sbytes"] = batch["sbytes"] * 3.5 + np.random.uniform(500, 2000, len(batch))
    batch["dur"] = batch["dur"] * 4.0 + np.random.exponential(2.0, len(batch))
    batch["sload"] = batch["sload"] * 5.0

    meta = SimulationScenarioMetadata(
        scenario_id="scenario_1_numerical_shift",
        scenario_name="Numerical Volume & Duration Shift",
        description="Simulated 3.5x-5x scaling on transport volume variables (`sbytes`, `dur`, `sload`).",
        target_features=["sbytes", "dur", "sload"],
        perturbation_type="multiplicative_scale_and_additive_shift",
        perturbation_magnitude="3.5x sbytes, 4x dur, 5x sload",
        affected_sample_count=len(batch),
        expected_drift_level="CRITICAL_DRIFT",
        expected_monitoring_behavior="Targeted features trigger PSI >= 0.25; system alerts on transport volume shift.",
    )
    return batch, meta


def generate_categorical_frequency_shift(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 2: Shift in categorical protocol/service distributions (e.g. heavy DNS/UDP surge)."""
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()

    # Overwrite 60% of traffic to UDP protocol and DNS service (simulating heavy DNS resolution storm)
    np.random.seed(random_state)
    mask = np.random.rand(len(batch)) < 0.60
    batch.loc[mask, "proto"] = "udp"
    batch.loc[mask, "service"] = "dns"

    meta = SimulationScenarioMetadata(
        scenario_id="scenario_2_categorical_shift",
        scenario_name="Protocol & Service Frequency Skew",
        description="Simulated protocol shift forcing 60% of batch flows to UDP / DNS traffic.",
        target_features=["proto", "service"],
        perturbation_type="categorical_frequency_overhaul",
        perturbation_magnitude="60% UDP/DNS surge",
        affected_sample_count=int(mask.sum()),
        expected_drift_level="CRITICAL_DRIFT",
        expected_monitoring_behavior="Categorical PSI on proto and service exceeds 0.25; overall status MODERATE/CRITICAL.",
    )
    return batch, meta


def generate_novel_categories_injection(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 3: Introduction of novel, previously unseen protocol and service categories."""
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()

    # Inject unseen protocols e.g. 'wireguard', 'quic', 'coap'
    np.random.seed(random_state)
    n = len(batch)
    novel_proto_indices = batch.sample(n=int(n * 0.12), random_state=random_state).index
    novel_service_indices = batch.sample(n=int(n * 0.15), random_state=random_state + 1).index

    batch.loc[novel_proto_indices, "proto"] = np.random.choice(["wireguard", "quic"], size=len(novel_proto_indices))
    batch.loc[novel_service_indices, "service"] = np.random.choice(["coap", "mqtt", "doh"], size=len(novel_service_indices))

    meta = SimulationScenarioMetadata(
        scenario_id="scenario_3_novel_categories",
        scenario_name="Novel Unseen Category Injection",
        description="Introduction of unseen IoT and modern protocols (`wireguard`, `quic`, `coap`, `mqtt`).",
        target_features=["proto", "service"],
        perturbation_type="novel_category_insertion",
        perturbation_magnitude="12% novel proto, 15% novel service",
        affected_sample_count=len(set(novel_proto_indices) | set(novel_service_indices)),
        expected_drift_level="CRITICAL_DRIFT",
        expected_monitoring_behavior="Novel category detector flags unseen categories; novel_rate > 10%.",
    )
    return batch, meta


def generate_corrupted_missing_and_invalid_values(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 4: Data pipeline corruption with missing (NaN) and non-finite (Inf) values."""
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()

    np.random.seed(random_state)
    n = len(batch)
    nan_indices = batch.sample(n=int(n * 0.08), random_state=random_state).index
    inf_indices = batch.sample(n=int(n * 0.05), random_state=random_state + 2).index

    batch.loc[nan_indices, "sbytes"] = np.nan
    batch.loc[nan_indices, "dur"] = np.nan
    batch.loc[inf_indices, "sload"] = np.inf
    batch.loc[inf_indices, "dload"] = -np.inf

    meta = SimulationScenarioMetadata(
        scenario_id="scenario_4_data_corruption",
        scenario_name="Missing & Non-Finite Pipeline Corruption",
        description="Sensor transmission failure injecting 8% NaNs and 5% Infs into telemetry columns.",
        target_features=["sbytes", "dur", "sload", "dload"],
        perturbation_type="corrupted_nulls_and_infinities",
        perturbation_magnitude="8% NaN, 5% Inf",
        affected_sample_count=len(set(nan_indices) | set(inf_indices)),
        expected_drift_level="CRITICAL_DRIFT",
        expected_monitoring_behavior="Missing-value rate and non-finite rate thresholds violated; alert triggered.",
    )
    return batch, meta


def generate_sttl_artifact_perturbation(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 5: Specific perturbation of the synthetic testbed artifact `sttl`

    Tests whether C8 (dependent on STTL) collapses while C10 (No-STTL) remains stable!
    """
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()

    # Adversary or OS change modifies STTL: randomize TTL to uniform [1, 255]
    np.random.seed(random_state)
    batch["sttl"] = np.random.randint(1, 255, size=len(batch))

    meta = SimulationScenarioMetadata(
        scenario_id="scenario_5_sttl_artifact_perturbation",
        scenario_name="STTL Synthetic Artifact Perturbation",
        description="Adversarial TTL spoofing / OS heterogeneity randomizing `sttl` across [1, 255].",
        target_features=["sttl"],
        perturbation_type="uniform_randomization",
        perturbation_magnitude="Complete randomization of STTL across [1, 255]",
        affected_sample_count=len(batch),
        expected_drift_level="CRITICAL_DRIFT",
        expected_monitoring_behavior="sttl triggers critical PSI >= 0.50; C8 model prediction shifts heavily while C10 stays invariant.",
    )
    return batch, meta


def generate_combined_multivariate_shift(
    df_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, SimulationScenarioMetadata]:
    """Scenario 6: Realistic combined multivariate enterprise shift."""
    batch = df_val.sample(n=min(len(df_val), sample_size), random_state=random_state).copy()

    np.random.seed(random_state)
    # 1. Volume shift
    batch["sbytes"] = batch["sbytes"] * 2.2 + 100
    batch["spkts"] = batch["spkts"] * 1.8 + 10
    # 2. Connection density shift
    batch["ct_srv_dst"] = np.clip(batch["ct_srv_dst"] * 2.0, 1, 50)
    # 3. Protocol skew
    skew_mask = np.random.rand(len(batch)) < 0.35
    batch.loc[skew_mask, "service"] = "http"

    meta = SimulationScenarioMetadata(
        scenario_id="scenario_6_multivariate_shift",
        scenario_name="Combined Multivariate Network Shift",
        description="Concurrent volume scaling, connection fan-out increase, and service concentration.",
        target_features=["sbytes", "spkts", "ct_srv_dst", "service"],
        perturbation_type="multivariate_simultaneous",
        perturbation_magnitude="2.2x bytes, 1.8x pkts, 2.0x fanout, 35% HTTP skew",
        affected_sample_count=len(batch),
        expected_drift_level="CRITICAL_DRIFT",
        expected_monitoring_behavior="Multiple numerical and categorical features trigger moderate/critical drift.",
    )
    return batch, meta
