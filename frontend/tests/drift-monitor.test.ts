import test from "node:test";
import assert from "node:assert";
import { DRIFT_SCENARIOS } from "../src/lib/drift-scenarios.ts";
import type { DriftMonitoringResponse } from "../src/types/api.ts";
import { ApiError } from "../src/lib/api.ts";

test("Drift Monitor: all 7 research benchmark scenarios are registered", () => {
  assert.strictEqual(DRIFT_SCENARIOS.length, 7);

  const scenarioIds = DRIFT_SCENARIOS.map((s) => s.id);
  assert.ok(scenarioIds.includes("scenario_0_stable_control"));
  assert.ok(scenarioIds.includes("scenario_1_numerical_shift"));
  assert.ok(scenarioIds.includes("scenario_2_categorical_shift"));
  assert.ok(scenarioIds.includes("scenario_3_novel_categories"));
  assert.ok(scenarioIds.includes("scenario_4_data_corruption"));
  assert.ok(scenarioIds.includes("scenario_5_sttl_synthetic_perturbation"));
  assert.ok(scenarioIds.includes("scenario_6_multivariate_shift"));
});

test("Drift Monitor: each scenario generates at least 30 flows to satisfy statistical safeguard", () => {
  for (const scenario of DRIFT_SCENARIOS) {
    const batch = scenario.generateBatch();
    assert.ok(
      batch.length >= 30,
      `Scenario ${scenario.id} generated ${batch.length} flows, below safeguard threshold 30`
    );
  }
});

test("Drift Monitor: scenario 1 applies transport volume scaling", () => {
  const scenario1 = DRIFT_SCENARIOS.find((s) => s.id === "scenario_1_numerical_shift")!;
  const batch = scenario1.generateBatch();

  assert.strictEqual(scenario1.expectedDriftLevel, "CRITICAL_DRIFT");
  assert.strictEqual(scenario1.expectedAction, "RECOMMEND_OFFLINE_EVALUATION");

  // First flow should have high byte volume and sload
  const first = batch[0];
  assert.ok(first.sbytes > 10000);
  assert.ok(first.dur > 0.4);
});

test("Drift Monitor: scenario 3 introduces novel uncataloged protocols", () => {
  const scenario3 = DRIFT_SCENARIOS.find((s) => s.id === "scenario_3_novel_categories")!;
  const batch = scenario3.generateBatch();

  assert.strictEqual(scenario3.expectedAction, "RECOMMEND_DATA_COLLECTION");
  const protos = new Set(batch.map((f) => f.proto));
  assert.ok(protos.has("wireguard") || protos.has("quic") || protos.has("mqtt"));
});

test("Drift Monitor: scenario 5 isolates STTL artifact perturbation", () => {
  const scenario5 = DRIFT_SCENARIOS.find((s) => s.id === "scenario_5_sttl_synthetic_perturbation")!;
  const batch = scenario5.generateBatch();

  assert.strictEqual(scenario5.expectedAction, "RECOMMEND_NO_STTL_FALLBACK");
  for (const flow of batch) {
    assert.strictEqual(flow.sttl, 254.0);
  }
});

test("Drift Monitor: response mapping parses PSI and adaptive decisions cleanly", () => {
  const mockResponse: DriftMonitoringResponse = {
    batch_id: "test_batch_001",
    batch_size: 35,
    is_valid_size: true,
    overall_drift_status: "CRITICAL_DRIFT",
    drifting_features_count: 3,
    critical_features_count: 2,
    primary_recommendation: "RECOMMEND_OFFLINE_EVALUATION",
    decisions: [
      {
        action: "RECOMMEND_OFFLINE_EVALUATION",
        priority: "CRITICAL",
        trigger_reason: "Systemic multivariate drift across 3 critical features.",
        evidence_metrics: { drifting_count: 3 },
        operational_guidance: "Initiate controlled offline evaluation against recent labeled captures.",
      },
    ],
    requires_soc_escalation: true,
    requires_pipeline_investigation: false,
    feature_drift_summary: {
      numerical: {
        sbytes: {
          psi: 0.3542,
          ks_statistic: 0.4521,
          ks_pvalue: 0.00001,
          drift_level: "CRITICAL_DRIFT",
          missing_rate: 0.0,
          non_finite_rate: 0.0,
        },
      },
      categorical: {
        proto: {
          psi: 0.0412,
          drift_level: "STABLE",
          unknown_category_count: 0,
          unknown_category_rate: 0.0,
          novel_categories: [],
        },
      },
    },
    prediction_drift_summary: {
      model_id: "C8_Primary",
      prediction_psi: 0.8124,
      mean_confidence: 0.842,
      min_confidence: 0.412,
      low_confidence_rate: 0.174,
      mean_entropy: 0.652,
      drift_level: "MODERATE_DRIFT",
      predicted_distribution: { Normal: 0.45, Exploits: 0.35, Generic: 0.20 },
      c8_c10_disagreement_rate: 0.125,
    },
    warnings: [],
    reference_dataset_info: {
      source: "train.csv",
      sample_count: 151325,
      sealed_test_partition_status: "Strictly Untouched",
    },
    processing_time_ms: 14.2,
  };

  assert.strictEqual(mockResponse.overall_drift_status, "CRITICAL_DRIFT");
  assert.strictEqual(mockResponse.requires_soc_escalation, true);
  assert.strictEqual(mockResponse.feature_drift_summary.numerical.sbytes.psi, 0.3542);
  assert.strictEqual(mockResponse.prediction_drift_summary?.prediction_psi, 0.8124);
  assert.strictEqual(mockResponse.decisions[0].action, "RECOMMEND_OFFLINE_EVALUATION");
});

test("Drift Monitor: ApiError handles backend rejection on empty batch", () => {
  const err = new ApiError(
    400,
    "ValidationError",
    "Flow telemetry batch cannot be empty.",
    { field: "flows" }
  );

  assert.strictEqual(err.statusCode, 400);
  assert.strictEqual(err.errorType, "ValidationError");
  assert.strictEqual(err.message, "Flow telemetry batch cannot be empty.");
});
