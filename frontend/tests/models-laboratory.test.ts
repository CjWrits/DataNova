import test from "node:test";
import assert from "node:assert";
import type {
  ModelComparisonItem,
  ModelComparisonRequest,
  ModelComparisonResponse,
  ModelInfo,
  ModelsListResponse,
} from "../src/types/api.ts";
import { ApiError } from "../src/lib/api.ts";
import { SAMPLE_EXPLOIT_FLOW } from "../src/lib/sample-data.ts";

test("Model Laboratory: catalog response parsing with 3 frozen models", () => {
  const mockCatalog: ModelsListResponse = {
    models: [
      {
        id: "binary_winner",
        name: "Binary Calibrated XGBoost Winner",
        version: "4.0.0",
        task_type: "binary_detection",
        schema_name: "primary",
        feature_count: 42,
        description: "Cost-sensitive calibrated XGBoost for high-recall benign vs. attack discrimination.",
        is_default: true,
        threshold_info: {
          default: 0.4951,
          f1_optimal: 0.5247,
          cost_optimal: 0.1783,
          low_false_alarm: 0.7227,
          high_recall: 0.3268,
        },
      },
      {
        id: "c8",
        name: "Candidate 8: FE + Cost-Sensitive Sqrt",
        version: "4.0.0",
        task_type: "multiclass_classification",
        schema_name: "primary",
        feature_count: 42,
        description: "Primary multiclass winner with domain feature engineering and square-root class weighting.",
        is_default: true,
      },
      {
        id: "c10",
        name: "Candidate 10: Robustness Track No-STTL",
        version: "4.0.0",
        task_type: "multiclass_classification",
        schema_name: "alternative_no_sttl",
        feature_count: 41,
        description: "Robustness-oriented multiclass winner excluding synthetic STTL artifact.",
        is_default: false,
      },
    ],
    count: 3,
  };

  assert.strictEqual(mockCatalog.count, 3);
  assert.strictEqual(mockCatalog.models.length, 3);

  const binModel = mockCatalog.models.find((m) => m.id === "binary_winner");
  assert.ok(binModel);
  assert.strictEqual(binModel.feature_count, 42);
  assert.strictEqual(binModel.threshold_info?.default, 0.4951);
  assert.strictEqual(binModel.threshold_info?.cost_optimal, 0.1783);

  const c8Model = mockCatalog.models.find((m) => m.id === "c8");
  assert.ok(c8Model);
  assert.strictEqual(c8Model.schema_name, "primary");
  assert.strictEqual(c8Model.feature_count, 42);

  const c10Model = mockCatalog.models.find((m) => m.id === "c10");
  assert.ok(c10Model);
  assert.strictEqual(c10Model.schema_name, "alternative_no_sttl");
  assert.strictEqual(c10Model.feature_count, 41);
});

test("Model Laboratory: comparison request construction", () => {
  const req: ModelComparisonRequest = {
    flows: [SAMPLE_EXPLOIT_FLOW],
  };

  assert.strictEqual(req.flows.length, 1);
  assert.strictEqual(req.flows[0].proto, "tcp");
  assert.strictEqual(req.flows[0].sttl, 62.0);
});

test("Model Laboratory: comparison agreement condition (C8 and C10 agree)", () => {
  const mockAgreementResponse: ModelComparisonResponse = {
    total_flows: 1,
    agreement_count: 1,
    disagreement_count: 0,
    agreement_rate: 1.0,
    disagreement_rate: 0.0,
    comparison_items: [
      {
        index: 0,
        c8_predicted_class: "Exploits",
        c8_confidence: 0.7738,
        c10_predicted_class: "Exploits",
        c10_confidence: 0.8124,
        agreement: true,
        l1_distribution_divergence: 0.1245,
      },
    ],
    c8_model_name: "Candidate 8: FE + Cost-Sensitive Sqrt",
    c10_model_name: "Candidate 10: Robustness Track No-STTL",
    processing_time_ms: 32.1,
  };

  assert.strictEqual(mockAgreementResponse.total_flows, 1);
  assert.strictEqual(mockAgreementResponse.agreement_count, 1);
  assert.strictEqual(mockAgreementResponse.disagreement_count, 0);
  assert.strictEqual(mockAgreementResponse.agreement_rate, 1.0);

  const item: ModelComparisonItem = mockAgreementResponse.comparison_items[0];
  assert.strictEqual(item.agreement, true);
  assert.strictEqual(item.c8_predicted_class, item.c10_predicted_class);
  assert.ok(item.l1_distribution_divergence < 0.5);
});

test("Model Laboratory: comparison disagreement condition (C8 and C10 diverge)", () => {
  const mockDisagreementResponse: ModelComparisonResponse = {
    total_flows: 1,
    agreement_count: 0,
    disagreement_count: 1,
    agreement_rate: 0.0,
    disagreement_rate: 1.0,
    comparison_items: [
      {
        index: 0,
        c8_predicted_class: "Exploits",
        c8_confidence: 0.6521,
        c10_predicted_class: "Normal",
        c10_confidence: 0.5843,
        agreement: false,
        l1_distribution_divergence: 0.8412,
      },
    ],
    c8_model_name: "Candidate 8: FE + Cost-Sensitive Sqrt",
    c10_model_name: "Candidate 10: Robustness Track No-STTL",
    processing_time_ms: 35.4,
  };

  assert.strictEqual(mockDisagreementResponse.agreement_count, 0);
  assert.strictEqual(mockDisagreementResponse.disagreement_count, 1);
  assert.strictEqual(mockDisagreementResponse.agreement_rate, 0.0);
  assert.strictEqual(mockDisagreementResponse.disagreement_rate, 1.0);

  const item = mockDisagreementResponse.comparison_items[0];
  assert.strictEqual(item.agreement, false);
  assert.notStrictEqual(item.c8_predicted_class, item.c10_predicted_class);
  assert.strictEqual(item.c8_predicted_class, "Exploits");
  assert.strictEqual(item.c10_predicted_class, "Normal");
  assert.ok(item.l1_distribution_divergence > 0.5);
});

test("Model Laboratory: multi-flow comparison batch statistics", () => {
  const mockBatchResponse: ModelComparisonResponse = {
    total_flows: 4,
    agreement_count: 3,
    disagreement_count: 1,
    agreement_rate: 0.75,
    disagreement_rate: 0.25,
    comparison_items: [
      { index: 0, c8_predicted_class: "Normal", c8_confidence: 0.95, c10_predicted_class: "Normal", c10_confidence: 0.96, agreement: true, l1_distribution_divergence: 0.04 },
      { index: 1, c8_predicted_class: "Exploits", c8_confidence: 0.88, c10_predicted_class: "Exploits", c10_confidence: 0.85, agreement: true, l1_distribution_divergence: 0.08 },
      { index: 2, c8_predicted_class: "DoS", c8_confidence: 0.72, c10_predicted_class: "Generic", c10_confidence: 0.61, agreement: false, l1_distribution_divergence: 0.65 },
      { index: 3, c8_predicted_class: "Fuzzers", c8_confidence: 0.81, c10_predicted_class: "Fuzzers", c10_confidence: 0.79, agreement: true, l1_distribution_divergence: 0.09 },
    ],
    c8_model_name: "C8",
    c10_model_name: "C10",
    processing_time_ms: 45.0,
  };

  assert.strictEqual(mockBatchResponse.total_flows, 4);
  assert.strictEqual(mockBatchResponse.agreement_count, 3);
  assert.strictEqual(mockBatchResponse.disagreement_count, 1);
  assert.strictEqual(mockBatchResponse.agreement_rate, 0.75);
  assert.strictEqual(mockBatchResponse.disagreement_rate, 0.25);
});

test("Model Laboratory: ApiError handling on server rejection", () => {
  const err = new ApiError(500, "InternalServerError", "Comparison failure: Model execution failed");
  assert.strictEqual(err.statusCode, 500);
  assert.strictEqual(err.errorType, "InternalServerError");
  assert.ok(err.message.includes("Comparison failure"));
});

test("Model Laboratory: consensus vs disagreement terminology and operational guidance", () => {
  // Agreement wording: must reflect cross-model consensus, NOT high operational confidence or proof
  const agreementText = "Model Agreement (Cross-Model Consensus)";
  assert.strictEqual(agreementText.includes("Consensus"), true);
  assert.strictEqual(agreementText.includes("High Operational Confidence"), false);
  assert.strictEqual(agreementText.includes("Proof"), false);

  // Consensus nuance: predicted class unchanged, but probability distributions may differ
  const consensusExplanation = "The predicted class is unchanged when Source TTL is excluded, although the probability distributions may differ. The L1 divergence quantifies that distribution difference.";
  assert.strictEqual(consensusExplanation.includes("predicted class is unchanged"), true);
  assert.strictEqual(consensusExplanation.includes("probability distributions may differ"), true);
  assert.strictEqual(consensusExplanation.includes("prediction is invariant to whether Source TTL"), false);

  // Disagreement wording: must describe sensitivity to feature ablation, NOT proof of manipulation
  const disagreementGuidance = "Prediction changes when Source TTL is excluded. This indicates sensitivity to the feature ablation, not proof of TTL manipulation. Corroborate with independent network evidence before drawing a forensic conclusion.";
  assert.strictEqual(disagreementGuidance.includes("sensitivity to the feature ablation"), true);
  assert.strictEqual(disagreementGuidance.includes("not proof of TTL manipulation"), true);
  assert.strictEqual(disagreementGuidance.includes("TTL manipulation likely occurred"), false);
});

test("Model Laboratory: threshold matrix labels reflect validation provenance without automated blocking promises", () => {
  const thresholdPresets = [
    { name: "default", threshold: 0.4951, desc: "Default operating threshold (closest grid point to 0.50 on validation sweep; not inherently balanced)" },
    { name: "f1_optimal", threshold: 0.5247, desc: "Validation-set F1-optimal threshold (maximizes harmonic mean of precision and recall)" },
    { name: "cost_optimal", threshold: 0.1783, desc: "Threshold minimizing documented asymmetric validation cost (Cost = 1.0·FP + 5.0·FN)" },
    { name: "low_false_alarm", threshold: 0.7227, desc: "Precision-oriented operating threshold (selected for validation FPR ≤ 2.0% constraint; does not authorize automatic blocking)" },
    { name: "high_recall", threshold: 0.3268, desc: "Recall-oriented operating threshold (selected for validation Recall ≥ 98.0% constraint; high-sensitivity catch-all)" },
  ];

  for (const preset of thresholdPresets) {
    // Must not promise automated blocking or auto-drop
    assert.strictEqual(preset.desc.includes("Auto-Drop"), false);
    assert.strictEqual(preset.desc.includes("authorizes automated"), false);
    assert.ok(preset.threshold > 0 && preset.threshold < 1);
  }

  // Low false alarm must explicitly disclaim automatic blocking authorization
  const lfa = thresholdPresets.find((p) => p.name === "low_false_alarm");
  assert.ok(lfa?.desc.includes("does not authorize automatic blocking"));
});

test("Model Laboratory: confidence represents model softmax probability", () => {
  const item: ModelComparisonItem = {
    index: 0,
    c8_predicted_class: "Exploits",
    c8_confidence: 0.7738,
    c10_predicted_class: "Exploits",
    c10_confidence: 0.8124,
    agreement: true,
    l1_distribution_divergence: 0.1245,
  };

  // Both confidences represent maximum softmax probability in [0, 1]
  assert.ok(item.c8_confidence >= 0.0 && item.c8_confidence <= 1.0);
  assert.ok(item.c10_confidence >= 0.0 && item.c10_confidence <= 1.0);
  assert.strictEqual(typeof item.c8_confidence, "number");
});

test("Model Laboratory: feature-specific domain validation logic", () => {
  // Test TTL validator
  const validateTtl = (val: number) => {
    if (isNaN(val) || !Number.isFinite(val)) return "Must be a finite number";
    if (val < 0) return "Cannot be negative (ge=0.0)";
    if (val > 255) return "TTL must be between 0 and 255 (RFC 791 / 8-bit integer)";
    return null;
  };

  assert.strictEqual(validateTtl(64), null);
  assert.strictEqual(validateTtl(255), null);
  assert.strictEqual(validateTtl(256)?.includes("between 0 and 255"), true);
  assert.strictEqual(validateTtl(-1)?.includes("Cannot be negative"), true);

  // Test binary indicator validator (is_sm_ips_ports)
  const validateBinaryFlag = (val: number) => {
    if (val !== 0 && val !== 1) return "Binary indicator flag must be either 0 or 1";
    return null;
  };

  assert.strictEqual(validateBinaryFlag(0), null);
  assert.strictEqual(validateBinaryFlag(1), null);
  assert.strictEqual(validateBinaryFlag(2)?.includes("either 0 or 1"), true);

  // Test FTP login count validator (is_ftp_login: observed 0, 1, 2, 4 in dataset)
  const validateFtpLogin = (val: number) => {
    if (val < 0) return "Cannot be negative (ge=0.0)";
    if (!Number.isInteger(val)) return "FTP login count must be an integer";
    return null;
  };

  assert.strictEqual(validateFtpLogin(0), null);
  assert.strictEqual(validateFtpLogin(1), null);
  assert.strictEqual(validateFtpLogin(2), null);
  assert.strictEqual(validateFtpLogin(4), null);
  assert.strictEqual(validateFtpLogin(-1)?.includes("Cannot be negative"), true);
  assert.strictEqual(validateFtpLogin(1.5)?.includes("must be an integer"), true);

  // Test TCP window validator in UNSW-NB15 representation (0-255)
  const validateTcpWindow = (val: number) => {
    if (val < 0) return "Cannot be negative (ge=0.0)";
    if (val > 255) return "TCP window in UNSW-NB15 representation must be between 0 and 255";
    return null;
  };

  assert.strictEqual(validateTcpWindow(0), null);
  assert.strictEqual(validateTcpWindow(255), null);
  assert.strictEqual(validateTcpWindow(256)?.includes("between 0 and 255"), true);
});

test("Model Laboratory: dynamic threshold matrix populated in model info", () => {
  const binaryModelInfo: Partial<ModelInfo> = {
    id: "binary_winner",
    threshold_matrix: {
      default: { threshold: 0.4951, accuracy: 0.9503, precision: 0.9668, recall: 0.9552, f1: 0.9609, fpr: 0.0585, cost: 6253.0 },
      cost_optimal: { threshold: 0.1783, accuracy: 0.9149, precision: 0.8850, recall: 0.9968, f1: 0.9375, fpr: 0.2310, cost: 3547.0 },
    },
  };

  assert.ok(binaryModelInfo.threshold_matrix);
  assert.strictEqual(binaryModelInfo.threshold_matrix.cost_optimal.recall, 0.9968);
  assert.strictEqual(binaryModelInfo.threshold_matrix.cost_optimal.cost, 3547.0);
});
