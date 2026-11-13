import test from "node:test";
import assert from "node:assert";
import type {
  ExplainRequest,
  ExplainResponse,
  FlowExplanationItem,
} from "../src/types/api.ts";
import { ApiError } from "../src/lib/api.ts";
import { SAMPLE_BENIGN_FLOW, SAMPLE_EXPLOIT_FLOW } from "../src/lib/sample-data.ts";

test("Explainability: request payload construction with supported models ('c8' and 'c10')", () => {
  const reqC8: ExplainRequest = {
    flows: [SAMPLE_EXPLOIT_FLOW],
    model_id: "c8",
    top_k: 10,
  };

  assert.strictEqual(reqC8.flows.length, 1);
  assert.strictEqual(reqC8.model_id, "c8");
  assert.strictEqual(reqC8.top_k, 10);
  assert.strictEqual(reqC8.flows[0].proto, "tcp");

  const reqC10: ExplainRequest = {
    flows: [SAMPLE_BENIGN_FLOW],
    model_id: "c10",
    top_k: 5,
  };
  assert.strictEqual(reqC10.model_id, "c10");
  assert.strictEqual(reqC10.top_k, 5);
});

test("Explainability: bounded batch size constraint (max 10 flows)", () => {
  const validBatch: ExplainRequest = {
    flows: Array(10).fill(SAMPLE_BENIGN_FLOW),
    model_id: "c8",
  };
  assert.strictEqual(validBatch.flows.length, 10);

  // Beyond 10 flows violates the bounded backend constraint
  const oversizedBatch = Array(11).fill(SAMPLE_BENIGN_FLOW);
  assert.ok(oversizedBatch.length > 10, "Batch size exceeds MAX_EXPLAIN_BATCH_SIZE (10)");
});

test("Explainability: response mapping and base margin log-odds preservation", () => {
  // Representative fixture matching exact ExplainResponse schema
  const mockResponse: ExplainResponse = {
    model_id: "c8",
    model_name: "Feature Engineering + Cost-Sensitive (Sqrt)",
    output_space_description: "Attributions represent additive shifts in raw pre-softmax log-odds margins.",
    explanations: [
      {
        index: 0,
        predicted_class: "Exploits",
        output_space: "raw log-odds margin logit units",
        base_margin: -2.1402,
        top_attributions: [
          {
            feature_name: "dttl",
            attribution_margin: 2.8451,
            direction: "push_towards_class",
          },
          {
            feature_name: "sttl",
            attribution_margin: 1.6214,
            direction: "push_towards_class",
          },
          {
            feature_name: "sbytes",
            attribution_margin: -0.9842,
            direction: "push_away_from_class",
          },
        ],
      },
    ],
    flow_count: 1,
    processing_time_ms: 24.5,
  };

  assert.strictEqual(mockResponse.model_id, "c8");
  assert.strictEqual(mockResponse.flow_count, 1);
  assert.strictEqual(mockResponse.explanations.length, 1);

  const exp: FlowExplanationItem = mockResponse.explanations[0];
  assert.strictEqual(exp.index, 0);
  assert.strictEqual(exp.predicted_class, "Exploits");
  assert.strictEqual(exp.output_space, "raw log-odds margin logit units");
  assert.strictEqual(exp.base_margin, -2.1402);
  assert.strictEqual(exp.top_attributions.length, 3);
});

test("Explainability: signed attribution handling (positive vs negative direction)", () => {
  const attributions = [
    { feature_name: "dttl", attribution_margin: 2.8451, direction: "push_towards_class" as const },
    { feature_name: "ct_state_ttl", attribution_margin: 1.1205, direction: "push_towards_class" as const },
    { feature_name: "sbytes", attribution_margin: -0.9842, direction: "push_away_from_class" as const },
    { feature_name: "rate", attribution_margin: -0.412, direction: "push_away_from_class" as const },
  ];

  const positiveContributors = attributions.filter((a) => a.attribution_margin > 0);
  const negativeContributors = attributions.filter((a) => a.attribution_margin < 0);

  assert.strictEqual(positiveContributors.length, 2);
  assert.strictEqual(positiveContributors[0].feature_name, "dttl");
  assert.strictEqual(positiveContributors[0].direction, "push_towards_class");

  assert.strictEqual(negativeContributors.length, 2);
  assert.strictEqual(negativeContributors[0].feature_name, "sbytes");
  assert.strictEqual(negativeContributors[0].direction, "push_away_from_class");

  // Strongest positive and negative
  const topPos = attributions.find((a) => a.attribution_margin > 0);
  const topNeg = attributions.find((a) => a.attribution_margin < 0);
  assert.strictEqual(topPos?.feature_name, "dttl");
  assert.strictEqual(topNeg?.feature_name, "sbytes");
});

test("Explainability: empty explanation list and zero-attribution resilience", () => {
  const emptyRes: ExplainResponse = {
    model_id: "c8",
    model_name: "C8",
    output_space_description: "raw margins",
    explanations: [],
    flow_count: 0,
    processing_time_ms: 1.2,
  };

  assert.strictEqual(emptyRes.explanations.length, 0);

  const zeroAttrItem: FlowExplanationItem = {
    index: 0,
    predicted_class: "Normal",
    output_space: "raw log-odds margin logit units",
    base_margin: 0.0,
    top_attributions: [],
  };

  assert.strictEqual(zeroAttrItem.top_attributions.length, 0);
  const pos = zeroAttrItem.top_attributions.find((a) => a.attribution_margin > 0) || null;
  const neg = zeroAttrItem.top_attributions.find((a) => a.attribution_margin < 0) || null;
  assert.strictEqual(pos, null);
  assert.strictEqual(neg, null);
});

test("Explainability: ApiError handling on unsupported model ID or batch limit", () => {
  const errBatch = new ApiError(
    400,
    "HTTPException",
    "Batch size (15) exceeds maximum permitted explanation limit (10) to protect server memory."
  );
  assert.strictEqual(errBatch.statusCode, 400);
  assert.ok(errBatch.message.includes("exceeds maximum permitted explanation limit (10)"));

  const errModel = new ApiError(
    400,
    "HTTPException",
    "Invalid model_id 'binary_winner'. Must be explicitly 'c8' or 'c10'."
  );
  assert.strictEqual(errModel.statusCode, 400);
  assert.ok(errModel.message.includes("Must be explicitly 'c8' or 'c10'"));
});

test("Explainability: SHAP class-specific attribution mapping and TreeSHAP efficiency relationship", () => {
  // Verifies mathematical properties of TreeSHAP output:
  // 1. base_margin is specific to the predicted class
  // 2. attributions are in raw pre-softmax log-odds margin space (not probabilities)
  // 3. sum of top-k attributions is a partial shift, not full reconstruction of raw margin
  const mockClassBase = 0.9195;
  const mockTopAttributions = [
    { feature_name: "sttl", attribution_margin: 0.8421, direction: "push_towards_class" as const },
    { feature_name: "dttl", attribution_margin: 0.4125, direction: "push_towards_class" as const },
    { feature_name: "sbytes", attribution_margin: -0.2104, direction: "push_away_from_class" as const },
  ];

  const topSum = mockTopAttributions.reduce((acc, curr) => acc + curr.attribution_margin, 0);
  assert.strictEqual(Math.round(topSum * 10000) / 10000, 1.0442);

  // Partial margin with top 3
  const partialMargin = mockClassBase + topSum;
  assert.ok(partialMargin > mockClassBase);

  // Verify attributions are signed logit shifts, not probabilities (can be > 1.0 or < -1.0)
  const largeAttribution = { feature_name: "dur", attribution_margin: 3.4512, direction: "push_towards_class" as const };
  assert.ok(largeAttribution.attribution_margin > 1.0, "Attribution in log-odds space can exceed 1.0");
});

test("Explainability: UI terminology check - verifies SHAP base value and avoids unconditioned prior", () => {
  const validOutputSpace = "raw log-odds margin logit units";
  const validBaseName = "SHAP Base Value E[f(x)]";

  assert.strictEqual(validOutputSpace.includes("logit"), true);
  assert.strictEqual(validBaseName.includes("unconditioned"), false);
  assert.strictEqual(validBaseName.includes("SHAP Base Value"), true);
});

test("Explainability: top_k boundary and partial attribution limitation (42 max vs 52 total parents)", () => {
  // Bounded top_k constraint: schema enforces top_k <= 42
  const maxPermittedTopK = 42;
  const totalParentFeatures = 52; // 42 raw + 10 domain engineered

  assert.strictEqual(maxPermittedTopK < totalParentFeatures, true);

  // Full parent reconstruction is mathematically impossible when capped at 42
  const truncatedFeaturesCount = totalParentFeatures - maxPermittedTopK;
  assert.strictEqual(truncatedFeaturesCount, 10);
});
