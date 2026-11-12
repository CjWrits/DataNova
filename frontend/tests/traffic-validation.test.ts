import test from "node:test";
import assert from "node:assert";
import type {
  BinaryPredictionRequest,
  BinaryPredictionResponse,
  MulticlassPredictionRequest,
  MulticlassPredictionResponse,
  NetworkFlowRecord,
} from "../src/types/api.ts";
import { ApiError } from "../src/lib/api.ts";
import { SAMPLE_BENIGN_FLOW, SAMPLE_EXPLOIT_FLOW, SAMPLE_DOS_FLOW } from "../src/lib/sample-data.ts";
import { exportBinaryPredictionsToCsv, exportMulticlassPredictionsToCsv } from "../src/lib/csv-export.ts";

test("Single flow: request payload construction with threshold presets", () => {
  const presets = ["default", "f1_optimal", "cost_optimal", "low_false_alarm", "high_recall"] as const;

  for (const preset of presets) {
    const req: BinaryPredictionRequest = {
      flows: [SAMPLE_BENIGN_FLOW],
      threshold_preset: preset,
    };

    assert.strictEqual(req.flows.length, 1);
    assert.strictEqual(req.threshold_preset, preset);
    assert.strictEqual(req.flows[0].proto, "tcp");
    assert.strictEqual(req.flows[0].dur, 0.1214);
  }
});

test("Single flow: multiclass candidate selection ('c8' vs 'c10')", () => {
  const c8Req: MulticlassPredictionRequest = {
    flows: [SAMPLE_EXPLOIT_FLOW],
    model_id: "c8",
  };
  assert.strictEqual(c8Req.model_id, "c8");
  assert.strictEqual(c8Req.flows[0].sttl, 62.0);

  const c10Req: MulticlassPredictionRequest = {
    flows: [SAMPLE_EXPLOIT_FLOW],
    model_id: "c10",
  };
  assert.strictEqual(c10Req.model_id, "c10");
});

test("Batch inference: row mapping with multiple distinct input flows", () => {
  // Create 3 clearly distinguishable input rows with unique byte counts
  const distinctFlow0: NetworkFlowRecord = { ...SAMPLE_BENIGN_FLOW, sbytes: 1111, dbytes: 2222 };
  const distinctFlow1: NetworkFlowRecord = { ...SAMPLE_EXPLOIT_FLOW, sbytes: 3333, dbytes: 4444 };
  const distinctFlow2: NetworkFlowRecord = { ...SAMPLE_DOS_FLOW, sbytes: 5555, dbytes: 6666 };
  const inputBatch = [distinctFlow0, distinctFlow1, distinctFlow2];

  // Backend response returns predictions indexed 0, 1, 2
  const mockResponse: BinaryPredictionResponse = {
    model_id: "binary_winner",
    model_name: "Binary Calibrated XGBoost Winner",
    threshold_preset: "default",
    threshold_applied: 0.4951,
    predictions: [
      {
        index: 0,
        label: 0,
        label_name: "Normal",
        attack_probability: 0.042,
        threshold_applied: 0.4951,
      },
      {
        index: 1,
        label: 1,
        label_name: "Attack",
        attack_probability: 0.912,
        threshold_applied: 0.4951,
      },
      {
        index: 2,
        label: 1,
        label_name: "Attack",
        attack_probability: 0.985,
        threshold_applied: 0.4951,
      },
    ],
    flow_count: 3,
    processing_time_ms: 12.5,
  };

  // Export to CSV
  const csvText = exportBinaryPredictionsToCsv(mockResponse, inputBatch);
  const lines = csvText.trim().split("\r\n");

  assert.strictEqual(lines.length, 4); // 1 header + 3 data rows
  const headers = lines[0].split(",");
  assert.strictEqual(headers[0], "csv_line_number");
  assert.strictEqual(headers[1], "data_row_index");

  // Verify Row 0 maps to line 2, index 0, Normal
  const row0 = lines[1].split(",");
  assert.strictEqual(row0[0], "2"); // physical CSV line 2 (line 1 is header)
  assert.strictEqual(row0[1], "0"); // zero-based index 0
  assert.strictEqual(row0[3], "Normal");

  // Verify Row 1 maps to line 3, index 1, Attack
  const row1 = lines[2].split(",");
  assert.strictEqual(row1[0], "3");
  assert.strictEqual(row1[1], "1");
  assert.strictEqual(row1[3], "Attack");

  // Verify Row 2 maps to line 4, index 2, Attack
  const row2 = lines[3].split(",");
  assert.strictEqual(row2[0], "4");
  assert.strictEqual(row2[1], "2");
  assert.strictEqual(row2[3], "Attack");
});

test("Batch inference: export preserves physical starting line with multiline records independently of data_row_index", () => {
  // Scenario:
  // Line 1: Header
  // Line 2: Record 0 (Normal)
  // Line 3-4: Record 1 (Attack) - spans 2 physical lines
  // Line 5: Record 2 (Attack) - starts on Line 5
  const mockResponse: BinaryPredictionResponse = {
    model_id: "binary_winner",
    model_name: "Binary Calibrated XGBoost Winner",
    threshold_preset: "default",
    threshold_applied: 0.4951,
    predictions: [
      {
        index: 0,
        label: 0,
        label_name: "Normal",
        attack_probability: 0.05,
        threshold_applied: 0.4951,
      },
      {
        index: 1,
        label: 1,
        label_name: "Attack",
        attack_probability: 0.88,
        threshold_applied: 0.4951,
      },
      {
        index: 2,
        label: 1,
        label_name: "Attack",
        attack_probability: 0.95,
        threshold_applied: 0.4951,
      },
    ],
    flow_count: 3,
    processing_time_ms: 10.0,
  };

  const rowMetadata = [
    { dataRowIndex: 0, physicalStartingLine: 2 },
    { dataRowIndex: 1, physicalStartingLine: 3 },
    { dataRowIndex: 2, physicalStartingLine: 5 }, // Physical line 5, NOT 4!
  ];

  const csvText = exportBinaryPredictionsToCsv(mockResponse, undefined, rowMetadata);
  const lines = csvText.trim().split("\r\n");

  assert.strictEqual(lines.length, 4); // 1 header + 3 rows
  const [hLine, r0, r1, r2] = lines;
  assert.ok(hLine.startsWith("csv_line_number,data_row_index"));

  // Record 0: Line 2, index 0
  const cols0 = r0.split(",");
  assert.strictEqual(cols0[0], "2");
  assert.strictEqual(cols0[1], "0");

  // Record 1: Line 3, index 1
  const cols1 = r1.split(",");
  assert.strictEqual(cols1[0], "3");
  assert.strictEqual(cols1[1], "1");

  // Record 2: Line 5, index 2 (NOT Line 4!)
  const cols2 = r2.split(",");
  assert.strictEqual(cols2[0], "5", "Physical line must be 5 due to previous multiline record");
  assert.strictEqual(cols2[1], "2", "Data row index must remain 2");
  assert.notStrictEqual(cols2[0], "4", "Cannot compute physical line number from array index alone");
});

test("Batch inference: multiclass 10-family probabilities and CSV export with row numbers", () => {
  const mockMulticlassRes: MulticlassPredictionResponse = {
    model_id: "c8",
    model_name: "Candidate 8: Imbalance-Aware Multiclass Primary",
    schema_used: "primary_42_features",
    predictions: [
      {
        index: 0,
        predicted_class_id: 2,
        predicted_class_name: "Exploits",
        confidence: 0.884,
        probabilities: {
          Normal: 0.01,
          Generic: 0.02,
          Exploits: 0.884,
          Fuzzers: 0.04,
          DoS: 0.01,
          Reconnaissance: 0.01,
          Analysis: 0.01,
          Backdoor: 0.01,
          Shellcode: 0.003,
          Worms: 0.003,
        },
      },
      {
        index: 1,
        predicted_class_id: 0,
        predicted_class_name: "Normal",
        confidence: 0.965,
        probabilities: {
          Normal: 0.965,
          Generic: 0.01,
          Exploits: 0.005,
          Fuzzers: 0.005,
          DoS: 0.005,
          Reconnaissance: 0.005,
          Analysis: 0.002,
          Backdoor: 0.001,
          Shellcode: 0.001,
          Worms: 0.001,
        },
      },
    ],
    flow_count: 2,
    processing_time_ms: 18.2,
  };

  const csvOut = exportMulticlassPredictionsToCsv(mockMulticlassRes);
  const lines = csvOut.trim().split("\r\n");
  assert.strictEqual(lines.length, 3); // 1 header + 2 data rows

  // Headers check
  assert.ok(lines[0].includes("csv_line_number"));
  assert.ok(lines[0].includes("data_row_index"));
  assert.ok(lines[0].includes("prob_exploits"));

  // Line 1: index 0, line 2, Exploits
  assert.ok(lines[1].startsWith("2,0,2,Exploits,0.884000"));
  // Line 2: index 1, line 3, Normal
  assert.ok(lines[2].startsWith("3,1,0,Normal,0.965000"));
});

test("Error handling: ApiError preserves HTTP status and structured validation details", () => {
  const validationDetail = [
    { field: "body -> flows -> 0 -> dur", message: "Input should be a valid number" },
  ];
  const err = new ApiError(422, "ValidationError", "Input validation failed.", validationDetail);

  assert.strictEqual(err.statusCode, 422);
  assert.strictEqual(err.errorType, "ValidationError");
  assert.strictEqual(err.message, "Input validation failed.");
  assert.deepStrictEqual(err.detail, validationDetail);
});

test("Error handling: ApiError handles network connection failure (status 0)", () => {
  const err = new ApiError(0, "NetworkError", "Backend connection failed: fetch failed");

  assert.strictEqual(err.statusCode, 0);
  assert.strictEqual(err.errorType, "NetworkError");
  assert.ok(err.message.includes("Backend connection failed"));
});
