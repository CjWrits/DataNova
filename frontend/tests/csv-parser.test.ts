import test from "node:test";
import assert from "node:assert";
import { parseRawCsvText, parseRawCsvTextWithErrors, validateAndParseCsv } from "../src/lib/csv-parser.ts";
import { SAMPLE_BENIGN_FLOW } from "../src/lib/sample-data.ts";
import { ALL_42_FEATURE_KEYS } from "../src/lib/feature-metadata.ts";

test("RFC 4180: parse quoted cells with commas and newlines", () => {
  const csv = 'proto,service,state\n"tcp,with,comma","http\nnewline","FIN"';
  const grid = parseRawCsvText(csv);
  assert.strictEqual(grid.length, 2);
  assert.strictEqual(grid[0].length, 3);
  assert.strictEqual(grid[1][0], "tcp,with,comma");
  assert.strictEqual(grid[1][1], "http\nnewline");
  assert.strictEqual(grid[1][2], "FIN");
});

test("RFC 4180: parse escaped double quotes in cells", () => {
  const csv = 'proto,service,state\n"tcp""quoted""protocol",http,FIN';
  const grid = parseRawCsvText(csv);
  assert.strictEqual(grid.length, 2);
  assert.strictEqual(grid[1][0], 'tcp"quoted"protocol');
});

test("RFC 4180: UTF-8 BOM is stripped cleanly", () => {
  const bomCsv = '\uFEFFproto,service,state\ntcp,http,FIN';
  const grid = parseRawCsvText(bomCsv);
  assert.strictEqual(grid.length, 2);
  assert.strictEqual(grid[0][0], "proto");
});

test("RFC 4180: detect unterminated quoted fields at EOF", () => {
  const malformed = 'proto,service,state\n"tcp,http,FIN';
  const res = validateAndParseCsv(malformed, "unterminated.csv");
  assert.strictEqual(res.isValid, false);
  const unclosedErr = res.errors.find((e) => e.message.includes("unterminated quoted field"));
  assert.ok(unclosedErr, "Should detect unterminated quote");
});

test("RFC 4180: detect invalid quote placement (characters after closing quote)", () => {
  const malformed = 'proto,service,state\n"tcp"invalid,http,FIN';
  const res = validateAndParseCsv(malformed, "bad_quote_pos.csv");
  assert.strictEqual(res.isValid, false);
  const badQuoteErr = res.errors.find((e) => e.message.includes("unexpected character"));
  assert.ok(badQuoteErr, "Should detect characters after closing quote");
});

test("RFC 4180: detect unescaped quotes inside unquoted fields", () => {
  const malformed = 'proto,service,state\ntcp"badquote,http,FIN';
  const res = parseRawCsvTextWithErrors(malformed);
  assert.ok(res.syntaxErrors.length > 0);
  assert.ok(res.syntaxErrors.some((e) => e.message.includes("unescaped quote")));
});

test("Schema validation: malformed row width mismatch rejected", () => {
  const header = ALL_42_FEATURE_KEYS.join(",");
  // Row with only 3 columns instead of 42
  const shortRow = "tcp,http,FIN";
  const csv = `${header}\n${shortRow}`;

  const res = validateAndParseCsv(csv, "short_row.csv");
  assert.strictEqual(res.isValid, false);
  const widthErr = res.errors.find((e) => e.message.includes("column count mismatch"));
  assert.ok(widthErr, "Should report column count mismatch on short row");
});

test("Schema validation: empty or blank numerical field rejected", () => {
  const header = ALL_42_FEATURE_KEYS.join(",");
  const rowVals = ALL_42_FEATURE_KEYS.map((k) => (k === "dur" ? "" : SAMPLE_BENIGN_FLOW[k]));
  const csv = `${header}\n${rowVals.join(",")}`;

  const res = validateAndParseCsv(csv, "blank_field.csv");
  assert.strictEqual(res.isValid, false);
  const blankErr = res.errors.find((e) => e.column === "dur" && e.message.includes("missing or blank"));
  assert.ok(blankErr, "Should reject empty numerical field");
});

test("Schema validation: valid 42-feature single row CSV passes", () => {
  const header = ALL_42_FEATURE_KEYS.join(",");
  const rowVals = ALL_42_FEATURE_KEYS.map((k) => SAMPLE_BENIGN_FLOW[k]);
  const csv = `${header}\n${rowVals.join(",")}`;

  const res = validateAndParseCsv(csv, "valid_sample.csv");
  assert.strictEqual(res.isValid, true);
  assert.strictEqual(res.summary.totalDataRows, 1);
  assert.strictEqual(res.summary.validRowCount, 1);
  assert.strictEqual(res.summary.errorCount, 0);
  assert.strictEqual(res.records.length, 1);
  assert.strictEqual(res.records[0].proto, "tcp");
  assert.strictEqual(res.records[0].sbytes, 2520);
});

test("Schema validation: missing required columns detected", () => {
  // Only 3 columns provided
  const csv = "proto,service,state\ntcp,http,FIN";
  const res = validateAndParseCsv(csv, "missing_cols.csv");

  assert.strictEqual(res.isValid, false);
  assert.strictEqual(res.summary.missingRequiredColumns.length, 39);
  assert.strictEqual(res.summary.missingRequiredColumns.includes("dur"), true);
  assert.strictEqual(res.summary.missingRequiredColumns.includes("sbytes"), true);
  assert.strictEqual(res.records.length, 0);
});

test("Schema validation: duplicate header detection", () => {
  const csv = "proto,service,state,proto\ntcp,http,FIN,tcp";
  const res = validateAndParseCsv(csv, "dup_header.csv");

  assert.strictEqual(res.isValid, false);
  const dupError = res.errors.find((e) => e.message.includes("duplicate column"));
  assert.ok(dupError, "Should report duplicate header error");
});

test("Schema validation: non-finite float rejection (NaN, Inf, -Inf)", () => {
  const header = ALL_42_FEATURE_KEYS.join(",");
  const rowValsNan = ALL_42_FEATURE_KEYS.map((k) => (k === "dur" ? "NaN" : SAMPLE_BENIGN_FLOW[k]));
  const rowValsInf = ALL_42_FEATURE_KEYS.map((k) => (k === "rate" ? "Inf" : SAMPLE_BENIGN_FLOW[k]));
  const rowValsNegInf = ALL_42_FEATURE_KEYS.map((k) => (k === "sload" ? "-Inf" : SAMPLE_BENIGN_FLOW[k]));
  const csv = `${header}\n${rowValsNan.join(",")}\n${rowValsInf.join(",")}\n${rowValsNegInf.join(",")}`;

  const res = validateAndParseCsv(csv, "non_finite_test.csv");
  assert.strictEqual(res.isValid, false);
  assert.ok(res.errors.some((e) => e.column === "dur" && e.message.includes("Non-finite")));
  assert.ok(res.errors.some((e) => e.column === "rate" && e.message.includes("Non-finite")));
  assert.ok(res.errors.some((e) => e.column === "sload" && e.message.includes("Non-finite")));
});

test("Schema validation: negative numeric value rejection (ge=0.0 constraint)", () => {
  const header = ALL_42_FEATURE_KEYS.join(",");
  const rowVals = ALL_42_FEATURE_KEYS.map((k) => (k === "sbytes" ? "-50" : SAMPLE_BENIGN_FLOW[k]));
  const csv = `${header}\n${rowVals.join(",")}`;

  const res = validateAndParseCsv(csv, "negative_test.csv");
  assert.strictEqual(res.isValid, false);
  const negError = res.errors.find((e) => e.column === "sbytes" && e.message.includes("negative"));
  assert.ok(negError, "Should reject negative byte count");
});

test("Schema validation: empty file and header-only file", () => {
  const emptyRes = validateAndParseCsv("", "empty.csv");
  assert.strictEqual(emptyRes.isValid, false);
  assert.strictEqual(emptyRes.summary.totalDataRows, 0);

  const headerOnlyRes = validateAndParseCsv(ALL_42_FEATURE_KEYS.join(","), "header_only.csv");
  assert.strictEqual(headerOnlyRes.isValid, false);
  assert.strictEqual(headerOnlyRes.summary.totalDataRows, 0);
});

test("Schema validation: batch size limit safeguard (rejects >1000 rows)", () => {
  const header = ALL_42_FEATURE_KEYS.join(",");
  const singleRow = ALL_42_FEATURE_KEYS.map((k) => SAMPLE_BENIGN_FLOW[k]).join(",");

  const rows = [header];
  for (let i = 0; i < 1005; i++) {
    rows.push(singleRow);
  }
  const bigCsv = rows.join("\n");

  const res = validateAndParseCsv(bigCsv, "oversized.csv");
  assert.strictEqual(res.summary.totalDataRows, 1005);
  assert.strictEqual(res.summary.isOversized, true);
  assert.strictEqual(res.isValid, false);
  const limitErr = res.errors.find((e) => e.message.includes("exceeds the backend maximum batch limit (1,000 rows)"));
  assert.ok(limitErr, "Should detect oversized batch limit violation");
});

test("Physical line tracking: multiline quoted field preserves true starting line and data_row_index independently", () => {
  // Construct a 42-feature valid CSV containing:
  // - Line 1: Header
  // - Line 2: Normal data record
  // - Line 3-4: Quoted multiline field spanning two physical lines
  // - Line 5: Another normal data record
  const header = ALL_42_FEATURE_KEYS.join(",");

  // Normal record 0
  const row0 = ALL_42_FEATURE_KEYS.map((k) => SAMPLE_BENIGN_FLOW[k]).join(",");

  // Multiline record 1: service field contains a newline: "http\ncustom"
  const row1Vals = ALL_42_FEATURE_KEYS.map((k) => {
    if (k === "service") return '"http\ncustom"';
    return SAMPLE_BENIGN_FLOW[k];
  });
  const row1 = row1Vals.join(",");

  // Normal record 2
  const row2Vals = ALL_42_FEATURE_KEYS.map((k) => {
    if (k === "sbytes") return 9999;
    return SAMPLE_BENIGN_FLOW[k];
  });
  const row2 = row2Vals.join(",");

  const csvText = `${header}\n${row0}\n${row1}\n${row2}`;

  // Direct raw parser verification
  const { grid, rowStartingLines, syntaxErrors } = parseRawCsvTextWithErrors(csvText);
  assert.strictEqual(syntaxErrors.length, 0);
  assert.strictEqual(grid.length, 4); // 1 header + 3 data rows
  assert.deepStrictEqual(rowStartingLines, [1, 2, 3, 5]);

  // Full validation & metadata verification
  const result = validateAndParseCsv(csvText, "multiline_test.csv");
  assert.strictEqual(result.isValid, true);
  assert.strictEqual(result.records.length, 3);
  assert.strictEqual(result.rowMetadata.length, 3);

  // Record 0 (Line 2, index 0)
  assert.strictEqual(result.rowMetadata[0].dataRowIndex, 0);
  assert.strictEqual(result.rowMetadata[0].physicalStartingLine, 2);

  // Record 1 (Starts on Line 3, multiline spanning to Line 4, index 1)
  assert.strictEqual(result.rowMetadata[1].dataRowIndex, 1);
  assert.strictEqual(result.rowMetadata[1].physicalStartingLine, 3);
  assert.strictEqual(result.records[1].service, "http\ncustom");

  // Record 2 (Final record starts on Line 5, index 2)
  // Crucial: array index alone (rawIdx + 2) would give 2 + 2 = 4, which is INCORRECT.
  // The true physical starting line must be 5!
  assert.strictEqual(result.rowMetadata[2].dataRowIndex, 2);
  assert.strictEqual(result.rowMetadata[2].physicalStartingLine, 5);
  assert.notStrictEqual(result.rowMetadata[2].physicalStartingLine, 4, "Must not calculate line number from array index alone");
  assert.strictEqual(result.records[2].sbytes, 9999);
});
