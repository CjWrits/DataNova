/**
 * RFC 4180 Compliant CSV Parser and UNSW-NB15 Schema Validator.
 * Handles quoted cells, commas, CRLF/LF linebreaks, and strict numerical/finite validation.
 */

import type { NetworkFlowRecord } from "@/types/api";
import { ALL_42_FEATURE_KEYS } from "./feature-metadata.ts";

export interface CsvValidationError {
  row: number; // 1-indexed, row 1 is header
  column?: string;
  message: string;
}

export interface CsvValidationSummary {
  filename: string;
  totalDataRows: number;
  validRowCount: number;
  errorCount: number;
  columnsFound: string[];
  missingRequiredColumns: string[];
  unexpectedColumns: string[];
  isOversized: boolean; // > 1000 rows
}

export interface CsvRowMetadata {
  dataRowIndex: number; // 0-based data record index
  physicalStartingLine: number; // 1-based physical CSV line where record started
}

export interface CsvParseResult {
  records: NetworkFlowRecord[];
  rowMetadata: CsvRowMetadata[];
  rawRows: Record<string, string>[];
  errors: CsvValidationError[];
  summary: CsvValidationSummary;
  isValid: boolean;
}

const CATEGORICAL_KEYS: (keyof NetworkFlowRecord)[] = ["proto", "service", "state"];
const NUMERICAL_KEYS: (keyof NetworkFlowRecord)[] = ALL_42_FEATURE_KEYS.filter(
  (k) => !CATEGORICAL_KEYS.includes(k)
);

/**
 * Parses raw CSV text into a 2D array of string cells following RFC 4180 rules,
 * capturing syntax errors like unterminated quotes and illegal quote positions,
 * while accurately tracking the physical starting line of every row.
 */
export function parseRawCsvTextWithErrors(csvText: string): {
  grid: string[][];
  rowStartingLines: number[];
  syntaxErrors: CsvValidationError[];
} {
  // Strip UTF-8 BOM if present
  let text = csvText;
  if (text.charCodeAt(0) === 0xfeff) {
    text = text.slice(1);
  }

  const rawRows: { cells: string[]; startingLine: number }[] = [];
  const syntaxErrors: CsvValidationError[] = [];
  let currentRow: string[] = [];
  let currentCell = "";
  let inQuotes = false;
  let justClosedQuote = false;
  let currentLineNumber = 1;
  let rowStartLine = 1;
  let i = 0;
  const len = text.length;

  while (i < len) {
    const char = text[i];

    if (inQuotes) {
      if (char === '"') {
        // Escaped quote ("")
        if (i + 1 < len && text[i + 1] === '"') {
          currentCell += '"';
          i += 2;
          continue;
        } else {
          // Closing quote
          inQuotes = false;
          justClosedQuote = true;
          i++;
          continue;
        }
      } else if (char === "\r") {
        if (i + 1 < len && text[i + 1] === "\n") {
          currentCell += "\r\n";
          currentLineNumber++;
          i += 2;
          continue;
        } else {
          currentCell += "\r";
          currentLineNumber++;
          i++;
          continue;
        }
      } else if (char === "\n") {
        currentCell += "\n";
        currentLineNumber++;
        i++;
        continue;
      } else {
        currentCell += char;
        i++;
        continue;
      }
    } else {
      if (justClosedQuote) {
        // After closing quote, only delimiter, newline, or whitespace is permitted
        if (char === ",") {
          currentRow.push(currentCell.trim());
          currentCell = "";
          justClosedQuote = false;
          i++;
          continue;
        } else if (char === "\r") {
          if (i + 1 < len && text[i + 1] === "\n") {
            i++;
          }
          currentLineNumber++;
          currentRow.push(currentCell.trim());
          rawRows.push({ cells: currentRow, startingLine: rowStartLine });
          currentRow = [];
          currentCell = "";
          justClosedQuote = false;
          rowStartLine = currentLineNumber;
          i++;
          continue;
        } else if (char === "\n") {
          currentLineNumber++;
          currentRow.push(currentCell.trim());
          rawRows.push({ cells: currentRow, startingLine: rowStartLine });
          currentRow = [];
          currentCell = "";
          justClosedQuote = false;
          rowStartLine = currentLineNumber;
          i++;
          continue;
        } else if (char === " " || char === "\t") {
          // Trailing whitespace after quote before comma is tolerated
          i++;
          continue;
        } else {
          // Invalid character after closing quote (e.g. "foo"bar)
          syntaxErrors.push({
            row: currentLineNumber,
            message: `Malformed CSV: unexpected character '${char}' immediately after closing quote in field.`,
          });
          currentCell += char;
          justClosedQuote = false;
          i++;
          continue;
        }
      }

      if (char === '"') {
        if (currentCell.trim().length > 0) {
          // Quote inside unquoted field (e.g. foo"bar)
          syntaxErrors.push({
            row: currentLineNumber,
            message: `Malformed CSV: unescaped quote character in unquoted field.`,
          });
        }
        inQuotes = true;
        i++;
        continue;
      } else if (char === ",") {
        currentRow.push(currentCell.trim());
        currentCell = "";
        i++;
        continue;
      } else if (char === "\r") {
        if (i + 1 < len && text[i + 1] === "\n") {
          i++;
        }
        currentLineNumber++;
        currentRow.push(currentCell.trim());
        rawRows.push({ cells: currentRow, startingLine: rowStartLine });
        currentRow = [];
        currentCell = "";
        rowStartLine = currentLineNumber;
        i++;
        continue;
      } else if (char === "\n") {
        currentLineNumber++;
        currentRow.push(currentCell.trim());
        rawRows.push({ cells: currentRow, startingLine: rowStartLine });
        currentRow = [];
        currentCell = "";
        rowStartLine = currentLineNumber;
        i++;
        continue;
      } else {
        currentCell += char;
        i++;
        continue;
      }
    }
  }

  // Check unterminated quote at EOF
  if (inQuotes) {
    syntaxErrors.push({
      row: currentLineNumber,
      message: `Malformed CSV: unterminated quoted field at end of file (missing closing quote).`,
    });
  }

  // Push last remaining cell & row
  if (currentCell.length > 0 || currentRow.length > 0) {
    currentRow.push(currentCell.trim());
    rawRows.push({ cells: currentRow, startingLine: rowStartLine });
  }

  // Filter out any purely empty trailing lines
  const filteredRows = rawRows.filter(
    (r) => r.cells.length > 1 || (r.cells.length === 1 && r.cells[0].length > 0)
  );
  return {
    grid: filteredRows.map((r) => r.cells),
    rowStartingLines: filteredRows.map((r) => r.startingLine),
    syntaxErrors,
  };
}

export function parseRawCsvText(csvText: string): string[][] {
  return parseRawCsvTextWithErrors(csvText).grid;
}

/**
 * Validates and converts parsed CSV rows into strict NetworkFlowRecord objects.
 */
export function validateAndParseCsv(
  csvText: string,
  filename: string = "uploaded.csv"
): CsvParseResult {
  const errors: CsvValidationError[] = [];
  const rawRows: Record<string, string>[] = [];
  const validRecords: NetworkFlowRecord[] = [];
  const validRowMetadata: CsvRowMetadata[] = [];

  if (!csvText || csvText.trim().length === 0) {
    return {
      records: [],
      rowMetadata: [],
      rawRows: [],
      errors: [{ row: 0, message: "Uploaded file is completely empty." }],
      summary: {
        filename,
        totalDataRows: 0,
        validRowCount: 0,
        errorCount: 1,
        columnsFound: [],
        missingRequiredColumns: ALL_42_FEATURE_KEYS as string[],
        unexpectedColumns: [],
        isOversized: false,
      },
      isValid: false,
    };
  }

  const { grid: rawGrid, rowStartingLines, syntaxErrors } = parseRawCsvTextWithErrors(csvText);
  if (syntaxErrors.length > 0) {
    errors.push(...syntaxErrors);
  }

  if (rawGrid.length < 2) {
    return {
      records: [],
      rowMetadata: [],
      rawRows: [],
      errors: [
        {
          row: 1,
          message:
            "CSV contains only a header line with no data rows (or is malformed).",
        },
      ],
      summary: {
        filename,
        totalDataRows: 0,
        validRowCount: 0,
        errorCount: 1,
        columnsFound: rawGrid.length > 0 ? rawGrid[0] : [],
        missingRequiredColumns: [],
        unexpectedColumns: [],
        isOversized: false,
      },
      isValid: false,
    };
  }

  const headers = rawGrid[0].map((h) => h.toLowerCase().trim());
  const headerColCount = headers.length;

  // Check duplicate headers
  const seenHeaders = new Set<string>();
  const duplicateHeaders = new Set<string>();
  for (const h of headers) {
    if (seenHeaders.has(h)) {
      duplicateHeaders.add(h);
    }
    seenHeaders.add(h);
  }

  if (duplicateHeaders.size > 0) {
    errors.push({
      row: 1,
      message: `CSV header contains duplicate column(s): ${Array.from(duplicateHeaders).join(", ")}.`,
    });
  }

  // Check required 42 features
  const headerSet = new Set(headers);
  const missingRequiredColumns = ALL_42_FEATURE_KEYS.filter((k) => !headerSet.has(k as string)) as string[];
  const unexpectedColumns = headers.filter(
    (h) => !ALL_42_FEATURE_KEYS.includes(h as keyof NetworkFlowRecord)
  );

  if (missingRequiredColumns.length > 0) {
    errors.push({
      row: 1,
      message: `Missing ${missingRequiredColumns.length} required flow predictor(s): ${missingRequiredColumns.slice(0, 10).join(", ")}${missingRequiredColumns.length > 10 ? "..." : ""}`,
    });
  }

  const headerIndexMap = new Map<string, number>();
  headers.forEach((h, idx) => headerIndexMap.set(h, idx));

  const totalDataRows = rawGrid.length - 1;

  for (let rowIndex = 1; rowIndex < rawGrid.length; rowIndex++) {
    const rowCells = rawGrid[rowIndex];
    const physicalStartingLine = rowStartingLines[rowIndex] !== undefined ? rowStartingLines[rowIndex] : rowIndex + 1;
    const dataRowIndex = rowIndex - 1; // 0-based data record index
    const userRowNumber = physicalStartingLine; // 1-indexed physical CSV line where record started

    // Build raw row dict for preview
    const rawDict: Record<string, string> = {};
    headers.forEach((h, idx) => {
      rawDict[h] = rowCells[idx] !== undefined ? rowCells[idx] : "";
    });
    rawRows.push(rawDict);

    // Mismatched column length check
    if (rowCells.length !== headerColCount) {
      errors.push({
        row: userRowNumber,
        message: `Row column count mismatch: expected ${headerColCount} cells, found ${rowCells.length}.`,
      });
      continue;
    }

    // If missing columns in header, row validation cannot produce valid records
    if (missingRequiredColumns.length > 0) {
      continue;
    }

    let rowHasError = false;
    const recordCandidate: Partial<NetworkFlowRecord> = {};

    // Validate categorical features
    for (const catKey of CATEGORICAL_KEYS) {
      const colIdx = headerIndexMap.get(catKey as string)!;
      const rawVal = rowCells[colIdx]?.trim();
      if (!rawVal || rawVal.length === 0) {
        errors.push({
          row: userRowNumber,
          column: catKey as string,
          message: `Categorical field '${catKey}' cannot be empty.`,
        });
        rowHasError = true;
      } else {
        (recordCandidate as Record<string, unknown>)[catKey] = rawVal;
      }
    }

    // Validate numerical features
    for (const numKey of NUMERICAL_KEYS) {
      const colIdx = headerIndexMap.get(numKey as string)!;
      const rawVal = rowCells[colIdx]?.trim();

      if (rawVal === undefined || rawVal === "" || rawVal === null) {
        errors.push({
          row: userRowNumber,
          column: numKey as string,
          message: `Numerical field '${numKey}' is missing or blank.`,
        });
        rowHasError = true;
        continue;
      }

      // Check non-finite words
      const lowerVal = rawVal.toLowerCase();
      if (lowerVal === "nan" || lowerVal === "inf" || lowerVal === "-inf" || lowerVal === "+inf") {
        errors.push({
          row: userRowNumber,
          column: numKey as string,
          message: `Non-finite numeric value '${rawVal}' is not permitted by model schema.`,
        });
        rowHasError = true;
        continue;
      }

      const parsedNum = Number(rawVal);
      if (isNaN(parsedNum) || !Number.isFinite(parsedNum)) {
        errors.push({
          row: userRowNumber,
          column: numKey as string,
          message: `Value '${rawVal}' is not a valid finite number.`,
        });
        rowHasError = true;
        continue;
      }

      if (parsedNum < 0) {
        errors.push({
          row: userRowNumber,
          column: numKey as string,
          message: `Value ${parsedNum} is negative (violates ge=0.0 schema constraint).`,
        });
        rowHasError = true;
        continue;
      }

      (recordCandidate as Record<string, unknown>)[numKey] = parsedNum;
    }

    if (!rowHasError) {
      validRecords.push(recordCandidate as NetworkFlowRecord);
      validRowMetadata.push({ dataRowIndex, physicalStartingLine });
    }
  }

  const isOversized = totalDataRows > 1000;
  if (isOversized) {
    errors.push({
      row: 0,
      message: `Batch contains ${totalDataRows} rows, which exceeds the backend maximum batch limit (1,000 rows). Please partition or reduce file size.`,
    });
  }

  const isValid = errors.length === 0 && validRecords.length === totalDataRows && !isOversized;

  return {
    records: validRecords,
    rowMetadata: validRowMetadata,
    rawRows,
    errors,
    summary: {
      filename,
      totalDataRows,
      validRowCount: validRecords.length,
      errorCount: errors.length,
      columnsFound: headers,
      missingRequiredColumns,
      unexpectedColumns,
      isOversized,
    },
    isValid,
  };
}
