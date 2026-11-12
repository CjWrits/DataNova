/**
 * CSV Export Utility for Batch Inference Predictions.
 * Generates RFC 4180 compliant CSV strings and triggers browser downloads.
 */

import type { BinaryPredictionResponse, MulticlassPredictionResponse, NetworkFlowRecord } from "@/types/api";
import type { CsvRowMetadata } from "./csv-parser";

function escapeCsvCell(val: unknown): string {
  if (val === null || val === undefined) return "";
  const str = String(val);
  if (str.includes(",") || str.includes('"') || str.includes("\n") || str.includes("\r")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function resolveMetadata(
  rawIdx: number,
  flowsOrMetadata?: NetworkFlowRecord[] | CsvRowMetadata[],
  rowMetadata?: CsvRowMetadata[]
): { physicalStartingLine: number; dataRowIndex: number } {
  let metaList: CsvRowMetadata[] | undefined = rowMetadata;
  if (!metaList && Array.isArray(flowsOrMetadata) && flowsOrMetadata.length > 0) {
    const item = flowsOrMetadata[0] as unknown as Record<string, unknown>;
    if ("physicalStartingLine" in item) {
      metaList = flowsOrMetadata as unknown as CsvRowMetadata[];
    }
  }

  const meta = metaList?.[rawIdx];
  return {
    physicalStartingLine:
      meta?.physicalStartingLine !== undefined ? meta.physicalStartingLine : rawIdx + 2,
    dataRowIndex:
      meta?.dataRowIndex !== undefined ? meta.dataRowIndex : rawIdx,
  };
}

export function exportBinaryPredictionsToCsv(
  response: BinaryPredictionResponse,
  flowsOrMetadata?: NetworkFlowRecord[] | CsvRowMetadata[],
  rowMetadata?: CsvRowMetadata[]
): string {
  const headers = [
    "csv_line_number",
    "data_row_index",
    "prediction_label",
    "prediction_name",
    "attack_probability",
    "threshold_applied",
    "margin_from_threshold",
    "model_id",
    "threshold_preset",
  ];

  const lines: string[] = [headers.join(",")];

  response.predictions.forEach((item, idx) => {
    const rawIdx = item.index !== undefined ? item.index : idx;
    const { physicalStartingLine, dataRowIndex } = resolveMetadata(rawIdx, flowsOrMetadata, rowMetadata);
    const margin = item.attack_probability - response.threshold_applied;
    const row = [
      physicalStartingLine,
      dataRowIndex,
      item.label,
      item.label_name,
      item.attack_probability.toFixed(6),
      response.threshold_applied.toFixed(4),
      margin.toFixed(6),
      response.model_id,
      response.threshold_preset,
    ];
    lines.push(row.map(escapeCsvCell).join(","));
  });

  return lines.join("\r\n");
}

export function exportMulticlassPredictionsToCsv(
  response: MulticlassPredictionResponse,
  flowsOrMetadata?: NetworkFlowRecord[] | CsvRowMetadata[],
  rowMetadata?: CsvRowMetadata[]
): string {
  const attackClasses = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
  ];

  const headers = [
    "csv_line_number",
    "data_row_index",
    "predicted_class_id",
    "predicted_class_name",
    "confidence",
    "model_id",
    "schema_used",
    ...attackClasses.map((c) => `prob_${c.toLowerCase()}`),
  ];

  const lines: string[] = [headers.join(",")];

  response.predictions.forEach((item, idx) => {
    const rawIdx = item.index !== undefined ? item.index : idx;
    const { physicalStartingLine, dataRowIndex } = resolveMetadata(rawIdx, flowsOrMetadata, rowMetadata);
    const probValues = attackClasses.map((cls) => {
      const p = item.probabilities?.[cls];
      return p !== undefined ? p.toFixed(6) : "0.000000";
    });

    const row = [
      physicalStartingLine,
      dataRowIndex,
      item.predicted_class_id,
      item.predicted_class_name,
      item.confidence.toFixed(6),
      response.model_id,
      response.schema_used,
      ...probValues,
    ];
    lines.push(row.map(escapeCsvCell).join(","));
  });

  return lines.join("\r\n");
}

export function triggerCsvDownload(csvContent: string, filename: string): void {
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
