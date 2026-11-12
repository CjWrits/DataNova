"use client";

import { useId, useMemo, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { FEATURE_GROUPS } from "@/lib/feature-metadata";
import { validateAndParseCsv, type CsvParseResult } from "@/lib/csv-parser";
import { exportBinaryPredictionsToCsv, exportMulticlassPredictionsToCsv, triggerCsvDownload } from "@/lib/csv-export";
import { SAMPLE_BENIGN_FLOW, SAMPLE_DOS_FLOW, SAMPLE_EXPLOIT_FLOW } from "@/lib/sample-data";
import type {
  BinaryPredictionResponse,
  MulticlassPredictionResponse,
  NetworkFlowRecord,
} from "@/types/api";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDownToLine,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Database,
  FileSpreadsheet,
  Filter,
  Layers,
  Network,
  RefreshCw,
  RotateCcw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type AnalysisMode = "single" | "batch";
type PredictionType = "binary" | "multiclass";
type ThresholdPreset = "default" | "f1_optimal" | "cost_optimal" | "low_false_alarm" | "high_recall";
type MulticlassModelId = "c8" | "c10";

const THRESHOLD_PRESET_OPTIONS: { id: ThresholdPreset; label: string; desc: string }[] = [
  { id: "default", label: "Default Calibrated (0.4951)", desc: "Balanced Phase 4 validation threshold" },
  { id: "f1_optimal", label: "F1-Optimal", desc: "Maximizes harmonic mean of precision and recall" },
  { id: "cost_optimal", label: "Cost-Optimal", desc: "Penalizes missed attacks heavily (lower threshold)" },
  { id: "low_false_alarm", label: "Low False Alarm", desc: "High precision operating point (fewer false alerts)" },
  { id: "high_recall", label: "High Recall (Catch-All)", desc: "Ultra-sensitive boundary to intercept subtle attacks" },
];

const HIGH_SEVERITY_ATTACK_CLASSES = ["Exploits", "Backdoor", "Shellcode", "Worms", "DoS"];

export default function TrafficAnalysisPage() {
  const binaryPresetSelectId = useId();
  const c8RadioId = useId();
  const c10RadioId = useId();
  const samplePresetSelectId = useId();
  const csvFileInputId = useId();

  // Top navigation states
  const [activeTab, setActiveTab] = useState<AnalysisMode>("single");
  const [predictionType, setPredictionType] = useState<PredictionType>("binary");
  const [thresholdPreset, setThresholdPreset] = useState<ThresholdPreset>("default");
  const [multiclassModel, setMulticlassModel] = useState<MulticlassModelId>("c8");

  // Single flow state
  const [flowRecord, setFlowRecord] = useState<NetworkFlowRecord>({ ...SAMPLE_BENIGN_FLOW });
  const [selectedPresetName, setSelectedPresetName] = useState<string>("benign");
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({
    volume: false,
    timing: true,
    application: true,
    density: true,
  });
  const [singleLoading, setSingleLoading] = useState<boolean>(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [singleValidationErrors, setSingleValidationErrors] = useState<Record<string, string>>({});
  const [binarySingleResult, setBinarySingleResult] = useState<BinaryPredictionResponse | null>(null);
  const [multiclassSingleResult, setMulticlassSingleResult] = useState<MulticlassPredictionResponse | null>(null);

  // Batch CSV state
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvParseResult, setCsvParseResult] = useState<CsvParseResult | null>(null);
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [binaryBatchResult, setBinaryBatchResult] = useState<BinaryPredictionResponse | null>(null);
  const [multiclassBatchResult, setMulticlassBatchResult] = useState<MulticlassPredictionResponse | null>(null);
  const [batchPage, setBatchPage] = useState<number>(1);
  const [batchPageSize] = useState<number>(15);
  const [batchFilter, setBatchFilter] = useState<string>("all");

  // Toggle group collapse
  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
  };

  // Preset loader
  const handlePresetChange = (name: string) => {
    setSelectedPresetName(name);
    setSingleError(null);
    setSingleValidationErrors({});
    if (name === "benign") {
      setFlowRecord({ ...SAMPLE_BENIGN_FLOW });
    } else if (name === "exploit") {
      setFlowRecord({ ...SAMPLE_EXPLOIT_FLOW });
    } else if (name === "dos") {
      setFlowRecord({ ...SAMPLE_DOS_FLOW });
    }
  };

  // Reset form
  const handleResetSingle = () => {
    handlePresetChange("benign");
    setBinarySingleResult(null);
    setMulticlassSingleResult(null);
    setSingleError(null);
  };

  // Update single field value
  const handleFieldChange = (key: keyof NetworkFlowRecord, value: string) => {
    const isCat = key === "proto" || key === "service" || key === "state";
    if (isCat) {
      setFlowRecord((prev) => ({ ...prev, [key]: value }));
      if (singleValidationErrors[key]) {
        setSingleValidationErrors((prev) => {
          const next = { ...prev };
          delete next[key];
          return next;
        });
      }
    } else {
      const num = Number(value);
      setFlowRecord((prev) => ({
        ...prev,
        [key]: isNaN(num) ? 0 : num,
      }));
      // Validate
      if (isNaN(num) || !Number.isFinite(num)) {
        setSingleValidationErrors((prev) => ({ ...prev, [key]: "Must be a finite number" }));
      } else if (num < 0) {
        setSingleValidationErrors((prev) => ({ ...prev, [key]: "Cannot be negative (ge=0.0)" }));
      } else {
        if (singleValidationErrors[key]) {
          setSingleValidationErrors((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
          });
        }
      }
    }
  };

  // Execute single flow analysis
  const handleExecuteSingle = async () => {
    setSingleError(null);
    setBinarySingleResult(null);
    setMulticlassSingleResult(null);

    // Validate in-memory record
    if (Object.keys(singleValidationErrors).length > 0) {
      setSingleError("Please resolve all input validation errors before submitting.");
      return;
    }

    setSingleLoading(true);
    try {
      if (predictionType === "binary") {
        const res = await api.predictBinary({
          flows: [flowRecord],
          threshold_preset: thresholdPreset,
        });
        setBinarySingleResult(res);
      } else {
        const res = await api.predictMulticlass({
          flows: [flowRecord],
          model_id: multiclassModel,
        });
        setMulticlassSingleResult(res);
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        let msg = `${err.errorType}: ${err.message}`;
        if (err.detail && Array.isArray(err.detail)) {
          msg += ` Details: ${err.detail.map((d: { field?: string; message?: string }) => `${d.field || ""}: ${d.message || ""}`).join(", ")}`;
        }
        setSingleError(msg);
      } else {
        setSingleError(err instanceof Error ? err.message : "Inference request failed");
      }
    } finally {
      setSingleLoading(false);
    }
  };

  // File upload handler
  const handleFileUpload = (file: File) => {
    setCsvFile(file);
    setCsvParseResult(null);
    setBatchError(null);
    setBinaryBatchResult(null);
    setMulticlassBatchResult(null);
    setBatchPage(1);

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const parseResult = validateAndParseCsv(text, file.name);
      setCsvParseResult(parseResult);
    };
    reader.onerror = () => {
      setBatchError("Failed to read the uploaded CSV file from browser storage.");
    };
    reader.readAsText(file);
  };

  // Remove uploaded file
  const handleRemoveFile = () => {
    setCsvFile(null);
    setCsvParseResult(null);
    setBatchError(null);
    setBinaryBatchResult(null);
    setMulticlassBatchResult(null);
    setBatchPage(1);
  };

  // Execute batch prediction
  const handleExecuteBatch = async () => {
    if (!csvParseResult || !csvParseResult.isValid || csvParseResult.records.length === 0) {
      setBatchError("Cannot execute batch prediction: CSV contains validation errors or is empty.");
      return;
    }
    if (csvParseResult.records.length > 1000) {
      setBatchError("Batch size exceeds the 1,000 flow backend maximum constraint.");
      return;
    }

    setBatchLoading(true);
    setBatchError(null);
    setBinaryBatchResult(null);
    setMulticlassBatchResult(null);
    setBatchPage(1);

    try {
      if (predictionType === "binary") {
        const res = await api.predictBinary({
          flows: csvParseResult.records,
          threshold_preset: thresholdPreset,
        });
        setBinaryBatchResult(res);
      } else {
        const res = await api.predictMulticlass({
          flows: csvParseResult.records,
          model_id: multiclassModel,
        });
        setMulticlassBatchResult(res);
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        let msg = `${err.errorType}: ${err.message}`;
        if (err.detail && Array.isArray(err.detail)) {
          msg += ` Details: ${err.detail.map((d: { field?: string; message?: string }) => `${d.field || ""}: ${d.message || ""}`).join(", ")}`;
        }
        setBatchError(msg);
      } else {
        setBatchError(err instanceof Error ? err.message : "Batch inference request failed");
      }
    } finally {
      setBatchLoading(false);
    }
  };

  // Download batch results as CSV
  const handleDownloadBatchCsv = () => {
    if (binaryBatchResult) {
      const csv = exportBinaryPredictionsToCsv(
        binaryBatchResult,
        csvParseResult?.records,
        csvParseResult?.rowMetadata
      );
      triggerCsvDownload(csv, `binary_predictions_${thresholdPreset}_${Date.now()}.csv`);
    } else if (multiclassBatchResult) {
      const csv = exportMulticlassPredictionsToCsv(
        multiclassBatchResult,
        csvParseResult?.records,
        csvParseResult?.rowMetadata
      );
      triggerCsvDownload(csv, `multiclass_predictions_${multiclassModel}_${Date.now()}.csv`);
    }
  };

  // Filtered batch rows
  const filteredPredictions = useMemo(() => {
    if (binaryBatchResult) {
      return binaryBatchResult.predictions.filter((p) => {
        if (batchFilter === "all") return true;
        if (batchFilter === "attack") return p.label === 1;
        if (batchFilter === "normal") return p.label === 0;
        return true;
      });
    }
    if (multiclassBatchResult) {
      return multiclassBatchResult.predictions.filter((p) => {
        if (batchFilter === "all") return true;
        return p.predicted_class_name.toLowerCase() === batchFilter.toLowerCase();
      });
    }
    return [];
  }, [binaryBatchResult, multiclassBatchResult, batchFilter]);

  // Multiclass chart data
  const multiclassChartData = useMemo(() => {
    if (!multiclassBatchResult) return [];
    const counts: Record<string, number> = {};
    multiclassBatchResult.predictions.forEach((p) => {
      counts[p.predicted_class_name] = (counts[p.predicted_class_name] || 0) + 1;
    });
    return Object.entries(counts).map(([name, count]) => ({
      name,
      count,
      isHighSeverity: HIGH_SEVERITY_ATTACK_CLASSES.includes(name),
    }));
  }, [multiclassBatchResult]);

  // Binary chart data
  const binaryChartData = useMemo(() => {
    if (!binaryBatchResult) return [];
    let normalCount = 0;
    let attackCount = 0;
    binaryBatchResult.predictions.forEach((p) => {
      if (p.label === 1) attackCount++;
      else normalCount++;
    });
    return [
      { name: "Normal", count: normalCount, color: "#10b981" },
      { name: "Attack", count: attackCount, color: "#ef4444" },
    ];
  }, [binaryBatchResult]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Breadcrumb & Header Banner */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                Phase 8B Active
              </span>
              <span className="text-xs text-slate-500">Research Inference Console</span>
            </div>
            <h1 className="text-xl font-bold text-slate-900 mt-1">
              Traffic Analysis: Single Flow & Batch CSV Classification
            </h1>
            <p className="text-xs text-slate-600 mt-1 max-w-3xl">
              Inspect tabular network flow vectors against frozen Phase 4 calibrated models.
              Analyze individual flow payloads or batch-evaluate research datasets up to 1,000 rows.
              <span className="font-medium text-slate-700 ml-1">
                (Scope: Pre-extracted UNSW-NB15 flow features; no raw packet capture or live sniffing).
              </span>
            </p>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex p-1 bg-slate-100 rounded-lg border border-slate-200 shrink-0 self-start md:self-auto">
            <button
              onClick={() => setActiveTab("single")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                activeTab === "single"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Network className="h-3.5 w-3.5" />
              <span>Single Flow</span>
            </button>
            <button
              onClick={() => setActiveTab("batch")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                activeTab === "batch"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <FileSpreadsheet className="h-3.5 w-3.5" />
              <span>Batch CSV</span>
            </button>
          </div>
        </div>
      </div>

      {/* Global Prediction Configuration Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-700">Classification Task:</span>
            <div className="flex p-0.5 bg-slate-100 rounded-lg border border-slate-200">
              <button
                onClick={() => setPredictionType("binary")}
                className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                  predictionType === "binary"
                    ? "bg-indigo-600 text-white shadow-xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Binary Detection
              </button>
              <button
                onClick={() => setPredictionType("multiclass")}
                className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                  predictionType === "multiclass"
                    ? "bg-indigo-600 text-white shadow-xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Multiclass (10 Families)
              </button>
            </div>
          </div>

          {/* Binary Threshold Selector */}
          {predictionType === "binary" && (
            <div className="flex items-center space-x-2">
              <label htmlFor={binaryPresetSelectId} className="font-semibold text-slate-700">Operating Threshold Preset:</label>
              <select
                id={binaryPresetSelectId}
                value={thresholdPreset}
                onChange={(e) => setThresholdPreset(e.target.value as ThresholdPreset)}
                className="bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-xs text-slate-800 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 font-medium"
              >
                {THRESHOLD_PRESET_OPTIONS.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Multiclass Model Selector */}
          {predictionType === "multiclass" && (
            <div className="flex items-center space-x-3">
              <span className="font-semibold text-slate-700">Candidate Architecture:</span>
              <label htmlFor={c8RadioId} className="inline-flex items-center space-x-1.5 cursor-pointer">
                <input
                  type="radio"
                  id={c8RadioId}
                  name="mc_model"
                  value="c8"
                  checked={multiclassModel === "c8"}
                  onChange={() => setMulticlassModel("c8")}
                  className="text-indigo-600 focus:ring-indigo-500 text-xs"
                />
                <span className="text-slate-800 font-medium">Candidate 8 (Primary w/ STTL)</span>
              </label>
              <label htmlFor={c10RadioId} className="inline-flex items-center space-x-1.5 cursor-pointer">
                <input
                  type="radio"
                  id={c10RadioId}
                  name="mc_model"
                  value="c10"
                  checked={multiclassModel === "c10"}
                  onChange={() => setMulticlassModel("c10")}
                  className="text-indigo-600 focus:ring-indigo-500 text-xs"
                />
                <span className="text-slate-800 font-medium">Candidate 10 (No-STTL Invariant)</span>
              </label>
            </div>
          )}
        </div>

        {/* Operating Presets Description Hint */}
        <div className="text-slate-500 text-[11px] flex items-center space-x-1.5 shrink-0">
          <Activity className="h-3.5 w-3.5 text-indigo-500" />
          <span>
            {predictionType === "binary"
              ? THRESHOLD_PRESET_OPTIONS.find((p) => p.id === thresholdPreset)?.desc
              : multiclassModel === "c8"
              ? "C8: Full 42-feature schema; highest validation Macro-F1 (0.8359)"
              : "C10: Excludes STTL; robust against TTL-spoofing distribution shifts"}
          </span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MODE 1: SINGLE FLOW ANALYSIS WORKSPACE */}
      {/* ========================================================================= */}
      {activeTab === "single" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Form: Feature Inputs */}
          <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="font-semibold text-sm text-slate-900">42-Feature Telemetry Vector</h3>
                <p className="text-[11px] text-slate-500">
                  Pre-populated with verified UNSW-NB15 flow distributions
                </p>
              </div>

              {/* Preset Selector & Reset */}
              <div className="flex items-center space-x-2">
                <label htmlFor={samplePresetSelectId} className="text-xs text-slate-500 font-medium">Load Preset:</label>
                <select
                  id={samplePresetSelectId}
                  value={selectedPresetName}
                  onChange={(e) => handlePresetChange(e.target.value)}
                  className="bg-slate-50 border border-slate-300 rounded-md px-2 py-1 text-xs text-slate-800 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 font-medium"
                >
                  <option value="benign">Normal Web Session (Benign)</option>
                  <option value="exploit">HTTP Exploit Pattern</option>
                  <option value="dos">TCP SYN Flood (DoS)</option>
                  <option value="custom">Custom Edit</option>
                </select>

                <button
                  onClick={handleResetSingle}
                  className="p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition"
                  title="Reset form to default benign flow"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Validation Error Banner */}
            {singleError && (
              <div className="m-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start space-x-2">
                <AlertCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="font-semibold">Inference Error</div>
                  <div className="text-[11px] text-rose-700">{singleError}</div>
                </div>
                <button
                  onClick={() => setSingleError(null)}
                  className="text-rose-500 hover:text-rose-700"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {/* Collapsible Feature Groups Form */}
            <div className="p-4 space-y-4 max-h-[640px] overflow-y-auto">
              {FEATURE_GROUPS.map((group) => {
                const isCollapsed = collapsedGroups[group.id] ?? false;
                return (
                  <div
                    key={group.id}
                    className="border border-slate-200 rounded-lg overflow-hidden bg-slate-50/50"
                  >
                    <button
                      type="button"
                      onClick={() => toggleGroup(group.id)}
                      className="w-full px-4 py-2.5 bg-slate-100/70 hover:bg-slate-100 transition flex items-center justify-between text-left text-xs font-semibold text-slate-800"
                    >
                      <div className="flex items-center space-x-2">
                        {isCollapsed ? (
                          <ChevronRight className="h-4 w-4 text-slate-500" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-slate-500" />
                        )}
                        <span>{group.title}</span>
                      </div>
                      <span className="text-[10px] text-slate-400 font-normal">
                        {group.features.length} features
                      </span>
                    </button>

                    {!isCollapsed && (
                      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 bg-white">
                        {group.features.map((feat) => {
                          const val = flowRecord[feat.key];
                          const errorMsg = singleValidationErrors[feat.key];

                          return (
                            <div key={feat.key} className="space-y-1">
                              <div className="flex items-center justify-between">
                                <label
                                  htmlFor={`input-${feat.key}`}
                                  className="text-[11px] font-medium text-slate-700 truncate"
                                  title={feat.description}
                                >
                                  {feat.label}
                                </label>
                                {feat.key === "sttl" && multiclassModel === "c10" && (
                                  <span className="text-[9px] font-semibold text-amber-600 bg-amber-50 px-1 rounded border border-amber-200">
                                    C10 Excluded
                                  </span>
                                )}
                              </div>

                              {feat.type === "select" ? (
                                <select
                                  id={`input-${feat.key}`}
                                  value={String(val)}
                                  onChange={(e) => handleFieldChange(feat.key, e.target.value)}
                                  className="w-full bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs text-slate-900 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                                >
                                  {feat.options?.map((opt) => (
                                    <option key={opt} value={opt}>
                                      {opt}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <input
                                  type="number"
                                  id={`input-${feat.key}`}
                                  value={val !== undefined ? val : ""}
                                  step={feat.step || "any"}
                                  min={feat.min !== undefined ? feat.min : 0}
                                  onChange={(e) => handleFieldChange(feat.key, e.target.value)}
                                  className={`w-full bg-slate-50 border rounded px-2 py-1 text-xs text-slate-900 focus:outline-hidden focus:ring-1 ${
                                    errorMsg
                                      ? "border-rose-400 focus:ring-rose-400"
                                      : "border-slate-300 focus:ring-indigo-500"
                                  }`}
                                />
                              )}

                              {errorMsg && (
                                <p className="text-[10px] text-rose-600">{errorMsg}</p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Submit Action Bar */}
            <div className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
              <span className="text-[11px] text-slate-500">
                Ready to evaluate: {predictionType === "binary" ? "Binary Intrusion" : `Multiclass (${multiclassModel.toUpperCase()})`}
              </span>
              <button
                type="button"
                onClick={handleExecuteSingle}
                disabled={singleLoading || Object.keys(singleValidationErrors).length > 0}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 disabled:opacity-50 transition shadow-sm"
              >
                {singleLoading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Inference Running...</span>
                  </>
                ) : (
                  <>
                    <Shield className="h-3.5 w-3.5" />
                    <span>Analyze Network Flow</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right Column: Prediction Results Card */}
          <div className="lg:col-span-5 space-y-6">
            {/* Empty state when no analysis performed yet */}
            {!binarySingleResult && !multiclassSingleResult && !singleLoading && (
              <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm text-center">
                <div className="mx-auto h-12 w-12 rounded-full bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600">
                  <Activity className="h-6 w-6" />
                </div>
                <h4 className="font-semibold text-slate-800 text-sm mt-3">Awaiting Flow Submission</h4>
                <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                  Click <span className="font-semibold text-slate-700">Analyze Network Flow</span> to dispatch
                  the 42-feature telemetry record to the FastAPI backend.
                </p>
                <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 gap-2 text-[11px] text-slate-500 text-left">
                  <div className="p-2 rounded bg-slate-50">
                    <span className="font-semibold text-slate-700 block">Binary Mode:</span>
                    Returns attack probability, applied threshold preset, and Normal vs. Attack label.
                  </div>
                  <div className="p-2 rounded bg-slate-50">
                    <span className="font-semibold text-slate-700 block">Multiclass Mode:</span>
                    Returns 10-family probabilities, predicted attack class, and SOC triage flag.
                  </div>
                </div>
              </div>
            )}

            {/* Loading state indicator */}
            {singleLoading && (
              <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm text-center">
                <RefreshCw className="h-8 w-8 text-indigo-600 animate-spin mx-auto" />
                <h4 className="font-semibold text-slate-800 text-sm mt-3">Evaluating Network Telemetry</h4>
                <p className="text-xs text-slate-500 mt-1">
                  Querying frozen model bundle at <code className="font-mono text-indigo-600">/api/v1/predict</code>...
                </p>
              </div>
            )}

            {/* Binary Prediction Single Result */}
            {binarySingleResult && !singleLoading && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-100 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <ShieldCheck className="h-4 w-4 text-indigo-600" />
                    <h3 className="font-semibold text-sm text-slate-900">Binary Detection Decision</h3>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">
                    {binarySingleResult.processing_time_ms.toFixed(2)} ms
                  </span>
                </div>

                <div className="p-5 space-y-5">
                  {binarySingleResult.predictions.map((pred) => {
                    const isAttack = pred.label === 1;
                    const margin = pred.attack_probability - binarySingleResult.threshold_applied;

                    return (
                      <div key={pred.index} className="space-y-4">
                        {/* Major Classification Badge */}
                        <div
                          className={`p-4 rounded-xl border flex items-center justify-between ${
                            isAttack
                              ? "bg-rose-50 border-rose-200 text-rose-900"
                              : "bg-emerald-50 border-emerald-200 text-emerald-900"
                          }`}
                        >
                          <div className="flex items-center space-x-3">
                            <div
                              className={`h-10 w-10 rounded-lg flex items-center justify-center text-white ${
                                isAttack ? "bg-rose-600" : "bg-emerald-600"
                              }`}
                            >
                              {isAttack ? (
                                <ShieldAlert className="h-5 w-5" />
                              ) : (
                                <ShieldCheck className="h-5 w-5" />
                              )}
                            </div>
                            <div>
                              <div className="text-xs font-semibold uppercase tracking-wider opacity-75">
                                Predicted Classification
                              </div>
                              <div className="text-xl font-bold">
                                {isAttack ? "Malicious Intrusion (Attack)" : "Benign Traffic (Normal)"}
                              </div>
                            </div>
                          </div>

                          <div className="text-right">
                            <div className="text-xs font-semibold opacity-75">Probability</div>
                            <div className="text-xl font-mono font-bold">
                              {(pred.attack_probability * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Probability Progress Bar */}
                        <div className="space-y-1.5">
                          <div className="flex justify-between text-xs font-medium text-slate-600">
                            <span>Estimated P(Attack | X): {(pred.attack_probability * 100).toFixed(2)}%</span>
                            <span>Operating Threshold (&tau;): {(binarySingleResult.threshold_applied * 100).toFixed(2)}%</span>
                          </div>
                          <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden relative">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isAttack ? "bg-rose-500" : "bg-emerald-500"
                              }`}
                              style={{ width: `${Math.min(100, pred.attack_probability * 100)}%` }}
                            />
                            {/* Vertical Line for Threshold */}
                            <div
                              className="absolute top-0 bottom-0 w-0.5 bg-slate-900 shadow-xs z-10"
                              style={{ left: `${binarySingleResult.threshold_applied * 100}%` }}
                              title={`Decision Cutoff (\u03c4): ${binarySingleResult.threshold_applied.toFixed(4)}`}
                            />
                          </div>
                          <div className="flex justify-between text-[10px] text-slate-400">
                            <span>0% (Benign)</span>
                            <span className="text-slate-600 font-medium">
                              Classification Margin (P - &tau;): {margin >= 0 ? `+${(margin * 100).toFixed(2)}%` : `${(margin * 100).toFixed(2)}%`}
                            </span>
                            <span>100% (Attack)</span>
                          </div>
                        </div>

                        {/* Telemetry Breakdown Details */}
                        <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-slate-500 text-[11px] block">Model Artifact:</span>
                            <span className="font-medium text-slate-800">{binarySingleResult.model_name}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 text-[11px] block">Threshold Preset:</span>
                            <span className="font-medium text-slate-800 capitalize">
                              {binarySingleResult.threshold_preset} (&tau; = {binarySingleResult.threshold_applied.toFixed(4)})
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 text-[11px] block">Decision Rule:</span>
                            <span className="font-medium text-slate-800">
                              {isAttack ? "P(Attack | X) \u2265 \u03c4 \u2192 Attack" : "P(Attack | X) < \u03c4 \u2192 Normal"}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 text-[11px] block">Runtime Latency:</span>
                            <span className="font-medium text-slate-800">
                              {binarySingleResult.processing_time_ms.toFixed(2)} ms
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Multiclass Single Result */}
            {multiclassSingleResult && !singleLoading && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-100 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Layers className="h-4 w-4 text-indigo-600" />
                    <h3 className="font-semibold text-sm text-slate-900">
                      Multiclass Attack Family Decision
                    </h3>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">
                    {multiclassSingleResult.processing_time_ms.toFixed(2)} ms
                  </span>
                </div>

                <div className="p-5 space-y-5">
                  {multiclassSingleResult.predictions.map((pred) => {
                    const isNormal = pred.predicted_class_name === "Normal";
                    const isHighSeverity = HIGH_SEVERITY_ATTACK_CLASSES.includes(pred.predicted_class_name);

                    // Sort probabilities descending
                    const sortedProbs = Object.entries(pred.probabilities).sort(
                      ([, a], [, b]) => b - a
                    );

                    return (
                      <div key={pred.index} className="space-y-4">
                        {/* Major Classification Badge */}
                        <div
                          className={`p-4 rounded-xl border flex items-center justify-between ${
                            isNormal
                              ? "bg-emerald-50 border-emerald-200 text-emerald-900"
                              : isHighSeverity
                              ? "bg-rose-50 border-rose-200 text-rose-900"
                              : "bg-amber-50 border-amber-200 text-amber-900"
                          }`}
                        >
                          <div className="flex items-center space-x-3">
                            <div
                              className={`h-10 w-10 rounded-lg flex items-center justify-center text-white ${
                                isNormal
                                  ? "bg-emerald-600"
                                  : isHighSeverity
                                  ? "bg-rose-600"
                                  : "bg-amber-600"
                              }`}
                            >
                              {isNormal ? <ShieldCheck className="h-5 w-5" /> : <AlertTriangle className="h-5 w-5" />}
                            </div>
                            <div>
                              <div className="text-xs font-semibold uppercase tracking-wider opacity-75">
                                Predicted Attack Family
                              </div>
                              <div className="text-xl font-bold">{pred.predicted_class_name}</div>
                            </div>
                          </div>

                          <div className="text-right">
                            <div className="text-[11px] font-semibold opacity-75">Max Posterior P(C | X)</div>
                            <div className="text-xl font-mono font-bold">
                              {(pred.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* SOC Severity Triage Indicator */}
                        {!isNormal && (
                          <div
                            className={`p-2.5 rounded-lg border text-xs flex items-center space-x-2 ${
                              isHighSeverity
                                ? "bg-rose-100/50 border-rose-200 text-rose-900"
                                : "bg-amber-100/50 border-amber-200 text-amber-900"
                            }`}
                          >
                            <AlertCircle className="h-4 w-4 shrink-0" />
                            <span>
                              <strong>Operational Triage Notice:</strong>{" "}
                              {isHighSeverity
                                ? "Heuristic priority flag for severe category (Exploits, DoS, Backdoor, Shellcode, Worms). This is an operational workflow flag, not a certainty guarantee."
                                : "Standard attack category (Generic, Fuzzers, Reconnaissance, Analysis). Routed to standard inspection queue."}
                            </span>
                          </div>
                        )}

                        {/* 10-Class Probability Breakdown */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-semibold text-slate-700">
                            10-Family Posterior Probabilities:
                          </h4>
                          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                            {sortedProbs.map(([clsName, prob]) => {
                              const isWinner = clsName === pred.predicted_class_name;
                              return (
                                <div key={clsName} className="text-xs space-y-0.5">
                                  <div className="flex justify-between items-center text-[11px]">
                                    <span
                                      className={`font-medium ${
                                        isWinner ? "text-indigo-900 font-bold" : "text-slate-600"
                                      }`}
                                    >
                                      {clsName} {isWinner && "✓"}
                                    </span>
                                    <span className="font-mono text-slate-500">
                                      {(prob * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                    <div
                                      className={`h-full rounded-full transition-all ${
                                        isWinner
                                          ? isNormal
                                            ? "bg-emerald-500"
                                            : "bg-indigo-600"
                                          : "bg-slate-300"
                                      }`}
                                      style={{ width: `${prob * 100}%` }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>

                        {/* Model Metadata Footnote */}
                        <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-slate-500 text-[11px] block">Model Evaluated:</span>
                            <span className="font-medium text-slate-800">{multiclassSingleResult.model_name}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 text-[11px] block">Feature Schema:</span>
                            <span className="font-medium text-slate-800 font-mono">
                              {multiclassSingleResult.schema_used}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODE 2: BATCH CSV ANALYSIS WORKSPACE */}
      {/* ========================================================================= */}
      {activeTab === "batch" && (
        <div className="space-y-6">
          {/* Upload Zone & Instructions */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
              <div className="flex-1 space-y-2">
                <h3 className="font-semibold text-sm text-slate-900 flex items-center space-x-2">
                  <UploadCloud className="h-4 w-4 text-indigo-600" />
                  <span>Batch Flow Telemetry Ingestion (.csv)</span>
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Upload a research CSV containing tabular network flow vectors. The client will parse RFC 4180
                  quoted values, validate all 42 required features, detect non-finite floats, and enforce the
                  1,000 flow backend batch limit.
                </p>
                <div className="text-[11px] text-slate-500 bg-slate-50 p-2.5 rounded-lg border border-slate-200 flex items-center space-x-2">
                  <Database className="h-3.5 w-3.5 text-indigo-500 shrink-0" />
                  <span>
                    Supported format: 42 UNSW-NB15 predictor features (<code className="font-mono text-slate-700">proto, service, state, dur, spkts, ...</code>).
                    Extra columns like <code className="font-mono text-slate-700">id, label, attack_cat</code> are safely ignored.
                  </span>
                </div>
              </div>

              {/* Upload Drop Area */}
              <div className="w-full md:w-80 shrink-0">
                {!csvFile ? (
                  <label
                    htmlFor={csvFileInputId}
                    className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-300 rounded-xl hover:border-indigo-500 hover:bg-indigo-50/20 transition cursor-pointer text-center"
                  >
                    <UploadCloud className="h-8 w-8 text-slate-400 mb-2" />
                    <span className="text-xs font-semibold text-slate-700">Select or Drag CSV File</span>
                    <span className="text-[11px] text-slate-400 mt-0.5">Up to 1,000 rows max</span>
                    <input
                      id={csvFileInputId}
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(file);
                      }}
                      className="hidden"
                    />
                  </label>
                ) : (
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-2 truncate">
                        <FileSpreadsheet className="h-5 w-5 text-indigo-600 shrink-0" />
                        <div className="truncate">
                          <div className="text-xs font-semibold text-slate-900 truncate">{csvFile.name}</div>
                          <div className="text-[11px] text-slate-400 font-mono">
                            {(csvFile.size / 1024).toFixed(1)} KB
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={handleRemoveFile}
                        className="p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-200 transition"
                        title="Remove file"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>

                    {csvParseResult && (
                      <div className="text-[11px] space-y-1 pt-2 border-t border-slate-200">
                        <div className="flex justify-between text-slate-600">
                          <span>Total Data Rows:</span>
                          <span className="font-semibold">{csvParseResult.summary.totalDataRows}</span>
                        </div>
                        <div className="flex justify-between text-slate-600">
                          <span>Columns Detected:</span>
                          <span className="font-semibold">{csvParseResult.summary.columnsFound.length}</span>
                        </div>
                        <div className="flex justify-between text-slate-600">
                          <span>Validation Status:</span>
                          <span
                            className={`font-semibold ${
                              csvParseResult.isValid ? "text-emerald-600" : "text-rose-600"
                            }`}
                          >
                            {csvParseResult.isValid ? "Valid Schema" : "Validation Errors"}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Validation Report Banner if errors exist */}
          {csvParseResult && !csvParseResult.isValid && (
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 shadow-sm text-xs text-rose-900 space-y-3">
              <div className="flex items-center space-x-2 font-semibold text-sm text-rose-800">
                <AlertTriangle className="h-4 w-4 text-rose-600" />
                <span>CSV Schema Validation Errors Detected ({csvParseResult.errors.length})</span>
              </div>

              {csvParseResult.summary.missingRequiredColumns.length > 0 && (
                <div>
                  <span className="font-semibold text-rose-900">
                    Missing Required Columns ({csvParseResult.summary.missingRequiredColumns.length}):
                  </span>{" "}
                  <span className="font-mono text-rose-700">
                    {csvParseResult.summary.missingRequiredColumns.join(", ")}
                  </span>
                </div>
              )}

              {csvParseResult.summary.isOversized && (
                <div className="font-semibold text-rose-800">
                  Batch contains {csvParseResult.summary.totalDataRows} rows, exceeding the 1,000 flow backend maximum constraint.
                </div>
              )}

              {csvParseResult.errors.length > 0 && (
                <div className="space-y-1 max-h-36 overflow-y-auto bg-white/70 p-2.5 rounded border border-rose-200">
                  {csvParseResult.errors.slice(0, 20).map((err, idx) => (
                    <div key={idx} className="text-[11px] text-rose-800 flex items-start space-x-2">
                      <span className="font-mono font-semibold text-rose-900 shrink-0">
                        {err.row > 0 ? `Row ${err.row}:` : "Header:"}
                      </span>
                      <span>{err.message}</span>
                    </div>
                  ))}
                  {csvParseResult.errors.length > 20 && (
                    <div className="text-[11px] text-rose-600 font-medium">
                      ...and {csvParseResult.errors.length - 20} more errors.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Success Validation Ready State */}
          {csvParseResult && csvParseResult.isValid && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 shadow-sm text-xs text-emerald-900 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" />
                <div>
                  <span className="font-semibold">Ready for Batch Evaluation</span>
                  <p className="text-[11px] text-emerald-700 mt-0.5">
                    Validated {csvParseResult.records.length} flow records against all 42 required schema features.
                  </p>
                </div>
              </div>

              <button
                onClick={handleExecuteBatch}
                disabled={batchLoading}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-700 text-white text-xs font-semibold hover:bg-emerald-800 disabled:opacity-50 transition shadow-sm"
              >
                {batchLoading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Processing Batch...</span>
                  </>
                ) : (
                  <>
                    <Shield className="h-3.5 w-3.5" />
                    <span>Execute Batch Inference ({csvParseResult.records.length} flows)</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Batch Error Banner */}
          {batchError && (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-3">
              <AlertCircle className="h-5 w-5 text-rose-600 shrink-0" />
              <div className="flex-1">
                <div className="font-semibold text-rose-900">Batch Inference Failed</div>
                <div className="text-[11px] text-rose-700">{batchError}</div>
              </div>
              <button onClick={() => setBatchError(null)} className="text-rose-500 hover:text-rose-700">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Upload Preview Table (First 5 Rows) */}
          {csvParseResult && !binaryBatchResult && !multiclassBatchResult && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between text-xs">
                <div className="font-semibold text-slate-800">
                  Data Preview (First {Math.min(5, csvParseResult.rawRows.length)} of {csvParseResult.summary.totalDataRows} rows)
                </div>
                <span className="text-slate-400 text-[11px]">RFC 4180 Parsed</span>
              </div>
              <div className="overflow-x-auto max-h-60">
                <table className="w-full text-[11px] text-left">
                  <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200 sticky top-0">
                    <tr>
                      <th className="px-3 py-2">#</th>
                      {csvParseResult.summary.columnsFound.slice(0, 15).map((col) => (
                        <th key={col} className="px-3 py-2 font-mono truncate max-w-[120px]">
                          {col}
                        </th>
                      ))}
                      {csvParseResult.summary.columnsFound.length > 15 && (
                        <th className="px-3 py-2 text-slate-400">
                          +{csvParseResult.summary.columnsFound.length - 15} more
                        </th>
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700 font-mono">
                    {csvParseResult.rawRows.slice(0, 5).map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/60">
                        <td className="px-3 py-1.5 font-sans font-semibold text-slate-400">{idx + 1}</td>
                        {csvParseResult.summary.columnsFound.slice(0, 15).map((col) => (
                          <td key={col} className="px-3 py-1.5 truncate max-w-[120px]">
                            {row[col] ?? ""}
                          </td>
                        ))}
                        {csvParseResult.summary.columnsFound.length > 15 && (
                          <td className="px-3 py-1.5 text-slate-400">...</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ========================================================================= */}
          {/* BATCH RESULTS DASHBOARD */}
          {/* ========================================================================= */}
          {(binaryBatchResult || multiclassBatchResult) && (
            <div className="space-y-6">
              {/* Batch KPI Overview Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                  <div className="text-xs text-slate-500 font-medium">Batch Evaluated</div>
                  <div className="text-2xl font-bold text-slate-900 mt-1">
                    {(binaryBatchResult || multiclassBatchResult)?.flow_count} flows
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1">
                    Latency: {(binaryBatchResult || multiclassBatchResult)?.processing_time_ms.toFixed(1)} ms
                  </div>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                  <div className="text-xs text-slate-500 font-medium">Model Architecture</div>
                  <div className="text-lg font-bold text-slate-900 mt-1 truncate">
                    {(binaryBatchResult || multiclassBatchResult)?.model_name}
                  </div>
                  <div className="text-[11px] text-indigo-600 mt-1 font-mono">
                    {binaryBatchResult
                      ? `Threshold Preset: ${binaryBatchResult.threshold_preset} (${binaryBatchResult.threshold_applied.toFixed(4)})`
                      : `Schema: ${multiclassBatchResult?.schema_used}`}
                  </div>
                </div>

                {binaryBatchResult && (
                  <>
                    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                      <div className="text-xs text-slate-500 font-medium">Detected Attacks</div>
                      <div className="text-2xl font-bold text-rose-600 mt-1">
                        {binaryBatchResult.predictions.filter((p) => p.label === 1).length}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">
                        {(
                          (binaryBatchResult.predictions.filter((p) => p.label === 1).length /
                            binaryBatchResult.flow_count) *
                          100
                        ).toFixed(1)}
                        % of batch
                      </div>
                    </div>

                    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                      <div className="text-xs text-slate-500 font-medium">Normal Traffic</div>
                      <div className="text-2xl font-bold text-emerald-600 mt-1">
                        {binaryBatchResult.predictions.filter((p) => p.label === 0).length}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">
                        {(
                          (binaryBatchResult.predictions.filter((p) => p.label === 0).length /
                            binaryBatchResult.flow_count) *
                          100
                        ).toFixed(1)}
                        % of batch
                      </div>
                    </div>
                  </>
                )}

                {multiclassBatchResult && (
                  <>
                    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                      <div className="text-xs text-slate-500 font-medium">Critical Threat Flows</div>
                      <div className="text-2xl font-bold text-rose-600 mt-1">
                        {
                          multiclassBatchResult.predictions.filter((p) =>
                            HIGH_SEVERITY_ATTACK_CLASSES.includes(p.predicted_class_name)
                          ).length
                        }
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">
                        Exploits, DoS, Backdoor, Shellcode, Worms
                      </div>
                    </div>

                    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                      <div className="text-xs text-slate-500 font-medium">Mean Confidence</div>
                      <div className="text-2xl font-bold text-indigo-600 mt-1 font-mono">
                        {(
                          (multiclassBatchResult.predictions.reduce((acc, p) => acc + p.confidence, 0) /
                            multiclassBatchResult.flow_count) *
                          100
                        ).toFixed(1)}
                        %
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">Across 10 families</div>
                    </div>
                  </>
                )}
              </div>

              {/* Visualization Section: Distribution Chart */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-sm text-slate-900">
                    {binaryBatchResult
                      ? "Binary Class Distribution (Normal vs. Attack)"
                      : "Multiclass Attack Family Distribution"}
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">Batch Size: {csvParseResult?.records.length}</span>
                </div>

                <div className="h-48 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={binaryBatchResult ? binaryChartData : multiclassChartData}
                      margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#64748b" }} />
                      <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#ffffff",
                          borderRadius: "8px",
                          border: "1px solid #e2e8f0",
                          fontSize: "12px",
                        }}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {binaryBatchResult
                          ? binaryChartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))
                          : multiclassChartData.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={
                                  entry.name === "Normal"
                                    ? "#10b981"
                                    : entry.isHighSeverity
                                    ? "#ef4444"
                                    : "#6366f1"
                                }
                              />
                            ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Row-Level Results Table */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="flex items-center space-x-3">
                    <span className="font-semibold text-slate-800">
                      Row-Level Classification Results ({filteredPredictions.length})
                    </span>

                    {/* Filter dropdown */}
                    <div className="flex items-center space-x-1.5 text-slate-500">
                      <Filter className="h-3.5 w-3.5" />
                      <select
                        value={batchFilter}
                        onChange={(e) => {
                          setBatchFilter(e.target.value);
                          setBatchPage(1);
                        }}
                        className="bg-slate-50 border border-slate-300 rounded px-2 py-0.5 text-xs text-slate-800 font-medium"
                      >
                        <option value="all">All Rows</option>
                        {binaryBatchResult ? (
                          <>
                            <option value="attack">Attack Only</option>
                            <option value="normal">Normal Only</option>
                          </>
                        ) : (
                          multiclassChartData.map((c) => (
                            <option key={c.name} value={c.name}>
                              {c.name} ({c.count})
                            </option>
                          ))
                        )}
                      </select>
                    </div>
                  </div>

                  {/* Export CSV Button */}
                  <button
                    onClick={handleDownloadBatchCsv}
                    className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 transition shadow-xs self-start sm:self-auto"
                  >
                    <ArrowDownToLine className="h-3.5 w-3.5" />
                    <span>Download Predictions CSV</span>
                  </button>
                </div>

                <div className="overflow-x-auto max-h-96">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200 sticky top-0">
                      <tr>
                        <th className="px-3 py-2.5">CSV Line</th>
                        <th className="px-3 py-2.5">Data Index</th>
                        <th className="px-4 py-2.5">Predicted Label</th>
                        {binaryBatchResult && (
                          <>
                            <th className="px-4 py-2.5">Attack Probability</th>
                            <th className="px-4 py-2.5">Applied Threshold</th>
                            <th className="px-4 py-2.5">Margin</th>
                          </>
                        )}
                        {multiclassBatchResult && (
                          <>
                            <th className="px-4 py-2.5">Confidence</th>
                            <th className="px-4 py-2.5">Top Posterior Probabilities</th>
                            <th className="px-4 py-2.5">SOC Severity Flag</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {filteredPredictions
                        .slice((batchPage - 1) * batchPageSize, batchPage * batchPageSize)
                        .map((item) => {
                          const meta = csvParseResult?.rowMetadata?.[item.index];
                          const physicalLine = meta?.physicalStartingLine ?? (item.index + 2);
                          const dataIndex = meta?.dataRowIndex ?? item.index;

                          if ("attack_probability" in item) {
                            // Binary Item
                            const isAttack = item.label === 1;
                            const margin = item.attack_probability - (binaryBatchResult?.threshold_applied || 0.5);
                            return (
                              <tr key={item.index} className="hover:bg-slate-50/60 font-mono text-[11px]">
                                <td className="px-3 py-2 text-slate-700 font-semibold font-mono">Line {physicalLine}</td>
                                <td className="px-3 py-2 text-slate-400 font-sans">[{dataIndex}]</td>
                                <td className="px-4 py-2">
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider font-sans ${
                                      isAttack
                                        ? "bg-rose-100 text-rose-800 border border-rose-200"
                                        : "bg-emerald-100 text-emerald-800 border border-emerald-200"
                                    }`}
                                  >
                                    {item.label_name}
                                  </span>
                                </td>
                                <td className="px-4 py-2 font-bold">
                                  {(item.attack_probability * 100).toFixed(2)}%
                                </td>
                                <td className="px-4 py-2 text-slate-500">
                                  {binaryBatchResult?.threshold_applied.toFixed(4)}
                                </td>
                                <td
                                  className={`px-4 py-2 font-semibold ${
                                    margin >= 0 ? "text-rose-600" : "text-emerald-600"
                                  }`}
                                >
                                  {margin >= 0 ? `+${(margin * 100).toFixed(2)}%` : `${(margin * 100).toFixed(2)}%`}
                                </td>
                              </tr>
                            );
                          } else {
                            // Multiclass Item
                            const isNormal = item.predicted_class_name === "Normal";
                            const isHighSeverity = HIGH_SEVERITY_ATTACK_CLASSES.includes(item.predicted_class_name);
                            return (
                              <tr key={item.index} className="hover:bg-slate-50/60 font-mono text-[11px]">
                                <td className="px-3 py-2 text-slate-700 font-semibold font-mono">Line {physicalLine}</td>
                                <td className="px-3 py-2 text-slate-400 font-sans">[{dataIndex}]</td>
                                <td className="px-4 py-2">
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold font-sans ${
                                      isNormal
                                        ? "bg-emerald-100 text-emerald-800 border border-emerald-200"
                                        : isHighSeverity
                                        ? "bg-rose-100 text-rose-800 border border-rose-200"
                                        : "bg-amber-100 text-amber-800 border border-amber-200"
                                    }`}
                                  >
                                    {item.predicted_class_name}
                                  </span>
                                </td>
                                <td className="px-4 py-2 font-bold">
                                  {(item.confidence * 100).toFixed(2)}%
                                </td>
                                <td className="px-4 py-2 text-[10px] text-slate-500 truncate max-w-xs">
                                  {Object.entries(item.probabilities)
                                    .sort(([, a], [, b]) => b - a)
                                    .slice(0, 3)
                                    .map(([name, p]) => `${name}: ${(p * 100).toFixed(1)}%`)
                                    .join(", ")}
                                </td>
                                <td className="px-4 py-2 font-sans">
                                  {isHighSeverity ? (
                                    <span className="text-[10px] font-semibold text-rose-700 bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200">
                                      High Severity
                                    </span>
                                  ) : isNormal ? (
                                    <span className="text-[10px] text-slate-400 font-medium">Benign</span>
                                  ) : (
                                    <span className="text-[10px] text-amber-700 font-medium">Standard</span>
                                  )}
                                </td>
                              </tr>
                            );
                          }
                        })}
                    </tbody>
                  </table>
                </div>

                {/* Pagination Controls */}
                <div className="px-5 py-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                  <div>
                    Showing{" "}
                    <span className="font-semibold text-slate-800">
                      {Math.min((batchPage - 1) * batchPageSize + 1, filteredPredictions.length)}
                    </span>{" "}
                    to{" "}
                    <span className="font-semibold text-slate-800">
                      {Math.min(batchPage * batchPageSize, filteredPredictions.length)}
                    </span>{" "}
                    of <span className="font-semibold text-slate-800">{filteredPredictions.length}</span> results
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setBatchPage((p) => Math.max(1, p - 1))}
                      disabled={batchPage === 1}
                      className="px-2.5 py-1 rounded border border-slate-200 hover:bg-slate-50 disabled:opacity-40 transition font-medium"
                    >
                      Previous
                    </button>
                    <span className="text-[11px] font-mono">
                      Page {batchPage} of {Math.max(1, Math.ceil(filteredPredictions.length / batchPageSize))}
                    </span>
                    <button
                      onClick={() =>
                        setBatchPage((p) =>
                          Math.min(Math.ceil(filteredPredictions.length / batchPageSize), p + 1)
                        )
                      }
                      disabled={batchPage >= Math.ceil(filteredPredictions.length / batchPageSize)}
                      className="px-2.5 py-1 rounded border border-slate-200 hover:bg-slate-50 disabled:opacity-40 transition font-medium"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
