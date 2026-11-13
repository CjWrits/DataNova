"use client";

import { useId, useMemo, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { DRIFT_SCENARIOS } from "@/lib/drift-scenarios";
import { validateAndParseCsv, type CsvParseResult } from "@/lib/csv-parser";
import type {
  DriftMonitoringResponse,
  NetworkFlowRecord,
} from "@/types/api";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Database,
  Filter,
  Info,
  Play,
  Search,
  Shield,
  Sliders,
  Upload,
} from "lucide-react";

export default function DriftMonitorPage() {
  const scenarioSelectId = useId();
  const searchInputId = useId();

  // State: Selected scenario or custom batch
  const [activeTab, setActiveTab] = useState<"preset" | "upload">("preset");
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>(
    DRIFT_SCENARIOS[0].id
  );
  const [customCsvResult, setCustomCsvResult] = useState<CsvParseResult | null>(null);
  const [customFileError, setCustomFileError] = useState<string | null>(null);
  const [batchIdInput, setBatchIdInput] = useState<string>("api_batch_001");

  // State: Execution & Results
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [driftResult, setDriftResult] = useState<DriftMonitoringResponse | null>(null);

  // State: Feature Table Filters
  const [driftLevelFilter, setDriftLevelFilter] = useState<string>("all");
  const [featureSearchQuery, setFeatureSearchQuery] = useState<string>("");

  const currentScenario = useMemo(
    () => DRIFT_SCENARIOS.find((s) => s.id === selectedScenarioId) || DRIFT_SCENARIOS[0],
    [selectedScenarioId]
  );

  // File upload handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCustomFileError(null);
    setCustomCsvResult(null);

    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result as string;
      try {
        const parsed = validateAndParseCsv(text);
        if (parsed.errors.length > 0 && parsed.records.length === 0) {
          setCustomFileError(
            `CSV validation failed: ${parsed.errors[0].message} (Row ${parsed.errors[0].row})`
          );
        } else {
          setCustomCsvResult(parsed);
          setBatchIdInput(`custom_${file.name.replace(/\.[^/.]+$/, "").slice(0, 20)}`);
        }
      } catch (err) {
        setCustomFileError(
          err instanceof Error ? err.message : "Failed to parse CSV file."
        );
      }
    };
    reader.onerror = () => setCustomFileError("Error reading file from disk.");
    reader.readAsText(file);
  };

  // Run drift evaluation
  const handleRunEvaluation = async () => {
    setEvaluating(true);
    setError(null);

    try {
      let flowsToEvaluate: NetworkFlowRecord[] = [];

      if (activeTab === "preset") {
        flowsToEvaluate = currentScenario.generateBatch();
      } else {
        if (!customCsvResult || customCsvResult.records.length === 0) {
          throw new Error("Please upload a valid CSV file with at least 30 flow records.");
        }
        flowsToEvaluate = customCsvResult.records;
      }

      const response = await api.monitorDrift({
        flows: flowsToEvaluate,
        batch_id: batchIdInput || "api_batch_001",
      });

      setDriftResult(response);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(`Drift API Error (${err.statusCode}): ${err.message}`);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unknown error occurred during drift evaluation.");
      }
    } finally {
      setEvaluating(false);
    }
  };

  // Flattened features list for filtering
  const featureList = useMemo(() => {
    if (!driftResult) return [];

    const list: Array<{
      name: string;
      type: "numerical" | "categorical";
      psi: number;
      driftLevel: string;
      ksStat?: number;
      ksPval?: number;
      missingRate?: number;
      nonFiniteRate?: number;
      novelRate?: number;
      novelCategories?: string[];
    }> = [];

    // Numerical
    const numMetrics = driftResult.feature_drift_summary.numerical || {};
    for (const [col, metric] of Object.entries(numMetrics)) {
      list.push({
        name: col,
        type: "numerical",
        psi: metric.psi,
        driftLevel: metric.drift_level,
        ksStat: metric.ks_statistic,
        ksPval: metric.ks_pvalue,
        missingRate: metric.missing_rate,
        nonFiniteRate: metric.non_finite_rate,
      });
    }

    // Categorical
    const catMetrics = driftResult.feature_drift_summary.categorical || {};
    for (const [col, metric] of Object.entries(catMetrics)) {
      list.push({
        name: col,
        type: "categorical",
        psi: metric.psi,
        driftLevel: metric.drift_level,
        novelRate: metric.unknown_category_rate,
        novelCategories: metric.novel_categories,
      });
    }

    // Filter
    return list.filter((item) => {
      if (driftLevelFilter === "drifting" && item.driftLevel === "STABLE") return false;
      if (driftLevelFilter === "critical" && item.driftLevel !== "CRITICAL_DRIFT") return false;
      if (
        featureSearchQuery &&
        !item.name.toLowerCase().includes(featureSearchQuery.toLowerCase())
      ) {
        return false;
      }
      return true;
    });
  }, [driftResult, driftLevelFilter, featureSearchQuery]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              Phase 8D Active
            </span>
            <span className="text-xs text-slate-500">Sliding-Window Statistical Drift Telemetry</span>
          </div>
          <h1 className="text-xl font-bold text-slate-900 mt-1">
            Drift Monitor & Adaptive Telemetry
          </h1>
          <p className="text-xs text-slate-600 mt-1 max-w-2xl">
            Evaluates streaming batches for Population Stability Index (PSI), Kolmogorov-Smirnov
            distribution divergence, prediction entropy collapse, and novel protocol injection
            against the baseline training distribution (151,325 rows).
          </p>
        </div>

        <button
          onClick={handleRunEvaluation}
          disabled={evaluating}
          className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition shadow-sm disabled:opacity-50 self-start md:self-auto shrink-0"
        >
          <Activity className={`h-4 w-4 ${evaluating ? "animate-spin" : ""}`} />
          <span>{evaluating ? "Evaluating Telemetry..." : "Evaluate Telemetry Batch"}</span>
        </button>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start space-x-3 text-xs">
          <AlertCircle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold text-rose-900">Drift Evaluation Error</h4>
            <p className="mt-0.5 text-rose-700">{error}</p>
          </div>
        </div>
      )}

      {/* Telemetry Ingestion Controls */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="border-b border-slate-200 bg-slate-50/70 px-5 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setActiveTab("preset")}
              className={`text-xs font-semibold pb-1 border-b-2 transition ${
                activeTab === "preset"
                  ? "border-emerald-600 text-emerald-700"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              Controlled Research Scenarios (7 Scenarios)
            </button>
            <button
              onClick={() => setActiveTab("upload")}
              className={`text-xs font-semibold pb-1 border-b-2 transition ${
                activeTab === "upload"
                  ? "border-emerald-600 text-emerald-700"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              Upload Custom Telemetry CSV
            </button>
          </div>

          <div className="flex items-center space-x-2 text-xs text-slate-500">
            <Sliders className="h-3.5 w-3.5" />
            <span>Safeguard Threshold: &ge; 30 flows</span>
          </div>
        </div>

        <div className="p-5">
          {activeTab === "preset" ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-1">
                  <label
                    htmlFor={scenarioSelectId}
                    className="block text-xs font-semibold text-slate-700 mb-1"
                  >
                    Select Drift Simulation Scenario
                  </label>
                  <select
                    id={scenarioSelectId}
                    value={selectedScenarioId}
                    onChange={(e) => setSelectedScenarioId(e.target.value)}
                    className="w-full text-xs rounded-lg border-slate-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500 bg-white"
                  >
                    {DRIFT_SCENARIOS.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="md:col-span-1">
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Telemetry Batch ID
                  </label>
                  <input
                    type="text"
                    value={batchIdInput}
                    onChange={(e) => setBatchIdInput(e.target.value)}
                    className="w-full text-xs rounded-lg border-slate-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500"
                    placeholder="api_batch_001"
                  />
                </div>

                <div className="md:col-span-1 flex items-end">
                  <button
                    onClick={handleRunEvaluation}
                    disabled={evaluating}
                    className="w-full py-2 px-3 rounded-lg bg-emerald-600 text-white text-xs font-medium hover:bg-emerald-700 transition shadow-sm flex items-center justify-center space-x-1.5 disabled:opacity-50"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" />
                    <span>Run Scenario Test</span>
                  </button>
                </div>
              </div>

              {/* Scenario details card */}
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-900">
                    {currentScenario.name}
                  </span>
                  <div className="flex items-center space-x-2">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        currentScenario.expectedDriftLevel === "STABLE"
                          ? "bg-emerald-100 text-emerald-800"
                          : currentScenario.expectedDriftLevel === "MODERATE_DRIFT"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-rose-100 text-rose-800"
                      }`}
                    >
                      Expected: {currentScenario.expectedDriftLevel}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-100 text-indigo-800">
                      Action: {currentScenario.expectedAction}
                    </span>
                  </div>
                </div>
                <p className="text-slate-600">{currentScenario.description}</p>
                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <span className="font-semibold text-slate-500">Target Features:</span>
                  {currentScenario.targetFeatures.map((f) => (
                    <span
                      key={f}
                      className="px-1.5 py-0.5 rounded bg-white border border-slate-300 text-[11px] text-slate-700 font-mono"
                    >
                      {f}
                    </span>
                  ))}
                </div>
                <p className="text-[11px] text-slate-500 pt-1">
                  <span className="font-semibold">SOC Rationale:</span>{" "}
                  {currentScenario.operationalRationale}
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center hover:border-emerald-400 transition bg-slate-50/50">
                <Upload className="h-8 w-8 text-slate-400 mx-auto mb-2" />
                <h4 className="text-xs font-semibold text-slate-800">
                  Upload Network Telemetry Batch CSV
                </h4>
                <p className="text-[11px] text-slate-500 mt-1 max-w-md mx-auto">
                  Accepts RFC 4180 CSV containing 42 UNSW-NB15 flow features. Minimum batch
                  safeguard is 30 flows for reliable statistical significance.
                </p>
                <label className="mt-3 inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-white border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-50 cursor-pointer shadow-sm">
                  <span>Browse CSV File</span>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>
              </div>

              {customFileError && (
                <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-700 text-xs flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  <span>{customFileError}</span>
                </div>
              )}

              {customCsvResult && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-800 text-xs flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    <span>
                      Parsed {customCsvResult.records.length} valid flow records cleanly.
                    </span>
                  </div>
                  <button
                    onClick={handleRunEvaluation}
                    disabled={evaluating}
                    className="px-3 py-1 bg-emerald-600 text-white rounded font-medium hover:bg-emerald-700"
                  >
                    Analyze Uploaded Batch
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Results Dashboard */}
      {driftResult && (
        <div className="space-y-6">
          {/* Status KPI Banner */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex items-center space-x-3">
              <div
                className={`h-11 w-11 rounded-lg flex items-center justify-center shrink-0 ${
                  driftResult.overall_drift_status === "STABLE"
                    ? "bg-emerald-100 text-emerald-700"
                    : driftResult.overall_drift_status === "MODERATE_DRIFT"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-rose-100 text-rose-700"
                }`}
              >
                <Activity className="h-6 w-6" />
              </div>
              <div>
                <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                  Overall Drift Status
                </p>
                <h3 className="text-base font-bold text-slate-900 mt-0.5">
                  {driftResult.overall_drift_status}
                </h3>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex items-center space-x-3">
              <div className="h-11 w-11 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center shrink-0">
                <Database className="h-6 w-6" />
              </div>
              <div>
                <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                  Batch Sample Size
                </p>
                <div className="flex items-center space-x-1.5 mt-0.5">
                  <h3 className="text-base font-bold text-slate-900">
                    {driftResult.batch_size}
                  </h3>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      driftResult.is_valid_size
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-rose-100 text-rose-700"
                    }`}
                  >
                    {driftResult.is_valid_size ? "Reliable (≥30)" : "Small (<30)"}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex items-center space-x-3">
              <div className="h-11 w-11 rounded-lg bg-amber-50 text-amber-700 flex items-center justify-center shrink-0">
                <Sliders className="h-6 w-6" />
              </div>
              <div>
                <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                  Drifting Features
                </p>
                <h3 className="text-base font-bold text-slate-900 mt-0.5">
                  {driftResult.drifting_features_count}{" "}
                  <span className="text-xs font-normal text-slate-500">
                    ({driftResult.critical_features_count} critical)
                  </span>
                </h3>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex items-center space-x-3">
              <div className="h-11 w-11 rounded-lg bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
                <Shield className="h-6 w-6" />
              </div>
              <div>
                <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                  Latency & Provenance
                </p>
                <h3 className="text-base font-bold text-slate-900 mt-0.5">
                  {driftResult.processing_time_ms} ms
                </h3>
              </div>
            </div>
          </div>

          {/* Adaptive Decision Engine Guidance Card */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-100 pb-3">
              <div className="flex items-center space-x-2">
                <Shield className="h-5 w-5 text-indigo-600" />
                <h2 className="text-sm font-bold text-slate-900">
                  Adaptive Response Engine — SOC Guidance
                </h2>
              </div>
              <div className="flex items-center space-x-2">
                {driftResult.requires_soc_escalation && (
                  <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-rose-100 text-rose-800 border border-rose-200 animate-pulse">
                    SOC Escalation Required
                  </span>
                )}
                {driftResult.requires_pipeline_investigation && (
                  <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-100 text-amber-800 border border-amber-200">
                    Pipeline Investigation Required
                  </span>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-1 bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
                <p className="text-[11px] font-semibold text-slate-500 uppercase">
                  Primary Recommendation
                </p>
                <span className="inline-block px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-indigo-600 text-white shadow-sm">
                  {driftResult.primary_recommendation}
                </span>
                <p className="text-xs text-slate-600 pt-2">
                  Deterministic, non-destructive policy response. Automatically avoids
                  unverified automated retraining while providing explicit operational guidance.
                </p>
              </div>

              <div className="md:col-span-2 space-y-3">
                {driftResult.decisions.map((decision, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-lg border border-slate-200 bg-white space-y-1.5 shadow-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-slate-900">
                        {decision.action}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          decision.priority === "CRITICAL"
                            ? "bg-rose-100 text-rose-800"
                            : decision.priority === "HIGH"
                            ? "bg-amber-100 text-amber-800"
                            : "bg-slate-100 text-slate-700"
                        }`}
                      >
                        {decision.priority}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600">{decision.trigger_reason}</p>
                    <p className="text-[11px] text-indigo-900 bg-indigo-50/70 p-2 rounded border border-indigo-100">
                      <span className="font-semibold">Operational Guidance:</span>{" "}
                      {decision.operational_guidance}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Prediction Drift & Model Degradation Telemetry */}
          {driftResult.prediction_drift_summary && (
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-emerald-600" />
                  <h2 className="text-sm font-bold text-slate-900">
                    Model Prediction Drift & Confidence Degradation
                  </h2>
                </div>
                <span className="text-xs font-mono text-slate-500">
                  Model: {driftResult.prediction_drift_summary.model_id}
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase">
                    Prediction PSI
                  </p>
                  <p className="text-base font-bold text-indigo-700 font-mono mt-0.5">
                    {driftResult.prediction_drift_summary.prediction_psi.toFixed(4)}
                  </p>
                  <p className="text-[10px] text-slate-400">Class probability redistribution</p>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase">
                    Mean Confidence
                  </p>
                  <p className="text-base font-bold text-slate-900 font-mono mt-0.5">
                    {(driftResult.prediction_drift_summary.mean_confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-slate-400">Average top softmax score</p>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase">
                    Low-Conf Rate (&lt;60%)
                  </p>
                  <p className="text-base font-bold text-amber-700 font-mono mt-0.5">
                    {(driftResult.prediction_drift_summary.low_confidence_rate * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-slate-400">Borderline flows</p>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase">
                    Prediction Entropy
                  </p>
                  <p className="text-base font-bold text-slate-900 font-mono mt-0.5">
                    {driftResult.prediction_drift_summary.mean_entropy.toFixed(3)}
                  </p>
                  <p className="text-[10px] text-slate-400">Shannon entropy H(P)</p>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase">
                    C8 vs C10 Divergence
                  </p>
                  <p className="text-base font-bold text-slate-900 font-mono mt-0.5">
                    {driftResult.prediction_drift_summary.c8_c10_disagreement_rate !== undefined
                      ? `${(driftResult.prediction_drift_summary.c8_c10_disagreement_rate * 100).toFixed(1)}%`
                      : "N/A"}
                  </p>
                  <p className="text-[10px] text-slate-400">Label disagreement</p>
                </div>
              </div>

              {/* Class distribution visualizer */}
              {driftResult.prediction_drift_summary.predicted_distribution && (
                <div className="space-y-2 pt-2">
                  <p className="text-xs font-semibold text-slate-700">
                    Batch Predicted Attack Family Distribution
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                    {Object.entries(
                      driftResult.prediction_drift_summary.predicted_distribution
                    ).map(([cls, ratio]) => (
                      <div
                        key={cls}
                        className="p-2 rounded bg-slate-50 border border-slate-200 text-xs flex justify-between items-center"
                      >
                        <span className="font-semibold text-slate-700">{cls}</span>
                        <span className="font-mono text-slate-900">
                          {(ratio * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Feature Drift Diagnostic Table */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="p-5 border-b border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Feature-Level Distribution Drift & Stability Table
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  10-bin Population Stability Index (PSI) and Kolmogorov-Smirnov test against baseline.
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <div className="relative">
                  <Search className="h-3.5 w-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                  <input
                    id={searchInputId}
                    type="text"
                    value={featureSearchQuery}
                    onChange={(e) => setFeatureSearchQuery(e.target.value)}
                    placeholder="Search feature..."
                    className="pl-8 pr-3 py-1.5 text-xs rounded-lg border-slate-300 focus:border-emerald-500 focus:ring-emerald-500 w-44"
                  />
                </div>

                <div className="flex items-center space-x-1.5 bg-slate-100 p-1 rounded-lg text-xs">
                  <Filter className="h-3.5 w-3.5 text-slate-500 ml-1" />
                  <button
                    onClick={() => setDriftLevelFilter("all")}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      driftLevelFilter === "all"
                        ? "bg-white text-slate-900 shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    All ({featureList.length})
                  </button>
                  <button
                    onClick={() => setDriftLevelFilter("drifting")}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      driftLevelFilter === "drifting"
                        ? "bg-white text-amber-700 shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    Drifting
                  </button>
                  <button
                    onClick={() => setDriftLevelFilter("critical")}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      driftLevelFilter === "critical"
                        ? "bg-white text-rose-700 shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    Critical
                  </button>
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-600">
                <thead className="bg-slate-50 text-[11px] uppercase tracking-wider text-slate-500 border-b border-slate-200">
                  <tr>
                    <th className="px-4 py-3 font-semibold">Feature Name</th>
                    <th className="px-4 py-3 font-semibold">Type</th>
                    <th className="px-4 py-3 font-semibold">PSI Score & Bar</th>
                    <th className="px-4 py-3 font-semibold">KS Statistic</th>
                    <th className="px-4 py-3 font-semibold">KS p-value</th>
                    <th className="px-4 py-3 font-semibold">Drift Level</th>
                    <th className="px-4 py-3 font-semibold">Data Quality</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {featureList.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="text-center py-6 text-slate-400">
                        No features match the selected filter criteria.
                      </td>
                    </tr>
                  ) : (
                    featureList.map((item) => (
                      <tr key={item.name} className="hover:bg-slate-50/70 transition">
                        <td className="px-4 py-2.5 font-mono font-semibold text-slate-900">
                          {item.name}
                        </td>
                        <td className="px-4 py-2.5">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                              item.type === "numerical"
                                ? "bg-slate-100 text-slate-700"
                                : "bg-purple-50 text-purple-700 border border-purple-200"
                            }`}
                          >
                            {item.type}
                          </span>
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center space-x-2">
                            <span className="font-mono font-bold text-slate-900 w-12 text-right">
                              {item.psi.toFixed(4)}
                            </span>
                            <div className="w-24 bg-slate-100 rounded-full h-2 overflow-hidden">
                              <div
                                className={`h-2 rounded-full ${
                                  item.psi < 0.1
                                    ? "bg-emerald-500"
                                    : item.psi < 0.25
                                    ? "bg-amber-500"
                                    : "bg-rose-500"
                                }`}
                                style={{
                                  width: `${Math.min(100, (item.psi / 0.5) * 100)}%`,
                                }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-2.5 font-mono">
                          {item.ksStat !== undefined ? item.ksStat.toFixed(4) : "—"}
                        </td>
                        <td className="px-4 py-2.5 font-mono">
                          {item.ksPval !== undefined ? (
                            <span
                              className={
                                item.ksPval < 0.01 ? "text-rose-600 font-bold" : ""
                              }
                            >
                              {item.ksPval.toExponential(2)}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="px-4 py-2.5">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              item.driftLevel === "STABLE"
                                ? "bg-emerald-100 text-emerald-800"
                                : item.driftLevel === "MODERATE_DRIFT"
                                ? "bg-amber-100 text-amber-800"
                                : "bg-rose-100 text-rose-800"
                            }`}
                          >
                            {item.driftLevel}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-[11px]">
                          {item.type === "numerical" ? (
                            <span>
                              Miss: {(item.missingRate || 0).toFixed(1)}% | Inf:{" "}
                              {(item.nonFiniteRate || 0).toFixed(1)}%
                            </span>
                          ) : (
                            <span>
                              Novel: {((item.novelRate || 0) * 100).toFixed(1)}%
                              {item.novelCategories && item.novelCategories.length > 0 && (
                                <span className="ml-1 text-purple-700 font-mono">
                                  ({item.novelCategories.join(", ")})
                                </span>
                              )}
                            </span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Reference Dataset & Research Scope Note */}
          <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 text-xs text-slate-600 space-y-1">
            <div className="flex items-center space-x-2 text-slate-800 font-semibold">
              <Info className="h-4 w-4 text-indigo-600" />
              <span>Reference Baseline & Scientific Isolation Note</span>
            </div>
            <p>
              Reference distributions are derived strictly from the training partition (
              <code className="font-mono">train.csv</code>, 151,325 rows). The test partition
              remains untouched. Drift recommendations follow safe deterministic rules to guide SOC
              triage and offline benchmarking without authorizing unverified model retraining.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
