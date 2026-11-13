"use client";

import { useEffect, useId, useState } from "react";
import { api, ApiError } from "@/lib/api";
import {
  FEATURE_GROUPS,
  PROTO_OPTIONS,
  SERVICE_OPTIONS,
  STATE_OPTIONS,
} from "@/lib/feature-metadata";
import {
  SAMPLE_BENIGN_FLOW,
  SAMPLE_DOS_FLOW,
  SAMPLE_EXPLOIT_FLOW,
} from "@/lib/sample-data";
import type {
  ModelComparisonResponse,
  ModelInfo,
  ModelsListResponse,
  NetworkFlowRecord,
  SystemStatusResponse,
} from "@/types/api";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  Info,
  Layers,
  RefreshCw,
  RotateCcw,
  Scale,
  Sliders,
} from "lucide-react";

export default function ModelLaboratoryPage() {
  const presetSelectId = useId();

  // Model catalog & system telemetry states
  const [modelsList, setModelsList] = useState<ModelsListResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [catalogLoading, setCatalogLoading] = useState<boolean>(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  // Comparison flow input state
  const [flowRecord, setFlowRecord] = useState<NetworkFlowRecord>({
    ...SAMPLE_EXPLOIT_FLOW,
  });
  const [selectedPreset, setSelectedPreset] = useState<string>("exploit");
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({
    protocol: false,
    volume: false,
    timing: true,
    application: true,
    density: true,
  });
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  // Comparison execution states
  const [comparisonLoading, setComparisonLoading] = useState<boolean>(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ModelComparisonResponse | null>(null);

  // Fetch model inventory and system status on mount
  useEffect(() => {
    let isMounted = true;
    const fetchCatalog = async () => {
      setCatalogLoading(true);
      setCatalogError(null);
      try {
        const [modelsRes, statusRes] = await Promise.all([
          api.getModels(),
          api.getSystemStatus().catch(() => null),
        ]);
        if (isMounted) {
          setModelsList(modelsRes);
          setSystemStatus(statusRes);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setCatalogError(
            err instanceof Error ? err.message : "Failed to load model inventory from backend."
          );
        }
      } finally {
        if (isMounted) {
          setCatalogLoading(false);
        }
      }
    };

    fetchCatalog();
    return () => {
      isMounted = false;
    };
  }, []);

  // Toggle group collapse
  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
  };

  // Preset selector
  const handlePresetChange = (name: string) => {
    setSelectedPreset(name);
    setComparisonError(null);
    setValidationErrors({});
    if (name === "benign") {
      setFlowRecord({ ...SAMPLE_BENIGN_FLOW });
    } else if (name === "exploit") {
      setFlowRecord({ ...SAMPLE_EXPLOIT_FLOW });
    } else if (name === "dos") {
      setFlowRecord({ ...SAMPLE_DOS_FLOW });
    }
  };

  // Field change handler
  const handleFieldChange = (key: keyof NetworkFlowRecord, value: string) => {
    const isCat = key === "proto" || key === "service" || key === "state";
    if (isCat) {
      setFlowRecord((prev) => ({ ...prev, [key]: value }));
      if (validationErrors[key]) {
        setValidationErrors((prev) => {
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

      if (isNaN(num) || !Number.isFinite(num)) {
        setValidationErrors((prev) => ({ ...prev, [key]: "Must be a finite number" }));
      } else if (num < 0) {
        setValidationErrors((prev) => ({ ...prev, [key]: "Cannot be negative (ge=0.0)" }));
      } else {
        if (validationErrors[key]) {
          setValidationErrors((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
          });
        }
      }
    }
  };

  // Execute C8 vs C10 comparison
  const handleExecuteComparison = async () => {
    setComparisonError(null);

    if (Object.keys(validationErrors).length > 0) {
      setComparisonError("Please correct numerical input errors before executing model comparison.");
      return;
    }

    setComparisonLoading(true);
    try {
      const res = await api.compareModels({
        flows: [flowRecord],
      });
      setComparisonResult(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        let msg = `${err.errorType}: ${err.message}`;
        if (err.detail && Array.isArray(err.detail)) {
          msg += ` Details: ${err.detail.map((d: { field?: string; message?: string }) => `${d.field || ""}: ${d.message || ""}`).join(", ")}`;
        }
        setComparisonError(msg);
      } else {
        setComparisonError(
          err instanceof Error ? err.message : "Model comparison execution failed."
        );
      }
    } finally {
      setComparisonLoading(false);
    }
  };

  const comparisonItem = comparisonResult?.comparison_items?.[0] || null;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Header Banner */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                Phase 8C Active
              </span>
              <span className="text-xs text-slate-500">Model Architecture & Audit Console</span>
            </div>
            <h1 className="text-xl font-bold text-slate-900 mt-1">
              Model Laboratory: Catalog Inventory & C8 vs. C10 Disagreement Audit
            </h1>
            <p className="text-xs text-slate-600 mt-1 max-w-3xl">
              Inspect active frozen model bundles, evaluate operating threshold calibration points,
              and perform side-by-side comparative inference between Candidate 8 (Primary w/ STTL)
              and Candidate 10 (Robustness Track No-STTL Invariant).
            </p>
          </div>

          <div className="flex items-center space-x-2 self-start md:self-auto shrink-0">
            <span className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
              <FlaskConical className="h-3.5 w-3.5 text-indigo-600" />
              <span>3 Frozen Models Loaded</span>
            </span>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 1: MODEL CATALOG INVENTORY */}
      {/* ========================================================================= */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
            <Layers className="h-4 w-4 text-indigo-600" />
            <span>Active Model Catalog (`/api/v1/models`)</span>
          </h2>
          {systemStatus && (
            <div className="text-[11px] text-slate-500 flex items-center space-x-3 font-mono">
              <span>Memory RSS: {systemStatus.memory_rss_mb.toFixed(1)} MB</span>
              <span>•</span>
              <span>Reference Training Rows: {systemStatus.reference_rows.toLocaleString()}</span>
            </div>
          )}
        </div>

        {catalogLoading ? (
          <div className="bg-white rounded-xl border border-slate-200 p-8 text-center text-xs text-slate-500 animate-pulse">
            Loading frozen model catalog metadata from backend...
          </div>
        ) : catalogError ? (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start space-x-3">
            <AlertCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Catalog Error</div>
              <div>{catalogError}</div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {modelsList?.models.map((m: ModelInfo) => {
              const isBinary = m.task_type === "binary_detection";
              const isC10 = m.id === "c10";

              return (
                <div
                  key={m.id}
                  className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-col justify-between space-y-3"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-slate-800 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                        {m.id}
                      </span>
                      {m.is_default && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
                          Primary Default
                        </span>
                      )}
                      {isC10 && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-200">
                          Robustness Invariant
                        </span>
                      )}
                    </div>

                    <div>
                      <h3 className="font-semibold text-xs text-slate-900 leading-snug">
                        {m.name}
                      </h3>
                      <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                        {m.description}
                      </p>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-slate-100 space-y-1.5 text-[11px] font-mono text-slate-600">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 font-sans">Task Type:</span>
                      <span className="font-semibold text-slate-800">{m.task_type}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 font-sans">Schema:</span>
                      <span>{m.schema_name}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 font-sans">Feature Count:</span>
                      <span className="font-bold text-indigo-700">{m.feature_count} features</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 font-sans">Artifact Version:</span>
                      <span>v{m.version}</span>
                    </div>

                    {isBinary && m.threshold_info && (
                      <div className="mt-2 pt-2 border-t border-slate-100">
                        <div className="text-[10px] font-sans font-semibold text-slate-500 mb-1">
                          Calibrated Operating Presets (τ):
                        </div>
                        <div className="grid grid-cols-2 gap-1 text-[10px]">
                          {Object.entries(m.threshold_info).map(([preset, val]) => (
                            <div key={preset} className="flex justify-between bg-slate-50 px-1.5 py-0.5 rounded">
                              <span className="text-slate-500 truncate">{preset}:</span>
                              <span className="font-bold text-slate-800">{val.toFixed(4)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: SIDE-BY-SIDE MODEL COMPARISON (C8 VS C10) */}
      {/* ========================================================================= */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
            <Scale className="h-4 w-4 text-indigo-600" />
            <span>Side-by-Side Model Comparison: Candidate 8 vs. Candidate 10</span>
          </h2>
          <span className="text-xs text-slate-500">
            Audit feature-sensitivity divergence on identical flow telemetry
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left: Input Telemetry Vector */}
          <div className="lg:col-span-5 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="font-semibold text-sm text-slate-900">Comparative Test Payload</h3>
                <p className="text-[11px] text-slate-500">
                  Feeds simultaneously into C8 (42 feats) and C10 (41 feats)
                </p>
              </div>

              {/* Preset Selector */}
              <div className="flex items-center space-x-2">
                <label htmlFor={presetSelectId} className="text-xs text-slate-600 font-medium">
                  Preset:
                </label>
                <select
                  id={presetSelectId}
                  value={selectedPreset}
                  onChange={(e) => handlePresetChange(e.target.value)}
                  className="bg-slate-50 border border-slate-200 rounded px-2 py-1 text-xs font-medium text-slate-800"
                >
                  <option value="exploit">Sample: Exploits Attack</option>
                  <option value="dos">Sample: DoS Attack</option>
                  <option value="benign">Sample: Benign Flow</option>
                </select>
              </div>
            </div>

            {/* Form Scroll Area */}
            <div className="p-4 max-h-[520px] overflow-y-auto space-y-4">
              {FEATURE_GROUPS.map((group) => {
                const isCollapsed = collapsedGroups[group.id] || false;
                return (
                  <div key={group.id} className="border border-slate-200 rounded-lg overflow-hidden">
                    <button
                      type="button"
                      onClick={() => toggleGroup(group.id)}
                      className="w-full px-3 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-left hover:bg-slate-100/70 transition"
                    >
                      <div>
                        <div className="font-semibold text-xs text-slate-800">{group.title}</div>
                        <div className="text-[10px] text-slate-500">{group.description}</div>
                      </div>
                      {isCollapsed ? (
                        <ChevronRight className="h-4 w-4 text-slate-400" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-slate-400" />
                      )}
                    </button>

                    {!isCollapsed && (
                      <div className="p-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs bg-white">
                        {group.features.map((feat) => {
                          const val = (flowRecord as unknown as Record<string, unknown>)[feat.key];
                          const err = validationErrors[feat.key as string];
                          const isSttl = feat.key === "sttl";

                          return (
                            <div key={feat.key} className="space-y-1">
                              <div className="flex items-center justify-between">
                                <label
                                  htmlFor={`comp-field-${feat.key}`}
                                  className="font-mono text-[11px] text-slate-700 truncate"
                                  title={feat.description}
                                >
                                  {feat.key}
                                </label>
                                {isSttl && (
                                  <span className="text-[9px] text-amber-700 bg-amber-50 px-1 rounded border border-amber-200">
                                    C8 Only (C10 Omits)
                                  </span>
                                )}
                              </div>

                              {feat.type === "select" ? (
                                <select
                                  id={`comp-field-${feat.key}`}
                                  value={String(val ?? "")}
                                  onChange={(e) => handleFieldChange(feat.key, e.target.value)}
                                  className="w-full bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs text-slate-800 font-mono"
                                >
                                  {(feat.key === "proto"
                                    ? PROTO_OPTIONS
                                    : feat.key === "service"
                                    ? SERVICE_OPTIONS
                                    : STATE_OPTIONS
                                  ).map((opt) => (
                                    <option key={opt} value={opt}>
                                      {opt}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <input
                                  id={`comp-field-${feat.key}`}
                                  type="text"
                                  inputMode="decimal"
                                  value={val !== undefined ? String(val) : ""}
                                  onChange={(e) => handleFieldChange(feat.key, e.target.value)}
                                  className={`w-full bg-slate-50 border rounded px-2 py-1 text-xs text-slate-800 font-mono ${
                                    err ? "border-rose-400 bg-rose-50/50" : "border-slate-300"
                                  }`}
                                />
                              )}

                              {err && <div className="text-[10px] text-rose-600 font-sans">{err}</div>}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Actions */}
            <div className="p-4 border-t border-slate-100 bg-slate-50 flex items-center justify-between gap-3">
              <button
                onClick={() => handlePresetChange("benign")}
                className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-xs font-medium text-slate-700 hover:bg-slate-100 transition"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset</span>
              </button>

              <button
                onClick={handleExecuteComparison}
                disabled={comparisonLoading || Object.keys(validationErrors).length > 0}
                className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 disabled:opacity-50 transition shadow-xs"
              >
                {comparisonLoading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Running C8 & C10...</span>
                  </>
                ) : (
                  <>
                    <Scale className="h-3.5 w-3.5" />
                    <span>Execute C8 vs. C10 Comparison</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right: Comparative Evaluation Results */}
          <div className="lg:col-span-7 space-y-4">
            {comparisonError && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start space-x-3">
                <AlertCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold">Comparison Execution Error</div>
                  <div>{comparisonError}</div>
                </div>
              </div>
            )}

            {comparisonLoading && (
              <div className="bg-white rounded-xl border border-slate-200 p-8 text-center space-y-3 animate-pulse">
                <RefreshCw className="h-8 w-8 text-indigo-600 animate-spin mx-auto" />
                <h3 className="font-semibold text-slate-800 text-sm">Evaluating Dual-Track Inference</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Routing flow simultaneously through Primary C8 (42 features) and Robustness C10 (41 features)...
                </p>
              </div>
            )}

            {!comparisonLoading && comparisonResult && comparisonItem && (
              <div className="space-y-4">
                {/* Agreement Status Banner */}
                <div
                  className={`p-4 rounded-xl border flex items-center justify-between ${
                    comparisonItem.agreement
                      ? "bg-emerald-50 border-emerald-200 text-emerald-900"
                      : "bg-amber-50 border-amber-200 text-amber-900"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {comparisonItem.agreement ? (
                      <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" />
                    ) : (
                      <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0" />
                    )}
                    <div>
                      <div className="font-bold text-sm">
                        {comparisonItem.agreement
                          ? "Model Agreement (Cross-Model Consensus)"
                          : "Prediction Disagreement Detected"}
                      </div>
                      <div className="text-xs mt-0.5 opacity-90">
                        {comparisonItem.agreement
                          ? `Both Candidate 8 and Candidate 10 assign the highest probability to '${comparisonItem.c8_predicted_class}'.`
                          : `Candidate 8 predicts '${comparisonItem.c8_predicted_class}', whereas Candidate 10 predicts '${comparisonItem.c10_predicted_class}'.`}
                      </div>
                    </div>
                  </div>

                  <div className="text-right font-mono text-xs shrink-0 pl-3">
                    <div className="text-[10px] uppercase tracking-wider opacity-70 font-sans">
                      L1 Probability Divergence
                    </div>
                    <div className="font-bold text-sm">
                      {comparisonItem.l1_distribution_divergence.toFixed(4)}
                    </div>
                  </div>
                </div>

                {/* Side-by-Side Comparison Columns */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Candidate 8 Card */}
                  <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                      <div>
                        <div className="font-bold text-xs text-slate-900">
                          Candidate 8 (Primary)
                        </div>
                        <div className="text-[10px] text-slate-500 font-mono">
                          42 Features • Includes STTL
                        </div>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                        Macro-F1 0.8359
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                        Predicted Attack Family
                      </div>
                      <div className="text-lg font-bold text-slate-900 font-mono">
                        {comparisonItem.c8_predicted_class}
                      </div>

                      <div className="pt-2 border-t border-slate-100 space-y-1 font-mono text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-400 font-sans">Max Softmax Prob:</span>
                          <span className="font-bold text-slate-800">
                            {(comparisonItem.c8_confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400 font-sans">Observed sttl:</span>
                          <span className="text-indigo-700 font-semibold">
                            {String(flowRecord.sttl)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Candidate 10 Card */}
                  <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                      <div>
                        <div className="font-bold text-xs text-slate-900">
                          Candidate 10 (Robustness)
                        </div>
                        <div className="text-[10px] text-slate-500 font-mono">
                          41 Features • Omits STTL
                        </div>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-200">
                        TTL Invariant
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                        Predicted Attack Family
                      </div>
                      <div className="text-lg font-bold text-slate-900 font-mono">
                        {comparisonItem.c10_predicted_class}
                      </div>

                      <div className="pt-2 border-t border-slate-100 space-y-1 font-mono text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-400 font-sans">Max Softmax Prob:</span>
                          <span className="font-bold text-slate-800">
                            {(comparisonItem.c10_confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400 font-sans">STTL Reliance:</span>
                          <span className="text-slate-400 italic">Excluded by design</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Analytical Interpretation Context */}
                <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 text-xs text-slate-600 space-y-2">
                  <div className="flex items-center space-x-2 font-semibold text-slate-800">
                    <Info className="h-4 w-4 text-indigo-500" />
                    <span>Analytical Interpretation: Operational Guidance</span>
                  </div>
                  {comparisonItem.agreement ? (
                    <div className="space-y-1.5 text-[11px] leading-relaxed">
                      <p>
                        <span className="font-semibold text-slate-800">Cross-Model Consensus:</span> Candidate 8 and Candidate 10 arrive at the same prediction (
                        <span className="font-semibold text-slate-800">
                          {comparisonItem.c8_predicted_class}
                        </span>
                        ). The predicted class is unchanged when Source TTL is excluded, although the probability distributions may differ. The L1 divergence ({comparisonItem.l1_distribution_divergence.toFixed(4)}) quantifies that distribution difference.
                      </p>
                      <p className="text-slate-500 italic">
                        Caution: Cross-model consensus between related tree ensembles does not prove prediction correctness, nor does it confirm whether the flow is benign or malicious. Independent sensor or forensic corroboration remains necessary.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-1.5 text-[11px] leading-relaxed">
                      <p>
                        <span className="font-semibold text-amber-800">Prediction Disagreement:</span> Candidate 8 predicts &apos;{comparisonItem.c8_predicted_class}&apos; while Candidate 10 predicts &apos;{comparisonItem.c10_predicted_class}&apos; (L1 divergence = {comparisonItem.l1_distribution_divergence.toFixed(4)}).
                      </p>
                      <p className="font-medium text-amber-900 bg-amber-50 p-2.5 rounded border border-amber-200">
                        Operational Notice: Prediction changes when Source TTL is excluded. This indicates sensitivity to the feature ablation, not proof of TTL manipulation. Corroborate with independent network evidence before drawing a forensic conclusion.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {!comparisonLoading && !comparisonResult && !comparisonError && (
              <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-3 shadow-sm">
                <Scale className="h-10 w-10 text-slate-400 mx-auto" />
                <h3 className="font-semibold text-slate-800 text-sm">No Comparison Executed</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Click <span className="font-semibold text-slate-700">&quot;Execute C8 vs. C10 Comparison&quot;</span> to
                  evaluate how Primary and Robustness architectures classify the flow.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 3: BINARY THRESHOLD OPERATING CALIBRATION MATRIX */}
      {/* ========================================================================= */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
        <div>
          <div className="flex items-center space-x-2">
            <Sliders className="h-4 w-4 text-indigo-600" />
            <h2 className="text-sm font-bold text-slate-900">
              Binary Threshold Operating Matrix (Validation Optimization)
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Operating points derived strictly on validation data (37,976 samples) by sweeping decision threshold &tau;
            against specific operational objectives. Shifting &tau; adjusts the decision boundary without altering
            the underlying model-estimated probability P(Attack | X).
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Note: Performance metrics reflect results on the frozen validation partition. Selecting an operating threshold is
            a detection policy choice and does not guarantee identical real-world false-alarm or detection rates, nor does it
            authorize automated blocking actions without independent operational verification.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
              <tr>
                <th className="px-4 py-2.5">Preset Name</th>
                <th className="px-4 py-2.5">Threshold (&tau;)</th>
                <th className="px-4 py-2.5">Validation Accuracy</th>
                <th className="px-4 py-2.5">Val Precision</th>
                <th className="px-4 py-2.5">Val Recall</th>
                <th className="px-4 py-2.5">Val F1</th>
                <th className="px-4 py-2.5">Val FPR</th>
                <th className="px-4 py-2.5">Optimization Objective / Provenance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700 font-mono text-[11px]">
              <tr className="hover:bg-slate-50/60">
                <td className="px-4 py-2.5 font-bold text-slate-900 font-sans">
                  Default (0.50 Baseline)
                </td>
                <td className="px-4 py-2.5 font-bold text-indigo-600">0.4951</td>
                <td className="px-4 py-2.5">95.03%</td>
                <td className="px-4 py-2.5">96.68%</td>
                <td className="px-4 py-2.5">95.52%</td>
                <td className="px-4 py-2.5 font-semibold text-slate-900">0.9609</td>
                <td className="px-4 py-2.5">5.85%</td>
                <td className="px-4 py-2.5 font-sans text-slate-500">Default operating threshold (closest grid point to 0.50 on validation sweep; not inherently balanced)</td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="px-4 py-2.5 font-bold text-slate-900 font-sans">
                  F1-Optimal
                </td>
                <td className="px-4 py-2.5 font-bold text-indigo-600">0.5247</td>
                <td className="px-4 py-2.5">95.05%</td>
                <td className="px-4 py-2.5">97.18%</td>
                <td className="px-4 py-2.5">95.04%</td>
                <td className="px-4 py-2.5 font-semibold text-emerald-700">0.9610 (Max)</td>
                <td className="px-4 py-2.5">4.92%</td>
                <td className="px-4 py-2.5 font-sans text-slate-500">Validation-set F1-optimal threshold (maximizes harmonic mean of precision and recall)</td>
              </tr>
              <tr className="hover:bg-slate-50/60 bg-indigo-50/20">
                <td className="px-4 py-2.5 font-bold text-indigo-900 font-sans">
                  Cost-Optimal (Heavy Miss Penalty)
                </td>
                <td className="px-4 py-2.5 font-bold text-indigo-600">0.1783</td>
                <td className="px-4 py-2.5">91.49%</td>
                <td className="px-4 py-2.5">88.50%</td>
                <td className="px-4 py-2.5 font-semibold text-emerald-700">99.68%</td>
                <td className="px-4 py-2.5">0.9375</td>
                <td className="px-4 py-2.5">23.10%</td>
                <td className="px-4 py-2.5 font-sans text-slate-700 font-semibold">
                  Threshold minimizing documented asymmetric validation cost (Cost = 1.0·FP + 5.0·FN)
                </td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="px-4 py-2.5 font-bold text-slate-900 font-sans">
                  Low False Alarm (Precision-Oriented)
                </td>
                <td className="px-4 py-2.5 font-bold text-indigo-600">0.7227</td>
                <td className="px-4 py-2.5">93.91%</td>
                <td className="px-4 py-2.5 font-semibold text-emerald-700">98.80%</td>
                <td className="px-4 py-2.5">91.61%</td>
                <td className="px-4 py-2.5">0.9507</td>
                <td className="px-4 py-2.5 font-bold text-emerald-700">1.98% (≤ 2.0%)</td>
                <td className="px-4 py-2.5 font-sans text-slate-500">Precision-oriented operating threshold (selected for validation FPR ≤ 2.0% constraint; does not authorize automatic blocking)</td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="px-4 py-2.5 font-bold text-slate-900 font-sans">
                  High Recall (Sensitivity-Oriented)
                </td>
                <td className="px-4 py-2.5 font-bold text-indigo-600">0.3268</td>
                <td className="px-4 py-2.5">94.11%</td>
                <td className="px-4 py-2.5">93.05%</td>
                <td className="px-4 py-2.5 font-semibold text-emerald-700">98.14% (≥ 98.0%)</td>
                <td className="px-4 py-2.5">0.9553</td>
                <td className="px-4 py-2.5">13.06%</td>
                <td className="px-4 py-2.5 font-sans text-slate-500">Recall-oriented operating threshold (selected for validation Recall ≥ 98.0% constraint; high-sensitivity catch-all)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
