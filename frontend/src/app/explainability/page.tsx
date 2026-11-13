"use client";

import { useId, useMemo, useState } from "react";
import { api, ApiError } from "@/lib/api";
import {
  FEATURE_GROUPS,
  FEATURE_METADATA_MAP,
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
  ExplainResponse,
  FlowExplanationItem,
  MulticlassModelId,
  NetworkFlowRecord,
} from "@/types/api";
import {
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  Clock,
  HelpCircle,
  Info,
  Layers,
  RefreshCw,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ChartBarData {
  name: string;
  label: string;
  value: number;
  absVal: number;
  direction: "push_towards_class" | "push_away_from_class";
  observedValue: string | number;
}

export default function ExplainabilityPage() {
  const modelSelectId = useId();
  const topKSelectId = useId();
  const presetSelectId = useId();

  // Model & configuration states
  const [selectedModel, setSelectedModel] = useState<MulticlassModelId>("c8");
  const [topK, setTopK] = useState<number>(10);
  const [selectedPreset, setSelectedPreset] = useState<string>("exploit");

  // Flow telemetry form state
  const [flowRecord, setFlowRecord] = useState<NetworkFlowRecord>({
    ...SAMPLE_EXPLOIT_FLOW,
  });
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({
    protocol: false,
    volume: false,
    timing: true,
    application: true,
    density: true,
  });
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  // API query states
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [explanationResponse, setExplanationResponse] = useState<ExplainResponse | null>(null);

  // Toggle group collapse
  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
  };

  // Preset selector
  const handlePresetChange = (name: string) => {
    setSelectedPreset(name);
    setError(null);
    setValidationErrors({});
    if (name === "benign") {
      setFlowRecord({ ...SAMPLE_BENIGN_FLOW });
    } else if (name === "exploit") {
      setFlowRecord({ ...SAMPLE_EXPLOIT_FLOW });
    } else if (name === "dos") {
      setFlowRecord({ ...SAMPLE_DOS_FLOW });
    }
  };

  // Field change handler with validation
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

  // Submit flow to TreeExplainer
  const handleExecuteExplain = async () => {
    setError(null);

    if (Object.keys(validationErrors).length > 0) {
      setError("Please correct numerical validation errors before generating explanations.");
      return;
    }

    setLoading(true);
    try {
      const res = await api.explainFlows({
        flows: [flowRecord],
        model_id: selectedModel,
        top_k: topK,
      });
      setExplanationResponse(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        let msg = `${err.errorType}: ${err.message}`;
        if (err.detail && Array.isArray(err.detail)) {
          msg += ` Details: ${err.detail.map((d: { field?: string; message?: string }) => `${d.field || ""}: ${d.message || ""}`).join(", ")}`;
        }
        setError(msg);
      } else {
        setError(err instanceof Error ? err.message : "Explanation generation failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Active explanation item
  const activeExplanation: FlowExplanationItem | null = useMemo(() => {
    if (!explanationResponse || explanationResponse.explanations.length === 0) {
      return null;
    }
    return explanationResponse.explanations[0];
  }, [explanationResponse]);

  // Chart data formatted for horizontal bar chart
  const chartData: ChartBarData[] = useMemo(() => {
    if (!activeExplanation) return [];

    return activeExplanation.top_attributions.map((attr) => {
      const observed = (flowRecord as unknown as Record<string, unknown>)[attr.feature_name];
      const meta = FEATURE_METADATA_MAP[attr.feature_name];
      return {
        name: attr.feature_name,
        label: meta ? meta.label : attr.feature_name,
        value: attr.attribution_margin,
        absVal: Math.abs(attr.attribution_margin),
        direction: attr.direction,
        observedValue: observed !== undefined ? (observed as string | number) : "-",
      };
    });
  }, [activeExplanation, flowRecord]);

  // Top positive and negative contributors
  const topContributors = useMemo(() => {
    if (!activeExplanation) return { positive: null, negative: null };
    const pos = activeExplanation.top_attributions.find((a) => a.attribution_margin > 0) || null;
    const neg = activeExplanation.top_attributions.find((a) => a.attribution_margin < 0) || null;
    return { positive: pos, negative: neg };
  }, [activeExplanation]);

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
              <span className="text-xs text-slate-500">Model Interpretability Console</span>
            </div>
            <h1 className="text-xl font-bold text-slate-900 mt-1">
              Explainability Lab: SHAP Feature Attribution & Local Diagnostics
            </h1>
            <p className="text-xs text-slate-600 mt-1 max-w-3xl">
              Inspect exact tree-based SHAP contributions computed via TreeExplainer for frozen
              XGBoost multiclass models. Examine how individual telemetry features push the
              pre-softmax log-odds margin toward or away from specific attack families.
              <span className="font-medium text-slate-700 ml-1">
                (Units: Raw pre-softmax logit margins; attributions reflect model sensitivity rather than physical causality).
              </span>
            </p>
          </div>

          <div className="flex items-center space-x-2 self-start md:self-auto shrink-0">
            <span className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
              <Sparkles className="h-3.5 w-3.5 text-indigo-600" />
              <span>SHAP TreeExplainer</span>
            </span>
          </div>
        </div>
      </div>

      {/* Global Configuration Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-4">
          {/* Target Model Selector */}
          <div className="flex items-center space-x-2">
            <label htmlFor={modelSelectId} className="font-semibold text-slate-700">
              Target Architecture:
            </label>
            <select
              id={modelSelectId}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as MulticlassModelId)}
              className="bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-xs text-slate-800 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 font-medium"
            >
              <option value="c8">Candidate 8 (Primary w/ STTL - 42 features)</option>
              <option value="c10">Candidate 10 (Robustness Track No-STTL - 41 features)</option>
            </select>
          </div>

          {/* Top-K Selector */}
          <div className="flex items-center space-x-2">
            <label htmlFor={topKSelectId} className="font-semibold text-slate-700">
              Top Features (k):
            </label>
            <select
              id={topKSelectId}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-xs text-slate-800 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 font-medium"
            >
              <option value={5}>Top 5 Features</option>
              <option value={10}>Top 10 Features (Default)</option>
              <option value={15}>Top 15 Features</option>
              <option value={20}>Top 20 Features</option>
              <option value={42}>All 42 Features</option>
            </select>
          </div>
        </div>

        <div className="text-slate-500 text-[11px] flex items-center space-x-1.5 shrink-0">
          <Info className="h-3.5 w-3.5 text-indigo-500" />
          <span>
            {selectedModel === "c8"
              ? "C8: Includes synthetic STTL artifact; highest validation Macro-F1 (0.8359)"
              : "C10: Invariant to STTL spoofing; robust under TTL distribution shifts"}
          </span>
        </div>
      </div>

      {/* Main Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: 42-Feature Telemetry Vector Input */}
        <div className="lg:col-span-5 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="font-semibold text-sm text-slate-900">Input Telemetry Vector</h3>
              <p className="text-[11px] text-slate-500">
                Inspect and modify individual flow attributes
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
          <div className="p-4 max-h-[620px] overflow-y-auto space-y-4">
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
                        const isSttlDisabled = selectedModel === "c10" && feat.key === "sttl";

                        return (
                          <div key={feat.key} className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label
                                htmlFor={`field-${feat.key}`}
                                className={`font-mono text-[11px] truncate ${
                                  isSttlDisabled ? "text-slate-400 line-through" : "text-slate-700"
                                }`}
                                title={feat.description}
                              >
                                {feat.key}
                              </label>
                              {isSttlDisabled && (
                                <span className="text-[9px] text-amber-600 bg-amber-50 px-1 rounded">
                                  C10 Invariant
                                </span>
                              )}
                            </div>

                            {feat.type === "select" ? (
                              <select
                                id={`field-${feat.key}`}
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
                                id={`field-${feat.key}`}
                                type="text"
                                inputMode="decimal"
                                value={val !== undefined ? String(val) : ""}
                                disabled={isSttlDisabled}
                                onChange={(e) => handleFieldChange(feat.key, e.target.value)}
                                className={`w-full bg-slate-50 border rounded px-2 py-1 text-xs text-slate-800 font-mono ${
                                  err
                                    ? "border-rose-400 bg-rose-50/50"
                                    : isSttlDisabled
                                    ? "border-slate-200 bg-slate-100 text-slate-400 cursor-not-allowed"
                                    : "border-slate-300"
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

          {/* Form Actions */}
          <div className="p-4 border-t border-slate-100 bg-slate-50 flex items-center justify-between gap-3">
            <button
              onClick={() => handlePresetChange("benign")}
              className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-xs font-medium text-slate-700 hover:bg-slate-100 transition"
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset</span>
            </button>

            <button
              onClick={handleExecuteExplain}
              disabled={loading || Object.keys(validationErrors).length > 0}
              className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 disabled:opacity-50 transition shadow-xs"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  <span>Computing SHAP Trees...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Generate SHAP Explanation</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: SHAP Attribution Visualization & Diagnostics */}
        <div className="lg:col-span-7 space-y-6">
          {/* Error Banner */}
          {error && (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-start space-x-3 text-xs text-rose-800">
              <AlertCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">Explanation Request Error</div>
                <div className="mt-0.5 leading-relaxed">{error}</div>
              </div>
            </div>
          )}

          {/* Loading Skeleton */}
          {loading && (
            <div className="bg-white rounded-xl border border-slate-200 p-8 text-center space-y-4 shadow-sm animate-pulse">
              <div className="h-10 w-10 mx-auto rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600">
                <RefreshCw className="h-5 w-5 animate-spin" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-800 text-sm">Evaluating Tree Path Attributions</h3>
                <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
                  Computing exact local Shapley values via TreeExplainer across the 10-family decision forest...
                </p>
              </div>
            </div>
          )}

          {/* Active Explanation Results */}
          {!loading && activeExplanation && (
            <div className="space-y-6">
              {/* Prediction Context Banner */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      Model Predicted Class
                    </div>
                    <div className="flex items-center space-x-3 mt-1">
                      <span className="text-xl font-bold text-slate-900">
                        {activeExplanation.predicted_class}
                      </span>
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                          activeExplanation.predicted_class === "Normal"
                            ? "bg-emerald-100 text-emerald-800 border border-emerald-200"
                            : "bg-rose-100 text-rose-800 border border-rose-200"
                        }`}
                      >
                        {activeExplanation.predicted_class === "Normal" ? "Classified as Normal (Benign)" : "Classified as Attack Category"}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
                    <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
                      <div className="text-[10px] text-slate-400 font-sans">SHAP Base Value E[f(x)]</div>
                      <div className="font-bold text-slate-700">
                        {activeExplanation.base_margin >= 0 ? `+${activeExplanation.base_margin.toFixed(4)}` : activeExplanation.base_margin.toFixed(4)}
                      </div>
                    </div>
                    <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
                      <div className="text-[10px] text-slate-400 font-sans">Inference Latency</div>
                      <div className="font-bold text-slate-700 flex items-center space-x-1">
                        <Clock className="h-3 w-3 text-slate-400" />
                        <span>{explanationResponse?.processing_time_ms} ms</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Output space guidance */}
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
                  <div className="flex items-center space-x-1.5">
                    <Layers className="h-3.5 w-3.5 text-indigo-500" />
                    <span>Space: {activeExplanation.output_space} (pre-softmax logit domain)</span>
                  </div>
                  <span className="text-[10px] text-slate-400">
                    Model: {explanationResponse?.model_name}
                  </span>
                </div>
              </div>

              {/* Attribution Horizontal Bar Chart */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold text-sm text-slate-900">
                      Ranked Feature Contributions (|SHAP| Margin Shift)
                    </h3>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Positive shifts (emerald) push toward predicted class; negative shifts (rose) push away
                    </p>
                  </div>
                  <div className="flex items-center space-x-3 text-[11px]">
                    <span className="flex items-center space-x-1">
                      <span className="h-2.5 w-2.5 rounded-xs bg-emerald-500"></span>
                      <span className="text-slate-600 font-medium">Towards Class (+)</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <span className="h-2.5 w-2.5 rounded-xs bg-rose-500"></span>
                      <span className="text-slate-600 font-medium">Away From Class (-)</span>
                    </span>
                  </div>
                </div>

                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                      <XAxis
                        type="number"
                        tick={{ fontSize: 10, fill: "#64748b" }}
                        domain={["auto", "auto"]}
                      />
                      <YAxis
                        dataKey="name"
                        type="category"
                        tick={{ fontSize: 11, fill: "#334155", fontFamily: "monospace" }}
                        width={80}
                      />
                      <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1.5} />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload as ChartBarData;
                            const isPositive = data.value >= 0;
                            return (
                              <div className="bg-slate-900 text-white rounded-lg p-3 text-xs shadow-lg space-y-1">
                                <div className="font-bold font-mono text-indigo-300">
                                  {data.name} ({data.label})
                                </div>
                                <div className="text-[11px] text-slate-300">
                                  Observed Telemetry:{" "}
                                  <span className="font-mono font-semibold text-white">
                                    {String(data.observedValue)}
                                  </span>
                                </div>
                                <div className="text-[11px] flex items-center space-x-1.5 pt-1 border-t border-slate-700">
                                  <span
                                    className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                      isPositive
                                        ? "bg-emerald-900/60 text-emerald-300 border border-emerald-500/40"
                                        : "bg-rose-900/60 text-rose-300 border border-rose-500/40"
                                    }`}
                                  >
                                    {isPositive ? "Pushes Toward Class" : "Pushes Away"}
                                  </span>
                                  <span className="font-mono font-semibold">
                                    {isPositive ? `+${data.value.toFixed(4)}` : data.value.toFixed(4)}{" "}
                                    log-odds
                                  </span>
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="value" radius={[3, 3, 3, 3]}>
                        {chartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.value >= 0 ? "#10b981" : "#f43f5e"}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Attribution Table & Feature Values */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 border-b border-slate-100 flex items-center justify-between">
                  <h3 className="font-semibold text-sm text-slate-900">
                    Feature Attribution Breakdown
                  </h3>
                  <span className="text-xs text-slate-500">
                    Showing top {activeExplanation.top_attributions.length} features
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                      <tr>
                        <th className="px-4 py-2.5">Rank</th>
                        <th className="px-4 py-2.5">Feature</th>
                        <th className="px-4 py-2.5">Observed Value</th>
                        <th className="px-4 py-2.5">Attribution Margin</th>
                        <th className="px-4 py-2.5">Impact Direction</th>
                        <th className="px-4 py-2.5">Description</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {activeExplanation.top_attributions.map((attr, idx) => {
                        const isPositive = attr.attribution_margin >= 0;
                        const observed = (flowRecord as unknown as Record<string, unknown>)[attr.feature_name];
                        const meta = FEATURE_METADATA_MAP[attr.feature_name];

                        return (
                          <tr key={attr.feature_name} className="hover:bg-slate-50/60 font-mono text-[11px]">
                            <td className="px-4 py-2 text-slate-400 font-sans">{idx + 1}</td>
                            <td className="px-4 py-2 font-bold text-slate-900">
                              {attr.feature_name}
                            </td>
                            <td className="px-4 py-2 text-indigo-700 font-semibold">
                              {observed !== undefined ? String(observed) : "-"}
                            </td>
                            <td
                              className={`px-4 py-2 font-bold ${
                                isPositive ? "text-emerald-600" : "text-rose-600"
                              }`}
                            >
                              {isPositive
                                ? `+${attr.attribution_margin.toFixed(4)}`
                                : attr.attribution_margin.toFixed(4)}
                            </td>
                            <td className="px-4 py-2 font-sans">
                              {isPositive ? (
                                <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
                                  <ArrowUpRight className="h-3 w-3" />
                                  <span>Towards Class</span>
                                </span>
                              ) : (
                                <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-100 text-rose-800 border border-rose-200">
                                  <ArrowDownRight className="h-3 w-3" />
                                  <span>Away From Class</span>
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-2 font-sans text-slate-500 text-[10px] truncate max-w-xs">
                              {meta?.description || "Engineered domain feature"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Analytical Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-emerald-50/60 rounded-xl border border-emerald-200 p-4 space-y-2">
                  <div className="flex items-center space-x-2 text-emerald-800 font-semibold text-xs">
                    <ArrowUpRight className="h-4 w-4 text-emerald-600" />
                    <span>Strongest Positive Contributor</span>
                  </div>
                  {topContributors.positive ? (
                    <div className="text-xs text-slate-700">
                      Feature{" "}
                      <span className="font-mono font-bold text-slate-900">
                        {topContributors.positive.feature_name}
                      </span>{" "}
                      (observed:{" "}
                      <span className="font-mono font-semibold">
                        {String(
                          (flowRecord as unknown as Record<string, unknown>)[
                            topContributors.positive.feature_name
                          ] ?? "-"
                        )}
                      </span>
                      ) added{" "}
                      <span className="font-mono font-bold text-emerald-700">
                        +{topContributors.positive.attribution_margin.toFixed(4)} log-odds
                      </span>{" "}
                      toward classifying this flow as {activeExplanation.predicted_class}.
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500">No positive contributors detected.</div>
                  )}
                </div>

                <div className="bg-rose-50/60 rounded-xl border border-rose-200 p-4 space-y-2">
                  <div className="flex items-center space-x-2 text-rose-800 font-semibold text-xs">
                    <ArrowDownRight className="h-4 w-4 text-rose-600" />
                    <span>Strongest Counter-Evidence</span>
                  </div>
                  {topContributors.negative ? (
                    <div className="text-xs text-slate-700">
                      Feature{" "}
                      <span className="font-mono font-bold text-slate-900">
                        {topContributors.negative.feature_name}
                      </span>{" "}
                      (observed:{" "}
                      <span className="font-mono font-semibold">
                        {String(
                          (flowRecord as unknown as Record<string, unknown>)[
                            topContributors.negative.feature_name
                          ] ?? "-"
                        )}
                      </span>
                      ) reduced the margin by{" "}
                      <span className="font-mono font-bold text-rose-700">
                        {topContributors.negative.attribution_margin.toFixed(4)} log-odds
                      </span>
                      , counteracting the predicted class.
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500">No negative contributors detected.</div>
                  )}
                </div>
              </div>

              {/* Methodological Context & Caveats Notice */}
              <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 text-xs text-slate-600 space-y-2">
                <div className="flex items-center space-x-2 font-semibold text-slate-800">
                  <HelpCircle className="h-4 w-4 text-indigo-500" />
                  <span>Interpretation Guidance & Methodological Scope</span>
                </div>
                <p className="text-[11px] leading-relaxed">
                  SHAP values represent local additive feature contributions in the model&apos;s raw pre-softmax logit margin space
                  for the predicted class. Over all features, attributions satisfy the TreeSHAP efficiency property (base value plus
                  sum of all attributions equals the raw class margin); when viewing top-k features, only the most influential subset
                  is displayed. Note: The backend explanation API bounds top-k to at most 42 features, whereas the pipeline contains
                  52 aggregated parent features (42 baseline + 10 domain engineered). Therefore, full parent-level margin reconstruction
                  is not available through the current API and attributions should be interpreted as partial rankings.
                </p>
                <p className="text-[11px] leading-relaxed text-slate-500">
                  SHAP attributions measure decision-tree feature sensitivity, not calibrated probabilities, confidence percentages, or
                  physical network causality. Features such as Source TTL (`sttl`) reflect model reliance rather than verified attacker manipulation.
                </p>
              </div>
            </div>
          )}

          {/* Empty State Prior to Generation */}
          {!loading && !activeExplanation && !error && (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-4 shadow-sm">
              <div className="h-12 w-12 mx-auto rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
                <Sparkles className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-800 text-sm">No Explanation Generated Yet</h3>
                <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto leading-relaxed">
                  Select a candidate model and click{" "}
                  <span className="font-semibold text-slate-700">&quot;Generate SHAP Explanation&quot;</span> to
                  compute tree-based feature attributions for the active flow vector.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
