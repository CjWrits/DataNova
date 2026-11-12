"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { HealthResponse, ModelsListResponse, SystemStatusResponse } from "@/types/api";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  FlaskConical,
  Layers,
  Network,
  RefreshCw,
  Server,
  Sparkles,
} from "lucide-react";

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [system, setSystem] = useState<SystemStatusResponse | null>(null);
  const [models, setModels] = useState<ModelsListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTelemetry = async () => {
    setLoading(true);
    setError(null);
    try {
      const [hRes, sRes, mRes] = await Promise.all([
        api.getHealth(),
        api.getSystemStatus(),
        api.getModels(),
      ]);
      setHealth(hRes);
      setSystem(sRes);
      setModels(mRes);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to connect to FastAPI backend at http://127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Banner / Hero */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              Phase 8 Foundation Active
            </span>
            <span className="text-xs text-slate-500">Research NIDS Dashboard</span>
          </div>
          <h1 className="text-xl font-bold text-slate-900 mt-1">
            Adaptive Threat Intelligence Platform
          </h1>
          <p className="text-xs text-slate-600 mt-1 max-w-2xl">
            Integrated machine learning security architecture evaluating UNSW-NB15 network flow telemetry.
            Provides calibrated binary detection, 10-family multiclass classification, SHAP tree explainability,
            and non-destructive drift monitoring.
          </p>
        </div>

        <button
          onClick={fetchTelemetry}
          disabled={loading}
          className="inline-flex items-center space-x-2 px-3 py-2 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 transition shadow-sm self-start md:self-auto shrink-0"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {/* Backend Connection Status Alert if Offline */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start space-x-3 text-xs">
          <AlertTriangle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold text-rose-900">Backend Connection Error</h4>
            <p className="mt-0.5 text-rose-700">{error}</p>
            <p className="mt-1 text-[11px] text-rose-600">
              Ensure the FastAPI backend is running via{" "}
              <code className="font-mono bg-rose-100 px-1 py-0.5 rounded text-rose-900">
                python -m uvicorn backend.app.main:app --port 8000
              </code>
            </p>
          </div>
        </div>
      )}

      {/* System Telemetry KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Service Status */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium">Inference Service</span>
            <Server className="h-4 w-4 text-indigo-500" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-slate-900 capitalize">
              {health?.status || (loading ? "Checking..." : "Offline")}
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center space-x-1">
            <span
              className={`h-2 w-2 rounded-full ${
                health?.models_ready ? "bg-emerald-500" : "bg-rose-500"
              }`}
            />
            <span>
              {health?.models_ready
                ? "Models preloaded in memory"
                : "Awaiting model readiness"}
            </span>
          </div>
        </div>

        {/* Card 2: Loaded Models */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium">Loaded Models</span>
            <Layers className="h-4 w-4 text-indigo-500" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-slate-900">
              {models?.count || 0}
            </span>
            <span className="text-xs text-slate-500">Active Artifacts</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 truncate">
            {system?.models_loaded?.join(", ") || "binary, c8, c10"}
          </div>
        </div>

        {/* Card 3: Reference Dataset */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium">Training Reference</span>
            <Database className="h-4 w-4 text-indigo-500" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-slate-900">
              {system?.reference_rows?.toLocaleString() || "151,325"}
            </span>
            <span className="text-xs text-slate-500">rows</span>
          </div>
          <div className="mt-2 text-[11px] text-emerald-600 font-medium flex items-center space-x-1">
            <CheckCircle2 className="h-3 w-3" />
            <span>Sealed Test Partition Untouched</span>
          </div>
        </div>

        {/* Card 4: Memory RSS */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium">Server Footprint</span>
            <Cpu className="h-4 w-4 text-indigo-500" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-slate-900">
              {system?.memory_rss_mb && system.memory_rss_mb > 0
                ? `${system.memory_rss_mb} MB`
                : "Active"}
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500">
            Uptime: {system?.uptime_seconds ? `${Math.round(system.uptime_seconds)}s` : "0s"}
          </div>
        </div>
      </div>

      {/* Model Registry Catalog */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Loaded NIDS Model Registry</h3>
            <p className="text-xs text-slate-500">
              Frozen model artifacts loaded from disk (zero runtime retraining or threshold mutation)
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">Joblib Serialized</span>
        </div>

        <div className="divide-y divide-slate-100">
          {models?.models?.map((model) => (
            <div key={model.id} className="p-4 hover:bg-slate-50/60 transition flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-slate-900 text-sm">{model.name}</span>
                  <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                    ID: {model.id}
                  </span>
                  {model.is_default && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                      Default
                    </span>
                  )}
                </div>
                <p className="text-slate-600 text-xs">{model.description}</p>
                <div className="flex items-center space-x-4 text-[11px] text-slate-500 pt-1">
                  <span>Task: <strong className="text-slate-700">{model.task_type}</strong></span>
                  <span>Schema: <strong className="text-slate-700">{model.schema_name}</strong> ({model.feature_count} features)</span>
                  <span>Version: <strong className="text-slate-700">{model.version}</strong></span>
                </div>
              </div>

              {model.threshold_info && (
                <div className="text-right shrink-0 bg-slate-50 p-2.5 rounded-lg border border-slate-200/80">
                  <div className="text-[11px] font-medium text-slate-500">Default Threshold</div>
                  <div className="text-base font-bold font-mono text-slate-900">
                    {model.threshold_info.default ?? 0.50}
                  </div>
                </div>
              )}
            </div>
          )) || (
            <div className="p-6 text-center text-slate-400 text-xs">
              Loading model registry metadata...
            </div>
          )}
        </div>
      </div>

      {/* Analyst Experience Navigation Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Nav 1: Traffic Analysis */}
        <Link
          href="/traffic"
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-md transition group flex flex-col justify-between"
        >
          <div>
            <div className="h-9 w-9 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3 group-hover:scale-105 transition">
              <Network className="h-5 w-5" />
            </div>
            <h4 className="text-sm font-semibold text-slate-900 group-hover:text-indigo-600 transition">
              Traffic Analysis
            </h4>
            <p className="text-xs text-slate-500 mt-1">
              Classify individual flow vectors or upload batch CSV captures.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-indigo-600 group-hover:translate-x-1 transition">
            <span>Analyze Flows</span>
            <ArrowRight className="h-3.5 w-3.5 ml-1" />
          </div>
        </Link>

        {/* Nav 2: Explainability Lab */}
        <Link
          href="/explainability"
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-md transition group flex flex-col justify-between"
        >
          <div>
            <div className="h-9 w-9 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center mb-3 group-hover:scale-105 transition">
              <Sparkles className="h-5 w-5" />
            </div>
            <h4 className="text-sm font-semibold text-slate-900 group-hover:text-purple-600 transition">
              Explainability Lab
            </h4>
            <p className="text-xs text-slate-500 mt-1">
              Inspect SHAP log-odds margins and parent feature attributions.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-purple-600 group-hover:translate-x-1 transition">
            <span>Explore SHAP</span>
            <ArrowRight className="h-3.5 w-3.5 ml-1" />
          </div>
        </Link>

        {/* Nav 3: Drift Monitor */}
        <Link
          href="/drift"
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-md transition group flex flex-col justify-between"
        >
          <div>
            <div className="h-9 w-9 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center mb-3 group-hover:scale-105 transition">
              <Activity className="h-5 w-5" />
            </div>
            <h4 className="text-sm font-semibold text-slate-900 group-hover:text-emerald-600 transition">
              Drift Monitor
            </h4>
            <p className="text-xs text-slate-500 mt-1">
              Track PSI/KS feature divergence and data-quality corruption alerts.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-emerald-600 group-hover:translate-x-1 transition">
            <span>Monitor Drift</span>
            <ArrowRight className="h-3.5 w-3.5 ml-1" />
          </div>
        </Link>

        {/* Nav 4: Model Laboratory */}
        <Link
          href="/models"
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-md transition group flex flex-col justify-between"
        >
          <div>
            <div className="h-9 w-9 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center mb-3 group-hover:scale-105 transition">
              <FlaskConical className="h-5 w-5" />
            </div>
            <h4 className="text-sm font-semibold text-slate-900 group-hover:text-amber-600 transition">
              Model Laboratory
            </h4>
            <p className="text-xs text-slate-500 mt-1">
              Compare Candidate 8 and Candidate 10 under STTL perturbation.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-amber-600 group-hover:translate-x-1 transition">
            <span>Audit Models</span>
            <ArrowRight className="h-3.5 w-3.5 ml-1" />
          </div>
        </Link>
      </div>
    </div>
  );
}
