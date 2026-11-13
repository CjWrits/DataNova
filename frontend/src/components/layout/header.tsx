"use client";

import { useEffect, useState } from "react";
import { api, getApiBaseUrl } from "@/lib/api";
import type { HealthResponse } from "@/types/api";
import { RefreshCw, Server, XCircle } from "lucide-react";

export function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const checkBackendHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getHealth();
      setHealth(res);
    } catch (err: unknown) {
      setHealth(null);
      setError(err instanceof Error ? err.message : "Backend unreachable");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000); // 30s poll
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between shrink-0">
      <div className="flex items-center space-x-3">
        <h2 className="text-base font-semibold text-slate-800">
          NIDS Threat Intelligence Platform
        </h2>
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
          UNSW-NB15 Benchmark
        </span>
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
          Simulated Flow Telemetry
        </span>
      </div>

      <div className="flex items-center space-x-4">
        {/* Backend Connectivity Status Badge */}
        <div className="flex items-center space-x-2 text-xs">
          <Server className="h-4 w-4 text-slate-400" />
          <span className="text-slate-500 font-medium">FastAPI:</span>

          {loading && !health ? (
            <span className="inline-flex items-center space-x-1 text-slate-400">
              <RefreshCw className="h-3 w-3 animate-spin" />
              <span>Connecting...</span>
            </span>
          ) : health?.models_ready ? (
            <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-medium border border-emerald-200">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>Ready (C8, C10, Binary)</span>
            </span>
          ) : (
            <span
              className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 font-medium border border-rose-200"
              title={`${error || "Offline"} — Target: ${getApiBaseUrl()}`}
            >
              <XCircle className="h-3 w-3 text-rose-500" />
              <span>Offline</span>
            </span>
          )}
        </div>

        {/* Manual Refresh Button */}
        <button
          onClick={checkBackendHealth}
          disabled={loading}
          className="p-1.5 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition"
          title="Refresh Backend Status"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>
    </header>
  );
}
