/**
 * Typed API Client for the Adaptive Threat Intelligence NIDS Backend.
 * Communicates with FastAPI endpoints under /api/v1.
 */

import type {
  ApiErrorResponse,
  BinaryPredictionRequest,
  BinaryPredictionResponse,
  DriftMonitoringRequest,
  DriftMonitoringResponse,
  ExplainRequest,
  ExplainResponse,
  HealthResponse,
  ModelComparisonRequest,
  ModelComparisonResponse,
  ModelsListResponse,
  MulticlassPredictionRequest,
  MulticlassPredictionResponse,
  SystemStatusResponse,
} from "@/types/api";

const PRODUCTION_DEFAULT_BACKEND = "https://datanova-backend-8ef2.onrender.com";

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    // If running on Render, use same-origin relative path to leverage Next.js rewrite proxy.
    // This routes requests server-to-server and 100% eliminates browser CORS / client fetch blocks!
    if (window.location.hostname.includes("onrender.com")) {
      const custom = localStorage.getItem("DATANOVA_API_URL");
      if (custom && custom.includes("direct=true")) {
        return custom.replace("?direct=true", "").replace(/\/$/, "");
      }
      return "";
    }

    const custom = localStorage.getItem("DATANOVA_API_URL");
    if (custom && custom.trim().length > 0) {
      return custom.trim().replace(/\/$/, "");
    }
  }

  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (
    envUrl &&
    !envUrl.includes("127.0.0.1") &&
    !envUrl.includes("localhost")
  ) {
    return envUrl.replace(/\/$/, "");
  }

  return envUrl?.replace(/\/$/, "") || PRODUCTION_DEFAULT_BACKEND;
}

export function setApiBaseUrl(url: string): void {
  if (typeof window !== "undefined") {
    if (!url || url.trim().length === 0) {
      localStorage.removeItem("DATANOVA_API_URL");
    } else {
      localStorage.setItem("DATANOVA_API_URL", url.trim().replace(/\/$/, ""));
    }
  }
}

export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly errorType: string;
  public readonly detail?: unknown;

  constructor(
    statusCode: number,
    errorType: string,
    message: string,
    detail?: unknown
  ) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.errorType = errorType;
    this.detail = detail;
  }
}

async function apiFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }

  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    if (!res.ok) {
      let errorData: ApiErrorResponse = {
        error_type: `HTTP_${res.status}`,
        message: `API request failed with status ${res.status}: ${res.statusText}`,
      };

      try {
        const parsed = await res.json();
        if (parsed.error_type || parsed.message || parsed.detail) {
          errorData = parsed;
        }
      } catch {
        // Fallback to generic message
      }

      throw new ApiError(
        res.status,
        errorData.error_type || `HTTP_${res.status}`,
        errorData.message || res.statusText,
        errorData.detail
      );
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : "Network error connecting to backend";
    throw new ApiError(0, "NetworkError", `Backend connection failed (${baseUrl}): ${message}`);
  }
}

export const api = {
  // Health & System Telemetry
  getHealth: () => apiFetch<HealthResponse>("/health"),
  getV1Health: () => apiFetch<HealthResponse>("/api/v1/health"),
  getSystemStatus: () => apiFetch<SystemStatusResponse>("/api/v1/system/status"),

  // Model Catalog
  getModels: () => apiFetch<ModelsListResponse>("/api/v1/models"),

  // Predictions
  predictBinary: (req: BinaryPredictionRequest) =>
    apiFetch<BinaryPredictionResponse>("/api/v1/predict/binary", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  predictMulticlass: (req: MulticlassPredictionRequest) =>
    apiFetch<MulticlassPredictionResponse>("/api/v1/predict/multiclass", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  compareModels: (req: ModelComparisonRequest) =>
    apiFetch<ModelComparisonResponse>("/api/v1/predict/compare", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // Drift Monitoring
  monitorDrift: (req: DriftMonitoringRequest) =>
    apiFetch<DriftMonitoringResponse>("/api/v1/monitor/drift", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // Explainability
  explainFlows: (req: ExplainRequest) =>
    apiFetch<ExplainResponse>("/api/v1/explain", {
      method: "POST",
      body: JSON.stringify(req),
    }),
};
