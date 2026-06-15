// frontend/lib/api.ts
// Centralized, typed API client for the FastAPI backend.
// All components import from here — never call fetch() directly.

import type {
  ExplainabilityResponse,
  ForecastResponse,
  KpiResponse,
  LocationsResponse,
} from "@/types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE_URL}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }

  const res = await fetch(url.toString(), {
    // Next.js App Router: opt out of the static cache for live data
    cache: "no-store",
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore JSON parse failure
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

// ── Public API functions ────────────────────────────────────────────────────

export async function fetchLocations(): Promise<LocationsResponse> {
  return get<LocationsResponse>("/api/locations");
}

export async function fetchKpis(): Promise<KpiResponse> {
  return get<KpiResponse>("/api/kpis");
}

export async function fetchForecast(
  location: string,
  horizon = 30,
): Promise<ForecastResponse> {
  return get<ForecastResponse>("/api/forecast", {
    location,
    horizon: String(horizon),
  });
}

export async function fetchExplainability(
  location: string,
  model = "XGBoost",
): Promise<ExplainabilityResponse> {
  return get<ExplainabilityResponse>("/api/explainability", {
    location,
    model,
  });
}

export { ApiError };
