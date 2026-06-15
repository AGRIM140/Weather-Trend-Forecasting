// frontend/types/index.ts
// Mirrors the Pydantic response models in backend/main.py exactly.

export interface HealthResponse {
  status: "ok" | "loading";
  dataset_ready: boolean;
  row_count: number | null;
}

export interface LocationsResponse {
  locations: string[];
  count: number;
}

export interface KpiResponse {
  total_observations: number;
  location_count: number;
  country_count: number;
  feature_count: number;
  avg_temperature_celsius: number;
  avg_humidity: number;
  avg_pressure_mb: number;
  avg_wind_kph: number;
  max_temperature_celsius: number;
  min_temperature_celsius: number;
  outlier_iqr_pct: number;
  outlier_isoforest_pct: number;
  date_range_start: string;
  date_range_end: string;
  most_observed_location: string;
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
}

export interface ModelMetrics {
  MAE: number;
  RMSE: number;
  MAPE: number;
  R2: number;
}

export interface ForecastResponse {
  location: string;
  target_col: string;
  historical: TimeSeriesPoint[];
  forecast: TimeSeriesPoint[];
  forecast_horizon_days: number;
  metrics: Record<string, ModelMetrics>;
  ensemble_weights: Record<string, number>;
  computation_time_seconds: number;
}

export interface ShapFeature {
  feature: string;
  combined_rank: number;
  combined_score: number;
  mean_abs_shap: number;
  shap_rank: number;
  permutation_importance: number;
  permutation_rank: number;
}

export interface ExplainabilityResponse {
  location: string;
  model: string;
  top_features: ShapFeature[];
  exogenous_shap_fraction: number;
  computation_time_seconds: number;
}

export type NavTab = "overview" | "forecast" | "explainability";

export interface ApiError {
  detail: string;
  status: number;
}
