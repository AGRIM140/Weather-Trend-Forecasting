"""
backend/main.py
===============

FastAPI application that wraps the src/ ML pipeline and exposes a clean
REST API consumed by the Next.js frontend.

Endpoints
---------
GET /api/health              — liveness probe
GET /api/locations           — list all available location names
GET /api/kpis                — dataset-level summary statistics
GET /api/forecast            — 30-day ensemble forecast for a location
GET /api/explainability      — SHAP + permutation importance rankings

Design decisions
----------------
- Data is loaded ONCE on startup via FastAPI's lifespan context and stored
  in module-level state, so no CSV is re-read per request.
- Forecast + explainability results are cached in a dict keyed by location
  (and model name for explainability) so the first request pays the
  training cost and every subsequent request is instant.
- All heavy computation runs in a thread pool via asyncio.get_event_loop()
  .run_in_executor so the async event loop is never blocked.
- CORS origins are read from the environment so local dev and the Vercel
  deployment URL both work without code changes.

Run locally
-----------
    cd <project-root>
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Make src/ importable regardless of working directory ───────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_prep import DataPrepConfig, WeatherDataPreprocessor
from src.explainability import (
    ExplainabilityConfig,
    ShapAnalyzer,
    build_global_feature_ranking,
    compute_permutation_importance,
)
from src.forecasting import ForecastConfig, MultivariateForecastingEngine

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", os.path.join(PROJECT_ROOT, "data", "GlobalWeatherRepository.csv"))
PROCESSED_DATA_PATH = os.getenv(
    "PROCESSED_DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "processed_weather_data.csv"),
)

# ── Module-level state (populated at startup) ───────────────────────────────
_state: Dict[str, Any] = {
    "df": None,           # processed DataFrame
    "preprocessor": None, # fitted WeatherDataPreprocessor
    "ready": False,
}

# In-memory result caches keyed by location / (location, model)
_forecast_cache: Dict[str, Dict] = {}
_explainability_cache: Dict[str, Dict] = {}
_forecast_locks: Dict[str, asyncio.Lock] = {}
_explain_locks: Dict[str, asyncio.Lock] = {}


# ── Lifespan: load + preprocess data once at startup ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading and preprocessing dataset …")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_data)
    logger.info("Startup complete — API is ready.")
    yield
    logger.info("Shutdown: clearing caches.")
    _forecast_cache.clear()
    _explainability_cache.clear()


def _load_data() -> None:
    """Synchronous data-loading function run once in a thread at startup."""
    import gdown

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH):
        logger.info("Downloading dataset from Google Drive …")
        gdown.download(
            "https://drive.google.com/file/d/17Bc2Cmo6Ao-Z1Gkxcybq8c3uryR-sfwD/view?usp=sharing",
            RAW_DATA_PATH,
            quiet=False,
        )

    if os.path.exists(PROCESSED_DATA_PATH):
        logger.info("Fast path: loading already-processed CSV.")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df["last_updated"] = pd.to_datetime(df["last_updated"])
        config = DataPrepConfig(
            raw_data_path=RAW_DATA_PATH,
            processed_data_path=PROCESSED_DATA_PATH,
        )
        preprocessor = WeatherDataPreprocessor(config=config)
    else:
        logger.info("Slow path: running full preprocessing pipeline …")
        from src.data_prep import run_data_prep_pipeline
        df, preprocessor = run_data_prep_pipeline(
            raw_path=RAW_DATA_PATH,
            save_path=PROCESSED_DATA_PATH,
        )
        df["last_updated"] = pd.to_datetime(df["last_updated"])

    _state["df"] = df
    _state["preprocessor"] = preprocessor
    _state["ready"] = True
    logger.info("Dataset loaded — shape: %s", df.shape)


# ── App factory ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Global Weather Forecasting API",
    description="REST backend for the multivariate weather forecasting dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Allow the Next.js dev server and any Vercel deployment URL.
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001",
)
CORS_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    dataset_ready: bool
    row_count: Optional[int] = None


class LocationsResponse(BaseModel):
    locations: List[str]
    count: int


class KpiResponse(BaseModel):
    total_observations: int
    location_count: int
    country_count: int
    feature_count: int
    avg_temperature_celsius: float
    avg_humidity: float
    avg_pressure_mb: float
    avg_wind_kph: float
    max_temperature_celsius: float
    min_temperature_celsius: float
    outlier_iqr_pct: float
    outlier_isoforest_pct: float
    date_range_start: str
    date_range_end: str
    most_observed_location: str


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class ModelMetrics(BaseModel):
    MAE: float
    RMSE: float
    MAPE: float
    R2: float


class ForecastResponse(BaseModel):
    location: str
    target_col: str = "temperature_celsius"
    historical: List[TimeSeriesPoint]
    forecast: List[TimeSeriesPoint]
    forecast_horizon_days: int
    metrics: Dict[str, ModelMetrics]
    ensemble_weights: Dict[str, float]
    computation_time_seconds: float


class ShapFeature(BaseModel):
    feature: str
    combined_rank: int
    combined_score: float
    mean_abs_shap: float
    shap_rank: int
    permutation_importance: float
    permutation_rank: int


class ExplainabilityResponse(BaseModel):
    location: str
    model: str
    top_features: List[ShapFeature]
    exogenous_shap_fraction: float  # fraction of total |SHAP| from non-target features
    computation_time_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# Guards
# ─────────────────────────────────────────────────────────────────────────────

def _require_ready() -> pd.DataFrame:
    if not _state["ready"] or _state["df"] is None:
        raise HTTPException(status_code=503, detail="Dataset not yet loaded. Retry in a moment.")
    return _state["df"]


def _require_location(df: pd.DataFrame, location: str) -> None:
    valid = df["location_name"].unique()
    if location not in valid:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location}' not found. Use GET /api/locations for valid names.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 as soon as the dataset is loaded."""
    df = _state.get("df")
    return HealthResponse(
        status="ok" if _state["ready"] else "loading",
        dataset_ready=_state["ready"],
        row_count=len(df) if df is not None else None,
    )


@app.get("/api/locations", response_model=LocationsResponse, tags=["Meta"])
async def get_locations() -> LocationsResponse:
    """Return a sorted list of all unique location names in the dataset."""
    df = _require_ready()
    locs = sorted(df["location_name"].dropna().unique().tolist())
    return LocationsResponse(locations=locs, count=len(locs))


@app.get("/api/kpis", response_model=KpiResponse, tags=["Overview"])
async def get_kpis() -> KpiResponse:
    """Dataset-level summary statistics for the Overview dashboard tab."""
    df = _require_ready()

    iqr_pct = float(df["is_outlier_iqr"].mean() * 100) if "is_outlier_iqr" in df.columns else 0.0
    iso_pct = float(df["is_outlier_isoforest"].mean() * 100) if "is_outlier_isoforest" in df.columns else 0.0

    dt_col = pd.to_datetime(df["last_updated"])
    top_loc = df["location_name"].value_counts().idxmax()

    return KpiResponse(
        total_observations=len(df),
        location_count=int(df["location_name"].nunique()),
        country_count=int(df["country"].nunique()),
        feature_count=int(df.shape[1]),
        avg_temperature_celsius=round(float(df["temperature_celsius"].mean()), 2),
        avg_humidity=round(float(df["humidity"].mean()), 2),
        avg_pressure_mb=round(float(df["pressure_mb"].mean()), 2),
        avg_wind_kph=round(float(df["wind_kph"].mean()), 2),
        max_temperature_celsius=round(float(df["temperature_celsius"].max()), 2),
        min_temperature_celsius=round(float(df["temperature_celsius"].min()), 2),
        outlier_iqr_pct=round(iqr_pct, 3),
        outlier_isoforest_pct=round(iso_pct, 3),
        date_range_start=dt_col.min().strftime("%Y-%m-%d"),
        date_range_end=dt_col.max().strftime("%Y-%m-%d"),
        most_observed_location=str(top_loc),
    )


@app.get("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def get_forecast(
    location: str = Query(..., description="Location name, e.g. 'London'"),
    horizon: int = Query(30, ge=7, le=90, description="Forecast horizon in days"),
) -> ForecastResponse:
    """
    Train the multivariate ensemble for the requested location and return
    the 30-day (or custom horizon) temperature forecast alongside
    per-model test-set metrics.

    Results are cached in memory — the first call for a location trains the
    models (15–60 seconds depending on hardware); subsequent calls are
    instant.
    """
    df = _require_ready()
    _require_location(df, location)

    cache_key = f"{location}::{horizon}"

    # Create a per-key lock to prevent parallel duplicate training jobs
    if cache_key not in _forecast_locks:
        _forecast_locks[cache_key] = asyncio.Lock()

    async with _forecast_locks[cache_key]:
        if cache_key in _forecast_cache:
            return ForecastResponse(**_forecast_cache[cache_key])

        logger.info("Training forecast engine for location='%s', horizon=%d …", location, horizon)
        t0 = time.perf_counter()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(_run_forecast_sync, df, location, horizon),
        )

        result["computation_time_seconds"] = round(time.perf_counter() - t0, 2)
        _forecast_cache[cache_key] = result
        logger.info(
            "Forecast cached for '%s' in %.1fs", location, result["computation_time_seconds"]
        )

    return ForecastResponse(**_forecast_cache[cache_key])


def _run_forecast_sync(df: pd.DataFrame, location: str, horizon: int) -> Dict:
    """Synchronous forecast computation — called in a thread-pool executor."""
    config = ForecastConfig(forecast_horizon_days=horizon)
    engine = MultivariateForecastingEngine(config=config)
    engine.prepare_data(df, location_name=location)
    metrics = engine.fit()
    forecast_df = engine.forecast_future(model_name="Ensemble")

    # Historical: last 90 days of the daily resampled series
    hist = engine.daily_df_[config.target_col].tail(90)
    historical = [
        {"date": str(ts.date()), "value": round(float(v), 3)}
        for ts, v in hist.items()
    ]

    # Forecast
    forecast = [
        {"date": str(ts.date()), "value": round(float(v), 3)}
        for ts, v in forecast_df["forecast_temperature_celsius"].items()
    ]

    # Metrics — filter None / non-dict entries
    clean_metrics: Dict[str, ModelMetrics] = {}
    for model_name, m in metrics.items():
        if isinstance(m, dict) and all(k in m for k in ("MAE", "RMSE", "MAPE", "R2")):
            clean_metrics[model_name] = ModelMetrics(
                MAE=round(m["MAE"], 4),
                RMSE=round(m["RMSE"], 4),
                MAPE=round(m["MAPE"], 4),
                R2=round(m["R2"], 4),
            )

    return {
        "location": location,
        "target_col": config.target_col,
        "historical": historical,
        "forecast": forecast,
        "forecast_horizon_days": horizon,
        "metrics": clean_metrics,
        "ensemble_weights": {
            k: round(v, 4) for k, v in engine.ensemble_weights_.items()
        },
        "computation_time_seconds": 0.0,  # filled by caller
    }


@app.get("/api/explainability", response_model=ExplainabilityResponse, tags=["Explainability"])
async def get_explainability(
    location: str = Query(..., description="Location name"),
    model: str = Query("XGBoost", description="Tree model for SHAP: XGBoost | LightGBM | RandomForest | ExtraTrees"),
) -> ExplainabilityResponse:
    """
    Compute SHAP values and permutation importance for the chosen model and
    location, then return a unified ranked feature importance list.

    Requires the forecast endpoint to have been called first for this location
    (so the engine + trained models are available in the forecast cache).
    """
    df = _require_ready()
    _require_location(df, location)

    forecast_key = f"{location}::30"
    if forecast_key not in _forecast_cache:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No trained model found for location='{location}'. "
                "Call GET /api/forecast?location={location} first."
            ),
        )

    cache_key = f"{location}::{model}"
    if cache_key not in _explain_locks:
        _explain_locks[cache_key] = asyncio.Lock()

    async with _explain_locks[cache_key]:
        if cache_key in _explainability_cache:
            return ExplainabilityResponse(**_explainability_cache[cache_key])

        logger.info("Computing explainability for location='%s', model='%s' …", location, model)
        t0 = time.perf_counter()

        loop = asyncio.get_event_loop()

        # Re-run the engine to get train/test splits and the fitted model.
        # (We do NOT re-train from scratch — the engine is rebuilt from the
        # cached processed df which is fast, then we call fit() again.
        # A production system would serialize the engine with joblib instead.)
        result = await loop.run_in_executor(
            None,
            partial(_run_explainability_sync, df, location, model),
        )

        result["computation_time_seconds"] = round(time.perf_counter() - t0, 2)
        _explainability_cache[cache_key] = result

    return ExplainabilityResponse(**_explainability_cache[cache_key])


def _run_explainability_sync(df: pd.DataFrame, location: str, model_name: str) -> Dict:
    """Synchronous SHAP computation — called in a thread-pool executor."""
    config = ForecastConfig()
    engine = MultivariateForecastingEngine(config=config)
    engine.prepare_data(df, location_name=location)
    engine.fit()

    available_tree_models = [
        k for k in engine.fitted_models_
        if k not in ("ARIMA", "Prophet")
    ]
    if model_name not in engine.fitted_models_:
        model_name = available_tree_models[0] if available_tree_models else None

    if model_name is None:
        raise HTTPException(status_code=503, detail="No tree-based model available for SHAP.")

    fitted_model = engine.fitted_models_[model_name]
    feature_names = engine.feature_cols_
    X_test = engine.test_df_[feature_names]
    y_test = engine.test_df_[config.target_col]

    exp_config = ExplainabilityConfig(
        shap_sample_size=300,
        permutation_n_repeats=8,
        top_n_features=20,
    )

    analyzer = ShapAnalyzer(fitted_model, feature_names, config=exp_config)
    analyzer.fit_explainer(X_test)
    shap_ranking = analyzer.get_shap_importance_ranking()

    perm_ranking = compute_permutation_importance(
        fitted_model, X_test, y_test, config=exp_config
    )
    global_ranking = build_global_feature_ranking(shap_ranking, perm_ranking)

    # Fraction of total SHAP mass from exogenous (non-temperature) features
    target = config.target_col
    total_shap = global_ranking["mean_abs_shap"].sum()
    exo_shap = global_ranking.loc[
        ~global_ranking["feature"].str.startswith(target), "mean_abs_shap"
    ].sum()
    exo_fraction = float(exo_shap / total_shap) if total_shap > 0 else 0.0

    top_features = []
    for _, row in global_ranking.head(20).iterrows():
        top_features.append(
            ShapFeature(
                feature=str(row["feature"]),
                combined_rank=int(row["combined_rank"]),
                combined_score=round(float(row["combined_score"]), 5),
                mean_abs_shap=round(float(row["mean_abs_shap"]), 5),
                shap_rank=int(row["shap_rank"]),
                permutation_importance=round(float(row["permutation_importance"]), 5),
                permutation_rank=int(row["permutation_rank"]),
            )
        )

    return {
        "location": location,
        "model": model_name,
        "top_features": top_features,
        "exogenous_shap_fraction": round(exo_fraction, 4),
        "computation_time_seconds": 0.0,
    }


# ── Dev entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
