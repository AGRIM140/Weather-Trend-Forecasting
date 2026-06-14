"""
app.py
======

Streamlit Dashboard — Global Weather Forecasting Engine
========================================================

Interactive, production-quality dashboard that orchestrates the full
ML pipeline:

    Tab 0  — Dataset Overview & KPIs
    Tab 1  — EDA & Spatial Analysis
    Tab 2  — Multivariate Forecasting (Ensemble + per-model metrics)
    Tab 3  — Model Explainability (SHAP + Permutation Importance)

All heavy computation (data loading, preprocessing, model training,
SHAP analysis) is protected by `@st.cache_data` / `@st.cache_resource`
so that UI interactions never trigger unnecessary re-runs.

Run:
    streamlit run app.py

Dependencies (beyond the project's own src/ modules):
    streamlit, plotly, folium, streamlit-folium, shap, pandas, numpy
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Project modules ──────────────────────────────────────────────────────────
from src.data_prep import DataPrepConfig, WeatherDataPreprocessor, run_data_prep_pipeline
from src.eda_spatial import (
    create_geographic_map,
    create_heatmap,
    plot_air_quality_relationships,
    plot_distributions,
    plot_lof_anomalies,
    plot_time_series_trends,
)
from src.explainability import (
    ExplainabilityConfig,
    ShapAnalyzer,
    build_global_feature_ranking,
    compute_permutation_importance,
    plot_permutation_importance,
)
from src.forecasting import (
    ForecastConfig,
    MultivariateForecastingEngine,
    run_forecasting_pipeline,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Weather Forecasting Engine",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark-teal / slate design language
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Palette ──────────────────────────────────────────────────────────
       Deep Navy   #0D1B2A   (page bg)
       Slate       #1E2F45   (card / sidebar bg)
       Teal        #00B4D8   (accent / headings)
       Teal-dim    #0077A8   (hover / secondary accent)
       Text-primary  #E8EDF2
       Text-muted    #8FA8C0
       Success     #4CAF82
       Warning     #F4A261
       Danger      #E63946
    ──────────────────────────────────────────────────────────────────── */

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #0D1B2A;
        color: #E8EDF2;
    }

    /* ── App background ── */
    .stApp { background-color: #0D1B2A; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1E2F45;
        border-right: 1px solid #2A3F57;
    }
    [data-testid="stSidebar"] * { color: #E8EDF2 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E2F45;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8FA8C0 !important;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 8px 18px;
        border: none;
        transition: all 0.18s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00B4D8 !important;
        color: #0D1B2A !important;
        font-weight: 700;
    }

    /* ── KPI / metric cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #1E2F45 0%, #152436 100%);
        border: 1px solid #2A3F57;
        border-left: 4px solid #00B4D8;
        border-radius: 10px;
        padding: 18px 20px 14px 20px;
        text-align: left;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #8FA8C0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00B4D8;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #8FA8C0;
        margin-top: 4px;
    }

    /* ── Section headings ── */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #00B4D8;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        border-bottom: 1px solid #2A3F57;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* ── Mission banner ── */
    .mission-banner {
        background: linear-gradient(135deg, #0077A8 0%, #005f85 100%);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 20px;
        border: 1px solid #00B4D8;
    }
    .mission-banner p {
        font-size: 0.78rem;
        color: #E8EDF2;
        margin: 0;
        line-height: 1.55;
    }
    .mission-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: #00B4D8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* ── Info / alert boxes ── */
    .info-box {
        background-color: #152436;
        border-left: 3px solid #00B4D8;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #8FA8C0;
        margin: 10px 0;
    }

    /* ── Model metric table styling ── */
    .metric-best { color: #4CAF82 !important; font-weight: 700; }
    .metric-worst { color: #E63946 !important; }

    /* ── Streamlit native element overrides ── */
    .stSelectbox label, .stSlider label, .stMultiSelect label,
    .stRadio label, .stCheckbox label {
        color: #8FA8C0 !important;
        font-size: 0.82rem;
    }
    .stButton > button {
        background-color: #00B4D8;
        color: #0D1B2A;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 8px 22px;
        transition: background-color 0.18s ease;
    }
    .stButton > button:hover {
        background-color: #0077A8;
        color: #E8EDF2;
    }
    div[data-testid="stMetric"] {
        background-color: #1E2F45;
        border: 1px solid #2A3F57;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label { color: #8FA8C0 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #00B4D8 !important;
    }

    /* ── Dataframe / table ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Spinner text ── */
    .stSpinner > div { border-top-color: #00B4D8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/GlobalWeatherRepository.csv"
PROCESSED_DATA_PATH = "data/processed_weather_data.csv"

PM_ACCELERATOR_MISSION = (
    "[Note to Agrim: Paste the exact PM Accelerator mission wording here before hitting send]"
)

PLOTLY_DARK_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#1E2F45",
    plot_bgcolor="#152436",
    font_color="#E8EDF2",
    font_family="Inter, Segoe UI, sans-serif",
)

ACCENT = "#00B4D8"
SUCCESS = "#4CAF82"
WARNING = "#F4A261"
DANGER = "#E63946"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — themed Plotly layout injector
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Apply the dashboard's dark theme to any Plotly figure in-place."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=ACCENT)),
        height=height,
        margin=dict(l=40, r=20, t=55, b=40),
        **PLOTLY_DARK_THEME,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#2A3F57",
            borderwidth=1,
            font=dict(size=11, color="#8FA8C0"),
        ),
    )
    fig.update_xaxes(gridcolor="#2A3F57", zerolinecolor="#2A3F57")
    fig.update_yaxes(gridcolor="#2A3F57", zerolinecolor="#2A3F57")
    return fig


def _section(label: str) -> None:
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)


def _kpi(col, label: str, value: str, sub: str = "") -> None:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _info(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Running preprocessing pipeline…")
def load_and_preprocess() -> tuple[pd.DataFrame, WeatherDataPreprocessor]:
    """Load raw CSV, run the full WeatherDataPreprocessor pipeline, and
    cache both the processed dataframe and the fitted preprocessor object.

    Uses `@st.cache_resource` (not `@st.cache_data`) because the
    preprocessor stores fitted scikit-learn objects (StandardScaler,
    LabelEncoders) that are not trivially serialisable by Arrow.
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        # Fast path: load the already-processed CSV produced by a prior run.
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df["last_updated"] = pd.to_datetime(df["last_updated"])
        config = DataPrepConfig(
            raw_data_path=RAW_DATA_PATH,
            processed_data_path=PROCESSED_DATA_PATH,
        )
        preprocessor = WeatherDataPreprocessor(config=config)
        # Re-fit encoders/scaler on the processed data so the preprocessor
        # object is usable for inference calls later in the session.
        numeric_cols = [c for c in config.columns_to_scale if c in df.columns]
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        preprocessor.scaler = StandardScaler().fit(df[numeric_cols])
        preprocessor.scaled_columns_ = numeric_cols
        for col in config.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                preprocessor.label_encoders[col] = le
        return df, preprocessor

    # Slow path: run the full pipeline (writes processed CSV to disk).
    df, preprocessor = run_data_prep_pipeline(
        raw_path=RAW_DATA_PATH,
        save_path=PROCESSED_DATA_PATH,
        drop_outliers=False,
    )
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    return df, preprocessor


@st.cache_data(show_spinner=False)
def get_raw_df() -> pd.DataFrame:
    """Return the raw (unprocessed) CSV as a lightweight reference for
    schema inspection and the Dataset Overview tab."""
    return pd.read_csv(RAW_DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training forecasting models…")
def run_forecast(location_name: str) -> tuple:
    """Fit the full MultivariateForecastingEngine for `location_name` and
    return the engine, per-model test metrics, and the 30-day forecast
    dataframe.

    Keyed by `location_name` so switching location triggers a re-run while
    re-selecting the same location hits the cache.
    """
    df, _ = load_and_preprocess()
    config = ForecastConfig()
    engine, metrics, forecast_df = run_forecasting_pipeline(
        df, location_name=location_name, config=config
    )
    return engine, metrics, forecast_df


# ─────────────────────────────────────────────────────────────────────────────
# CACHED EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Computing SHAP values…")
def run_explainability(location_name: str, model_name: str) -> tuple:
    """Compute SHAP values and permutation importance for `model_name`,
    given a previously-fitted engine for `location_name`.

    Returns (global_ranking_df, shap_values_array, X_explained_df,
    feature_names, permutation_ranking_df).
    """
    engine, _, _ = run_forecast(location_name)

    if model_name not in engine.fitted_models_:
        available = [
            k for k in engine.fitted_models_ if k not in ("ARIMA", "Prophet")
        ]
        model_name = available[0] if available else None

    if model_name is None:
        return None, None, None, None, None

    model = engine.fitted_models_[model_name]
    feature_names = engine.feature_cols_
    X_train = engine.train_df_[feature_names]
    X_test = engine.test_df_[feature_names]
    y_test = engine.test_df_[engine.config.target_col]

    exp_config = ExplainabilityConfig(
        output_dir="images",
        shap_sample_size=300,
        permutation_n_repeats=8,
        top_n_features=15,
    )

    # SHAP
    analyzer = ShapAnalyzer(model, feature_names, config=exp_config)
    analyzer.fit_explainer(X_test)
    shap_ranking = analyzer.get_shap_importance_ranking()

    # Permutation importance
    perm_ranking = compute_permutation_importance(
        model, X_test, y_test, config=exp_config
    )

    global_ranking = build_global_feature_ranking(shap_ranking, perm_ranking)

    return (
        global_ranking,
        analyzer.shap_values_,
        analyzer.X_explained_,
        feature_names,
        perm_ranking,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    """Render the sidebar navigation and return a dict of user selections."""

    with st.sidebar:
        # ── Logo / Title ──
        st.markdown(
            """
            <div style="text-align:center; padding: 8px 0 20px 0;">
                <span style="font-size:2.4rem;">🌦️</span>
                <div style="font-size:1.05rem; font-weight:700;
                            color:#00B4D8; letter-spacing:0.04em;">
                    Weather Forecast Engine
                </div>
                <div style="font-size:0.72rem; color:#8FA8C0;
                            margin-top:3px;">
                    Global Weather Repository · Kaggle
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── PM Accelerator Mission ──
        st.markdown(
            f"""
            <div class="mission-banner">
                <div class="mission-title">🚀 PM Accelerator</div>
                <p>{PM_ACCELERATOR_MISSION}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Location selector ──
        st.markdown(
            '<div style="font-size:0.78rem; color:#8FA8C0; font-weight:600;'
            ' text-transform:uppercase; letter-spacing:0.07em;'
            ' margin-bottom:6px;">📍 Location</div>',
            unsafe_allow_html=True,
        )
        locations = sorted(df["location_name"].dropna().unique().tolist())
        default_loc = (
            locations[0]
            if "London" not in locations
            else "London"
        )
        selected_location = st.selectbox(
            "Select location",
            locations,
            index=locations.index(default_loc) if default_loc in locations else 0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── EDA variable selectors ──
        st.markdown(
            '<div style="font-size:0.78rem; color:#8FA8C0; font-weight:600;'
            ' text-transform:uppercase; letter-spacing:0.07em;'
            ' margin-bottom:6px;">📊 EDA Variable</div>',
            unsafe_allow_html=True,
        )
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        eda_cols_defaults = [
            c for c in [
                "temperature_celsius", "humidity", "pressure_mb",
                "wind_kph", "precip_mm",
            ]
            if c in numeric_cols
        ]
        eda_variable = st.selectbox(
            "Variable for time-series & map",
            numeric_cols,
            index=numeric_cols.index(eda_cols_defaults[0])
            if eda_cols_defaults else 0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Forecasting model selector ──
        st.markdown(
            '<div style="font-size:0.78rem; color:#8FA8C0; font-weight:600;'
            ' text-transform:uppercase; letter-spacing:0.07em;'
            ' margin-bottom:6px;">🤖 Explainability Model</div>',
            unsafe_allow_html=True,
        )
        tree_models = ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees"]
        explainability_model = st.selectbox(
            "Tree model for SHAP / permutation importance",
            tree_models,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Dataset path info ──
        st.markdown(
            f"""
            <div style="font-size:0.72rem; color:#8FA8C0; line-height:1.6;">
            <b style="color:#00B4D8;">Data path</b><br>
            {RAW_DATA_PATH}<br><br>
            <b style="color:#00B4D8;">Processed path</b><br>
            {PROCESSED_DATA_PATH}
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "location": selected_location,
        "eda_variable": eda_variable,
        "explainability_model": explainability_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0  —  Dataset Overview & KPIs
# ─────────────────────────────────────────────────────────────────────────────

def render_overview_tab(df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """Render Dataset Overview & KPI cards."""

    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <div style="font-size:1.6rem; font-weight:800; color:#00B4D8;">
                Global Weather Repository
            </div>
            <div style="font-size:0.88rem; color:#8FA8C0; margin-top:4px;">
                Panel of hourly weather readings across 200+ locations worldwide ·
                Multivariate temperature forecasting pipeline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Row 1: Dataset-level KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    _kpi(c1, "Total Observations", f"{len(df):,}", "after preprocessing")
    _kpi(c2, "Raw Rows", f"{len(raw_df):,}", "before deduplication")
    _kpi(
        c3,
        "Locations",
        f"{df['location_name'].nunique():,}",
        f"{df['country'].nunique()} countries",
    )
    _kpi(c4, "Features (engineered)", f"{df.shape[1]:,}", "incl. lag / rolling")
    date_range = (
        f"{pd.to_datetime(df['last_updated']).min().strftime('%b %Y')} – "
        f"{pd.to_datetime(df['last_updated']).max().strftime('%b %Y')}"
    )
    _kpi(c5, "Date Range", date_range, "parsed from last_updated")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Weather variable KPIs ──
    c6, c7, c8, c9, c10 = st.columns(5)
    _kpi(
        c6,
        "Avg Temperature",
        f"{df['temperature_celsius'].mean():.1f} °C",
        f"σ = {df['temperature_celsius'].std():.1f} °C",
    )
    _kpi(
        c7,
        "Avg Humidity",
        f"{df['humidity'].mean():.0f}%",
        f"range {df['humidity'].min():.0f}–{df['humidity'].max():.0f}%",
    )
    _kpi(
        c8,
        "Avg Wind Speed",
        f"{df['wind_kph'].mean():.1f} kph",
        f"max {df['wind_kph'].max():.0f} kph",
    )
    iqr_pct = (df["is_outlier_iqr"].sum() / len(df) * 100) if "is_outlier_iqr" in df else 0
    _kpi(c9, "IQR Outlier Rows", f"{df['is_outlier_iqr'].sum():,}" if "is_outlier_iqr" in df else "N/A", f"{iqr_pct:.2f}% of dataset")
    iso_pct = (df["is_outlier_isoforest"].sum() / len(df) * 100) if "is_outlier_isoforest" in df else 0
    _kpi(c10, "IsoForest Outliers", f"{df['is_outlier_isoforest'].sum():,}" if "is_outlier_isoforest" in df else "N/A", f"{iso_pct:.2f}% of dataset")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Schema snapshot ──
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        _section("Processed Dataset Snapshot (first 100 rows)")
        display_cols = [
            "location_name", "country", "last_updated", "temperature_celsius",
            "humidity", "pressure_mb", "wind_kph", "precip_mm",
            "condition_text",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].head(100),
            use_container_width=True,
            height=320,
        )

    with col_right:
        _section("Missing Value Analysis (raw data)")
        missing = (
            raw_df.isnull()
            .sum()
            .reset_index()
            .rename(columns={"index": "column", 0: "missing_count"})
        )
        missing["missing_%"] = (missing["missing_count"] / len(raw_df) * 100).round(2)
        missing = missing[missing["missing_count"] > 0].sort_values(
            "missing_%", ascending=False
        )
        if missing.empty:
            st.success("✅ No missing values found in the raw dataset.")
        else:
            fig_miss = go.Figure(
                go.Bar(
                    x=missing["missing_%"],
                    y=missing["column"],
                    orientation="h",
                    marker_color=ACCENT,
                )
            )
            _apply_dark_theme(
                fig_miss, title="Missing % per Column", height=320
            )
            fig_miss.update_xaxes(title_text="Missing (%)")
            st.plotly_chart(fig_miss, use_container_width=True)

    # ── Numeric summary table ──
    _section("Descriptive Statistics — Core Numeric Variables")
    stat_cols = [
        "temperature_celsius", "humidity", "pressure_mb",
        "wind_kph", "precip_mm", "cloud", "uv_index",
        "feels_like_celsius", "gust_kph",
    ]
    stat_cols = [c for c in stat_cols if c in df.columns]
    desc = df[stat_cols].describe().T.round(3)
    st.dataframe(desc, use_container_width=True, height=320)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  —  EDA & Spatial Analysis
# ─────────────────────────────────────────────────────────────────────────────

def render_eda_tab(df: pd.DataFrame, eda_variable: str, location: str) -> None:

    # ── Distribution plots ──
    _section("Variable Distributions")
    dist_cols = [
        "temperature_celsius", "precip_mm", "humidity",
        "wind_kph", "pressure_mb", "uv_index",
    ]
    dist_cols = [c for c in dist_cols if c in df.columns]

    n_cols = 3
    rows = [dist_cols[i : i + n_cols] for i in range(0, len(dist_cols), n_cols)]
    for row_vars in rows:
        row_container = st.columns(len(row_vars))
        for col_widget, var in zip(row_container, row_vars):
            with col_widget:
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=df[var].dropna(),
                        nbinsx=60,
                        marker_color=ACCENT,
                        opacity=0.85,
                        name=var,
                    )
                )
                fig.add_vline(
                    x=df[var].mean(),
                    line_dash="dash",
                    line_color=WARNING,
                    annotation_text="mean",
                    annotation_font_color=WARNING,
                    annotation_font_size=10,
                )
                _apply_dark_theme(fig, title=var, height=240)
                fig.update_layout(margin=dict(l=30, r=10, t=40, b=30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time-series trend for selected variable ──
    _section(f"Time-Series Trend — {eda_variable}")
    _info(
        "Daily, monthly, and seasonal aggregations computed globally across "
        "all locations. Use the sidebar to switch variables."
    )
    try:
        ts_fig = plot_time_series_trends(df, target_col=eda_variable)
        ts_fig.update_layout(**PLOTLY_DARK_THEME, height=780)
        ts_fig.update_xaxes(gridcolor="#2A3F57")
        ts_fig.update_yaxes(gridcolor="#2A3F57")
        st.plotly_chart(ts_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render time-series trend: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-location line chart for selected variable ──
    _section(f"Location Comparison — {eda_variable} (top 10 locations by observation count)")
    top_locs = (
        df["location_name"].value_counts().head(10).index.tolist()
    )
    daily_loc = (
        df[df["location_name"].isin(top_locs)]
        .set_index("last_updated")
        .groupby("location_name")[eda_variable]
        .resample("W")
        .mean()
        .reset_index()
    )
    fig_lc = px.line(
        daily_loc,
        x="last_updated",
        y=eda_variable,
        color="location_name",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    _apply_dark_theme(fig_lc, title=f"Weekly mean {eda_variable} — top 10 locations", height=400)
    st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation heatmap ──
    _section("Correlation Heatmap — Core Weather Variables")
    corr_vars = [
        "temperature_celsius", "humidity", "pressure_mb",
        "wind_kph", "precip_mm", "cloud", "uv_index",
        "feels_like_celsius", "visibility_km",
    ]
    corr_vars = [c for c in corr_vars if c in df.columns]
    corr = df[corr_vars].corr().round(2)
    fig_corr = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            colorbar=dict(title="r", tickfont=dict(color="#8FA8C0")),
        )
    )
    _apply_dark_theme(fig_corr, title="Pearson Correlation Matrix", height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Air Quality vs Weather ──
    _section("Air Quality vs Weather Relationships")
    try:
        aq_fig = plot_air_quality_relationships(df)
        aq_fig.update_layout(**PLOTLY_DARK_THEME, height=880)
        aq_fig.update_xaxes(gridcolor="#2A3F57")
        aq_fig.update_yaxes(gridcolor="#2A3F57")
        st.plotly_chart(aq_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Air quality plot unavailable: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOF Anomaly Detection ──
    _section("Multivariate Anomaly Detection — Local Outlier Factor")
    _info(
        "LOF flags anomalies that are unusual relative to their local "
        "neighborhood in the joint feature space, even when each variable "
        "individually looks unremarkable."
    )
    try:
        lof_fig = plot_lof_anomalies(df, contamination=0.02)
        lof_fig.update_layout(**PLOTLY_DARK_THEME, height=500)
        lof_fig.update_xaxes(gridcolor="#2A3F57")
        lof_fig.update_yaxes(gridcolor="#2A3F57")
        st.plotly_chart(lof_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"LOF anomaly plot unavailable: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Geographic Maps (Folium via streamlit-folium) ──
    _section(f"Geographic Map — {eda_variable} (latest reading per location)")
    _info(
        "Each marker represents the most recent observed value for that "
        "location. Color scale: blue (low) → yellow (median) → red (high)."
    )
    try:
        from streamlit_folium import st_folium  # optional dep

        geo_map = create_geographic_map(df, value_col=eda_variable)
        st_folium(geo_map, width="100%", height=480, returned_objects=[])
    except ImportError:
        st.info(
            "Install `streamlit-folium` (`pip install streamlit-folium`) "
            "to view interactive geographic maps."
        )
    except Exception as exc:
        st.warning(f"Geographic map unavailable: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Heatmap ──
    _section(f"Density Heatmap — {eda_variable}")
    try:
        from streamlit_folium import st_folium  # noqa: F811

        heat_map = create_heatmap(df, value_col=eda_variable)
        st_folium(heat_map, width="100%", height=420, returned_objects=[])
    except ImportError:
        st.info("Install `streamlit-folium` to view the density heatmap.")
    except Exception as exc:
        st.warning(f"Heatmap unavailable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  —  Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def render_forecasting_tab(location: str) -> None:

    st.markdown(
        f"""
        <div style="margin-bottom:18px;">
            <span style="font-size:1.2rem; font-weight:700; color:#E8EDF2;">
                30-Day Temperature Forecast
            </span>
            <span style="font-size:0.82rem; color:#8FA8C0; margin-left:12px;">
                📍 {location}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _info(
        "The ensemble combines ARIMA, Prophet, Random Forest, Extra Trees, "
        "XGBoost, and LightGBM using inverse-RMSE weighting. Tabular models "
        "consume lag and rolling features for temperature, humidity, pressure, "
        "wind speed, precipitation, cloud cover, and PM 2.5."
    )

    if st.button("▶  Run Forecasting Pipeline", key="run_forecast_btn"):
        st.session_state["forecast_run"] = True

    # Auto-run if location changes
    if "forecast_location" not in st.session_state:
        st.session_state["forecast_location"] = location
    if st.session_state.get("forecast_location") != location:
        st.session_state["forecast_location"] = location
        st.session_state["forecast_run"] = True

    if not st.session_state.get("forecast_run", False):
        st.markdown(
            '<div style="color:#8FA8C0; font-size:0.85rem; margin-top:20px;">'
            'Click "Run Forecasting Pipeline" to train models and generate the '
            "30-day forecast for the selected location.</div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Fitting models and projecting 30 days ahead…"):
        try:
            engine, metrics, forecast_df = run_forecast(location)
        except Exception as exc:
            st.error(f"Forecasting failed: {exc}")
            return

    # ── Model Metrics Table ──
    _section("Test-Set Evaluation Metrics")

    metrics_df = pd.DataFrame(metrics).T.round(4)
    metrics_df.index.name = "Model"
    metrics_df = metrics_df.reset_index()

    # Highlight best (lowest MAE/RMSE/MAPE, highest R²) per column
    def _highlight_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
        styled = df_in.copy()
        for col in ["MAE", "RMSE", "MAPE"]:
            if col in styled.columns:
                min_idx = styled[col].idxmin()
                styled.loc[min_idx, col] = f"✅ {styled.loc[min_idx, col]}"
        if "R2" in styled.columns:
            max_idx = styled["R2"].idxmax()
            styled.loc[max_idx, "R2"] = f"✅ {styled.loc[max_idx, 'R2']}"
        return styled

    st.dataframe(
        _highlight_metrics(metrics_df),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Bar Charts ──
    metric_cols = st.columns(4)
    for i, (metric, label) in enumerate(
        [("MAE", "MAE (°C)"), ("RMSE", "RMSE (°C)"), ("MAPE", "MAPE (%)"), ("R2", "R²")]
    ):
        if metric not in metrics_df.columns:
            continue
        raw_df_m = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
        colors_list = [
            SUCCESS if raw_df_m.loc[raw_df_m["Model"] == m, metric].values[0]
            == (raw_df_m[metric].min() if metric != "R2" else raw_df_m[metric].max())
            else ACCENT
            for m in raw_df_m["Model"]
        ]
        fig_m = go.Figure(
            go.Bar(
                x=raw_df_m["Model"],
                y=raw_df_m[metric].round(4),
                marker_color=colors_list,
                text=raw_df_m[metric].round(4),
                textposition="outside",
                textfont=dict(size=10, color="#E8EDF2"),
            )
        )
        _apply_dark_theme(fig_m, title=label, height=280)
        fig_m.update_layout(margin=dict(l=20, r=10, t=45, b=30), showlegend=False)
        with metric_cols[i]:
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actual vs Predicted (test period) ──
    _section("Actual vs Predicted — Test Period")
    test_df = engine.test_df_
    y_test = test_df[engine.config.target_col]

    fig_avp = go.Figure()
    fig_avp.add_trace(
        go.Scatter(
            x=test_df.index,
            y=y_test,
            mode="lines",
            name="Actual",
            line=dict(color="#E8EDF2", width=2),
        )
    )
    model_colors = {
        "Ensemble": ACCENT,
        "XGBoost": SUCCESS,
        "LightGBM": WARNING,
        "RandomForest": "#9B59B6",
        "ExtraTrees": "#E74C3C",
        "ARIMA": "#F39C12",
        "Prophet": "#1ABC9C",
    }
    for model_name, preds in engine.test_predictions_.items():
        if len(preds) == len(y_test):
            fig_avp.add_trace(
                go.Scatter(
                    x=test_df.index,
                    y=preds,
                    mode="lines",
                    name=model_name,
                    line=dict(
                        color=model_colors.get(model_name, "#8FA8C0"),
                        width=2.5 if model_name == "Ensemble" else 1.2,
                        dash="solid" if model_name == "Ensemble" else "dot",
                    ),
                    opacity=1.0 if model_name == "Ensemble" else 0.7,
                )
            )

    _apply_dark_theme(
        fig_avp,
        title=f"Actual vs Predicted — {location} (test period)",
        height=430,
    )
    fig_avp.update_yaxes(title_text="Temperature (°C)")
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 30-Day Ensemble Forecast ──
    _section("30-Day Ensemble Forecast")

    fig_fc = go.Figure()

    # Historical tail (last 60 observations from daily series)
    hist_tail = engine.daily_df_[engine.config.target_col].tail(60)
    fig_fc.add_trace(
        go.Scatter(
            x=hist_tail.index,
            y=hist_tail.values,
            mode="lines",
            name="Historical (last 60 days)",
            line=dict(color="#8FA8C0", width=1.5),
        )
    )

    # Forecast
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["forecast_temperature_celsius"],
            mode="lines+markers",
            name="Ensemble Forecast",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5, color=ACCENT),
        )
    )

    # Confidence band (±1 std of historical variability as a proxy)
    hist_std = hist_tail.std()
    fig_fc.add_trace(
        go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df["forecast_temperature_celsius"] + hist_std)
            + list((forecast_df["forecast_temperature_celsius"] - hist_std)[::-1]),
            fill="toself",
            fillcolor=f"rgba(0,180,216,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1 Std Band",
            hoverinfo="skip",
        )
    )

    # Vertical separator between history and forecast
    split_date = forecast_df.index[0]
    fig_fc.add_vline(
        x=split_date,
        line_dash="dash",
        line_color=WARNING,
        annotation_text="Forecast start",
        annotation_font_color=WARNING,
        annotation_font_size=10,
    )

    _apply_dark_theme(
        fig_fc,
        title=f"30-Day Temperature Forecast — {location}",
        height=460,
    )
    fig_fc.update_yaxes(title_text="Temperature (°C)")
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Export forecast CSV ──
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Export Forecast")

    export_df = forecast_df.reset_index().rename(
        columns={"index": "date", "forecast_temperature_celsius": "forecast_temp_celsius"}
    )
    export_df["location"] = location

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    st.download_button(
        label="⬇  Download 30-Day Forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{location.replace(' ', '_').lower()}_30day.csv",
        mime="text/csv",
    )
    st.dataframe(export_df, use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  —  Explainability
# ─────────────────────────────────────────────────────────────────────────────

def render_explainability_tab(location: str, model_name: str) -> None:

    st.markdown(
        f"""
        <div style="margin-bottom:18px;">
            <span style="font-size:1.2rem; font-weight:700; color:#E8EDF2;">
                Model Interpretability
            </span>
            <span style="font-size:0.82rem; color:#8FA8C0; margin-left:12px;">
                📍 {location} · 🤖 {model_name}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _info(
        "SHAP (SHapley Additive exPlanations) via TreeExplainer provides "
        "exact Shapley values for tree ensembles. Permutation importance "
        "cross-checks SHAP by measuring test-set RMSE degradation when each "
        "feature is randomly shuffled, giving a model-agnostic, outcome-based "
        "view of feature contribution."
    )

    # Ensure forecasting has run first
    if not st.session_state.get("forecast_run", False):
        st.warning(
            "Run the **Forecasting** pipeline first (Tab 3) before computing "
            "SHAP values — the fitted model objects are required."
        )
        return

    with st.spinner(f"Computing SHAP values for {model_name}…"):
        try:
            (
                global_ranking,
                shap_values,
                X_explained,
                feature_names,
                perm_ranking,
            ) = run_explainability(location, model_name)
        except Exception as exc:
            st.error(f"Explainability computation failed: {exc}")
            return

    if global_ranking is None:
        st.warning(
            f"'{model_name}' was not fitted during the forecasting step "
            "(may not be installed). Select a different model in the sidebar."
        )
        return

    # ── Global Feature Ranking Table ──
    _section("Unified Feature Importance Ranking (SHAP + Permutation)")
    _info(
        "Both metrics are normalized to [0, 1] and averaged into a combined "
        "score. Rank 1 = most important overall. Features consistent across "
        "both methods are robustly informative."
    )
    top15 = global_ranking.head(15)[
        ["combined_rank", "feature", "combined_score",
         "mean_abs_shap", "shap_rank",
         "permutation_importance", "permutation_rank"]
    ].copy()
    top15.columns = [
        "Rank", "Feature", "Combined Score",
        "Mean |SHAP|", "SHAP Rank",
        "Perm. Importance", "Perm. Rank",
    ]
    st.dataframe(top15.round(5), use_container_width=True, hide_index=True, height=420)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Beeswarm Plot ──
    _section("SHAP Summary — Global Feature Impact (Beeswarm)")
    _info(
        "Each dot is one (sample, feature) SHAP value. "
        "Horizontal position = impact on temperature prediction (°C). "
        "Color = feature value (blue = low, red = high). "
        "Sorted top-to-bottom by mean |SHAP|."
    )

    shap_arr = np.array(shap_values)
    mean_abs = np.abs(shap_arr).mean(axis=0)
    top_n = min(15, len(feature_names))
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    fig_beeswarm = go.Figure()
    color_scale = [[0, "#3B82F6"], [0.5, "#A78BFA"], [1, "#EF4444"]]
    for fi in top_idx[::-1]:  # bottom-to-top so most important is at top
        feat = feature_names[fi]
        sv = shap_arr[:, fi]
        fv = X_explained.iloc[:, fi].values
        fv_norm = (fv - fv.min()) / (fv.ptp() + 1e-9)
        colors = [
            f"rgb({int((1 - t) * 59 + t * 239)}, "
            f"{int((1 - t) * 130 + t * 68)}, "
            f"{int((1 - t) * 246 + t * 68)})"
            for t in fv_norm
        ]
        fig_beeswarm.add_trace(
            go.Scatter(
                x=sv,
                y=[feat] * len(sv),
                mode="markers",
                marker=dict(
                    size=4,
                    color=fv_norm,
                    colorscale=color_scale,
                    opacity=0.55,
                    line=dict(width=0),
                ),
                name=feat,
                showlegend=False,
                hovertemplate=(
                    f"<b>{feat}</b><br>"
                    "SHAP: %{x:.4f}<br>"
                    "Feature value: %{customdata:.4f}<extra></extra>"
                ),
                customdata=fv,
            )
        )

    fig_beeswarm.add_vline(x=0, line_color="#8FA8C0", line_width=1)
    _apply_dark_theme(
        fig_beeswarm,
        title=f"SHAP Beeswarm — {model_name} ({location})",
        height=max(380, top_n * 28),
    )
    fig_beeswarm.update_xaxes(title_text="SHAP value (impact on temperature °C)")
    fig_beeswarm.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(fig_beeswarm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Bar Chart ──
    _section("SHAP Mean |Impact| per Feature — Bar Chart")
    shap_bar_data = (
        global_ranking[["feature", "mean_abs_shap"]]
        .head(15)
        .sort_values("mean_abs_shap")
    )
    fig_bar = go.Figure(
        go.Bar(
            x=shap_bar_data["mean_abs_shap"],
            y=shap_bar_data["feature"],
            orientation="h",
            marker=dict(
                color=shap_bar_data["mean_abs_shap"],
                colorscale=[[0, "#0077A8"], [1, "#00B4D8"]],
                showscale=False,
            ),
            text=shap_bar_data["mean_abs_shap"].round(4),
            textposition="outside",
            textfont=dict(size=10, color="#E8EDF2"),
        )
    )
    _apply_dark_theme(fig_bar, title="Mean |SHAP Value| per Feature (top 15)", height=480)
    fig_bar.update_xaxes(title_text="Mean |SHAP Value| (°C)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Permutation Importance ──
    _section("Permutation Feature Importance (Test Set, RMSE-based)")
    _info(
        "Importance = increase in test-set RMSE when the feature's values "
        "are randomly shuffled across samples. Error bars show variability "
        "across 8 repeat permutations."
    )
    perm_plot = perm_ranking.head(15).sort_values("importance_mean")
    fig_perm = go.Figure(
        go.Bar(
            x=perm_plot["importance_mean"],
            y=perm_plot["feature"],
            orientation="h",
            error_x=dict(
                type="data",
                array=perm_plot["importance_std"],
                color="#8FA8C0",
                thickness=1.5,
                width=4,
            ),
            marker=dict(
                color=perm_plot["importance_mean"],
                colorscale=[[0, "#1E2F45"], [1, SUCCESS]],
                showscale=False,
            ),
            text=perm_plot["importance_mean"].round(4),
            textposition="outside",
            textfont=dict(size=10, color="#E8EDF2"),
        )
    )
    _apply_dark_theme(
        fig_perm,
        title="Permutation Importance — RMSE increase when shuffled (top 15)",
        height=480,
    )
    fig_perm.update_xaxes(title_text="Increase in RMSE")
    st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Dependence — top lag feature vs humidity ──
    _section("SHAP Dependence: Lag-1 Temperature vs Humidity")
    _info(
        "Shows how the SHAP contribution of yesterday's temperature "
        "(lag_1) changes across its value range, colored by humidity — "
        "revealing the interaction effect between the two variables."
    )
    lag1_col = "temperature_celsius_lag_1"
    hum_col = "humidity"
    if lag1_col in feature_names and hum_col in X_explained.columns:
        lag1_idx = list(feature_names).index(lag1_col)
        hum_vals = X_explained[hum_col].values
        lag1_shap = shap_arr[:, lag1_idx]
        lag1_feat = X_explained[lag1_col].values

        fig_dep = go.Figure(
            go.Scatter(
                x=lag1_feat,
                y=lag1_shap,
                mode="markers",
                marker=dict(
                    size=5,
                    color=hum_vals,
                    colorscale="Viridis",
                    colorbar=dict(
                        title="Humidity (%)",
                        tickfont=dict(color="#8FA8C0"),
                        titlefont=dict(color="#8FA8C0"),
                    ),
                    opacity=0.65,
                ),
                hovertemplate=(
                    "Lag-1 Temp: %{x:.2f} °C<br>"
                    "SHAP: %{y:.4f}<br>"
                    "Humidity: %{marker.color:.1f}%<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig_dep.add_hline(y=0, line_color="#8FA8C0", line_dash="dot", line_width=1)
        _apply_dark_theme(
            fig_dep,
            title="SHAP Dependence: temperature_celsius_lag_1 (colored by humidity)",
            height=420,
        )
        fig_dep.update_xaxes(title_text="temperature_celsius_lag_1 (°C)")
        fig_dep.update_yaxes(title_text="SHAP value (°C)")
        st.plotly_chart(fig_dep, use_container_width=True)
    else:
        st.info("Lag-1 temperature or humidity feature not available in the feature matrix.")

    # ── SHAP vs Permutation rank comparison ──
    st.markdown("<br>", unsafe_allow_html=True)
    _section("SHAP Rank vs Permutation Rank — Agreement Scatter")
    _info(
        "Points near the diagonal indicate strong agreement between the "
        "two importance methods (robust features). Points far from the "
        "diagonal may indicate multicollinearity effects captured "
        "differently by each method."
    )
    plot_rank = global_ranking.head(25).copy()
    fig_rank = go.Figure(
        go.Scatter(
            x=plot_rank["shap_rank"],
            y=plot_rank["permutation_rank"],
            mode="markers+text",
            text=plot_rank["feature"].str.replace("temperature_celsius", "temp", regex=False),
            textposition="top center",
            textfont=dict(size=8, color="#8FA8C0"),
            marker=dict(
                size=10,
                color=plot_rank["combined_score"],
                colorscale=[[0, "#0077A8"], [1, ACCENT]],
                showscale=True,
                colorbar=dict(
                    title="Combined Score",
                    tickfont=dict(color="#8FA8C0"),
                    titlefont=dict(color="#8FA8C0"),
                ),
            ),
        )
    )
    max_rank = max(plot_rank["shap_rank"].max(), plot_rank["permutation_rank"].max())
    fig_rank.add_trace(
        go.Scatter(
            x=[1, max_rank],
            y=[1, max_rank],
            mode="lines",
            line=dict(color="#8FA8C0", dash="dash", width=1),
            name="Perfect agreement",
            showlegend=True,
        )
    )
    _apply_dark_theme(
        fig_rank,
        title="SHAP Rank vs Permutation Rank (top 25 features)",
        height=460,
    )
    fig_rank.update_xaxes(title_text="SHAP Rank", autorange="reversed")
    fig_rank.update_yaxes(title_text="Permutation Rank", autorange="reversed")
    st.plotly_chart(fig_rank, use_container_width=True)

    # ── Download global ranking ──
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Export Feature Ranking")
    csv_buf = io.StringIO()
    global_ranking.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇  Download Global Feature Ranking CSV",
        data=csv_buf.getvalue().encode(),
        file_name=f"feature_ranking_{location.replace(' ', '_').lower()}_{model_name}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Data loading guard ──
    if not os.path.exists(RAW_DATA_PATH):
        st.error(
            f"Raw data not found at `{RAW_DATA_PATH}`. "
            "Place `GlobalWeatherRepository.csv` inside the `data/` directory "
            "and restart the app."
        )
        st.stop()

    # ── Load & preprocess (cached) ──
    with st.spinner("Loading and preprocessing dataset…"):
        df, preprocessor = load_and_preprocess()
        raw_df = get_raw_df()

    # ── Sidebar: returns user selections ──
    selections = render_sidebar(df)
    location = selections["location"]
    eda_variable = selections["eda_variable"]
    explainability_model = selections["explainability_model"]

    # ── Page header ──
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #2A3F57;
                    margin-bottom: 24px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <span style="font-size:2rem;">🌦️</span>
                <div>
                    <div style="font-size:1.5rem; font-weight:800;
                                color:#00B4D8; letter-spacing:-0.01em;">
                        Global Weather Forecasting Engine
                    </div>
                    <div style="font-size:0.82rem; color:#8FA8C0; margin-top:2px;">
                        End-to-end multivariate ML pipeline ·
                        Data prep → EDA → Forecasting → Explainability
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    tab0, tab1, tab2, tab3 = st.tabs(
        [
            "📋  Dataset Overview",
            "🗺️  EDA & Spatial Analysis",
            "📈  Forecasting",
            "🔍  Explainability",
        ]
    )

    with tab0:
        render_overview_tab(df, raw_df)

    with tab1:
        render_eda_tab(df, eda_variable, location)

    with tab2:
        render_forecasting_tab(location)

    with tab3:
        render_explainability_tab(location, explainability_model)


if __name__ == "__main__":
    main()
