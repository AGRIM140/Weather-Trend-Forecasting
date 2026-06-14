"""
eda_spatial.py
==============

Exploratory Data Analysis (EDA) and Spatial/Geographic analysis module for
the Global Weather Repository dataset.

Generates publication-quality, interactive visualizations using Plotly
(distributions, time series, climate/AQ relationships, anomaly detection)
and Folium (interactive geographic maps, heatmaps).

All functions accept the processed dataframe produced by
`src.data_prep.WeatherDataPreprocessor.run_pipeline` (or any dataframe with
the original Global Weather Repository columns plus a parsed
`last_updated` datetime column) and either return a Plotly `Figure` object
or a Folium `Map` object. Saving to `images/` / `report/` is handled by the
calling script (e.g., `main.py` / notebooks) via the provided `save_path`
arguments.
"""

from __future__ import annotations

import os
from typing import List, Optional

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
from plotly.subplots import make_subplots
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# 1. Distribution plots
# --------------------------------------------------------------------------- #
def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Generate a grid of histogram + box-plot distribution panels.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the weather columns.
    columns : Optional[List[str]]
        Columns to plot. Defaults to the 7 variables requested:
        temperature_celsius, precip_mm, humidity, wind_kph, pressure_mb,
        uv_index, air_quality_PM2.5.
    save_path : Optional[str]
        If provided, the figure is saved as an HTML file at this path.

    Returns
    -------
    go.Figure
        A Plotly figure with one histogram subplot per column.
    """
    columns = columns or [
        "temperature_celsius",
        "precip_mm",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "uv_index",
        "air_quality_PM2.5",
    ]
    columns = [c for c in columns if c in df.columns]

    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Distribution of {c}" for c in columns],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    colors = px.colors.qualitative.Set2

    for idx, col in enumerate(columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker_color=colors[idx % len(colors)],
                nbinsx=50,
                showlegend=False,
            ),
            row=row,
            col=col_pos,
        )
        # Add a vertical line for the mean to aid quick visual reasoning
        mean_val = df[col].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="black",
            row=row,
            col=col_pos,
        )

    fig.update_layout(
        title_text="Distributions of Key Weather and Air Quality Variables",
        height=350 * n_rows,
        width=1100,
        template="plotly_white",
        showlegend=False,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


# --------------------------------------------------------------------------- #
# 2. Time-series trend analysis (daily, monthly, seasonal)
# --------------------------------------------------------------------------- #
def plot_time_series_trends(
    df: pd.DataFrame,
    datetime_col: str = "last_updated",
    target_col: str = "temperature_celsius",
    save_path: Optional[str] = None,
) -> go.Figure:
    """Plot daily, monthly, and seasonal (month-of-year) trends for a target
    variable, aggregated globally across all locations.

    Parameters
    ----------
    df : pd.DataFrame
    datetime_col : str
        Name of the parsed datetime column.
    target_col : str
        Variable to aggregate (default: temperature_celsius).
    save_path : Optional[str]

    Returns
    -------
    go.Figure
        Figure with 3 stacked subplots: daily mean trend, monthly mean
        trend, and seasonal (average by calendar month across years)
        profile with std-dev band.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # --- Daily aggregation ---
    daily = (
        df.set_index(datetime_col)[target_col]
        .resample("D")
        .mean()
        .reset_index()
    )

    # --- Monthly aggregation ---
    monthly = (
        df.set_index(datetime_col)[target_col]
        .resample("ME")
        .mean()
        .reset_index()
    )

    # --- Seasonal profile: average by calendar month (across all years) ---
    seasonal = df.copy()
    seasonal["month"] = seasonal[datetime_col].dt.month
    seasonal_stats = (
        seasonal.groupby("month")[target_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    seasonal_stats["month_name"] = seasonal_stats["month"].apply(
        lambda m: month_names[m - 1]
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"Daily Mean {target_col} (Global Average)",
            f"Monthly Mean {target_col} (Global Average)",
            f"Seasonal Profile of {target_col} (Mean +/- 1 Std by Month)",
        ),
        vertical_spacing=0.1,
    )

    # Daily trend
    fig.add_trace(
        go.Scatter(
            x=daily[datetime_col],
            y=daily[target_col],
            mode="lines",
            name="Daily Mean",
            line=dict(color="#1f77b4", width=1),
        ),
        row=1,
        col=1,
    )

    # Monthly trend
    fig.add_trace(
        go.Scatter(
            x=monthly[datetime_col],
            y=monthly[target_col],
            mode="lines+markers",
            name="Monthly Mean",
            line=dict(color="#ff7f0e", width=2),
        ),
        row=2,
        col=1,
    )

    # Seasonal profile with std-dev band
    fig.add_trace(
        go.Scatter(
            x=seasonal_stats["month_name"],
            y=seasonal_stats["mean"] + seasonal_stats["std"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seasonal_stats["month_name"],
            y=seasonal_stats["mean"] - seasonal_stats["std"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.2)",
            name="+/- 1 Std Dev",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seasonal_stats["month_name"],
            y=seasonal_stats["mean"],
            mode="lines+markers",
            name="Seasonal Mean",
            line=dict(color="#2ca02c", width=2),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title_text=f"Time Series Trend Analysis: {target_col}",
        height=1000,
        width=1100,
        template="plotly_white",
        showlegend=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


# --------------------------------------------------------------------------- #
# 3. Geographic & spatial analysis (Folium)
# --------------------------------------------------------------------------- #
def create_geographic_map(
    df: pd.DataFrame,
    value_col: str = "temperature_celsius",
    save_path: Optional[str] = None,
) -> folium.Map:
    """Create an interactive Folium map with markers color-coded by a
    target variable (e.g., current temperature per location).

    Each unique `location_name` is plotted once at its (latitude,
    longitude), using the most recent reading's value for `value_col`. The
    marker color is mapped through a simple quantile-based color scale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `location_name`, `latitude`, `longitude`,
        `last_updated`, and `value_col`.
    value_col : str
        The variable used to color-code markers.
    save_path : Optional[str]
        If provided, the map is saved as a standalone HTML file.

    Returns
    -------
    folium.Map
    """
    # Take the most recent observation per location for a "current
    # snapshot" map.
    latest = (
        df.sort_values("last_updated")
        .groupby("location_name", as_index=False)
        .tail(1)
    )

    # Compute quantile-based color bins for a robust color scale (avoids
    # extreme outliers compressing the color range).
    q_low, q_high = latest[value_col].quantile([0.05, 0.95])

    def _value_to_color(value: float) -> str:
        """Map a numeric value to a color on a blue -> yellow -> red scale."""
        if q_high == q_low:
            ratio = 0.5
        else:
            ratio = np.clip((value - q_low) / (q_high - q_low), 0.0, 1.0)

        if ratio < 0.5:
            # Blue -> Yellow
            t = ratio / 0.5
            r = int(0 + t * 255)
            g = int(102 + t * (255 - 102))
            b = int(255 - t * 255)
        else:
            # Yellow -> Red
            t = (ratio - 0.5) / 0.5
            r = 255
            g = int(255 - t * 255)
            b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    world_map = folium.Map(
        location=[latest["latitude"].mean(), latest["longitude"].mean()],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    for _, row in latest.iterrows():
        popup_html = (
            f"<b>{row['location_name']}, {row['country']}</b><br>"
            f"{value_col}: {row[value_col]:.2f}<br>"
            f"Condition: {row.get('condition_text', 'N/A')}<br>"
            f"Last Updated: {row['last_updated']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=folium.Popup(popup_html, max_width=300),
            color=_value_to_color(row[value_col]),
            fill=True,
            fill_color=_value_to_color(row[value_col]),
            fill_opacity=0.8,
            weight=1,
        ).add_to(world_map)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        world_map.save(save_path)

    return world_map


def create_heatmap(
    df: pd.DataFrame,
    value_col: str = "temperature_celsius",
    save_path: Optional[str] = None,
) -> folium.Map:
    """Create a Folium HeatMap (e.g., for temperature or rainfall
    intensity) using latitude/longitude weighted by `value_col`.

    For variables that can be negative (e.g., `temperature_celsius`), the
    weights are shifted to be non-negative since `folium.plugins.HeatMap`
    requires non-negative weights.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `latitude`, `longitude`, `last_updated`, and
        `value_col`.
    value_col : str
        Variable used as the heat intensity (e.g., `temperature_celsius`
        or `precip_mm`).
    save_path : Optional[str]

    Returns
    -------
    folium.Map
    """
    latest = (
        df.sort_values("last_updated")
        .groupby("location_name", as_index=False)
        .tail(1)
    )

    values = latest[value_col].astype(float)
    min_val = values.min()
    # Shift to non-negative range for HeatMap weighting; add a small
    # epsilon so zero-value points still register a (minimal) heat signal.
    shifted_values = values - min_val + 1e-6

    heat_data = [
        [row["latitude"], row["longitude"], weight]
        for row, weight in zip(latest.to_dict("records"), shifted_values)
    ]

    heat_map = folium.Map(
        location=[latest["latitude"].mean(), latest["longitude"].mean()],
        zoom_start=2,
        tiles="CartoDB dark_matter",
    )

    HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=20,
        blur=15,
        max_zoom=4,
    ).add_to(heat_map)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        heat_map.save(save_path)

    return heat_map


# --------------------------------------------------------------------------- #
# 4. Climate & environmental impact: AQ vs weather relationships
# --------------------------------------------------------------------------- #
def plot_air_quality_relationships(
    df: pd.DataFrame,
    aq_cols: Optional[List[str]] = None,
    weather_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Visualize relationships between air quality indicators and weather
    variables: a correlation heatmap plus scatter plots with trendlines.

    Parameters
    ----------
    df : pd.DataFrame
    aq_cols : Optional[List[str]]
        Air quality columns (default: Ozone, PM2.5, Nitrogen_dioxide,
        Carbon_Monoxide).
    weather_cols : Optional[List[str]]
        Weather columns to correlate against (default: temperature_celsius,
        humidity, wind_kph, pressure_mb).
    save_path : Optional[str]

    Returns
    -------
    go.Figure
        A figure combining a correlation heatmap (top) with scatter plots
        of the two strongest AQ-weather relationships (bottom).
    """
    aq_cols = aq_cols or [
        "air_quality_Ozone",
        "air_quality_PM2.5",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Carbon_Monoxide",
        "air_quality_PM10",
        "air_quality_Sulphur_dioxide",
    ]
    weather_cols = weather_cols or [
        "temperature_celsius",
        "humidity",
        "wind_kph",
        "pressure_mb",
    ]

    aq_cols = [c for c in aq_cols if c in df.columns]
    weather_cols = [c for c in weather_cols if c in df.columns]
    all_cols = aq_cols + weather_cols

    corr_matrix = df[all_cols].corr()
    # Extract the AQ x weather sub-block of the correlation matrix for the
    # heatmap (most relevant cross-relationships).
    cross_corr = corr_matrix.loc[aq_cols, weather_cols]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Correlation: Air Quality vs Weather Variables",
            "",
            "Strongest Positive Relationship",
            "Strongest Negative Relationship",
        ),
        specs=[
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Heatmap(
            z=cross_corr.values,
            x=cross_corr.columns,
            y=cross_corr.index,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(cross_corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Corr"),
        ),
        row=1,
        col=1,
    )

    # Identify strongest positive and negative correlations to scatter-plot.
    flat_corr = cross_corr.stack()
    strongest_pos = flat_corr.idxmax()
    strongest_neg = flat_corr.idxmin()

    # Subsample large datasets for scatter performance/readability.
    sample_df = df.sample(min(5000, len(df)), random_state=42)

    aq_pos, w_pos = strongest_pos
    fig.add_trace(
        go.Scatter(
            x=sample_df[w_pos],
            y=sample_df[aq_pos],
            mode="markers",
            marker=dict(
                size=4,
                color=sample_df[aq_pos],
                colorscale="Viridis",
                opacity=0.5,
            ),
            name=f"{aq_pos} vs {w_pos}",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text=w_pos, row=2, col=1)
    fig.update_yaxes(title_text=aq_pos, row=2, col=1)

    aq_neg, w_neg = strongest_neg
    fig.add_trace(
        go.Scatter(
            x=sample_df[w_neg],
            y=sample_df[aq_neg],
            mode="markers",
            marker=dict(
                size=4,
                color=sample_df[aq_neg],
                colorscale="Plasma",
                opacity=0.5,
            ),
            name=f"{aq_neg} vs {w_neg}",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text=w_neg, row=2, col=2)
    fig.update_yaxes(title_text=aq_neg, row=2, col=2)

    fig.update_layout(
        title_text="Climate & Environmental Impact: Air Quality vs Weather",
        height=900,
        width=1100,
        template="plotly_white",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


# --------------------------------------------------------------------------- #
# 5. Anomaly visualization using Local Outlier Factor (LOF)
# --------------------------------------------------------------------------- #
def plot_lof_anomalies(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_neighbors: int = 20,
    contamination: float = 0.02,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Detect and visualize multivariate anomalies using Local Outlier
    Factor (LOF).

    LOF compares the local density of a point to the local densities of its
    `n_neighbors` nearest neighbors. Points in regions of substantially
    lower density than their neighbors (LOF score >> 1) are flagged as
    anomalies -- this captures *local* anomalies that a global method might
    miss (e.g., a point that's normal globally but anomalous relative to
    its immediate climatic neighborhood).

    Features are standardized before computing LOF, since LOF relies on
    distance metrics that are sensitive to feature scale.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : Optional[List[str]]
        Features used for LOF (default: temperature_celsius, humidity,
        wind_kph, pressure_mb, precip_mm).
    n_neighbors : int
        Number of neighbors used by LOF (default 20).
    contamination : float
        Expected proportion of outliers (default 0.02).
    save_path : Optional[str]

    Returns
    -------
    go.Figure
        A 2D scatter plot (temperature vs humidity, or the first two
        feature columns) with anomalies highlighted, plus a histogram of
        LOF (negative outlier) scores.
    """
    feature_cols = feature_cols or [
        "temperature_celsius",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "precip_mm",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    X_scaled = StandardScaler().fit_transform(X)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination
    )
    # fit_predict: -1 = outlier, 1 = inlier
    labels = lof.fit_predict(X_scaled)
    # negative_outlier_factor_: more negative => more anomalous
    scores = lof.negative_outlier_factor_

    plot_df = df.copy()
    plot_df["lof_label"] = np.where(labels == -1, "Anomaly", "Normal")
    plot_df["lof_score"] = scores

    x_col, y_col = feature_cols[0], feature_cols[1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"LOF Anomalies: {y_col} vs {x_col}",
            "Distribution of LOF Scores",
        ),
        column_widths=[0.6, 0.4],
    )

    for label, color in [("Normal", "#1f77b4"), ("Anomaly", "#d62728")]:
        subset = plot_df[plot_df["lof_label"] == label]
        fig.add_trace(
            go.Scatter(
                x=subset[x_col],
                y=subset[y_col],
                mode="markers",
                name=label,
                marker=dict(
                    size=5 if label == "Normal" else 8,
                    color=color,
                    opacity=0.5 if label == "Normal" else 0.9,
                    symbol="circle" if label == "Normal" else "x",
                ),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=plot_df["lof_score"],
            nbinsx=50,
            marker_color="#9467bd",
            name="LOF Score",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Negative Outlier Factor (LOF Score)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    n_anomalies = int((labels == -1).sum())
    fig.update_layout(
        title_text=(
            f"Multivariate Anomaly Detection via Local Outlier Factor "
            f"({n_anomalies} anomalies flagged out of {len(plot_df)} rows, "
            f"contamination={contamination})"
        ),
        height=500,
        width=1100,
        template="plotly_white",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


# --------------------------------------------------------------------------- #
# Orchestration: generate all EDA/spatial outputs
# --------------------------------------------------------------------------- #
def generate_all_eda_outputs(
    df: pd.DataFrame, output_dir: str = "images"
) -> dict:
    """Generate and save all EDA/spatial visualizations to `output_dir`.

    Returns a dictionary mapping a short name to the saved file path for
    each generated visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs: dict = {}

    dist_path = os.path.join(output_dir, "distributions.html")
    plot_distributions(df, save_path=dist_path)
    outputs["distributions"] = dist_path

    trend_path = os.path.join(output_dir, "time_series_trends.html")
    plot_time_series_trends(df, save_path=trend_path)
    outputs["time_series_trends"] = trend_path

    map_path = os.path.join(output_dir, "geographic_map.html")
    create_geographic_map(df, save_path=map_path)
    outputs["geographic_map"] = map_path

    heatmap_temp_path = os.path.join(output_dir, "heatmap_temperature.html")
    create_heatmap(df, value_col="temperature_celsius", save_path=heatmap_temp_path)
    outputs["heatmap_temperature"] = heatmap_temp_path

    heatmap_precip_path = os.path.join(output_dir, "heatmap_precipitation.html")
    create_heatmap(df, value_col="precip_mm", save_path=heatmap_precip_path)
    outputs["heatmap_precipitation"] = heatmap_precip_path

    aq_path = os.path.join(output_dir, "air_quality_relationships.html")
    plot_air_quality_relationships(df, save_path=aq_path)
    outputs["air_quality_relationships"] = aq_path

    lof_path = os.path.join(output_dir, "lof_anomalies.html")
    plot_lof_anomalies(df, save_path=lof_path)
    outputs["lof_anomalies"] = lof_path

    return outputs


if __name__ == "__main__":
    data_path = "data/processed_weather_data.csv"
    if os.path.exists(data_path):
        weather_df = pd.read_csv(data_path)
        weather_df["last_updated"] = pd.to_datetime(weather_df["last_updated"])
        generated = generate_all_eda_outputs(weather_df)
        print("Generated EDA outputs:")
        for name, path in generated.items():
            print(f"  {name}: {path}")
    else:
        print(
            f"Processed data not found at '{data_path}'. "
            "Run src/data_prep.py first."
        )
