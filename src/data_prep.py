"""
data_prep.py
============

Production-grade data preprocessing and multivariate feature-engineering
pipeline for the Kaggle "Global Weather Repository" dataset.

This module is responsible for:
    1. Loading the raw CSV.
    2. Missing value analysis and imputation.
    3. Duplicate removal and exact datetime conversion of `last_updated`,
       with the data sorted chronologically (per-location).
    4. Outlier detection using both the IQR rule and an Isolation Forest.
    5. Scaling (StandardScaler) and categorical encoding.
    6. Multivariate feature engineering: lag features and rolling
       statistics for `temperature_celsius` AND key exogenous variables
       (`humidity`, `pressure_mb`, `wind_kph`, `precip_mm`).

The dataset is a *panel* of time series: each `location_name` has its own
chronological sequence of observations (roughly hourly readings, ~550-758
timestamps per location). Consequently, every time-aware operation
(sorting, lagging, rolling statistics, outlier detection on temporal
structure) is performed *within each location group* (`groupby`) to avoid
mixing unrelated time series and to prevent data leakage across locations.

All transformations are encapsulated in scikit-learn-style classes/methods
so they can be composed into a pipeline from `main.py`.

Author: Senior Data Science Team
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class DataPrepConfig:
    """Centralized configuration for the preprocessing pipeline.

    Keeping all "magic numbers" and column lists in a single dataclass makes
    the pipeline easy to audit, reproduce, and tune.
    """

    # ---- File paths --------------------------------------------------- #
    raw_data_path: str = "data/GlobalWeatherRepository.csv"
    processed_data_path: str = "data/processed_weather_data.csv"

    # ---- Core columns ---------------------------------------------------- #
    datetime_col: str = "last_updated"
    epoch_col: str = "last_updated_epoch"
    group_col: str = "location_name"
    target_col: str = "temperature_celsius"

    # ---- Exogenous (multivariate) columns used for lag / rolling features - #
    # `temperature_celsius` is included because it is the autoregressive
    # target itself; the remaining variables are the exogenous drivers
    # explicitly requested for the multivariate forecasting approach.
    exogenous_cols: List[str] = field(
        default_factory=lambda: [
            "temperature_celsius",
            "humidity",
            "pressure_mb",
            "wind_kph",
            "precip_mm",
        ]
    )

    # ---- Numeric columns considered for IQR / Isolation Forest outlier
    # detection. Restricted to physically meaningful continuous sensor
    # readings (excludes lat/lon, epoch timestamps, and indices which are
    # not "outliers" in the statistical sense).
    numeric_outlier_cols: List[str] = field(
        default_factory=lambda: [
            "temperature_celsius",
            "wind_kph",
            "pressure_mb",
            "precip_mm",
            "humidity",
            "cloud",
            "feels_like_celsius",
            "visibility_km",
            "uv_index",
            "gust_kph",
            "air_quality_PM2.5",
            "air_quality_PM10",
        ]
    )

    # ---- Columns to scale with StandardScaler --------------------------- #
    # We scale the same set of continuous predictors used for outlier
    # detection plus engineered cyclical/lag/rolling features at a later
    # stage (handled dynamically -- see `scale_features`).
    columns_to_scale: List[str] = field(
        default_factory=lambda: [
            "temperature_celsius",
            "wind_kph",
            "pressure_mb",
            "precip_mm",
            "humidity",
            "cloud",
            "feels_like_celsius",
            "visibility_km",
            "uv_index",
            "gust_kph",
            "air_quality_PM2.5",
            "air_quality_PM10",
            "latitude",
            "longitude",
        ]
    )

    # ---- Categorical columns to label-encode ----------------------------- #
    categorical_cols: List[str] = field(
        default_factory=lambda: [
            "country",
            "location_name",
            "condition_text",
            "wind_direction",
            "moon_phase",
        ]
    )

    # ---- Lag / rolling window configuration ------------------------------ #
    # Lags are expressed in *number of observations* per location. Since the
    # data is recorded roughly hourly, `lag_1` ~= "1 reading ago" and
    # `lag_7` ~= "7 readings ago" (the de-facto daily-cycle-ish lag commonly
    # used in time series literature, e.g. weekly seasonality analogue).
    lag_periods: List[int] = field(default_factory=lambda: [1, 7])

    # Rolling window sizes (in observations) for mean/std aggregations.
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7])

    # ---- IQR multiplier ---------------------------------------------------- #
    # The classic Tukey's fences multiplier. 1.5 is the standard "mild
    # outlier" threshold; values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR are
    # flagged.
    iqr_multiplier: float = 1.5

    # ---- Isolation Forest hyperparameters ----------------------------------- #
    isolation_forest_contamination: float = 0.01  # expected outlier fraction
    isolation_forest_random_state: int = 42
    isolation_forest_n_estimators: int = 200


# --------------------------------------------------------------------------- #
# Main preprocessing class
# --------------------------------------------------------------------------- #
class WeatherDataPreprocessor:
    """End-to-end preprocessing & multivariate feature engineering pipeline.

    The class is stateful: it stores fitted encoders/scalers as attributes so
    that the *exact* same transformations can later be applied to new
    (unseen) data at inference time (e.g., from `app.py`).
    """

    def __init__(self, config: Optional[DataPrepConfig] = None) -> None:
        self.config = config or DataPrepConfig()

        # Fitted transformers are stored here after `fit`-style calls so
        # they can be reused/serialized (e.g., with joblib) for inference.
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.scaled_columns_: List[str] = []

    # ------------------------------------------------------------------ #
    # 1. Loading
    # ------------------------------------------------------------------ #
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load the raw CSV from `data/`.

        Parameters
        ----------
        path : Optional[str]
            Override path; defaults to `config.raw_data_path`.

        Returns
        -------
        pd.DataFrame
            The raw, unmodified dataframe.
        """
        data_path = path or self.config.raw_data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Could not find raw data at '{data_path}'. "
                "Place 'GlobalWeatherRepository.csv' inside the 'data/' folder."
            )

        logger.info("Loading raw data from '%s' ...", data_path)
        df = pd.read_csv(data_path)
        logger.info(
            "Loaded dataframe with shape %s and %d columns.",
            df.shape,
            df.shape[1],
        )
        return df

    # ------------------------------------------------------------------ #
    # 2. Missing value analysis & imputation
    # ------------------------------------------------------------------ #
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary table of missing values per column.

        Statistical reasoning
        ----------------------
        Before deciding on an imputation strategy we must understand *how
        much* and *where* data is missing. A column with >40-50% missing
        values is generally a poor candidate for simple imputation and
        might warrant dropping or a missing-indicator flag instead.
        """
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        summary = (
            pd.DataFrame(
                {"missing_count": missing_count, "missing_pct": missing_pct}
            )
            .sort_values("missing_pct", ascending=False)
            .query("missing_count > 0")
        )
        if summary.empty:
            logger.info("No missing values detected in the dataframe.")
        else:
            logger.info("Missing value summary:\n%s", summary)
        return summary

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using statistically-justified strategies.

        Strategy
        --------
        - Numeric columns: imputed with the **median** rather than the mean.
          The median is robust to the skew/outliers that are common in
          meteorological data (e.g., `precip_mm` is heavily right-skewed
          with many zeros and occasional extreme downpour values, so the
          mean would be pulled upward and would not represent a "typical"
          reading).
        - Numeric columns are imputed *within each `location_name` group*
          first (a location's own median is the most relevant local
          climate baseline), and only fall back to the **global** median
          if an entire location-group is missing for that column (e.g., a
          sensor that never reported air quality).
        - Categorical columns: imputed with the **mode** (most frequent
          category), again computed per-location first with a global
          fallback, since categorical weather conditions (e.g.,
          `condition_text`, `wind_direction`) are strongly location- and
          climate-dependent.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Dataframe with missing values imputed in place (copy returned).
        """
        df = df.copy()
        group_col = self.config.group_col

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # --- Numeric imputation: per-group median, then global median --- #
        for col in numeric_cols:
            if df[col].isnull().sum() == 0:
                continue

            logger.info("Imputing numeric column '%s' with median strategy.", col)

            # Per-location median (transform broadcasts the group median
            # back to every row in that group).
            group_median = df.groupby(group_col)[col].transform("median")
            df[col] = df[col].fillna(group_median)

            # Fallback: if a location had *no* non-null values for this
            # column, group_median will itself be NaN -> use the global
            # median as a last resort.
            global_median = df[col].median()
            df[col] = df[col].fillna(global_median)

        # --- Categorical imputation: per-group mode, then global mode --- #
        for col in categorical_cols:
            if df[col].isnull().sum() == 0:
                continue

            logger.info("Imputing categorical column '%s' with mode strategy.", col)

            def _mode_or_nan(series: pd.Series) -> object:
                mode_vals = series.mode(dropna=True)
                return mode_vals.iloc[0] if not mode_vals.empty else np.nan

            group_mode = df.groupby(group_col)[col].transform(_mode_or_nan)
            df[col] = df[col].fillna(group_mode)

            global_mode_series = df[col].mode(dropna=True)
            if not global_mode_series.empty:
                df[col] = df[col].fillna(global_mode_series.iloc[0])

        remaining_na = df.isnull().sum().sum()
        logger.info(
            "Imputation complete. Remaining missing values: %d", remaining_na
        )
        return df

    # ------------------------------------------------------------------ #
    # 3. Duplicate removal & datetime conversion / sorting
    # ------------------------------------------------------------------ #
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows and duplicate (location, timestamp)
        readings.

        Statistical reasoning
        ----------------------
        - Exact full-row duplicates add no new information and can bias
          aggregate statistics (e.g., inflate the apparent sample size).
        - Duplicate (location, last_updated) pairs represent the *same*
          sensor reading recorded twice; keeping both would create
          artificial "zero-distance" pairs in the time series, which can
          corrupt lag/rolling-window computations (e.g., `lag_1` would
          equal the current value).
        """
        df = df.copy()
        n_before = len(df)

        df = df.drop_duplicates()
        n_after_full = len(df)
        if n_before != n_after_full:
            logger.info(
                "Removed %d exact duplicate rows.", n_before - n_after_full
            )

        subset_cols = [self.config.group_col, self.config.datetime_col]
        df = df.drop_duplicates(subset=subset_cols, keep="first")
        n_after_subset = len(df)
        if n_after_full != n_after_subset:
            logger.info(
                "Removed %d duplicate (location, timestamp) rows.",
                n_after_full - n_after_subset,
            )

        return df.reset_index(drop=True)

    def convert_and_sort_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert `last_updated` to a proper `datetime64` dtype and sort
        the dataframe chronologically *within each location*.

        Statistical reasoning
        ----------------------
        Lag and rolling-window features are only meaningful if the rows
        feeding into `.shift()` / `.rolling()` are in strict chronological
        order. We therefore:
            1. Parse `last_updated` (e.g., "2024-05-16 13:15") into
               `datetime64[ns]` using an explicit format string for
               performance and correctness (avoids pandas' format
               inference, which can silently mis-parse ambiguous
               day/month orderings).
            2. Sort by (`location_name`, `last_updated`) so each location's
               time series is contiguous and increasing in time -- this is
               the exact ordering `groupby().shift()` /
               `groupby().rolling()` rely on.

        We additionally cross-validate against `last_updated_epoch`
        (a Unix timestamp) as a sanity check, since both fields encode the
        same moment in time.
        """
        df = df.copy()
        dt_col = self.config.datetime_col
        epoch_col = self.config.epoch_col
        group_col = self.config.group_col

        # Explicit format avoids ambiguous inference; matches
        # "YYYY-MM-DD HH:MM" observed in the dataset sample.
        df[dt_col] = pd.to_datetime(
            df[dt_col], format="%Y-%m-%d %H:%M", errors="coerce"
        )

        # Cross-check with epoch timestamps: rebuild datetime from
        # `last_updated_epoch` (seconds since 1970-01-01 UTC) and use it to
        # fill any rows where the string parse failed.
        epoch_dt = pd.to_datetime(df[epoch_col], unit="s", errors="coerce")
        n_missing = df[dt_col].isnull().sum()
        if n_missing > 0:
            logger.warning(
                "Found %d rows where '%s' failed to parse; "
                "falling back to '%s'.",
                n_missing,
                dt_col,
                epoch_col,
            )
            df[dt_col] = df[dt_col].fillna(epoch_dt)

        # Drop any rows where datetime is *still* unresolvable -- these are
        # unusable for time-series modeling.
        n_before = len(df)
        df = df.dropna(subset=[dt_col])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(
                "Dropped %d rows with unresolvable timestamps.", n_dropped
            )

        # Sort chronologically within each location group, then reset the
        # index so positional operations (shift/rolling) align correctly.
        df = df.sort_values(by=[group_col, dt_col]).reset_index(drop=True)
        logger.info(
            "Datetime conversion complete. Date range: %s to %s.",
            df[dt_col].min(),
            df[dt_col].max(),
        )
        return df

    # ------------------------------------------------------------------ #
    # 4. Outlier detection
    # ------------------------------------------------------------------ #
    def detect_outliers_iqr(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Flag outliers using Tukey's IQR (interquartile range) rule.

        Statistical reasoning
        ----------------------
        For each numeric column, compute:
            Q1 = 25th percentile, Q3 = 75th percentile
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR  (k = `iqr_multiplier`, default 1.5)

        Any value outside [lower_bound, upper_bound] is flagged. The IQR
        method is distribution-free (no normality assumption) and robust to
        extreme values, making it well-suited to skewed meteorological
        variables like `precip_mm` or `wind_kph`.

        A boolean column `<col>_outlier_iqr` is added for every column in
        `columns`, plus a combined `is_outlier_iqr` flag (True if the row is
        an outlier on *any* monitored variable).

        Parameters
        ----------
        df : pd.DataFrame
        columns : Optional[List[str]]
            Defaults to `config.numeric_outlier_cols`.

        Returns
        -------
        pd.DataFrame
            Copy of `df` with outlier flag columns appended.
        """
        df = df.copy()
        columns = columns or self.config.numeric_outlier_cols
        k = self.config.iqr_multiplier

        combined_flag = pd.Series(False, index=df.index)

        for col in columns:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr

            flag_col = f"{col}_outlier_iqr"
            df[flag_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            combined_flag |= df[flag_col]

            n_outliers = int(df[flag_col].sum())
            logger.info(
                "IQR outliers in '%s': %d rows (bounds: [%.3f, %.3f])",
                col,
                n_outliers,
                lower_bound,
                upper_bound,
            )

        df["is_outlier_iqr"] = combined_flag
        logger.info(
            "Total rows flagged as IQR outliers (any column): %d",
            int(combined_flag.sum()),
        )
        return df

    def detect_outliers_isolation_forest(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Flag multivariate outliers using an Isolation Forest.

        Statistical reasoning
        ----------------------
        The IQR method evaluates each variable *independently* and cannot
        detect outliers that arise from unusual *combinations* of
        variables (e.g., very high temperature combined with very high
        humidity AND very low pressure simultaneously -- each value
        individually plausible, but the joint combination is anomalous).

        Isolation Forest addresses this by building an ensemble of random
        binary trees that recursively partition the feature space.
        Anomalous points require fewer partitions ("splits") to isolate
        because they lie in sparse regions of the joint distribution -- the
        average path length to isolate a point is therefore *shorter* for
        outliers, and the model converts this into an anomaly score.

        We set `contamination=0.01` (expecting ~1% of rows to be
        multivariate anomalies) -- a conservative default appropriate for
        sensor data where genuine extreme weather events are rare but real.

        A boolean column `is_outlier_isoforest` is added (True = anomaly).

        Parameters
        ----------
        df : pd.DataFrame
        columns : Optional[List[str]]
            Defaults to `config.numeric_outlier_cols`.

        Returns
        -------
        pd.DataFrame
            Copy of `df` with the `is_outlier_isoforest` column appended.
        """
        df = df.copy()
        columns = columns or self.config.numeric_outlier_cols
        available_cols = [c for c in columns if c in df.columns]

        # Isolation Forest cannot handle NaNs -- by this stage missing
        # values should already be imputed, but we guard defensively.
        feature_matrix = df[available_cols].fillna(df[available_cols].median())

        iso_forest = IsolationForest(
            n_estimators=self.config.isolation_forest_n_estimators,
            contamination=self.config.isolation_forest_contamination,
            random_state=self.config.isolation_forest_random_state,
            n_jobs=-1,
        )

        # `fit_predict` returns -1 for anomalies, 1 for inliers.
        predictions = iso_forest.fit_predict(feature_matrix)
        df["is_outlier_isoforest"] = predictions == -1

        n_outliers = int(df["is_outlier_isoforest"].sum())
        logger.info(
            "Isolation Forest flagged %d rows (%.2f%%) as multivariate "
            "outliers across columns: %s",
            n_outliers,
            100 * n_outliers / len(df),
            available_cols,
        )
        return df

    # ------------------------------------------------------------------ #
    # 5. Scaling & categorical encoding
    # ------------------------------------------------------------------ #
    def encode_categorical_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Label-encode categorical columns and store fitted encoders.

        Statistical reasoning
        ----------------------
        Tree-based models (Random Forest, XGBoost, LightGBM) -- the primary
        candidates for the downstream multivariate forecasting task -- can
        natively handle integer-encoded categoricals via axis-aligned
        splits, without imposing a false ordinal relationship the way a
        naive numeric cast would for a *linear* model. Label encoding is
        therefore a compact, memory-efficient choice here, while one-hot
        encoding is avoided for high-cardinality columns such as
        `location_name` (268 unique values) and `country` (211 unique
        values), which would otherwise explode the feature space.

        For each encoded column `<col>`, a new integer column
        `<col>_encoded` is created and the fitted `LabelEncoder` is stored
        in `self.label_encoders[<col>]` for later inverse-transformation or
        application to new data.

        Unseen categories at inference time are mapped to a dedicated
        "unknown" bucket (-1) handled by `transform_new_categorical`.
        """
        df = df.copy()
        columns = columns or self.config.categorical_cols

        for col in columns:
            if col not in df.columns:
                continue
            encoder = LabelEncoder()
            df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
            self.label_encoders[col] = encoder
            logger.info(
                "Label-encoded '%s' -> '%s_encoded' (%d unique categories).",
                col,
                col,
                len(encoder.classes_),
            )

        return df

    def transform_new_categorical(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply *already-fitted* label encoders to new/unseen data.

        Categories not seen during `fit` are mapped to -1 rather than
        raising a `ValueError`, which is the default (and brittle) behavior
        of `LabelEncoder.transform`.
        """
        df = df.copy()
        columns = columns or self.config.categorical_cols

        for col in columns:
            if col not in df.columns or col not in self.label_encoders:
                continue
            encoder = self.label_encoders[col]
            class_to_idx = {cls: idx for idx, cls in enumerate(encoder.classes_)}
            df[f"{col}_encoded"] = (
                df[col].astype(str).map(class_to_idx).fillna(-1).astype(int)
            )

        return df

    def scale_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, fit: bool = True
    ) -> pd.DataFrame:
        """Standardize numeric columns using z-score scaling.

        Statistical reasoning
        ----------------------
        StandardScaler transforms each feature to zero mean and unit
        variance: z = (x - mu) / sigma. This is important for:
            - Linear baseline models (e.g., Ridge/Lasso regression), whose
              coefficients are directly comparable only on a common scale.
            - Gradient-based optimizers, which converge faster on
              normalized inputs.
            - Distance-based diagnostics.

        Tree-based ensembles are scale-invariant, so scaling does not harm
        them either -- meaning a single scaled feature set can serve both
        the linear baseline and the tree-based candidate models, keeping
        the pipeline simple.

        New columns are suffixed with `_scaled` so that the *original,
        human-interpretable* values are preserved alongside the scaled
        versions (useful for EDA, plotting, and reporting).

        Parameters
        ----------
        df : pd.DataFrame
        columns : Optional[List[str]]
            Defaults to `config.columns_to_scale`.
        fit : bool
            If True, fit a new `StandardScaler` on `df[columns]` (training
            time). If False, reuse `self.scaler` (must already be fitted) --
            used for transforming validation/test/inference data without
            leaking statistics from those sets.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        columns = columns or self.config.columns_to_scale
        available_cols = [c for c in columns if c in df.columns]

        if fit:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(df[available_cols])
            self.scaled_columns_ = available_cols
            logger.info("Fitted StandardScaler on columns: %s", available_cols)
        else:
            if self.scaler is None:
                raise RuntimeError(
                    "scale_features called with fit=False but no scaler "
                    "has been fitted yet. Call with fit=True first."
                )
            # Ensure the same column order used at fit time.
            available_cols = self.scaled_columns_
            scaled_values = self.scaler.transform(df[available_cols])

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=[f"{c}_scaled" for c in available_cols],
            index=df.index,
        )
        df = pd.concat([df, scaled_df], axis=1)
        return df

    # ------------------------------------------------------------------ #
    # 6. Multivariate feature engineering: lag & rolling statistics
    # ------------------------------------------------------------------ #
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive calendar/time-of-day features from `last_updated`.

        Statistical reasoning
        ----------------------
        Weather exhibits strong periodicity at multiple scales (diurnal
        cycle, annual seasonality). Raw integers like "hour=23" and
        "hour=0" are numerically far apart despite being adjacent in time,
        which misleads models that assume linear/Euclidean relationships.
        We therefore add both the raw integer features (useful for
        tree-based splits, which handle discontinuities natively) AND
        sine/cosine cyclical encodings (useful for linear/distance-based
        models), e.g.:

            hour_sin = sin(2*pi*hour/24)
            hour_cos = cos(2*pi*hour/24)

        This maps each hour onto a point on a unit circle, so 23:00 and
        00:00 are correctly represented as "close" to one another.
        """
        df = df.copy()
        dt_col = self.config.datetime_col

        df["year"] = df[dt_col].dt.year
        df["month"] = df[dt_col].dt.month
        df["day"] = df[dt_col].dt.day
        df["hour"] = df[dt_col].dt.hour
        df["day_of_week"] = df[dt_col].dt.dayofweek
        df["day_of_year"] = df[dt_col].dt.dayofyear

        # Cyclical encodings for diurnal (24h) and annual (~365.25 day)
        # periodicity.
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        logger.info(
            "Created temporal calendar and cyclical features from '%s'.", dt_col
        )
        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create lag features for the target and exogenous variables.

        Statistical reasoning
        ----------------------
        A lag feature `var_lag_k` represents the value of `var` exactly `k`
        observations in the past. These features encode the
        *autocorrelation structure* of the series: weather variables are
        highly autocorrelated at short lags (e.g., the temperature an hour
        ago is the single best predictor of the temperature now), and may
        show weekly-cycle echoes at longer lags.

        Including lags of *exogenous* variables (`humidity`, `pressure_mb`,
        `wind_kph`, `precip_mm`) -- not just the target
        `temperature_celsius` -- allows the model to learn cross-variable
        dynamics, e.g., a sharp pressure drop preceding a temperature change
        (a classic precursor to frontal weather systems).

        CRITICAL: `.shift()` is applied *within each `location_name`
        group* (via `groupby().shift()`), on data already sorted
        chronologically per location. This guarantees that:
            (a) lag values never leak information from a *different*
                location, and
            (b) lag values represent strictly *past* observations for that
                same location.

        The first `max(lags)` rows of each location group will contain NaN
        for these features (no history available yet) -- these are handled
        downstream (typically dropped before model training).

        Parameters
        ----------
        df : pd.DataFrame
            Must already be sorted by (location_name, last_updated).
        columns : Optional[List[str]]
            Defaults to `config.exogenous_cols`.
        lags : Optional[List[int]]
            Defaults to `config.lag_periods` (e.g., [1, 7]).

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        columns = columns or self.config.exogenous_cols
        lags = lags or self.config.lag_periods
        group_col = self.config.group_col

        grouped = df.groupby(group_col, sort=False)

        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                df[feature_name] = grouped[col].shift(lag)
                logger.info("Created lag feature '%s'.", feature_name)

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create rolling mean/std features for the target and exogenous
        variables.

        Statistical reasoning
        ----------------------
        Rolling statistics summarize *recent trend and volatility*:
            - Rolling mean (`var_roll_mean_w`) acts as a low-pass filter,
              smoothing short-term noise and capturing the local trend
              level over the last `w` observations.
            - Rolling std (`var_roll_std_w`) captures local volatility --
              e.g., a spike in `wind_kph` rolling std may signal an
              incoming storm system, independent of the mean wind level.

        As with lag features, rolling windows are computed strictly on
        *past* values to avoid lookahead bias:
            1. We first shift the series by 1 (`.shift(1)`) so that the
               rolling window for row `t` only includes observations up to
               and including `t-1` -- the row at time `t` itself is
               excluded. Without this shift, the rolling window for row `t`
               would include the value at `t`, meaning the "feature" would
               partially encode the target it's meant to help predict
               (data leakage).
            2. `.rolling(window=w, min_periods=1)` is then applied within
               each location group, again relying on the chronological sort
               performed earlier.

        `min_periods=1` allows rolling statistics to be computed even near
        the start of a location's series (with a smaller effective window),
        rather than producing NaN until `w` observations have accumulated --
        this preserves more usable rows while `create_lag_features` still
        introduces the necessary NaNs for the strict lag features.

        Parameters
        ----------
        df : pd.DataFrame
            Must already be sorted by (location_name, last_updated).
        columns : Optional[List[str]]
            Defaults to `config.exogenous_cols`.
        windows : Optional[List[int]]
            Defaults to `config.rolling_windows` (e.g., [3, 7]).

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        columns = columns or self.config.exogenous_cols
        windows = windows or self.config.rolling_windows
        group_col = self.config.group_col

        for col in columns:
            if col not in df.columns:
                continue

            # Shift by 1 BEFORE rolling so the window for row t covers
            # [t-w, ..., t-1], strictly excluding the current observation.
            shifted = df.groupby(group_col, sort=False)[col].shift(1)

            for window in windows:
                mean_name = f"{col}_roll_mean_{window}"
                std_name = f"{col}_roll_std_{window}"

                # Re-group the shifted series so rolling windows respect
                # location boundaries.
                rolling_obj = shifted.groupby(df[group_col]).rolling(
                    window=window, min_periods=1
                )

                df[mean_name] = rolling_obj.mean().reset_index(level=0, drop=True)
                df[std_name] = rolling_obj.std().reset_index(level=0, drop=True)

                # For windows of size 1 (or the very first observation per
                # group), std is undefined (NaN) -- fill with 0, indicating
                # "no observed volatility yet" rather than missingness.
                df[std_name] = df[std_name].fillna(0.0)

                logger.info(
                    "Created rolling features '%s' and '%s' (window=%d).",
                    mean_name,
                    std_name,
                    window,
                )

        return df

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_pipeline(
        self,
        raw_path: Optional[str] = None,
        save_path: Optional[str] = None,
        drop_outliers: bool = False,
        drop_na_from_lags: bool = True,
    ) -> pd.DataFrame:
        """Execute the full preprocessing & feature-engineering pipeline.

        Steps (in order):
            1. Load raw CSV.
            2. Analyze + impute missing values.
            3. Remove duplicates.
            4. Convert/sort datetime.
            5. Detect outliers (IQR + Isolation Forest) -- flags only by
               default.
            6. Encode categorical features.
            7. Create temporal (calendar/cyclical) features.
            8. Create lag features (target + exogenous).
            9. Create rolling mean/std features (target + exogenous).
           10. Scale numeric features.
           11. (Optional) drop outlier-flagged rows.
           12. (Optional) drop rows with NaNs introduced by lagging.
           13. Persist to `processed_data_path` if `save_path` given.

        Parameters
        ----------
        raw_path : Optional[str]
            Path to raw CSV (defaults to config).
        save_path : Optional[str]
            Path to write the processed CSV (defaults to config). Pass
            `None` explicitly via empty string to skip saving.
        drop_outliers : bool
            If True, rows flagged by *either* IQR or Isolation Forest as
            outliers are removed. Default False -- outliers are flagged but
            retained, since extreme weather events may be genuine and
            informative for forecasting, and downstream models can decide
            how to weight them.
        drop_na_from_lags : bool
            If True (default), drop rows where the longest lag feature is
            NaN (i.e., the first `max(lag_periods)` rows of each location's
            series), since these rows cannot be used for supervised
            training with lag features.

        Returns
        -------
        pd.DataFrame
            The fully processed, feature-engineered dataframe.
        """
        df = self.load_data(raw_path)

        self.analyze_missing_values(df)
        df = self.impute_missing_values(df)

        df = self.remove_duplicates(df)
        df = self.convert_and_sort_datetime(df)

        df = self.detect_outliers_iqr(df)
        df = self.detect_outliers_isolation_forest(df)

        df = self.encode_categorical_features(df)
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.scale_features(df, fit=True)

        if drop_outliers:
            n_before = len(df)
            df = df[~(df["is_outlier_iqr"] | df["is_outlier_isoforest"])]
            logger.info(
                "Dropped %d outlier rows (drop_outliers=True). New shape: %s",
                n_before - len(df),
                df.shape,
            )

        if drop_na_from_lags:
            max_lag = max(self.config.lag_periods)
            lag_cols = [
                f"{c}_lag_{max_lag}"
                for c in self.config.exogenous_cols
                if f"{c}_lag_{max_lag}" in df.columns
            ]
            n_before = len(df)
            df = df.dropna(subset=lag_cols)
            logger.info(
                "Dropped %d rows with insufficient lag history. New shape: %s",
                n_before - len(df),
                df.shape,
            )

        df = df.reset_index(drop=True)

        if save_path != "":
            out_path = save_path or self.config.processed_data_path
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False)
            logger.info("Saved processed dataframe to '%s'.", out_path)

        logger.info("Pipeline complete. Final dataframe shape: %s", df.shape)
        return df


# --------------------------------------------------------------------------- #
# Convenience function for scripting / `main.py`
# --------------------------------------------------------------------------- #
def run_data_prep_pipeline(
    raw_path: str = "data/GlobalWeatherRepository.csv",
    save_path: str = "data/processed_weather_data.csv",
    drop_outliers: bool = False,
) -> Tuple[pd.DataFrame, WeatherDataPreprocessor]:
    """Top-level convenience function: run the full pipeline and return both
    the processed dataframe and the fitted preprocessor (for reuse on
    new/inference data).

    Returns
    -------
    Tuple[pd.DataFrame, WeatherDataPreprocessor]
    """
    config = DataPrepConfig(raw_data_path=raw_path, processed_data_path=save_path)
    preprocessor = WeatherDataPreprocessor(config=config)
    df_processed = preprocessor.run_pipeline(
        raw_path=raw_path, save_path=save_path, drop_outliers=drop_outliers
    )
    return df_processed, preprocessor


if __name__ == "__main__":
    # Allow running this module standalone for quick testing:
    #   python -m src.data_prep
    processed_df, fitted_preprocessor = run_data_prep_pipeline()
    print(processed_df.head())
    print(f"Final shape: {processed_df.shape}")
