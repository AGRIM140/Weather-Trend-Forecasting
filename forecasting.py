"""
forecasting.py
===============

Multivariate Forecasting Engine for the Global Weather Repository dataset.

This module forecasts `temperature_celsius` over a 30-day horizon by
combining:

    1. Traditional univariate time-series models (ARIMA, Prophet) operating
       on the resampled daily target series.
    2. Tabular Machine Learning models (XGBoost, LightGBM, Random Forest,
       Extra Trees) consuming a fully engineered multivariate feature
       matrix that includes lag and rolling-statistic features for the
       target AND key exogenous variables:
           ['humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud',
            'air_quality_PM2.5']
    3. A weighted-average ensemble of all candidate models, with weights
       derived from inverse validation RMSE (better models get higher
       weight).
    4. A recursive multi-step forecasting loop that projects 30 days into
       the future, dynamically updating lag/rolling features at each step
       so the multivariate context evolves realistically.

Design notes
------------
- The raw (hourly-ish) observations are resampled to a **regular daily
  frequency** by taking the daily mean -- both for the target and for every
  exogenous variable -- so that all models operate on a consistent daily
  time index and multivariate context is preserved at every step.
- All time-series train/test splitting is **strictly chronological**
  (no shuffling), eliminating lookahead bias.
- Tabular models are trained on lag/rolling features that are themselves
  built only from *past* observations (via `.shift()` before `.rolling()`),
  consistent with the leakage-safe approach used in `src/data_prep.py`.

Author: Senior Data Science / ML Engineering Team
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Optional heavy dependencies: imported defensively so the module degrades
# gracefully (skipping that model) if a library is unavailable in a given
# environment, rather than crashing the whole pipeline.
# --------------------------------------------------------------------------- #
try:
    import xgboost as xgb

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    _HAS_LIGHTGBM = False

try:
    from statsmodels.tsa.arima.model import ARIMA

    _HAS_ARIMA = True
except ImportError:  # pragma: no cover
    _HAS_ARIMA = False

try:
    from prophet import Prophet

    _HAS_PROPHET = True
except ImportError:  # pragma: no cover
    _HAS_PROPHET = False


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy convergence/seasonality warnings from statsmodels/Prophet
# that would otherwise clutter pipeline logs without being actionable.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ForecastConfig:
    """Centralized configuration for the multivariate forecasting engine."""

    # ---- Core columns ---------------------------------------------------- #
    datetime_col: str = "last_updated"
    group_col: str = "location_name"
    target_col: str = "temperature_celsius"

    # ---- Exogenous variables used for multivariate context ---------------- #
    exogenous_cols: List[str] = field(
        default_factory=lambda: [
            "humidity",
            "wind_kph",
            "pressure_mb",
            "precip_mm",
            "cloud",
            "air_quality_PM2.5",
        ]
    )

    # ---- Lag periods (in days, after daily resampling) -------------------- #
    lag_periods: List[int] = field(default_factory=lambda: [1, 7])

    # ---- Rolling window sizes (in days) ------------------------------------ #
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7])

    # ---- Forecast horizon --------------------------------------------------- #
    forecast_horizon_days: int = 30

    # ---- Train/test split ---------------------------------------------------- #
    # Fraction of the chronologically-ordered daily series reserved for the
    # *test* set (the most recent observations). E.g. 0.2 -> the last 20% of
    # days form the held-out evaluation period.
    test_size_ratio: float = 0.2

    # ---- Random Forest hyperparameters -------------------------------------- #
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = 10
    rf_random_state: int = 42

    # ---- Extra Trees hyperparameters ----------------------------------------- #
    et_n_estimators: int = 300
    et_max_depth: Optional[int] = 10
    et_random_state: int = 42

    # ---- XGBoost hyperparameters ---------------------------------------------- #
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_random_state: int = 42

    # ---- LightGBM hyperparameters ----------------------------------------------- #
    lgb_n_estimators: int = 300
    lgb_max_depth: int = -1  # -1 = no limit (LightGBM default)
    lgb_learning_rate: float = 0.05
    lgb_random_state: int = 42

    # ---- ARIMA order (p, d, q) ---------------------------------------------------- #
    # A modest (2, 1, 2) order is a reasonable default for daily
    # temperature series: d=1 differences out a slow trend, while p=2/q=2
    # capture short-range autocorrelation/moving-average effects.
    arima_order: Tuple[int, int, int] = (2, 1, 2)

    # ---- Small epsilon to avoid division-by-zero in MAPE / inverse-RMSE -------- #
    epsilon: float = 1e-6


# --------------------------------------------------------------------------- #
# Step 1: Resample to a regular daily frequency (multivariate)
# --------------------------------------------------------------------------- #
def resample_to_daily(
    df: pd.DataFrame,
    config: ForecastConfig,
    location_name: Optional[str] = None,
) -> pd.DataFrame:
    """Resample raw (sub-daily) observations to a regular daily frequency.

    For both the target (`temperature_celsius`) and every exogenous
    variable, the **daily mean** is computed, preserving the multivariate
    context at the daily granularity required for forecasting.

    Statistical reasoning
    ----------------------
    The raw data contains multiple readings per day at irregular
    intervals. Forecasting models (ARIMA, Prophet, and the lag/rolling
    feature engineering used by the tabular models) require a *regular*
    time index. The daily mean is a natural, interpretable aggregation that
    smooths out intra-day noise while retaining the day-to-day signal that
    drives a 30-day forecast horizon.

    If `location_name` is provided, the dataframe is first filtered to that
    single location (forecasting is performed per-location, since pooling
    all locations into one global daily mean would average away meaningful
    geographic climate differences). If `location_name` is `None` and the
    dataframe contains multiple locations, a global (all-location) daily
    mean is computed instead -- useful for a "world average" baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `config.datetime_col`, `config.target_col`, and all
        columns in `config.exogenous_cols`. May also contain
        `config.group_col`.
    config : ForecastConfig
    location_name : Optional[str]
        If provided, filter to this location before resampling.

    Returns
    -------
    pd.DataFrame
        A daily-indexed dataframe (DatetimeIndex with freq='D') containing
        the target column and all exogenous columns, with any gaps
        forward-filled (then backward-filled for any leading gaps).
    """
    df = df.copy()
    df[config.datetime_col] = pd.to_datetime(df[config.datetime_col])

    if location_name is not None and config.group_col in df.columns:
        df = df[df[config.group_col] == location_name]
        if df.empty:
            raise ValueError(
                f"No rows found for location_name='{location_name}'."
            )

    cols_to_keep = [config.target_col] + [
        c for c in config.exogenous_cols if c in df.columns
    ]

    daily = (
        df.set_index(config.datetime_col)[cols_to_keep]
        .resample("D")
        .mean()
    )

    # Fill any gap days (e.g., a location with no readings on a given day)
    # using forward-fill (carry the last known daily average forward), then
    # back-fill any remaining leading NaNs (e.g., if the very first day in
    # the index has no data).
    n_missing_before = daily.isnull().sum().sum()
    daily = daily.ffill().bfill()
    n_missing_after = daily.isnull().sum().sum()

    logger.info(
        "Resampled to daily frequency: %d days, %d->%d missing cell(s) "
        "after ffill/bfill (location=%s).",
        len(daily),
        n_missing_before,
        n_missing_after,
        location_name or "ALL",
    )

    return daily


# --------------------------------------------------------------------------- #
# Step 2: Multivariate feature engineering (lags + rolling stats)
# --------------------------------------------------------------------------- #
def build_multivariate_features(
    daily_df: pd.DataFrame, config: ForecastConfig
) -> pd.DataFrame:
    """Create lag and rolling-statistic features for the target AND all
    exogenous variables, on a daily-resampled dataframe.

    Statistical reasoning
    ----------------------
    - **Lag features** (`<col>_lag_k`): the value of `<col>` exactly `k`
      days ago. `lag_1` captures the immediate day-to-day persistence
      typical of temperature series (today's temperature is highly
      correlated with yesterday's); `lag_7` captures a weekly-cycle echo
      (useful when weather patterns have a roughly weekly periodicity due
      to synoptic-scale weather systems).
    - **Rolling mean** (`<col>_roll_mean_w`): a smoothed local trend level
      over the last `w` days, computed on the series *shifted by 1* so the
      window for day `t` covers days `[t-w, ..., t-1]` -- the value at day
      `t` itself is never included, preventing leakage.
    - **Rolling std** (`<col>_roll_std_w`): local volatility over the same
      shifted window; a rising rolling std in `pressure_mb` or `wind_kph`
      often precedes a shift in temperature.

    These features are generated for `config.target_col` and every column
    in `config.exogenous_cols`, giving the tabular ML models a rich
    multivariate, leakage-safe feature matrix.

    Calendar features (day-of-year, month, cyclical encodings) are also
    added, since temperature has strong annual seasonality.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of `resample_to_daily` -- a daily-indexed dataframe
        containing the target and exogenous columns.
    config : ForecastConfig

    Returns
    -------
    pd.DataFrame
        `daily_df` with additional lag/rolling/calendar feature columns.
        Rows at the start of the series with insufficient history for the
        largest lag will contain NaNs in the lag columns (handled by the
        caller, typically via `dropna`).
    """
    df = daily_df.copy()

    feature_cols = [config.target_col] + [
        c for c in config.exogenous_cols if c in df.columns
    ]

    # --- Lag features --- #
    for col in feature_cols:
        for lag in config.lag_periods:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # --- Rolling mean/std (leakage-safe: shift(1) before rolling) --- #
    for col in feature_cols:
        shifted = df[col].shift(1)
        for window in config.rolling_windows:
            roll = shifted.rolling(window=window, min_periods=1)
            df[f"{col}_roll_mean_{window}"] = roll.mean()
            df[f"{col}_roll_std_{window}"] = roll.std().fillna(0.0)

    # --- Calendar / seasonality features --- #
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info(
        "Built multivariate feature matrix: %d rows x %d columns "
        "(before dropna).",
        df.shape[0],
        df.shape[1],
    )
    return df


# --------------------------------------------------------------------------- #
# Step 3: Chronological train/test split
# --------------------------------------------------------------------------- #
def chronological_train_test_split(
    df: pd.DataFrame, config: ForecastConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed dataframe into train/test sets *chronologically*.

    Statistical reasoning
    ----------------------
    Shuffling time-series data before splitting (as in a standard
    `train_test_split`) would allow the model to "see the future" relative
    to test points during training (lookahead bias), producing
    overly-optimistic validation metrics that do not generalize to real
    forecasting. Instead, the **most recent** `test_size_ratio` fraction of
    days is held out as the test set, and everything before it is used for
    training -- mirroring how the model would actually be deployed
    (trained on history, evaluated on the immediate future).

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed (DatetimeIndex) dataframe, sorted ascending by date.
    config : ForecastConfig

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    df = df.sort_index()
    n = len(df)
    n_test = max(1, int(np.ceil(n * config.test_size_ratio)))
    n_train = n - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough data ({n} rows) for the requested "
            f"test_size_ratio={config.test_size_ratio}."
        )

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    logger.info(
        "Chronological split: %d train days (%s -> %s), %d test days "
        "(%s -> %s).",
        len(train_df),
        train_df.index.min().date(),
        train_df.index.max().date(),
        len(test_df),
        test_df.index.min().date(),
        test_df.index.max().date(),
    )
    return train_df, test_df


# --------------------------------------------------------------------------- #
# Step 4: Evaluation metrics
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6
) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R^2 for a set of predictions.

    Statistical reasoning
    ----------------------
    - **MAE** (Mean Absolute Error): average absolute deviation, in the
      same units as the target (degrees Celsius) -- easy to interpret.
    - **RMSE** (Root Mean Squared Error): like MAE but penalizes large
      errors more heavily (squared term), useful for detecting models that
      occasionally make large mistakes.
    - **MAPE** (Mean Absolute Percentage Error): error as a percentage of
      the true value -- scale-independent, but unstable when `y_true` is
      near zero. A small `epsilon` is added to the denominator to avoid
      division-by-zero (temperature in Celsius can legitimately be 0 or
      negative, which is why MAPE is reported alongside, not instead of,
      MAE/RMSE).
    - **R^2** (Coefficient of Determination): fraction of variance in
      `y_true` explained by `y_pred`, relative to a naive "predict the
      mean" baseline. R^2 = 1 is a perfect fit; R^2 = 0 matches the
      mean-baseline; R^2 < 0 is worse than the mean-baseline.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    epsilon : float
        Small constant added to `|y_true|` in the MAPE denominator.

    Returns
    -------
    Dict[str, float]
        {"MAE": ..., "RMSE": ..., "MAPE": ..., "R2": ...}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    )
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# --------------------------------------------------------------------------- #
# Step 5: Model wrappers
# --------------------------------------------------------------------------- #
class ARIMAModel:
    """Thin wrapper around `statsmodels.tsa.arima.model.ARIMA`.

    ARIMA operates **univariately** on the target series only (it does not
    consume the exogenous feature matrix), serving as a classical
    statistical baseline.
    """

    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)) -> None:
        self.order = order
        self.fitted_model = None
        self._history: Optional[pd.Series] = None

    def fit(self, y_train: pd.Series) -> "ARIMAModel":
        if not _HAS_ARIMA:
            raise ImportError(
                "statsmodels is required for ARIMAModel but is not installed."
            )
        self._history = y_train.copy()
        model = ARIMA(y_train.values, order=self.order)
        self.fitted_model = model.fit()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.fitted_model is None:
            raise RuntimeError("ARIMAModel.fit() must be called before predict().")
        forecast = self.fitted_model.forecast(steps=n_periods)
        return np.asarray(forecast)


class ProphetModel:
    """Thin wrapper around Facebook/Meta's `Prophet` univariate forecaster.

    Like ARIMA, Prophet is fit purely on the target series (with its own
    internal trend + seasonality decomposition), serving as a second
    classical baseline that explicitly models yearly/weekly seasonality.
    """

    def __init__(self) -> None:
        self.model = None

    def fit(self, y_train: pd.Series) -> "ProphetModel":
        if not _HAS_PROPHET:
            raise ImportError("prophet is required for ProphetModel but is not installed.")

        prophet_df = pd.DataFrame(
            {"ds": y_train.index, "y": y_train.values}
        )
        # Daily data spanning ~1-2 years: enable yearly seasonality if
        # enough history exists, always enable weekly seasonality.
        yearly_seasonality = len(y_train) >= 365
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        # Suppress Prophet's verbose stan/cmdstanpy logging.
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        self.model.fit(prophet_df)
        return self

    def predict(self, n_periods: int, last_date: pd.Timestamp) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ProphetModel.fit() must be called before predict().")

        future = pd.DataFrame(
            {
                "ds": pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=n_periods,
                    freq="D",
                )
            }
        )
        forecast = self.model.predict(future)
        return forecast["yhat"].values


# --------------------------------------------------------------------------- #
# Step 6: Tabular model factory
# --------------------------------------------------------------------------- #
def build_tabular_models(config: ForecastConfig) -> Dict[str, object]:
    """Instantiate all tabular ML regressors with hyperparameters from
    `config`.

    Returns
    -------
    Dict[str, object]
        Mapping from model name -> unfitted scikit-learn-compatible
        regressor. XGBoost/LightGBM are included only if the corresponding
        package is installed.
    """
    models: Dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            random_state=config.rf_random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=config.et_n_estimators,
            max_depth=config.et_max_depth,
            random_state=config.et_random_state,
            n_jobs=-1,
        ),
    }

    if _HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.xgb_random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
        )
    else:  # pragma: no cover
        logger.warning("xgboost not installed; skipping XGBoost model.")

    if _HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=config.lgb_n_estimators,
            max_depth=config.lgb_max_depth,
            learning_rate=config.lgb_learning_rate,
            random_state=config.lgb_random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    else:  # pragma: no cover
        logger.warning("lightgbm not installed; skipping LightGBM model.")

    return models


# --------------------------------------------------------------------------- #
# Step 7: Main forecasting engine
# --------------------------------------------------------------------------- #
class MultivariateForecastingEngine:
    """End-to-end multivariate forecasting engine for `temperature_celsius`.

    Workflow
    --------
    1. `prepare_data`: resample raw data to daily frequency and build the
       leakage-safe multivariate feature matrix.
    2. `fit`: chronologically split into train/test, fit ARIMA, Prophet,
       and all tabular ML models, evaluate each on the test set, and build
       an inverse-RMSE-weighted ensemble.
    3. `forecast_future`: recursively forecast `config.forecast_horizon_days`
       days beyond the end of the available data, dynamically updating
       lag/rolling features at each step for the tabular models.
    """

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        self.config = config or ForecastConfig()

        # Populated by `prepare_data`
        self.daily_df_: Optional[pd.DataFrame] = None
        self.feature_df_: Optional[pd.DataFrame] = None
        self.feature_cols_: List[str] = []

        # Populated by `fit`
        self.train_df_: Optional[pd.DataFrame] = None
        self.test_df_: Optional[pd.DataFrame] = None
        self.fitted_models_: Dict[str, object] = {}
        self.test_predictions_: Dict[str, np.ndarray] = {}
        self.metrics_: Dict[str, Dict[str, float]] = {}
        self.ensemble_weights_: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def prepare_data(
        self, df: pd.DataFrame, location_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Resample to daily frequency, engineer multivariate features, and
        drop rows with insufficient lag history.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or lightly-processed dataframe with the original schema
            (must include `last_updated`, `temperature_celsius`, and the
            configured exogenous columns; optionally `location_name`).
        location_name : Optional[str]
            If provided, forecast for this single location only.

        Returns
        -------
        pd.DataFrame
            The leakage-safe daily feature matrix (NaN rows from lagging
            dropped), stored as `self.feature_df_`.
        """
        self.daily_df_ = resample_to_daily(df, self.config, location_name)
        feature_df = build_multivariate_features(self.daily_df_, self.config)

        # Drop rows where the longest-lag feature is still NaN (i.e., the
        # first `max(lag_periods)` days of the series).
        max_lag = max(self.config.lag_periods)
        ref_lag_col = f"{self.config.target_col}_lag_{max_lag}"
        n_before = len(feature_df)
        feature_df = feature_df.dropna(subset=[ref_lag_col])
        logger.info(
            "Dropped %d rows with insufficient lag history (%d -> %d rows).",
            n_before - len(feature_df),
            n_before,
            len(feature_df),
        )

        # All columns except the target itself are candidate features for
        # the tabular models.
        self.feature_cols_ = [
            c for c in feature_df.columns if c != self.config.target_col
        ]
        self.feature_df_ = feature_df
        return feature_df

    # ------------------------------------------------------------------ #
    # Fitting + evaluation
    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, Dict[str, float]]:
        """Fit ARIMA, Prophet, and all tabular models; evaluate each on a
        chronological test split; build the ensemble.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping from model name -> {"MAE", "RMSE", "MAPE", "R2"}.
        """
        if self.feature_df_ is None:
            raise RuntimeError("Call prepare_data() before fit().")

        train_df, test_df = chronological_train_test_split(
            self.feature_df_, self.config
        )
        self.train_df_ = train_df
        self.test_df_ = test_df

        target = self.config.target_col
        y_train = train_df[target]
        y_test = test_df[target]
        X_train = train_df[self.feature_cols_]
        X_test = test_df[self.feature_cols_]

        # ------------------------------------------------------------ #
        # 1. ARIMA (univariate)
        # ------------------------------------------------------------ #
        if _HAS_ARIMA:
            try:
                arima = ARIMAModel(order=self.config.arima_order)
                arima.fit(y_train)
                arima_preds = arima.predict(n_periods=len(y_test))
                self.fitted_models_["ARIMA"] = arima
                self.test_predictions_["ARIMA"] = arima_preds
                self.metrics_["ARIMA"] = evaluate_predictions(
                    y_test.values, arima_preds, self.config.epsilon
                )
                logger.info("ARIMA fit complete. Metrics: %s", self.metrics_["ARIMA"])
            except Exception as exc:  # pragma: no cover
                logger.warning("ARIMA failed to fit: %s", exc)
        else:  # pragma: no cover
            logger.warning("statsmodels not installed; skipping ARIMA.")

        # ------------------------------------------------------------ #
        # 2. Prophet (univariate)
        # ------------------------------------------------------------ #
        if _HAS_PROPHET:
            try:
                prophet = ProphetModel()
                prophet.fit(y_train)
                prophet_preds = prophet.predict(
                    n_periods=len(y_test), last_date=y_train.index[-1]
                )
                self.fitted_models_["Prophet"] = prophet
                self.test_predictions_["Prophet"] = prophet_preds
                self.metrics_["Prophet"] = evaluate_predictions(
                    y_test.values, prophet_preds, self.config.epsilon
                )
                logger.info(
                    "Prophet fit complete. Metrics: %s", self.metrics_["Prophet"]
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Prophet failed to fit: %s", exc)
        else:  # pragma: no cover
            logger.warning("prophet not installed; skipping Prophet.")

        # ------------------------------------------------------------ #
        # 3. Tabular ML models (multivariate)
        # ------------------------------------------------------------ #
        tabular_models = build_tabular_models(self.config)
        for name, model in tabular_models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                self.fitted_models_[name] = model
                self.test_predictions_[name] = preds
                self.metrics_[name] = evaluate_predictions(
                    y_test.values, preds, self.config.epsilon
                )
                logger.info("%s fit complete. Metrics: %s", name, self.metrics_[name])
            except Exception as exc:  # pragma: no cover
                logger.warning("%s failed to fit: %s", name, exc)

        # ------------------------------------------------------------ #
        # 4. Ensemble: inverse-RMSE weighted average
        # ------------------------------------------------------------ #
        self._build_ensemble(y_test)

        return self.metrics_

    def _build_ensemble(self, y_test: pd.Series) -> None:
        """Construct a weighted-average ensemble from all successfully
        fitted models, with weights proportional to inverse RMSE.

        Statistical reasoning
        ----------------------
        A simple (equal-weight) average treats every model as equally
        trustworthy, even if one model is far more accurate than another.
        Weighting by **inverse RMSE** (1 / RMSE, normalized to sum to 1)
        gives more influence to models with lower test-set error, while
        still allowing weaker models to contribute (a soft form of
        stacking without requiring a second-level meta-model). Models that
        failed to fit are simply excluded from the weight computation.
        """
        if not self.test_predictions_:
            logger.warning("No models were successfully fitted; skipping ensemble.")
            return

        rmses = {
            name: self.metrics_[name]["RMSE"]
            for name in self.test_predictions_
            if name in self.metrics_
        }

        inverse_rmses = {
            name: 1.0 / (rmse + self.config.epsilon) for name, rmse in rmses.items()
        }
        total = sum(inverse_rmses.values())
        self.ensemble_weights_ = {
            name: weight / total for name, weight in inverse_rmses.items()
        }

        ensemble_preds = np.zeros(len(y_test))
        for name, weight in self.ensemble_weights_.items():
            ensemble_preds += weight * np.asarray(self.test_predictions_[name])

        self.test_predictions_["Ensemble"] = ensemble_preds
        self.metrics_["Ensemble"] = evaluate_predictions(
            y_test.values, ensemble_preds, self.config.epsilon
        )

        logger.info("Ensemble weights (inverse-RMSE normalized): %s", self.ensemble_weights_)
        logger.info("Ensemble metrics: %s", self.metrics_["Ensemble"])

    # ------------------------------------------------------------------ #
    # Recursive multi-step future forecasting
    # ------------------------------------------------------------------ #
    def forecast_future(
        self, model_name: str = "Ensemble"
    ) -> pd.DataFrame:
        """Recursively forecast `config.forecast_horizon_days` days beyond
        the end of the available data.

        Statistical reasoning / methodology
        -------------------------------------
        Multi-step forecasting with lag features faces a fundamental
        challenge: at forecast step `t+1`, the `lag_1` feature requires the
        *actual* value at step `t` -- but for `t > 0` (i.e., beyond the
        last observed day), that value doesn't exist yet. The standard
        solution is a **recursive (iterative) forecasting loop**:

            1. Start with the last `max(lag_periods, rolling_windows)` days
               of *observed* history for the target and every exogenous
               variable.
            2. For exogenous variables, since we have no future
               observations, we **persist** them using their own most
               recent rolling mean (a "no-change-from-recent-normal"
               assumption -- a common, defensible baseline for variables
               that are not the primary forecast target).
            3. At each step, recompute lag and rolling features from the
               (growing) history buffer -- which now includes previously
               *forecasted* target values for steps already produced.
            4. Feed the resulting feature vector to the selected fitted
               model (default: the "Ensemble") to predict the next day's
               `temperature_celsius`.
            5. Append the new prediction (and the persisted exogenous
               values) to the history buffer and repeat until
               `forecast_horizon_days` predictions have been produced.

        For ARIMA and Prophet (pure univariate models that were already fit
        with their own internal `.predict(n_periods=...)` interfaces), this
        function instead directly calls their multi-step `predict` methods,
        since those models do not use the lag/rolling feature matrix.

        Parameters
        ----------
        model_name : str
            Which fitted model to use for forecasting. Use "Ensemble"
            (default) for the weighted ensemble, or the name of any
            individual fitted model (e.g., "XGBoost", "ARIMA", "Prophet").

        Returns
        -------
        pd.DataFrame
            Indexed by future date (DatetimeIndex, `freq='D'`), with a
            single column `"forecast_temperature_celsius"` containing the
            30-day-ahead (or `config.forecast_horizon_days`-ahead)
            predictions.
        """
        if self.feature_df_ is None or self.daily_df_ is None:
            raise RuntimeError("Call prepare_data() and fit() before forecast_future().")

        horizon = self.config.forecast_horizon_days
        last_date = self.daily_df_.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )

        # ------------------------------------------------------------ #
        # Pure univariate models: ARIMA / Prophet have native multi-step
        # predict() methods and don't need the recursive feature loop.
        # ------------------------------------------------------------ #
        if model_name == "ARIMA":
            if "ARIMA" not in self.fitted_models_:
                raise RuntimeError("ARIMA model was not fitted successfully.")
            # Refit on the FULL daily target series (train+test) so the
            # future forecast starts exactly at `last_date`.
            full_arima = ARIMAModel(order=self.config.arima_order)
            full_arima.fit(self.daily_df_[self.config.target_col])
            preds = full_arima.predict(n_periods=horizon)
            return pd.DataFrame(
                {"forecast_temperature_celsius": preds}, index=future_dates
            )

        if model_name == "Prophet":
            if "Prophet" not in self.fitted_models_:
                raise RuntimeError("Prophet model was not fitted successfully.")
            full_prophet = ProphetModel()
            full_prophet.fit(self.daily_df_[self.config.target_col])
            preds = full_prophet.predict(n_periods=horizon, last_date=last_date)
            return pd.DataFrame(
                {"forecast_temperature_celsius": preds}, index=future_dates
            )

        # ------------------------------------------------------------ #
        # Tabular models (and Ensemble): recursive feature-based loop.
        # ------------------------------------------------------------ #
        if model_name == "Ensemble":
            if not self.ensemble_weights_:
                raise RuntimeError("Ensemble was not built; call fit() first.")
            available = set(self.ensemble_weights_.keys())
            # Only tabular models (with predict(X)) participate in the
            # recursive loop; ARIMA/Prophet predictions for the *future*
            # horizon are computed separately and blended in afterward
            # using their own ensemble weight (applied to their univariate
            # multi-step forecast).
            tabular_names = [
                n for n in available if n in self.fitted_models_ and hasattr(
                    self.fitted_models_[n], "predict"
                ) and n not in ("ARIMA", "Prophet")
            ]
        elif model_name in self.fitted_models_:
            tabular_names = [model_name]
        else:
            raise ValueError(f"Unknown or unfitted model_name: '{model_name}'")

        # --- Build the rolling history buffer from the full daily series --- #
        # We need enough trailing history to compute the largest lag/rolling
        # window at every future step.
        target_col = self.config.target_col
        exo_cols = [c for c in self.config.exogenous_cols if c in self.daily_df_.columns]
        max_window = max(max(self.config.lag_periods), max(self.config.rolling_windows))

        history = self.daily_df_[[target_col] + exo_cols].copy()

        # Pre-compute each exogenous variable's "persistence value" -- the
        # mean of its last `max(rolling_windows)` observed days. Future
        # exogenous values are held at this level for the entire horizon
        # (a neutral "recent normal" assumption in the absence of their own
        # forecasts).
        persistence_window = max(self.config.rolling_windows)
        exo_persistence = {
            col: history[col].iloc[-persistence_window:].mean() for col in exo_cols
        }

        future_predictions: List[float] = []

        for step, future_date in enumerate(future_dates):
            # Append a new row to the history buffer for `future_date`.
            # Exogenous columns: persisted value. Target column: placeholder
            # (NaN) -- will be filled in immediately after prediction.
            new_row = {col: exo_persistence[col] for col in exo_cols}
            new_row[target_col] = np.nan
            history.loc[future_date] = new_row

            # Build features for this single new row using the SAME
            # leakage-safe lag/rolling logic as build_multivariate_features,
            # but only need the last row's feature values.
            feature_row = self._build_single_step_features(history, future_date)

            # Predict with each tabular model in `tabular_names`, combining
            # via ensemble weights if applicable.
            if model_name == "Ensemble":
                pred = 0.0
                tabular_weight_sum = 0.0
                for name in tabular_names:
                    model = self.fitted_models_[name]
                    X_row = feature_row[self.feature_cols_].values.reshape(1, -1)
                    model_pred = float(model.predict(X_row)[0])
                    weight = self.ensemble_weights_.get(name, 0.0)
                    pred += weight * model_pred
                    tabular_weight_sum += weight

                # Blend in ARIMA/Prophet's own multi-step forecast for this
                # step, weighted by their ensemble weight (computed once,
                # cached on first iteration).
                for uni_name in ("ARIMA", "Prophet"):
                    if uni_name in self.ensemble_weights_:
                        if step == 0:
                            # Compute the full univariate horizon forecast once.
                            uni_forecast_df = self.forecast_future(model_name=uni_name)
                            setattr(
                                self,
                                f"_cached_{uni_name.lower()}_forecast",
                                uni_forecast_df["forecast_temperature_celsius"].values,
                            )
                        cached = getattr(self, f"_cached_{uni_name.lower()}_forecast")
                        pred += self.ensemble_weights_[uni_name] * cached[step]
                        tabular_weight_sum += self.ensemble_weights_[uni_name]

                # Re-normalize in case some models were unavailable for the
                # recursive loop (defensive; normally weights already sum to 1).
                if tabular_weight_sum > 0:
                    pred /= tabular_weight_sum
            else:
                model = self.fitted_models_[tabular_names[0]]
                X_row = feature_row[self.feature_cols_].values.reshape(1, -1)
                pred = float(model.predict(X_row)[0])

            future_predictions.append(pred)

            # Fill in the predicted target value for this future date so
            # subsequent steps' lag/rolling features see it as "history".
            history.loc[future_date, target_col] = pred

        # Clean up any cached single-use attributes from the ensemble blend.
        for uni_name in ("arima", "prophet"):
            attr = f"_cached_{uni_name}_forecast"
            if hasattr(self, attr):
                delattr(self, attr)

        return pd.DataFrame(
            {"forecast_temperature_celsius": future_predictions}, index=future_dates
        )

    def _build_single_step_features(
        self, history: pd.DataFrame, target_date: pd.Timestamp
    ) -> pd.Series:
        """Compute the lag/rolling/calendar feature vector for a single
        `target_date`, given the (growing) `history` dataframe.

        This mirrors `build_multivariate_features` but is optimized to
        compute only the features needed for the single most recent row
        (`target_date`), which is what the recursive forecasting loop
        requires at each step.

        Parameters
        ----------
        history : pd.DataFrame
            DatetimeIndex dataframe containing the target column and
            exogenous columns, with `target_date` as its last row (target
            value may be NaN at this point -- it is not used to compute
            lag/rolling features for `target_date` itself, only for *past*
            rows, consistent with the leakage-safe design).
        target_date : pd.Timestamp
            The date for which to compute features.

        Returns
        -------
        pd.Series
            Feature values for `target_date`, indexed by feature name
            (matching `self.feature_cols_` plus the target column).
        """
        config = self.config
        target_col = config.target_col
        exo_cols = [c for c in config.exogenous_cols if c in history.columns]
        feature_cols_for_loop = [target_col] + exo_cols

        row: Dict[str, float] = {}

        # --- Lag features: value `lag` rows before target_date --- #
        target_idx = history.index.get_loc(target_date)
        for col in feature_cols_for_loop:
            for lag in config.lag_periods:
                src_idx = target_idx - lag
                row[f"{col}_lag_{lag}"] = (
                    history[col].iloc[src_idx] if src_idx >= 0 else np.nan
                )

        # --- Rolling mean/std: window of `window` rows ending at
        # target_idx - 1 (i.e., shift(1) then rolling, matching training) --- #
        for col in feature_cols_for_loop:
            for window in config.rolling_windows:
                start_idx = max(0, target_idx - window)
                end_idx = target_idx  # exclusive of target_idx itself
                window_vals = history[col].iloc[start_idx:end_idx]
                row[f"{col}_roll_mean_{window}"] = (
                    window_vals.mean() if len(window_vals) > 0 else np.nan
                )
                row[f"{col}_roll_std_{window}"] = (
                    window_vals.std() if len(window_vals) > 1 else 0.0
                )

        # --- Calendar / cyclical features --- #
        row["day_of_year"] = target_date.dayofyear
        row["month"] = target_date.month
        row["day_of_week"] = target_date.dayofweek
        row["day_of_year_sin"] = np.sin(2 * np.pi * row["day_of_year"] / 365.25)
        row["day_of_year_cos"] = np.cos(2 * np.pi * row["day_of_year"] / 365.25)
        row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
        row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)

        # --- Raw exogenous values for target_date (persisted values) are
        # also part of the feature row in `build_multivariate_features`
        # (since `feature_cols_` includes raw exogenous columns, not just
        # their lag/rolling derivatives). --- #
        for col in exo_cols:
            row[col] = history[col].iloc[target_idx]

        return pd.Series(row)


# --------------------------------------------------------------------------- #
# Convenience top-level function
# --------------------------------------------------------------------------- #
def run_forecasting_pipeline(
    df: pd.DataFrame,
    location_name: Optional[str] = None,
    config: Optional[ForecastConfig] = None,
) -> Tuple[MultivariateForecastingEngine, Dict[str, Dict[str, float]], pd.DataFrame]:
    """Run the full forecasting pipeline for a single location (or the
    global average if `location_name` is None): prepare data, fit all
    models, and produce the 30-day-ahead ensemble forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the original Global Weather Repository schema
        (must include `last_updated`, `temperature_celsius`, the
        configured exogenous columns, and optionally `location_name`).
    location_name : Optional[str]
        Location to forecast for. If None, a global daily average across
        all locations is used.
    config : Optional[ForecastConfig]

    Returns
    -------
    Tuple[MultivariateForecastingEngine, Dict[str, Dict[str, float]], pd.DataFrame]
        (fitted engine, test-set metrics per model, 30-day future forecast)
    """
    config = config or ForecastConfig()
    engine = MultivariateForecastingEngine(config=config)
    engine.prepare_data(df, location_name=location_name)
    metrics = engine.fit()
    future_forecast = engine.forecast_future(model_name="Ensemble")
    return engine, metrics, future_forecast


if __name__ == "__main__":
    import os

    data_path = "data/processed_weather_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at '{data_path}'. Run src/data_prep.py first."
        )

    weather_df = pd.read_csv(data_path)
    weather_df["last_updated"] = pd.to_datetime(weather_df["last_updated"])

    # Demonstrate the pipeline on a single, data-rich location.
    sample_location = weather_df["location_name"].value_counts().idxmax()
    logger.info("Running forecasting pipeline for location: %s", sample_location)

    fc_config = ForecastConfig()
    engine, metrics, forecast_df = run_forecasting_pipeline(
        weather_df, location_name=sample_location, config=fc_config
    )

    print("\n=== Test-set Evaluation Metrics ===")
    for model_name, m in metrics.items():
        print(
            f"{model_name:>15s} | MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
            f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.3f}"
        )

    print(f"\n=== {fc_config.forecast_horizon_days}-Day Forecast for {sample_location} ===")
    print(forecast_df)
