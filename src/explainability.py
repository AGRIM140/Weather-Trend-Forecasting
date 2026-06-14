"""
explainability.py
==================

Model interpretability and explainability module for the Global Weather
Repository multivariate temperature forecasting engine.

This module ingests a fitted tree-based regressor (e.g., the XGBoost,
LightGBM, RandomForest, or ExtraTrees models produced by
`src.forecasting.MultivariateForecastingEngine`) together with its
multivariate train/test feature matrices, and provides:

    1. SHAP (SHapley Additive exPlanations) analysis via
       `shap.TreeExplainer`, including:
         - Global summary / beeswarm plots (overall feature impact).
         - Dependence plots showing how SHAP values for one feature vary
           with the value of another (e.g., temperature lag vs. humidity).
    2. Permutation importance via
       `sklearn.inspection.permutation_importance`, an alternate,
       model-agnostic perspective on feature contributions computed by
       measuring the drop in test-set performance when a feature's values
       are randomly shuffled.
    3. A unified, ranked `pd.DataFrame` combining SHAP-based and
       permutation-based importance metrics, suitable for direct
       consumption by a Streamlit dashboard.

All plots are saved as image files (PNG via Matplotlib for SHAP plots,
which is SHAP's native rendering backend) to a configurable output
directory (default: `images/`).

Author: Senior Data Science / ML Engineering Team
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib

# Use a non-interactive backend so this module runs cleanly in headless
# environments (servers, CI, notebooks without a display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExplainabilityConfig:
    """Configuration for the explainability module."""

    # Directory where SHAP / importance plots are saved.
    output_dir: str = "images"

    # Number of background samples used by SHAP's TreeExplainer when the
    # test set is large (keeps SHAP computation tractable on big feature
    # matrices). `None` uses the full test set.
    shap_sample_size: Optional[int] = 500

    # Number of repeats for permutation importance (more repeats -> more
    # stable estimates, at the cost of compute time).
    permutation_n_repeats: int = 10

    # Random state for reproducibility (sampling, permutation shuffles).
    random_state: int = 42

    # Number of top features to display in summary/bar plots.
    top_n_features: int = 15


# --------------------------------------------------------------------------- #
# SHAP analysis
# --------------------------------------------------------------------------- #
class ShapAnalyzer:
    """Wraps `shap.TreeExplainer` for tree-based regressors (XGBoost,
    LightGBM, RandomForest, ExtraTrees) and produces global summary and
    dependence plots.

    Parameters
    ----------
    model : object
        A fitted tree-based regressor exposing a `.predict()` method and
        compatible with `shap.TreeExplainer` (e.g., XGBRegressor,
        LGBMRegressor, RandomForestRegressor, ExtraTreesRegressor).
    feature_names : List[str]
        Names of the columns in the feature matrix, in the same order used
        for training/prediction.
    config : Optional[ExplainabilityConfig]
    """

    def __init__(
        self,
        model: object,
        feature_names: List[str],
        config: Optional[ExplainabilityConfig] = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.config = config or ExplainabilityConfig()

        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.X_explained_: Optional[pd.DataFrame] = None

    def fit_explainer(self, X_test: pd.DataFrame) -> "ShapAnalyzer":
        """Initialize a `shap.TreeExplainer` on `self.model` and compute
        SHAP values for (a subsample of) `X_test`.

        Statistical reasoning
        ----------------------
        `shap.TreeExplainer` computes *exact* Shapley values for tree
        ensembles in polynomial time by exploiting the tree structure
        (the "Tree SHAP" algorithm), rather than the exponential-time exact
        computation required for arbitrary models. Each SHAP value
        represents the contribution of a single feature to the difference
        between a single prediction and the model's average prediction
        (the expected value over the background dataset) -- SHAP values
        for a given row sum (approximately) to
        `prediction - expected_value`.

        To keep computation tractable on large test sets, we optionally
        subsample `X_test` down to `config.shap_sample_size` rows (using a
        fixed `random_state` for reproducibility). SHAP values computed on
        a representative random subsample still give a statistically valid
        picture of *global* feature importance (the summary/beeswarm plots
        aggregate over many rows), while dependence plots benefit from
        having enough points to reveal the underlying relationship.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix with columns matching `self.feature_names`.

        Returns
        -------
        ShapAnalyzer
            `self`, with `self.explainer`, `self.shap_values_`, and
            `self.X_explained_` populated.
        """
        X = X_test[self.feature_names].copy()

        sample_size = self.config.shap_sample_size
        if sample_size is not None and len(X) > sample_size:
            X = X.sample(n=sample_size, random_state=self.config.random_state)
            logger.info(
                "Subsampled test set from %d to %d rows for SHAP computation.",
                len(X_test),
                len(X),
            )

        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values_ = self.explainer.shap_values(X)
        self.X_explained_ = X

        logger.info(
            "Computed SHAP values: shape=%s for %d features.",
            np.asarray(self.shap_values_).shape,
            len(self.feature_names),
        )
        return self

    # ------------------------------------------------------------------ #
    # Global summary / beeswarm plot
    # ------------------------------------------------------------------ #
    def plot_summary(self, save_path: Optional[str] = None) -> str:
        """Generate and save a SHAP beeswarm summary plot showing the
        global impact of every feature across all explained samples.

        Each point represents one (sample, feature) SHAP value: its
        horizontal position is the SHAP value (impact on the prediction,
        in degrees Celsius), its color encodes the feature's actual value
        (low=blue, high=red), and features are sorted top-to-bottom by mean
        absolute SHAP value (overall importance).

        Parameters
        ----------
        save_path : Optional[str]
            Output PNG path. Defaults to
            `<config.output_dir>/shap_summary_beeswarm.png`.

        Returns
        -------
        str
            The path the plot was saved to.
        """
        self._ensure_fitted()
        save_path = save_path or os.path.join(
            self.config.output_dir, "shap_summary_beeswarm.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure()
        shap.summary_plot(
            self.shap_values_,
            self.X_explained_,
            feature_names=self.feature_names,
            max_display=self.config.top_n_features,
            show=False,
        )
        plt.title("SHAP Summary (Beeswarm) Plot: Global Feature Impact")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Saved SHAP summary/beeswarm plot to '%s'.", save_path)
        return save_path

    def plot_bar_importance(self, save_path: Optional[str] = None) -> str:
        """Generate and save a SHAP bar plot of mean |SHAP value| per
        feature -- a compact global-importance ranking.

        Parameters
        ----------
        save_path : Optional[str]
            Output PNG path. Defaults to
            `<config.output_dir>/shap_bar_importance.png`.

        Returns
        -------
        str
        """
        self._ensure_fitted()
        save_path = save_path or os.path.join(
            self.config.output_dir, "shap_bar_importance.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure()
        shap.summary_plot(
            self.shap_values_,
            self.X_explained_,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=self.config.top_n_features,
            show=False,
        )
        plt.title("SHAP Mean |Impact| per Feature")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Saved SHAP bar importance plot to '%s'.", save_path)
        return save_path

    # ------------------------------------------------------------------ #
    # Dependence plots
    # ------------------------------------------------------------------ #
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate and save a SHAP dependence plot for `feature`.

        A dependence plot shows, for every explained sample, the value of
        `feature` (x-axis) against its SHAP value (y-axis, the impact on
        the predicted temperature). Points are colored by
        `interaction_feature` (if provided, or auto-selected by SHAP if
        `None`), revealing how the effect of `feature` on the prediction
        *interacts* with another variable.

        Example
        -------
        `plot_dependence("temperature_celsius_lag_1", "humidity")` shows how
        the impact of yesterday's temperature on today's predicted
        temperature varies depending on the humidity level -- e.g., the
        lag-1 effect might be stronger (steeper slope) under low-humidity
        conditions.

        Parameters
        ----------
        feature : str
            The primary feature to plot on the x-axis. Must be present in
            `self.feature_names`.
        interaction_feature : Optional[str]
            The feature used for point coloring. If `None`, SHAP
            auto-selects the feature with the strongest interaction.
        save_path : Optional[str]
            Output PNG path. Defaults to
            `<config.output_dir>/shap_dependence_<feature>.png`.

        Returns
        -------
        str
        """
        self._ensure_fitted()
        if feature not in self.feature_names:
            raise ValueError(
                f"Feature '{feature}' not found in feature_names: "
                f"{self.feature_names}"
            )
        if interaction_feature is not None and interaction_feature not in self.feature_names:
            raise ValueError(
                f"Interaction feature '{interaction_feature}' not found in "
                f"feature_names: {self.feature_names}"
            )

        safe_feature_name = feature.replace("/", "_").replace(" ", "_")
        save_path = save_path or os.path.join(
            self.config.output_dir, f"shap_dependence_{safe_feature_name}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure()
        shap.dependence_plot(
            feature,
            self.shap_values_,
            self.X_explained_,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.title(f"SHAP Dependence Plot: {feature}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Saved SHAP dependence plot for '%s' to '%s'.", feature, save_path)
        return save_path

    # ------------------------------------------------------------------ #
    # SHAP-based ranking
    # ------------------------------------------------------------------ #
    def get_shap_importance_ranking(self) -> pd.DataFrame:
        """Return a `pd.DataFrame` ranking features by mean |SHAP value|.

        Returns
        -------
        pd.DataFrame
            Columns: ["feature", "mean_abs_shap"], sorted descending by
            `mean_abs_shap`.
        """
        self._ensure_fitted()
        shap_array = np.asarray(self.shap_values_)
        mean_abs_shap = np.abs(shap_array).mean(axis=0)

        ranking = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return ranking

    def _ensure_fitted(self) -> None:
        if self.explainer is None or self.shap_values_ is None:
            raise RuntimeError(
                "ShapAnalyzer.fit_explainer() must be called before "
                "generating plots or rankings."
            )


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #
def compute_permutation_importance(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Optional[ExplainabilityConfig] = None,
    scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """Compute permutation feature importance on the held-out test set.

    Statistical reasoning
    ----------------------
    Permutation importance measures, for each feature, the increase in
    model error (or decrease in score) when that feature's values are
    randomly shuffled (permuted) across the test samples -- breaking the
    relationship between the feature and the target while preserving the
    feature's marginal distribution. A large drop in performance indicates
    the model relied heavily on that feature.

    Unlike SHAP (which explains individual predictions based on the
    model's *internal* structure), permutation importance is a purely
    *outcome-based* / model-agnostic measure computed directly from
    held-out predictive performance, making it a useful **cross-check**
    against SHAP: features that rank highly under both methods are
    robustly important, while large disagreements may indicate
    multicollinearity (SHAP can "split" importance among correlated
    features in ways permutation importance does not).

    `n_repeats` independent permutations are performed per feature and the
    results averaged, with the standard deviation reported as an estimate
    of the importance score's sampling variability.

    Parameters
    ----------
    model : object
        A fitted regressor exposing `.predict()`.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target values.
    config : Optional[ExplainabilityConfig]
    scoring : str
        A scikit-learn scoring string. Default
        "neg_root_mean_squared_error" -- importance is reported as the
        *increase* in RMSE (positive = feature is helpful) when permuted.

    Returns
    -------
    pd.DataFrame
        Columns: ["feature", "importance_mean", "importance_std"], sorted
        descending by `importance_mean`.
    """
    config = config or ExplainabilityConfig()

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=config.permutation_n_repeats,
        random_state=config.random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    # For "neg_*" scorers, sklearn's permutation_importance reports the
    # decrease in the (negated) score from permuting -- i.e.,
    # importances_mean is the drop in (negative-RMSE), which is positive
    # when permutation makes RMSE worse (a good thing for "importance"
    # interpretation: higher = more important). We keep the sign as-is so
    # that "higher importance_mean = more important feature" holds.
    ranking = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    logger.info(
        "Computed permutation importance (%d repeats, scoring='%s') for %d features.",
        config.permutation_n_repeats,
        scoring,
        len(ranking),
    )
    return ranking


def plot_permutation_importance(
    ranking: pd.DataFrame,
    config: Optional[ExplainabilityConfig] = None,
    save_path: Optional[str] = None,
) -> str:
    """Generate and save a horizontal bar chart of permutation importance
    (top-N features, with error bars showing `importance_std`).

    Parameters
    ----------
    ranking : pd.DataFrame
        Output of `compute_permutation_importance`.
    config : Optional[ExplainabilityConfig]
    save_path : Optional[str]
        Output PNG path. Defaults to
        `<config.output_dir>/permutation_importance.png`.

    Returns
    -------
    str
    """
    config = config or ExplainabilityConfig()
    save_path = save_path or os.path.join(
        config.output_dir, "permutation_importance.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    top = ranking.head(config.top_n_features).iloc[::-1]  # reverse for barh

    plt.figure(figsize=(8, max(4, 0.4 * len(top))))
    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#4C72B0",
        ecolor="#333333",
        capsize=3,
    )
    plt.xlabel("Permutation Importance (Increase in RMSE when shuffled)")
    plt.title("Permutation Feature Importance (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved permutation importance plot to '%s'.", save_path)
    return save_path


# --------------------------------------------------------------------------- #
# Unified global ranking export
# --------------------------------------------------------------------------- #
def build_global_feature_ranking(
    shap_ranking: pd.DataFrame,
    permutation_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Combine SHAP-based and permutation-based importance rankings into a
    single, unified `pd.DataFrame` for dashboard consumption.

    Both importance metrics are normalized to a [0, 1] scale (dividing by
    their respective maxima) so they are directly comparable despite being
    on different natural scales (SHAP values are in degrees Celsius;
    permutation importance is in RMSE units). A combined score is computed
    as the simple average of the two normalized scores, and the resulting
    table is sorted by this combined score.

    Parameters
    ----------
    shap_ranking : pd.DataFrame
        Output of `ShapAnalyzer.get_shap_importance_ranking()`. Columns:
        ["feature", "mean_abs_shap"].
    permutation_ranking : pd.DataFrame
        Output of `compute_permutation_importance()`. Columns:
        ["feature", "importance_mean", "importance_std"].

    Returns
    -------
    pd.DataFrame
        Columns: ["feature", "mean_abs_shap", "shap_rank",
        "permutation_importance", "permutation_rank",
        "normalized_shap", "normalized_permutation", "combined_score",
        "combined_rank"], sorted by `combined_rank` ascending (rank 1 =
        most important).
    """
    merged = pd.merge(
        shap_ranking,
        permutation_ranking[["feature", "importance_mean"]].rename(
            columns={"importance_mean": "permutation_importance"}
        ),
        on="feature",
        how="outer",
    )

    # Fill any features missing from one ranking (e.g., due to SHAP
    # subsampling differences) with 0 -- treated as "no measured impact"
    # for that metric.
    merged["mean_abs_shap"] = merged["mean_abs_shap"].fillna(0.0)
    merged["permutation_importance"] = merged["permutation_importance"].fillna(0.0)

    # Permutation importance can be negative (shuffling occasionally
    # *improves* the score by chance); clip at 0 before normalizing since
    # negative values don't represent meaningful "importance".
    clipped_perm = merged["permutation_importance"].clip(lower=0.0)

    max_shap = merged["mean_abs_shap"].max()
    max_perm = clipped_perm.max()

    merged["normalized_shap"] = (
        merged["mean_abs_shap"] / max_shap if max_shap > 0 else 0.0
    )
    merged["normalized_permutation"] = (
        clipped_perm / max_perm if max_perm > 0 else 0.0
    )

    merged["combined_score"] = (
        merged["normalized_shap"] + merged["normalized_permutation"]
    ) / 2.0

    merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    merged["permutation_rank"] = merged["permutation_importance"].rank(
        ascending=False, method="min"
    ).astype(int)
    merged["combined_rank"] = merged["combined_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    merged = merged.sort_values("combined_rank").reset_index(drop=True)

    column_order = [
        "feature",
        "combined_rank",
        "combined_score",
        "mean_abs_shap",
        "shap_rank",
        "permutation_importance",
        "permutation_rank",
        "normalized_shap",
        "normalized_permutation",
    ]
    return merged[column_order]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_explainability_pipeline(
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    config: Optional[ExplainabilityConfig] = None,
    dependence_feature_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[pd.DataFrame, ShapAnalyzer]:
    """Run the full explainability pipeline: SHAP analysis, permutation
    importance, plot generation, and unified ranking export.

    Parameters
    ----------
    model : object
        Fitted tree-based regressor (e.g., from
        `src.forecasting.MultivariateForecastingEngine.fitted_models_`).
    X_train : pd.DataFrame
        Training feature matrix (not directly used by SHAP/permutation
        here, but accepted for API symmetry / future extensions such as
        background-distribution-based explainers).
    X_test : pd.DataFrame
        Test feature matrix, columns matching `feature_names`.
    y_test : pd.Series
        Test target values (for permutation importance).
    feature_names : List[str]
        Ordered list of feature column names.
    config : Optional[ExplainabilityConfig]
    dependence_feature_pairs : Optional[List[Tuple[str, str]]]
        List of (feature, interaction_feature) pairs for SHAP dependence
        plots. If `None`, defaults to pairing each of the configured
        target-lag features with `humidity` (when present), e.g.
        `("temperature_celsius_lag_1", "humidity")`.

    Returns
    -------
    Tuple[pd.DataFrame, ShapAnalyzer]
        (global feature ranking dataframe, fitted ShapAnalyzer instance)
    """
    config = config or ExplainabilityConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    # --- SHAP analysis --- #
    shap_analyzer = ShapAnalyzer(model, feature_names, config=config)
    shap_analyzer.fit_explainer(X_test)
    shap_analyzer.plot_summary()
    shap_analyzer.plot_bar_importance()

    if dependence_feature_pairs is None:
        dependence_feature_pairs = []
        candidate_features = [
            f for f in feature_names if f.startswith("temperature_celsius_lag_")
        ]
        for feat in candidate_features:
            interaction = "humidity" if "humidity" in feature_names else None
            dependence_feature_pairs.append((feat, interaction))

    for feature, interaction in dependence_feature_pairs:
        if feature in feature_names:
            try:
                shap_analyzer.plot_dependence(feature, interaction_feature=interaction)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to generate dependence plot for '%s' (interaction='%s'): %s",
                    feature,
                    interaction,
                    exc,
                )

    shap_ranking = shap_analyzer.get_shap_importance_ranking()

    # --- Permutation importance --- #
    permutation_ranking = compute_permutation_importance(
        model, X_test[feature_names], y_test, config=config
    )
    plot_permutation_importance(permutation_ranking, config=config)

    # --- Unified ranking --- #
    global_ranking = build_global_feature_ranking(shap_ranking, permutation_ranking)

    ranking_path = os.path.join(config.output_dir, "global_feature_ranking.csv")
    global_ranking.to_csv(ranking_path, index=False)
    logger.info("Saved global feature ranking to '%s'.", ranking_path)

    return global_ranking, shap_analyzer


if __name__ == "__main__":
    from src.forecasting import ForecastConfig, MultivariateForecastingEngine

    data_path = "data/processed_weather_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at '{data_path}'. Run src/data_prep.py first."
        )

    weather_df = pd.read_csv(data_path)
    weather_df["last_updated"] = pd.to_datetime(weather_df["last_updated"])

    sample_location = weather_df["location_name"].value_counts().idxmax()
    logger.info("Running explainability pipeline for location: %s", sample_location)

    fc_config = ForecastConfig()
    engine = MultivariateForecastingEngine(config=fc_config)
    engine.prepare_data(weather_df, location_name=sample_location)
    engine.fit()

    # Use XGBoost if available, otherwise fall back to RandomForest.
    model_name = "XGBoost" if "XGBoost" in engine.fitted_models_ else "RandomForest"
    selected_model = engine.fitted_models_[model_name]
    logger.info("Selected model for explainability: %s", model_name)

    X_train_ = engine.train_df_[engine.feature_cols_]
    X_test_ = engine.test_df_[engine.feature_cols_]
    y_test_ = engine.test_df_[engine.config.target_col]

    exp_config = ExplainabilityConfig(output_dir="images")
    ranking, analyzer = run_explainability_pipeline(
        selected_model,
        X_train_,
        X_test_,
        y_test_,
        feature_names=engine.feature_cols_,
        config=exp_config,
    )

    print("\n=== Top 15 Globally Ranked Features ===")
    print(ranking.head(15).to_string(index=False))
