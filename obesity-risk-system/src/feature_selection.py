# Allows Python to evaluate type annotations lazily.
from __future__ import annotations

# Imports warnings to notify users when a SHAP ranking cannot be calculated.
import warnings

# Imports Optional for type annotations that allow None values.
from typing import Optional

# Imports NumPy for numerical calculations and array manipulation.
import numpy as np

# Imports utilities for detecting and handling sparse matrices.
from scipy import sparse

# Imports base classes for creating a custom scikit-learn transformer.
from sklearn.base import BaseEstimator, TransformerMixin, clone

# Imports Random Forest for models without native RFE feature importances.
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# Imports feature selection methods: RFE, Chi-Square, and ANOVA F-test.
from sklearn.feature_selection import RFE, chi2, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Imports MinMaxScaler to scale values into a non-negative range for Chi-Square testing.
from sklearn.preprocessing import MinMaxScaler


# Creates a custom transformer that combines four feature-ranking methods.
# Borda ranking combines their rankings to identify the most useful features.
class BordaFeatureSelector(BaseEstimator, TransformerMixin):
    """Hybrid feature selector using Chi-Square, ANOVA, RFE and SHAP rankings.

    Borda points are summed across the four ranking methods. The selector is
    intended to live inside an imbalanced-learn pipeline, so it is fitted only
    on the training portion of each cross-validation fold.
    """

    # Initializes the feature selector and its configuration settings.
    def __init__(
        self,
        n_features_to_select: int = 12,  # Number of features to keep.
        random_state: int = 42,          # Seed for reproducible random operations.
        shap_sample_size: int = 300,     # Maximum number of samples used for SHAP importance.
        ranking_estimator=None,          # Model-specific estimator used for RFE and SHAP.
    ) -> None:
        # Stores the number of features to retain.
        self.n_features_to_select = n_features_to_select

        # Stores the random seed.
        self.random_state = random_state

        # Stores the maximum SHAP sample size.
        self.shap_sample_size = shap_sample_size

        # Stores the candidate model used to produce model-specific rankings.
        self.ranking_estimator = ranking_estimator

    # Converts sparse matrices into regular NumPy arrays when necessary.
    @staticmethod
    def _to_dense(X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)

    # Converts raw importance values into ranking scores.
    # Higher original values receive higher ranking scores.
    @staticmethod
    def _rank_scores(values: np.ndarray) -> np.ndarray:
        # Converts values to floating-point numbers and replaces NaN with negative infinity.
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)

        # Sorts feature indices from the highest value to the lowest.
        order = np.argsort(-values, kind="stable")

        # Creates an array to store each feature's ranking points.
        scores = np.zeros(len(values), dtype=float)

        # Assigns the highest points to the best-ranked feature.
        scores[order] = np.arange(len(values), 0, -1, dtype=float)

        # Returns the ranking scores.
        return scores

    @staticmethod
    def _bounded_clone(estimator, max_trees: int, max_iterations: int):
        """Limit auxiliary ranking fits while keeping the candidate estimator type."""
        bounded = clone(estimator)
        params = bounded.get_params(deep=True)
        updates = {}
        for name, value in params.items():
            parameter = name.rsplit("__", 1)[-1]
            if parameter in {"n_estimators", "iterations"} and isinstance(value, int):
                updates[name] = min(value, max_trees)
            elif parameter == "max_iter" and isinstance(value, int):
                updates[name] = min(value, max_iterations)
            elif parameter == "n_jobs":
                parent_name = name.rpartition("__")[0]
                target = (
                    bounded.get_params(deep=True)[parent_name]
                    if parent_name
                    else bounded
                )
                if not isinstance(target, LogisticRegression):
                    updates[name] = 1
        if updates:
            bounded.set_params(**updates)
        return bounded

    @staticmethod
    def _as_feature_importance(values, n_features: int) -> Optional[np.ndarray]:
        importance = np.abs(np.asarray(values, dtype=float))
        if importance.ndim == 1 and importance.size == n_features:
            return importance
        if importance.ndim == 2:
            if importance.shape[1] == n_features:
                return np.mean(importance, axis=0)
            if importance.shape[0] == n_features:
                return np.mean(importance, axis=1)
        return None

    @classmethod
    def _rfe_feature_importance(cls, estimator) -> np.ndarray:
        """Extract feature importances for estimators supported by this selector."""
        n_features = estimator.n_features_in_
        importances = []

        direct = getattr(estimator, "feature_importances_", None)
        if direct is not None:
            importance = cls._as_feature_importance(direct, n_features)
            if importance is not None:
                importances.append(importance)

        coefficients = getattr(estimator, "coef_", None)
        if coefficients is not None:
            importance = cls._as_feature_importance(coefficients, n_features)
            if importance is not None:
                importances.append(importance)

        first_layer = getattr(estimator, "coefs_", None)
        if first_layer:
            importance = cls._as_feature_importance(first_layer[0], n_features)
            if importance is not None:
                importances.append(importance)

        for sub_estimator in getattr(estimator, "named_estimators_", {}).values():
            direct = getattr(sub_estimator, "feature_importances_", None)
            coefficients = getattr(sub_estimator, "coef_", None)
            if direct is not None:
                importance = cls._as_feature_importance(direct, n_features)
            elif coefficients is not None:
                importance = cls._as_feature_importance(coefficients, n_features)
            else:
                importance = None
            if importance is not None:
                scale = float(np.max(importance))
                importances.append(importance / scale if scale > 0 else importance)

        if not importances:
            raise ValueError(
                f"{type(estimator).__name__} did not expose feature importances."
            )

        return np.mean(importances, axis=0)

    # Calculates global SHAP importance for the candidate model.
    def _safe_shap_importance(
        self,
        estimator,
        X_background: np.ndarray,
        X_sample: np.ndarray,
    ) -> np.ndarray:
        import shap

        # CatBoost's native SHAP adapter can corrupt memory with some
        # CatBoost/SHAP version combinations; stacking models are black boxes.
        model_agnostic = (
            type(estimator).__module__.startswith("catboost")
            or isinstance(
                estimator,
                (LogisticRegression, MLPClassifier, StackingClassifier),
            )
        )
        if model_agnostic:
            explainer = shap.Explainer(
                estimator.predict_proba,
                X_background,
                algorithm="permutation",
            )
            explanation = explainer(
                X_sample,
                max_evals=2 * X_sample.shape[1] + 1,
                silent=True,
            )
        else:
            try:
                explainer = shap.Explainer(estimator, X_background)
                if type(explainer).__name__ == "Permutation":
                    explanation = explainer(
                        X_sample,
                        max_evals=2 * X_sample.shape[1] + 1,
                        silent=True,
                    )
                elif type(explainer).__name__ == "TreeExplainer":
                    explanation = explainer(X_sample)
                else:
                    explanation = explainer(X_sample, silent=True)
            except Exception as native_error:
                warnings.warn(
                    f"Native SHAP explanation failed for {type(estimator).__name__}; "
                    "retrying with a model-agnostic SHAP explainer: "
                    f"{native_error}",
                    RuntimeWarning,
                )
                explainer = shap.Explainer(
                    estimator.predict_proba,
                    X_background,
                    algorithm="permutation",
                )
                explanation = explainer(
                    X_sample,
                    max_evals=2 * X_sample.shape[1] + 1,
                    silent=True,
                )

        values = explanation.values

        # Some explainers return one feature-value array per output class.
        if isinstance(values, list):
            values = np.stack(values, axis=-1)

        values = np.asarray(values)
        if values.ndim == 3:
            return np.mean(np.abs(values), axis=(0, 2))
        if values.ndim == 2:
            return np.mean(np.abs(values), axis=0)

        raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    # Learns feature rankings from the training data and selects the best features.
    def fit(self, X, y):
        # Converts input features to a dense floating-point NumPy array.
        Xd = self._to_dense(X).astype(float)

        # Gets the total number of input features.
        n_features = Xd.shape[1]

        # Ensures the requested number of selected features is between 1 and the available feature count.
        k = min(max(int(self.n_features_to_select), 1), n_features)

        # Converts the target labels into a NumPy array.
        y_arr = np.asarray(y)

        # Stores the original number of input features for later use.
        self.n_input_features_ = n_features

        # 1. CHI-SQUARE FEATURE RANKING

        # Scales a copy of the features to non-negative values because Chi-Square requires non-negative inputs.
        chi_input = MinMaxScaler().fit_transform(Xd)

        # Calculates Chi-Square scores and p-values for the features and target labels.
        chi_scores, _ = chi2(chi_input, y_arr)

        # 2. ANOVA F-TEST FEATURE RANKING

        # Measures how strongly each feature's values differ across target classes.
        anova_scores, _ = f_classif(Xd, y_arr)

        # Replaces invalid ANOVA scores with zero to prevent ranking problems.
        anova_scores = np.nan_to_num(anova_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. RECURSIVE FEATURE ELIMINATION (RFE)
        candidate = self.ranking_estimator
        if candidate is None:
            candidate = RandomForestClassifier(
                n_estimators=120,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
                max_depth=10,
            )

        rfe = RFE(
            estimator=self._bounded_clone(candidate, max_trees=80, max_iterations=300),
            n_features_to_select=max(1, int(np.ceil(n_features / 2))),
            step=max(1, int(n_features / 5)),
            importance_getter=self._rfe_feature_importance,
        )
        try:
            rfe.fit(Xd, y_arr)
        except (ValueError, TypeError, AttributeError) as exc:
            if self.ranking_estimator is None:
                raise
            warnings.warn(
                f"RFE could not use {type(candidate).__name__}; "
                f"using Random Forest for this ranking: {exc}",
                RuntimeWarning,
            )
            rfe = RFE(
                estimator=RandomForestClassifier(
                    n_estimators=120,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    max_depth=10,
                ),
                n_features_to_select=max(1, int(np.ceil(n_features / 2))),
                step=max(1, int(n_features / 5)),
            )
            rfe.fit(Xd, y_arr)

        # Retrieves RFE rankings, where rank 1 indicates the most important features.
        rfe_scores = rfe.ranking_.astype(float)

        # Reverses the ranking so better RFE ranks receive higher scores.
        rfe_scores = (rfe_scores.max() + 1.0) - rfe_scores

        # 4. SHAP FEATURE IMPORTANCE

        # Creates a random number generator using the configured seed.
        rng = np.random.default_rng(self.random_state)

        shap_candidate = (
            self.ranking_estimator
            if self.ranking_estimator is not None
            else candidate
        )
        model_agnostic = (
            type(shap_candidate).__module__.startswith("catboost")
            or isinstance(shap_candidate, (StackingClassifier, MLPClassifier))
        )
        sample_limit = 30 if model_agnostic else 100
        sample_n = min(self.shap_sample_size, Xd.shape[0], sample_limit)
        background_n = min(50, Xd.shape[0])

        sample_idx = rng.choice(Xd.shape[0], size=sample_n, replace=False)
        background_idx = rng.choice(Xd.shape[0], size=background_n, replace=False)

        ranking_model = self._bounded_clone(
            shap_candidate,
            max_trees=120,
            max_iterations=400,
        )
        ranking_model.fit(Xd, y_arr)
        shap_scores = self._safe_shap_importance(
            ranking_model,
            Xd[background_idx],
            Xd[sample_idx],
        )

        # Stores the Chi-Square scores and replaces invalid values with zero.
        self.chi2_scores_ = np.nan_to_num(chi_scores, nan=0.0)

        # Stores the ANOVA F-test scores.
        self.anova_scores_ = anova_scores

        # Stores the converted RFE ranking scores.
        self.rfe_scores_ = rfe_scores

        # Stores SHAP importance scores and replaces invalid values with zero.
        self.shap_scores_ = np.nan_to_num(shap_scores, nan=0.0)

        # Adds the ranking points from all four feature-selection methods.
        borda = (
            self._rank_scores(self.chi2_scores_)
            + self._rank_scores(self.anova_scores_)
            + self._rank_scores(self.rfe_scores_)
            + self._rank_scores(self.shap_scores_)
        )

        # Saves the combined Borda score for every feature.
        self.borda_scores_ = borda

        # Sorts feature indices from the highest combined score to the lowest.
        order = np.argsort(-borda, kind="stable")

        # Selects the indices of the top-ranked features.
        self.selected_indices_ = order[:k]

        # Creates a Boolean mask marking every feature as unselected initially.
        self.support_mask_ = np.zeros(n_features, dtype=bool)

        # Marks the selected feature positions as True.
        self.support_mask_[self.selected_indices_] = True

        # Stores the complete feature ranking from best to worst.
        self.ranking_order_ = order

        # Records the number of features retained by the selector.
        self.n_features_out_ = k

        # Initializes feature names; they can be supplied separately if available.
        self.feature_names_in_ = None

        # Returns the fitted selector so it can be used in a pipeline.
        return self

    # Reduces the dataset to only the features selected during fit().
    def transform(self, X):
        # Prevents transformation before the selector has been fitted.
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("BordaFeatureSelector must be fitted before transform().")

        # Converts the input into a dense array if necessary.
        Xd = self._to_dense(X)

        # Returns only the columns marked True in the selection mask.
        return Xd[:, self.support_mask_]

    # Returns the selected features as a Boolean mask or as their column indices.
    def get_support(self, indices: bool = False):
        # Returns the indices of selected features when indices=True.
        if indices:
            return np.flatnonzero(self.support_mask_)

        # Otherwise, returns a copy of the Boolean selection mask.
        return self.support_mask_.copy()

    # Returns the names of the selected features.
    def get_feature_names_out(self, input_features: Optional[np.ndarray] = None):
        # Uses stored feature names or generates default names if none are provided.
        if input_features is None:

            # Generates names such as feature_0 and feature_1 when original names are unavailable.
            if self.feature_names_in_ is None:
                input_features = np.array(
                    [f"feature_{i}" for i in range(self.n_input_features_)], dtype=object
                )

            # Uses previously stored feature names when available.
            else:
                input_features = np.asarray(self.feature_names_in_, dtype=object)

        # Uses feature names supplied directly to this function.
        else:
            input_features = np.asarray(input_features, dtype=object)

        # Returns only the names corresponding to selected features.
        return input_features[self.support_mask_]

    # Stores the original feature names so selected features can be identified later.
    def set_feature_names(self, names):
        self.feature_names_in_ = np.asarray(names, dtype=object)

        # Returns the selector to support method chaining.
        return self

    # Creates a table showing each feature's individual and combined ranking scores.
    def ranking_table(self, feature_names) -> "object":
        # Imports pandas locally to create a readable results table.
        import pandas as pd

        # Converts the supplied feature names into a NumPy array.
        names = np.asarray(feature_names, dtype=object)

        # Combines feature names, ranking scores, and selection status into a table.
        df = pd.DataFrame(
            {
                "feature": names,                  # Feature name.
                "chi2": self.chi2_scores_,         # Chi-Square score.
                "anova_f": self.anova_scores_,     # ANOVA F-test score.
                "rfe_score": self.rfe_scores_,     # Converted RFE ranking score.
                "shap_importance": self.shap_scores_,# SHAP importance score.
                "borda_score": self.borda_scores_, # Combined ranking score.
                "selected": self.support_mask_,    # Whether the feature was selected.
            }
        )

        # Sorts features by combined score, placing the highest-ranked features first.
        return df.sort_values("borda_score", ascending=False).reset_index(drop=True)