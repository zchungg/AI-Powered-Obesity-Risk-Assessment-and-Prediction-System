# Allows Python to evaluate type annotations lazily.
from __future__ import annotations

# Imports type hints used by the model factories.
from typing import Dict, Tuple

# Imports Random Forest and Stacking Ensemble classification algorithms.
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)

# Imports Logistic Regression, a linear classification algorithm.
from sklearn.linear_model import LogisticRegression

# Imports MLPClassifier, which implements a feedforward neural network.
from sklearn.neural_network import MLPClassifier

# Imports Decision Tree, a tree-based classification algorithm.
from sklearn.tree import DecisionTreeClassifier

# Imports CatBoost, a gradient-boosting algorithm based on decision trees.
from catboost import CatBoostClassifier

# Imports LightGBM, an efficient gradient-boosting algorithm.
from lightgbm import LGBMClassifier

# Imports XGBoost, a gradient-boosting algorithm based on decision trees.
from xgboost import XGBClassifier


# Records which models are intended to support SHAP explanations in this project.
# True means the model is designated as compatible; False means it is not.
SHAP_COMPATIBLE = {
    "Logistic Regression": True,
    "Decision Tree": True,
    "Random Forest": True,
    "XGBoost": True,
    "CatBoost": True,
    "LightGBM": True,
    "Stacking Ensemble": False,
    "Artificial Neural Network": False,
}


# Creates and returns the candidate machine learning models for comparison.
def get_models(random_state: int = 42) -> Dict[str, object]:
    """Return the candidate classifiers from the proposal."""

    # Creates a Logistic Regression model that accounts for class imbalance.
    lr = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=random_state)

    # Creates a Random Forest consisting of 400 decision trees.
    rf = RandomForestClassifier(
        n_estimators=400,                  # Number of trees in the forest.
        random_state=random_state,        # Makes randomized operations more reproducible.
        n_jobs=-1,                         # Uses all available CPU cores.
        class_weight="balanced_subsample", # Adjusts class weights for each tree's sample.
        max_depth=None,                    # Allows trees to grow without a fixed depth limit.
    )

    # Creates a LightGBM model for predicting multiple obesity categories.
    lgbm = LGBMClassifier(
        n_estimators=400,      # Number of boosting iterations.
        learning_rate=0.05,    # Controls how much each boosting iteration contributes.
        num_leaves=31,         # Limits the number of leaves in each tree.
        subsample=0.9,         # Uses 90% of the training rows per boosting iteration when enabled.
        colsample_bytree=0.9,  # Uses 90% of the features for each tree.
        objective="multiclass",# Specifies a multi-class classification task.
        random_state=random_state,
        n_jobs=-1,             # Uses all available CPU cores.
        verbosity=-1,          # Suppresses most LightGBM training messages.
    )

    # Stores all candidate models in a dictionary for training and comparison.
    models = {
        # Logistic Regression provides a linear classification baseline.
        "Logistic Regression": LogisticRegression(
            max_iter=2500,                  # Allows up to 2,500 optimization iterations.
            class_weight="balanced",        # Helps account for unequal class frequencies.
            random_state=random_state,
        ),

        # Decision Tree learns classification rules through a tree structure.
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,           # Limits tree depth to reduce overfitting.
            min_samples_leaf=2,     # Requires at least two training samples in each leaf.
            class_weight="balanced",# Gives greater weight to underrepresented classes.
            random_state=random_state,
        ),

        # Random Forest combines predictions from multiple decision trees.
        "Random Forest": rf,

        # XGBoost builds trees sequentially to improve classification performance.
        "XGBoost": XGBClassifier(
            n_estimators=400,           # Number of boosting rounds.
            max_depth=6,                # Maximum depth of each tree.
            learning_rate=0.05,         # Controls each boosting update's contribution.
            subsample=0.9,              # Samples 90% of training rows per boosting round.
            colsample_bytree=0.9,       # Samples 90% of features per tree.
            objective="multi:softprob", # Predicts probabilities for all target classes.
            eval_metric="mlogloss",     # Uses multiclass log loss as the evaluation metric.
            tree_method="hist",         # Uses histogram-based tree construction.
            random_state=random_state,
            n_jobs=-1,                  # Uses all available CPU cores.
        ),

        # CatBoost uses gradient boosting with techniques designed for robust tree training.
        "CatBoost": CatBoostClassifier(
            iterations=400,             # Number of boosting iterations.
            depth=7,                    # Maximum depth of each tree.
            learning_rate=0.05,         # Controls the contribution of each boosting iteration.
            l2_leaf_reg=3.0,            # Makes the CatBoost default explicit for tuning.
            loss_function="MultiClass", # Specifies multiclass classification.
            verbose=False,              # Suppresses training progress output.
            random_seed=random_state,
            thread_count=1,              # Avoids unstable nested parallelism in selector fits.
        ),

        # Adds the LightGBM model created earlier.
        "LightGBM": lgbm,

        # Combines several different models into a Stacking Ensemble.
        "Stacking Ensemble": StackingClassifier(
            estimators=[
                ("lr", lr),     # Uses Logistic Regression as a base model.
                ("rf", rf),     # Uses Random Forest as a base model.
                ("lgbm", lgbm), # Uses LightGBM as a base model.
            ],
            final_estimator=LogisticRegression(max_iter=2000), # Learns how to combine base-model predictions.
            cv=5,                           # Uses five-fold cross-validation to generate training predictions.
            stack_method="predict_proba",   # Combines predicted class probabilities.
            passthrough=False,              # Sends base-model predictions, not original features, to the final estimator.
            n_jobs=-1,                      # Uses available CPU cores where supported.
        ),

        # Creates an Artificial Neural Network for learning complex feature relationships.
        "Artificial Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), # Uses three hidden layers with 128, 64, and 32 neurons.
            activation="relu",                 # Uses ReLU activation in hidden layers.
            solver="adam",                     # Uses Adam to optimize network weights.
            alpha=1e-4,                        # Applies L2 regularization to help reduce overfitting.
            batch_size=32,                     # Processes 32 training samples per mini-batch.
            learning_rate_init=1e-3,            # Sets the initial learning rate to 0.001.
            max_iter=600,                       # Allows up to 600 training iterations.
            early_stopping=True,                # Stops training when validation performance stops improving.
            validation_fraction=0.15,           # Reserves 15% of the training data for validation.
            n_iter_no_change=25,                 # Stops after 25 iterations without sufficient improvement.
            random_state=random_state,
        ),
    }

    # Returns the dictionary containing all eight candidate models.
    return models


# Checks whether a model is designated as compatible with SHAP explanations.
def shap_compatibility(name: str) -> bool:

    # Returns the stored compatibility status; unknown model names default to False.
    return bool(SHAP_COMPATIBLE.get(name, False))


def get_tuning_space(name: str) -> Tuple[object, dict]:
    """Return a model-specific estimator and its randomized search space."""
    if name == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
        params = {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        }
    elif name == "Decision Tree":
        estimator = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        )
        params = {
            "model__max_depth": [4, 6, 8, 10, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 4, 8, 12],
            "model__criterion": ["gini", "entropy", "log_loss"],
        }
    elif name == "Random Forest":
        estimator = RandomForestClassifier(
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        params = {
            "model__n_estimators": [250, 400, 600],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.7, 1.0],
        }
    elif name == "XGBoost":
        estimator = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )
        params = {
            "model__n_estimators": [250, 400, 600],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.75, 0.9, 1.0],
            "model__min_child_weight": [1, 3, 5],
        }
    elif name == "CatBoost":
        estimator = CatBoostClassifier(
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
            thread_count=1,
        )
        params = {
            "model__iterations": [250, 400, 600],
            "model__depth": [4, 5, 6, 7, 8, 10],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
            "model__l2_leaf_reg": [1, 3, 5, 8, 12],
        }
    elif name == "LightGBM":
        estimator = LGBMClassifier(
            objective="multiclass",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
        params = {
            "model__n_estimators": [250, 400, 600],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
            "model__num_leaves": [15, 31, 63, 127],
            "model__max_depth": [-1, 5, 8, 12],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.75, 0.9, 1.0],
        }
    elif name == "Artificial Neural Network":
        estimator = MLPClassifier(
            solver="adam",
            activation="relu",
            early_stopping=True,
            random_state=42,
            max_iter=600,
        )
        params = {
            "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [5e-4, 1e-3, 5e-3],
        }
    elif name == "Stacking Ensemble":
        estimator = get_models(42)[name]
        params = {
            "model__lr__C": [0.1, 1.0, 10.0],
            "model__rf__n_estimators": [200, 400, 600],
            "model__rf__max_depth": [None, 8, 16],
            "model__lgbm__num_leaves": [15, 31, 63],
            "model__final_estimator__C": [0.1, 1.0, 10.0],
        }
    else:
        raise ValueError(f"No tuning space configured for {name}")

    return estimator, params