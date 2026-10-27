from __future__ import annotations

# argparse lets the user pass options when running the script.
import argparse

# json saves model information in a readable JSON file.
import json

# time measures how long model evaluation takes.
import time
from tempfile import TemporaryDirectory

from pathlib import Path

# joblib saves the trained machine-learning pipeline for later use.
import joblib

import numpy as np
import pandas as pd

# SMOTE creates synthetic training examples to help balance underrepresented classes.
from imblearn.over_sampling import SMOTE

# This pipeline supports SMOTE alongside normal preprocessing and model steps.
from imblearn.pipeline import Pipeline as ImbPipeline

# These metrics measure model accuracy and classification performance.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

# These functions split the data and create cross-validation folds.
from sklearn.model_selection import (
    GridSearchCV,
    ParameterSampler,
    StratifiedKFold,
    train_test_split,
)

# Import the project's settings, including file paths and training parameters.
from .config import (
    BEST_MODEL_FILE,
    COMPARISON_FILE,
    CV_FOLDS,
    DATA_FILE,
    METADATA_FILE,
    MODEL_DIR,
    N_FEATURES_TO_SELECT,
    RANDOM_SEARCH_ITERATIONS,
    RANDOM_STATE,
    RESULTS_DIR,
    SHAP_SAMPLE_SIZE,
    SMOTE_K_NEIGHBORS,
    TARGET_CLASSES,
    TARGET_RAW_TO_ID,
    TARGET_COLUMN,
    TEST_SIZE,
    TOP_MODELS_TO_TUNE,
)

# Import the feature selector, model factory, and preprocessing functions.
from .feature_selection import BordaFeatureSelector
from .models import get_models, get_tuning_space, shap_compatibility
from .preprocessing import build_preprocessor, clean_columns, validate_dataset


# Combine preprocessing, class balancing, feature selection, and classification
# into one pipeline so the same steps are applied consistently during training.
def build_pipeline(model, memory=None):
    return ImbPipeline(
        steps=[
            # Clean and encode the input features and scale numerical values.
            ("preprocess", build_preprocessor()),

            # Balance the training classes by generating synthetic examples.
            # During cross-validation, SMOTE only sees the current training fold.
            ("smote", SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=SMOTE_K_NEIGHBORS,
            )),

            # Select the most useful features using the project's Borda method.
            ("feature_select", BordaFeatureSelector(
                n_features_to_select=N_FEATURES_TO_SELECT,
                random_state=RANDOM_STATE,
                shap_sample_size=SHAP_SAMPLE_SIZE,
                ranking_estimator=model,
            )),

            # Train the selected machine-learning classifier.
            ("model", model),
        ],
        memory=memory,
    )


# Calculate evaluation metrics and return them as one dictionary/table row.
def _metric_row(
    name: str,
    y_true,
    y_pred,
    y_prob,
    elapsed,
    cv_f1=np.nan,
    class_labels=None,
):
    # Use the supplied class order, or fall back to the project's seven classes.
    labels = (
        list(class_labels)
        if class_labels is not None
        else list(range(len(TARGET_CLASSES)))
    )

    return {
        "model": name,

        # Percentage of predictions that are correct.
        "accuracy": accuracy_score(y_true, y_pred),

        # Macro F1 gives each class equal importance, regardless of its size.
        "macro_f1": f1_score(y_true, y_pred, average="macro"),

        # Measures the quality of predicted probabilities; lower is better.
        "log_loss": log_loss(y_true, y_prob, labels=labels),

        # Time taken for cross-validation, final fitting, and test predictions.
        "inference_time_sec": elapsed,

        # Average macro F1 across the cross-validation validation folds.
        "cv_macro_f1": cv_f1,

        # Record whether this model is marked as suitable for the project's
        # intended SHAP explanation workflow.
        "shap_compatible": shap_compatibility(name),
    }


# Evaluate a model using cross-validation, then train and test its final pipeline.
def evaluate_pipeline(name, pipeline, X_train, y_train, X_test, y_test, cv):
    # Start timing the entire evaluation process.
    start = time.perf_counter()

    # Cross-validation uses only the training set, leaving the test set untouched.
    # SMOTE and feature selection run inside each fold to reduce data leakage.
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        # Build a fresh pipeline so each fold learns preprocessing, feature
        # selection, and the model using only that fold's training data.
        fold = build_pipeline(get_models(RANDOM_STATE)[name])

        # Fit on the fold's training rows and predict its validation rows.
        fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = fold.predict(X_train.iloc[val_idx])

        # Record this fold's macro F1 score.
        cv_scores.append(
            f1_score(y_train.iloc[val_idx], pred, average="macro")
        )

    # Calculate the average cross-validation macro F1 score.
    cv_f1 = float(np.mean(cv_scores))

    # Fit the supplied pipeline using the complete training set.
    pipeline.fit(X_train, y_train)

    # Predict the classes and class probabilities for the held-out test set.
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)

    # Calculate the total elapsed evaluation time.
    elapsed = time.perf_counter() - start

    # Return the trained pipeline and its test-set performance metrics.
    return pipeline, _metric_row(
        name, y_test, pred, prob, elapsed, cv_f1, pipeline.classes_
    )


# Create and save a visual matrix showing correct and incorrect predictions.
def save_confusion_matrix(name: str, y_true, y_pred) -> None:
    # Import plotting libraries only when this function is called.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Keep all seven obesity categories in the same order.
    class_ids = list(range(len(TARGET_CLASSES)))

    # Count actual-versus-predicted class combinations.
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    # Create a figure and draw the confusion matrix as a heatmap.
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    # Label the chart so actual and predicted classes are easy to identify.
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(TARGET_CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(TARGET_CLASSES, rotation=0)

    # Adjust spacing so the labels fit, then ensure the results folder exists.
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert the model name into a filename-friendly format.
    safe_name = name.lower().replace(" ", "_").replace("/", "_")

    # Save the chart as a high-resolution PNG image and close the figure.
    fig.savefig(
        RESULTS_DIR / f"confusion_matrix_{safe_name}.png",
        dpi=160,
    )
    plt.close(fig)


def _model_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def save_model_metric_chart(row) -> Path:
    """Save a compact metric bar chart for one fitted model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = str(row["model"])
    score_names = ["accuracy", "macro_f1", "cv_macro_f1"]
    score_values = [float(row[key]) for key in score_names]
    labels = ["Accuracy", "Macro F1", "CV Macro F1"]

    fig, (score_ax, loss_ax) = plt.subplots(
        1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [3, 1]}
    )
    bars = score_ax.bar(labels, score_values, color=["#4C78A8", "#59A14F", "#F28E2B"])
    score_ax.set_ylim(0, 1)
    score_ax.set_ylabel("Score")
    score_ax.set_title("Classification scores")
    score_ax.bar_label(bars, fmt="%.3f", padding=3)
    score_ax.tick_params(axis="x", rotation=15)

    loss = float(row["log_loss"])
    loss_ax.bar(["Log loss"], [loss], color="#E15759")
    loss_ax.set_ylabel("Lower is better")
    loss_ax.set_title("Probability loss")
    loss_ax.text(0, loss, f"{loss:.3f}", ha="center", va="bottom")

    fig.suptitle(name)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = RESULTS_DIR / f"model_metrics_{_model_slug(name)}.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


# Return the fitted selector's ranking, optionally exporting a per-model report.
def save_feature_ranking(
    best_pipeline,
    output_name: str | None = None,
) -> pd.DataFrame:
    # Retrieve the fitted preprocessing and feature-selection steps.
    preprocessor = best_pipeline.named_steps["preprocess"]
    selector = best_pipeline.named_steps["feature_select"]

    # Get the feature names after encoding and other preprocessing.
    names = preprocessor.get_feature_names_out()

    # Give the selector the names so its ranking can identify each feature.
    selector.set_feature_names(names)

    # Create a table containing feature rankings and selection information.
    ranking = selector.ranking_table(names)

    if output_name is not None:
        ranking.to_csv(RESULTS_DIR / output_name, index=False)

    return ranking


# Tune the strongest baseline candidates using training-only cross-validation.
def tune_top_models(results, X_train, y_train, X_test, y_test, cv):
    results_df = pd.DataFrame(results)
    candidates = results_df.sort_values(
        "cv_macro_f1",
        ascending=False,
        kind="stable",
    ).head(TOP_MODELS_TO_TUNE)
    tuned = []

    for name in candidates["model"].tolist():
        print(f"\nTuning {name} ...")
        try:
            _, search_space = get_tuning_space(name)
            baseline_estimator = get_models(RANDOM_STATE)[name]
            baseline_parameters = baseline_estimator.get_params(deep=True)
            baseline_candidate = {}
            for parameter_name in search_space:
                model_parameter = parameter_name.removeprefix("model__")
                if model_parameter not in baseline_parameters:
                    raise ValueError(
                        f"{name} baseline does not define {model_parameter!r}."
                    )
                baseline_candidate[parameter_name] = [
                    baseline_parameters[model_parameter]
                ]

            random_candidates = list(
                ParameterSampler(
                    search_space,
                    n_iter=RANDOM_SEARCH_ITERATIONS,
                    random_state=RANDOM_STATE,
                )
            )
            candidate_grids = [baseline_candidate] + [
                {key: [value] for key, value in candidate.items()}
                for candidate in random_candidates
            ]

            with TemporaryDirectory(
                prefix="obesity-feature-ranking-cache-"
            ) as cache_dir:
                pipeline = build_pipeline(baseline_estimator, memory=cache_dir)
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=candidate_grids,
                    scoring="f1_macro",
                    cv=cv,
                    n_jobs=1,
                    refit=True,
                    verbose=2,
                )
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_
                best_estimator.set_params(memory=None)
                best_score = float(search.best_score_)
                best_params = search.best_params_
                baseline_score = float(search.cv_results_["mean_test_score"][0])

            if best_score + 1e-12 < baseline_score:
                raise RuntimeError(
                    f"{name} search selected CV Macro F1 {best_score:.6f}, "
                    f"below its baseline score {baseline_score:.6f}."
                )

            pred = best_estimator.predict(X_test)
            prob = best_estimator.predict_proba(X_test)
            tuned.append({
                "model": f"{name} (Tuned)",
                "base_model": name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "log_loss": log_loss(
                    y_test,
                    prob,
                    labels=best_estimator.classes_,
                ),
                "cv_macro_f1": best_score,
                "baseline_cv_macro_f1": baseline_score,
                "inference_time_sec": 0.0,
                "shap_compatible": shap_compatibility(name),
                "best_estimator": best_estimator,
                "best_params": best_params,
            })
        except Exception as exc:
            raise RuntimeError(f"Hyperparameter tuning failed for {name}.") from exc

    if len(tuned) != len(candidates):
        raise RuntimeError(
            f"Expected to tune {len(candidates)} models, but tuned {len(tuned)}."
        )
    return tuned


# Main function: load the dataset, train models, compare results, and save
# the best deployable model and its metadata.
def main(data_path: Path = DATA_FILE):
    # Create folders for the saved models and evaluation results.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stop early if the requested dataset file does not exist.
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run `python download_data.py` first."
        )

    # Read the CSV, clean its column names and text values, and check its schema.
    df = clean_columns(pd.read_csv(data_path))
    validate_dataset(df, TARGET_COLUMN)

    # Separate input features (X) from the target labels (y).
    X = df.drop(columns=[TARGET_COLUMN])
    raw_y = df[TARGET_COLUMN].astype(str)

    # Check that every target value belongs to the expected seven categories.
    unknown_classes = sorted(set(raw_y.unique()) - set(TARGET_RAW_TO_ID))
    if unknown_classes:
        raise ValueError(
            f"Unexpected target class values: {unknown_classes}. "
            f"Expected: {list(TARGET_RAW_TO_ID)}"
        )

    # Convert category names into stable integer IDs.
    # This format is required by XGBoost and standardizes class labels.
    y = raw_y.map(TARGET_RAW_TO_ID).astype(int)

    # Split the dataset into training and testing portions.
    # Stratification preserves approximately the same class proportions in both.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Create reproducible cross-validation folds that preserve class proportions.
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Create all candidate models and containers for their results.
    models = get_models(RANDOM_STATE)
    results = []
    fitted = {}

    print(
        f"Dataset: {len(df)} rows | "
        f"train={len(X_train)} | test={len(X_test)}"
    )
    print("\nTraining candidate models...\n")

    # Train and evaluate each candidate model using the same pipeline structure.
    for name, estimator in models.items():
        print(f"Training {name} ...")

        pipeline = build_pipeline(estimator)

        try:
            # Run cross-validation and evaluate the final pipeline on test data.
            fitted_pipeline, metrics = evaluate_pipeline(
                name, pipeline, X_train, y_train, X_test, y_test, cv
            )

            # Keep the fitted model and its performance metrics.
            fitted[name] = fitted_pipeline
            results.append(metrics)
            save_feature_ranking(
                fitted_pipeline,
                output_name=f"feature_ranking_{_model_slug(name)}.csv",
            )

            # Generate predictions for detailed evaluation reports.
            y_pred = fitted_pipeline.predict(X_test)

            # Save a visual summary of correct and incorrect predictions.
            save_confusion_matrix(name, y_test, y_pred)

            # Generate precision, recall, F1, and support for every class.
            class_ids = list(range(len(TARGET_CLASSES)))
            report = classification_report(
                y_test,
                y_pred,
                labels=class_ids,
                target_names=TARGET_CLASSES,
                output_dict=True,
                zero_division=0,
            )

            # Export the detailed classification report to CSV.
            pd.DataFrame(report).transpose().to_csv(
                RESULTS_DIR
                / f"classification_report_{name.lower().replace(' ', '_')}.csv"
            )

        except Exception as exc:
            raise RuntimeError(f"Training or evaluation failed for {name}.") from exc

    # Tune the top baseline candidates selected by training-set CV Macro F1.
    tuned = tune_top_models(results, X_train, y_train, X_test, y_test, cv)
    for item in tuned:
        fitted[item["model"]] = item["best_estimator"]
        results.append({
            key: value
            for key, value in item.items()
            if key not in {"best_estimator", "best_params"}
        })
        save_confusion_matrix(
            item["model"],
            y_test,
            item["best_estimator"].predict(X_test),
        )
        save_feature_ranking(
            item["best_estimator"],
            output_name=f"feature_ranking_{_model_slug(item['model'])}.csv",
        )

    # Rank by training-only CV Macro F1. Test metrics are not used to select.
    all_results = pd.DataFrame(results).sort_values(
        ["cv_macro_f1", "macro_f1", "accuracy", "log_loss"],
        ascending=[False, False, False, True],
        kind="stable",
    )

    # Select the highest-CV-scoring baseline or tuned candidate. The held-out
    # test set remains for evaluation and does not determine model selection.
    best_row = all_results.loc[all_results["cv_macro_f1"].idxmax()]
    best_name = best_row["model"]
    all_results = (
        all_results.assign(_is_deployed=all_results["model"].eq(best_name))
        .sort_values(
            ["_is_deployed", "cv_macro_f1", "macro_f1", "accuracy", "log_loss"],
            ascending=[False, False, False, False, True],
            kind="stable",
        )
        .drop(columns="_is_deployed")
    )

    # Find the name and fitted pipeline of the chosen deployment model.
    best_pipeline = fitted[best_name]

    # Save the complete fitted pipeline, including preprocessing and feature selection.
    joblib.dump(best_pipeline, BEST_MODEL_FILE)

    # Save the final model's feature rankings for later inspection.
    ranking = save_feature_ranking(best_pipeline)

    # Export one bar chart for every baseline and tuned model.
    chart_files = {}
    for _, result_row in all_results.iterrows():
        chart_path = save_model_metric_chart(result_row)
        chart_files[str(result_row["model"])] = chart_path.name

    # Build a compact model card containing details the application may display.
    metadata = {
        "project_title": "AI-Powered Obesity Risk Assessment and Prediction System",
        "target_column": TARGET_COLUMN,
        "classes": TARGET_CLASSES,
        "class_ids": list(range(len(TARGET_CLASSES))),
        "raw_target_values": list(TARGET_RAW_TO_ID.keys()),
        "best_model": str(best_name),

        # Convert NumPy or pandas numeric values into standard Python floats
        # so they can be serialized to JSON.
        "best_metrics": {
            "accuracy": float(best_row["accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "log_loss": float(best_row["log_loss"]),
            "cv_macro_f1": float(best_row["cv_macro_f1"]),
        },

        "shap_compatible": bool(best_row["shap_compatible"]),

        # Store only the features that the fitted selector chose to keep.
        "selected_features": ranking.loc[
            ranking["selected"], "feature"
        ].tolist(),
        "n_selected_features": int(ranking["selected"].sum()),
        "model_metric_charts": chart_files,

        # Record training settings to help reproduce the experiment.
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "smote_inside_pipeline": True,

        # Describe the feature-selection method and dataset used.
        "selection_method": (
            "Model-specific Hybrid Borda count combining Chi-Square, "
            "ANOVA F-test, RFE and SHAP"
        ),
        "per_model_feature_selection": True,
        "dataset": (
            "UCI Estimation of Obesity Levels Based On Eating Habits "
            "and Physical Condition"
        ),
        "tuned_models": [str(item["model"]) for item in tuned],
        "top_two_candidates": [str(item["base_model"]) for item in tuned],
        "tuned_best_params": {
            str(item["model"]): item["best_params"] for item in tuned
        },
        "tuned_baseline_cv_macro_f1": {
            str(item["model"]): float(item["baseline_cv_macro_f1"])
            for item in tuned
        },
        "deployment_selection_metric": (
            "highest cross-validation macro F1 among baseline and tuned models"
        ),
    }

    # Write the model metadata in an easy-to-read JSON format.
    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Save the complete comparison table with the deployed model first.
    all_results.to_csv(COMPARISON_FILE, index=False)

    # Print a summary of the selected model and where its files were saved.
    print("\nFinal model selection")
    print(f"Best deployable model: {best_name}")
    print(json.dumps(metadata["best_metrics"], indent=2))
    print("\nSelected features:")

    for feature in metadata["selected_features"]:
        print(f" - {feature}")

    print(f"\nSaved: {BEST_MODEL_FILE}")
    print(f"Saved: {METADATA_FILE}")
    print(f"Saved: {COMPARISON_FILE}")


# Run the training process only when this file is executed as a script,
# rather than when another Python file imports it.
if __name__ == "__main__":
    # Allow the user to provide a custom dataset path from the command line.
    parser = argparse.ArgumentParser(
        description="Train the obesity classification system."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="Path to the CSV dataset",
    )

    # Read the command-line arguments and start training.
    args = parser.parse_args()
    main(args.data)