from __future__ import annotations

# json reads the saved model metadata from a JSON file.
import json

from pathlib import Path
from typing import Dict, Tuple

# joblib loads the trained machine-learning pipeline from disk.
import joblib

# Pandas converts user input into a table the model can process.
import pandas as pd

# Import the default saved model, metadata paths, and class names.
from .config import BEST_MODEL_FILE, METADATA_FILE, TARGET_CLASSES


def _risk_group(class_id: int) -> str:
    if class_id == 0:
        return "Low risk"
    if class_id == 1:
        return "No risk (normal weight)"
    return "High risk"


# Load the trained model and its metadata so the application can make predictions.
def load_artifacts(
    model_path: Path = BEST_MODEL_FILE,
    metadata_path: Path = METADATA_FILE,
):
    # Load the complete trained pipeline, including preprocessing and feature selection.
    model = joblib.load(model_path)

    # Read the model's saved information, such as its name and evaluation metrics.
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    # Return both objects so other functions can use them.
    return model, metadata


# Predict the obesity-risk category for one person's input data.
def predict_one(model, input_data: Dict) -> Dict:
    # Convert the user's answers into a one-row DataFrame.
    # The input keys must match the feature columns expected by the pipeline.
    df = pd.DataFrame([input_data])

    # Predict the category ID, such as 0 for Insufficient Weight.
    prediction_id = int(model.predict(df)[0])

    # Get the model's predicted probability for each class.
    probabilities = model.predict_proba(df)[0]

    # Retrieve the class IDs in the same order as the probability values.
    classes = [
        int(cls)
        for cls in model.named_steps["model"].classes_
    ]

    # Pair each human-readable category name with its predicted probability.
    probability_map = {
        TARGET_CLASSES[cls]: float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    # Group the seven output categories into the risk labels shown in the website.
    risk_probabilities = {
        "Low risk": probability_map[TARGET_CLASSES[0]],
        "High risk": float(sum(probability_map[label] for label in TARGET_CLASSES[2:])),
        "No risk (normal weight)": probability_map[TARGET_CLASSES[1]],
    }
    risk_level = _risk_group(prediction_id)

    # Return the prediction and supporting information for the application.
    return {
        "prediction_id": prediction_id,

        # Convert the predicted ID into a readable category name.
        "prediction": TARGET_CLASSES[prediction_id],

        "risk_level": risk_level,
        "risk_probability": risk_probabilities[risk_level],
        "risk_probabilities": risk_probabilities,

        # Probabilities for all obesity categories.
        "probabilities": probability_map,

        # Keep the original input so the application can display or review it.
        "input": input_data,
    }


# Transform the user's input into the selected feature values used by the model.
def transformed_selected_input(
    model,
    input_data: Dict,
) -> Tuple[pd.DataFrame, list]:
    """Return the final feature vector entering the deployed estimator."""

    # Convert the input dictionary into a one-row DataFrame.
    df = pd.DataFrame([input_data])

    # Retrieve the fitted preprocessing and feature-selection steps.
    preprocessor = model.named_steps["preprocess"]
    selector = model.named_steps["feature_select"]

    # Apply the same encoding and scaling learned during model training.
    X_transformed = preprocessor.transform(df)

    # Get the names of the features produced by preprocessing.
    all_feature_names = preprocessor.get_feature_names_out()

    # Give the selector the feature names so it can identify selected columns.
    selector.set_feature_names(all_feature_names)

    # Keep only the features chosen by the trained feature selector.
    X_selected = selector.transform(X_transformed)

    # Retrieve the names corresponding to those selected features.
    selected_names = selector.get_feature_names_out(
        all_feature_names
    ).tolist()

    # Return the selected values as a named DataFrame and the feature names.
    # This is useful when displaying explanations or feature values in the app.
    return pd.DataFrame(
        X_selected,
        columns=selected_names,
    ), selected_names