from __future__ import annotations

# warnings displays a warning if the system cannot generate a SHAP explanation.
import warnings

from typing import Dict, Tuple

# NumPy handles numerical arrays and SHAP value calculations.
import numpy as np

# Pandas stores feature contributions in a readable table.
import pandas as pd

# SHAP explains how individual features influence a model's predictions.
import shap


# Extract SHAP values and the base value for one predicted class.
def _prediction_class_shap(
    explainer,
    X_selected: pd.DataFrame,
    class_index: int,
):
    # Generate SHAP explanations for the supplied input features.
    explanation = explainer(X_selected)

    # Convert the SHAP values into a NumPy array for easier processing.
    values = np.asarray(explanation.values)

    # The base value represents the starting point of the SHAP explanation.
    base = explanation.base_values

    # Handle SHAP output shaped as (samples, features, classes).
    if values.ndim == 3:
        # Select the first sample and the requested class's feature contributions.
        shap_values = values[0, :, class_index]

        # Extract the base value for the same sample and class.
        if np.ndim(base) == 2:
            base_value = float(np.asarray(base)[0, class_index])
        elif np.ndim(base) == 1:
            base_value = float(np.asarray(base)[class_index])
        else:
            # Use the first available base value for other output shapes.
            base_value = float(np.ravel(base)[0])

        return shap_values, base_value

    # Handle SHAP output shaped as (samples, features), such as a
    # single-output explanation.
    if values.ndim == 2:
        shap_values = values[0, :]
        base_value = float(np.ravel(base)[0])
        return shap_values, base_value

    # Stop if SHAP returns a format this function does not support.
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


# Explain why the model predicted a particular class for one input.
def explain_prediction(
    pipeline,
    input_df: pd.DataFrame,
    predicted_class: int,
) -> Tuple[pd.DataFrame, float, str]:
    """Return feature contributions, a base value, and the method used."""

    # Retrieve the fitted preprocessing, feature-selection, and model steps.
    preprocessor = pipeline.named_steps["preprocess"]
    selector = pipeline.named_steps["feature_select"]
    estimator = pipeline.named_steps["model"]

    # Apply the same preprocessing used during training.
    X_transformed = preprocessor.transform(input_df)

    # Get the names of the features created by preprocessing.
    all_names = preprocessor.get_feature_names_out()

    # Provide feature names to the selector for identifying selected features.
    selector.set_feature_names(all_names)

    # Keep only the features selected by the trained feature selector.
    X_selected = selector.transform(X_transformed)

    # Retrieve the names of the selected features.
    selected_names = selector.get_feature_names_out(all_names).tolist()

    # Create a DataFrame so SHAP explanations retain readable feature names.
    X_selected_df = pd.DataFrame(
        X_selected,
        columns=selected_names,
    )

    # Get the model's class order and find the index of the predicted class.
    classes = [
        int(cls)
        for cls in pipeline.named_steps["model"].classes_
    ]
    class_index = classes.index(int(predicted_class))

    # Record which explanation method was used.
    shap_method = "SHAP"

    try:
        # TreeExplainer is designed for tree-based models such as Random Forest
        # and supported gradient-boosting models.
        explainer = shap.TreeExplainer(estimator)

        # Extract the selected class's feature contributions and base value.
        values, base_value = _prediction_class_shap(
            explainer,
            X_selected_df,
            class_index,
        )

    except Exception:
        try:
            # If TreeExplainer fails, try a more general SHAP explainer.
            # This can support more model types but may be slower.
            background = X_selected_df.copy()

            # Limit the background data to reduce explanation time.
            if len(background) > 40:
                background = background.iloc[:40].copy()

            # Explain the estimator's predicted probabilities using the
            # selected input features as background data.
            generic_explainer = shap.Explainer(
                estimator.predict_proba,
                background,
            )

            # Calculate SHAP values, allowing enough evaluations for the
            # number of selected features. The explainer may require more.
            explanation = generic_explainer(
                X_selected_df,
                max_evals=max(200, 2 * len(selected_names) + 1),
            )

            values = np.asarray(explanation.values)

            # Handle output shaped as (samples, features, classes).
            if values.ndim == 3:
                # Extract the first sample's contributions for the predicted class.
                values = values[0, :, class_index]

                # Retrieve the corresponding base value when available.
                base_arr = np.asarray(explanation.base_values)
                if base_arr.ndim == 2:
                    base_value = float(base_arr[0, class_index])
                else:
                    base_value = float(np.ravel(base_arr)[0])

            # Handle output shaped as (samples, features).
            elif values.ndim == 2:
                values = values[0]
                base_value = float(
                    np.ravel(explanation.base_values)[0]
                )

            else:
                # Raise an error if the generic explainer returns an
                # unsupported array shape.
                raise ValueError(
                    f"Unsupported generic SHAP shape: {values.shape}"
                )

            # Mark that the general-purpose explainer was used.
            shap_method = "SHAP (generic fallback)"

        except Exception as exc:
            # Warn the developer if both explanation methods fail.
            warnings.warn(f"SHAP explanation failed: {exc}")

            # Mark the explanation as unavailable rather than crashing the app.
            shap_method = "Unavailable"

            # Return zero contributions as placeholders.
            # These zeros are NOT genuine SHAP explanations.
            values = np.zeros(len(selected_names), dtype=float)
            base_value = 0.0

    # Combine feature names, SHAP values, and absolute impact sizes into a table.
    result = pd.DataFrame({
        "feature": selected_names,

        # Positive or negative values indicate the direction of contribution
        # relative to the explanation's output scale.
        "shap_value": values,

        # Absolute values show contribution magnitude without its direction.
        "absolute_shap": np.abs(values),
    })

    # Sort features from largest to smallest absolute contribution.
    result = result.sort_values(
        "absolute_shap",
        ascending=False,
    )

    # Return the ranked table, base value, and explanation method.
    return result.reset_index(drop=True), float(base_value), shap_method


# Convert raw dataset column names into clearer labels for the application UI.
def pretty_feature_name(name: str) -> str:

    # Map technical feature names to user-friendly descriptions.
    mapping = {
        "family_history_with_overweight": "Family history of overweight",
        "FAVC": "High-calorie food consumption",
        "FCVC": "Vegetable consumption",
        "NCP": "Main meals per day",
        "CAEC": "Food between meals",
        "SMOKE": "Smoking",
        "CH2O": "Water consumption",
        "SCC": "Calorie-monitoring behaviour",
        "FAF": "Physical activity frequency",
        "TUE": "Technology usage time",
        "CALC": "Alcohol consumption",
        "MTRANS": "Transport",
        "Gender": "Gender",
        "Age": "Age",
        "Height": "Height",
        "Weight": "Weight",
    }

    # Check whether the feature matches a known name or an encoded category.
    for key, label in mapping.items():

        # Return the general label if the feature matches exactly.
        if name == key:
            return label

        # Add the category value for one-hot encoded features.
        # Example: MTRANS_Public_Transportation becomes
        # "Transport: Public Transportation".
        if name.startswith(key + "_"):
            return (
                f"{label}: "
                f"{name[len(key) + 1:].replace('_', ' ')}"
            )

    # For unmapped features, replace underscores with spaces.
    return name.replace("_", " ")


# Convert one feature's SHAP value into a user-friendly explanation.
def format_contribution(
    feature: str,
    shap_value: float,
) -> Dict[str, str]:

    # Label the direction according to the sign of the SHAP value.
    # Positive and negative directions depend on the model output being explained.
    direction = (
        "Increased Risk Contribution"
        if shap_value >= 0
        else "Reduced Risk Contribution"
    )

    return {
        # Display the feature using a readable name.
        "feature": pretty_feature_name(feature),

        # Store the direction and numerical contribution.
        "direction": direction,
        "shap_value": float(shap_value),

        # Create a short explanation that the application can display.
        "impact_text": (
            "This feature pushed the model more toward the predicted category."
            if shap_value >= 0
            else "This feature pushed the model away from the predicted category."
        ),
    }