# Allows Python to evaluate type annotations lazily, which helps with forward references.
from __future__ import annotations

# Imports pandas for handling and manipulating dataset tables.
import pandas as pd

# Imports ColumnTransformer to apply different preprocessing methods to different columns.
from sklearn.compose import ColumnTransformer

# Imports Pipeline to combine multiple preprocessing steps into a single workflow.
from sklearn.pipeline import Pipeline

# Imports encoders to convert categorical data into numerical representations.
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Imports SimpleImputer to fill missing values in the dataset.
from sklearn.impute import SimpleImputer

# Imports feature lists and category definitions from the config.py.
from .config import (
    BINARY_CATEGORIES,      # Defines the allowed categories for binary features.
    BINARY_FEATURES,        # Lists features with two possible categories.
    CONTINUOUS_FEATURES,    # Lists numerical features.
    NOMINAL_FEATURES,       # Lists categorical features without a natural order.
    ORDINAL_CATEGORIES,     # Defines the ordered categories for ordinal features.
    ORDINAL_FEATURES,       # Lists categorical features with a natural order.
)


# Cleans column names and text values to prevent problems caused by extra whitespace.
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values."""

    # Creates a copy so the original DataFrame is not modified.
    out = df.copy()

    # Removes leading and trailing whitespace from every column name.
    out.columns = [str(c).strip() for c in out.columns]

    # Checks each column to identify columns containing text data.
    for col in out.columns:

        # Cleans text values by converting them to strings and removing surrounding whitespace.
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()

    # Returns the cleaned DataFrame.
    return out


# Checks whether the dataset contains all the columns required by the system.
def validate_dataset(df: pd.DataFrame, target_column: str) -> None:

    # Combines all expected input features and the target column into a set.
    required = set(
        CONTINUOUS_FEATURES
        + BINARY_FEATURES
        + ORDINAL_FEATURES
        + NOMINAL_FEATURES
        + [target_column]
    )

    # Identifies required columns that are missing from the dataset.
    missing = sorted(required - set(df.columns))

    # Stops processing and reports the missing columns if any are found.
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")


# Builds a preprocessing pipeline that prepares different feature types for machine learning.
def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline described in the proposal."""

    # Creates a pipeline for numerical features.
    continuous_pipe = Pipeline(
        steps=[
            # Replaces missing numerical values with the median of each feature.
            ("imputer", SimpleImputer(strategy="median")),

            # Standardizes numerical values to have approximately zero mean and unit variance.
            ("scaler", StandardScaler()),
        ]
    )

    # Creates a pipeline for binary categorical features.
    binary_pipe = Pipeline(
        steps=[
            # Replaces missing categorical values with the most frequent category.
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # Converts binary categories into numerical values based on their defined order.
            (
                "encoder",
                OrdinalEncoder(
                    categories=BINARY_CATEGORIES,  # Specifies the allowed categories for each feature.
                    handle_unknown="use_encoded_value",  # Handles categories not seen during training.
                    unknown_value=-1,  # Encodes unrecognized categories as -1.
                ),
            ),
        ]
    )

    # Creates a pipeline for categorical features that have a meaningful order.
    ordinal_pipe = Pipeline(
        steps=[
            # Fills missing ordinal values with the most frequent category.
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # Converts ordered categories into numerical values while preserving their order.
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,  # Defines the order of categories for each feature.
                    handle_unknown="use_encoded_value",  # Allows previously unseen categories to be handled.
                    unknown_value=-1,  # Assigns -1 to unrecognized categories.
                ),
            ),
        ]
    )

    # Creates a pipeline for categorical features without a natural order.
    nominal_pipe = Pipeline(
        steps=[
            # Fills missing nominal values with the most frequent category.
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # Converts categories into separate binary columns using one-hot encoding.
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",  # Ignores categories not encountered during training.
                    sparse_output=False,  # Returns a regular array instead of a sparse matrix.
                    dtype=float,  # Stores encoded values as floating-point numbers.
                ),
            ),
        ]
    )

    # Combines the four preprocessing pipelines and applies each to its designated columns.
    return ColumnTransformer(
        transformers=[
            # Applies imputation and scaling to continuous numerical features.
            ("continuous", continuous_pipe, CONTINUOUS_FEATURES),

            # Applies imputation and ordinal encoding to binary features.
            ("binary", binary_pipe, BINARY_FEATURES),

            # Applies imputation and ordinal encoding to ordered categorical features.
            ("ordinal", ordinal_pipe, ORDINAL_FEATURES),

            # Applies imputation and one-hot encoding to nominal categorical features.
            ("nominal", nominal_pipe, NOMINAL_FEATURES),
        ],

        # Excludes columns that are not explicitly assigned to a transformer.
        remainder="drop",

        # Uses the original feature names instead of adding transformer prefixes to them.
        verbose_feature_names_out=False,
    )