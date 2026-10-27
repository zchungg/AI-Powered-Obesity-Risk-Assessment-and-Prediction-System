from __future__ import annotations

import streamlit as st

from src.config import TARGET_CLASSES
from website_app.shared import disclaimer


def render(metadata: dict) -> None:
    st.header("About the System")
    st.write(
        "The system follows the proposed workflow: dataset ingestion → preprocessing → "
        "stratified train/test split → SMOTE within the training pipeline → "
        "model-specific hybrid Borda feature selection → cross-validated "
        "hyperparameter tuning → model comparison → "
        "SHAP explanation → Streamlit interface."
    )
    st.subheader("Prediction target")
    st.write(", ".join(TARGET_CLASSES))
    st.subheader("Feature selection")
    st.write(
        "Each candidate model gets its own feature rankings. Chi-Square, ANOVA F-test, "
        "RFE and SHAP rankings are combined with Borda-count voting."
    )
    st.subheader("Limitations")
    st.write(
        "The underlying UCI dataset is a classification dataset based on eating habits "
        "and physical condition. It includes self-reported behavioural information and "
        "a mixture of real and synthetic records, so this system is not a clinical "
        "diagnostic tool."
    )
    st.subheader("Current deployed model")
    st.json(metadata)
    disclaimer()
