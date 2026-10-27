from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import BEST_MODEL_FILE, METADATA_FILE
from src.predict import load_artifacts


@st.cache_resource(show_spinner=False)
def load_system():
    return load_artifacts(BEST_MODEL_FILE, METADATA_FILE)


def show_header() -> None:
    st.title("AI-powered obesity risk assessment")
    st.caption("An interactive classification and explainability application")


def disclaimer() -> None:
    st.warning(
        "This application provides an educational prediction based on the information "
        "entered. It is not a medical diagnosis or a substitute for professional "
        "healthcare advice. SHAP explanations describe model associations, not direct "
        "medical causation.",
        title="Important",
    )


def show_result(result: dict) -> None:
    st.subheader("Prediction")
    result_col, chart_col = st.columns([1, 2])
    with result_col:
        st.metric("Predicted category", result["prediction"])
        st.metric("Model risk group", result["risk_level"])
    with chart_col:
        risk_frame = pd.DataFrame(
            {"Probability": result["risk_probabilities"]}
        )
        st.bar_chart(
            risk_frame,
            alt="Probability grouped into low risk, high risk, and normal-weight categories",
        )
    st.caption(
        "The risk group is mapped from the predicted category. Bars aggregate model "
        "probabilities: insufficient weight is Low risk, normal weight is No risk, "
        "and overweight/obesity categories are High risk. These are not clinical "
        "risk estimates."
    )


def show_no_assessment_message() -> bool:
    result = st.session_state.get("assessment_result")
    if result is None:
        st.info("Complete a risk assessment first.")
        return True
    return False
