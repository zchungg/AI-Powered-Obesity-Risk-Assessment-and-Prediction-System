from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src.config import VALIDATION_RANGES
from src.predict import predict_one
from src.recommendations import generate_recommendations
from website_app.shared import show_result


def _assessment_form() -> Optional[dict]:
    with st.form("assessment_form"):
        st.subheader("1. Personal information")
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        age = col2.number_input("Age (years)", 10.0, 100.0, 22.0, 1.0)
        family_history = col1.selectbox("Family history with overweight", ["no", "yes"])
        smoke = col2.selectbox("Do you smoke?", ["no", "yes"])

        st.subheader("2. Physical information")
        col1, col2 = st.columns(2)
        height = col1.number_input("Height (metres)", 1.20, 2.20, 1.70, 0.01, format="%.2f")
        weight = col2.number_input("Weight (kg)", 25.0, 250.0, 65.0, 0.5, format="%.1f")
        faf = col1.number_input("Physical activity frequency (0–5)", 0.0, 5.0, 1.0, 0.1, format="%.1f")
        tue = col2.number_input("Technology usage time (hours/day)", 0.0, 12.0, 2.0, 0.1, format="%.1f")

        st.subheader("3. Dietary habits")
        col1, col2 = st.columns(2)
        favc = col1.selectbox("Frequent high-calorie food consumption", ["no", "yes"])
        fcvc = col2.number_input("Vegetable consumption frequency (1–5)", 1.0, 5.0, 2.0, 0.1, format="%.1f")
        ncp = col1.number_input("Number of main meals per day", 1.0, 8.0, 3.0, 0.5, format="%.1f")
        caec = col2.selectbox("Food consumption between meals", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = col1.number_input("Daily water consumption (relative scale)", 0.0, 8.0, 2.0, 0.1, format="%.1f")
        scc = col2.selectbox("Calorie consumption monitoring", ["no", "yes"])

        st.subheader("4. Lifestyle & activities")
        col1, col2 = st.columns(2)
        calc = col1.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = col2.selectbox(
            "Main mode of transportation",
            ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
        )
        submitted = st.form_submit_button("Generate prediction", width="stretch")

    if not submitted:
        return None

    return {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
    }


def render(model) -> None:
    st.header("Risk Assessment")
    st.write(
        "Complete the four sections below. No personally identifiable information "
        "is required by this application."
    )
    raw = _assessment_form()
    if raw is None:
        return

    for field, (low, high) in VALIDATION_RANGES.items():
        value = float(raw[field])
        if not low <= value <= high:
            st.error(f"{field} must be between {low} and {high}.")
            return

    result = predict_one(model, raw)
    input_frame = pd.DataFrame([raw])
    from src.explain import explain_prediction

    contributions, base_value, shap_method = explain_prediction(
        model, input_frame, result["prediction_id"]
    )
    result["contributions"] = contributions.to_dict(orient="records")
    result["shap_method"] = shap_method
    result["base_value"] = base_value
    result["recommendations"] = generate_recommendations(raw)
    st.session_state["assessment_result"] = result
    st.session_state["assessment_input"] = raw
    st.success("Assessment completed.")
    show_result(result)
