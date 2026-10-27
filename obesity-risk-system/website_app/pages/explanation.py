from __future__ import annotations

import pandas as pd
import streamlit as st

from website_app.shared import show_no_assessment_message


def render() -> None:
    if show_no_assessment_message():
        return
    from src.explain import format_contribution

    result = st.session_state["assessment_result"]
    st.header("Why This Prediction?")
    st.write(
        "The table below shows local SHAP contributions for the predicted category. "
        "Positive values mean the feature moved the model toward the predicted category; "
        "negative values mean it moved the model away from it."
    )
    st.caption(f"Explanation method: {result.get('shap_method', 'SHAP')}")
    rows = []
    for item in result["contributions"]:
        feature = item["feature"]
        shap_value = float(item["shap_value"])
        formatted = format_contribution(feature, shap_value)
        rows.append(
            {
                "Feature": formatted["feature"],
                "Contribution": shap_value,
                "Direction": formatted["direction"],
                "Interpretation": formatted["impact_text"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows).head(10),
        hide_index=True,
        alt="Top SHAP contributions for the current prediction",
    )
    st.info(
        "SHAP values explain how the model used the supplied variables for this prediction. "
        "They are not evidence that a variable directly causes a health outcome."
    )
