from __future__ import annotations

import pandas as pd
import streamlit as st

from website_app.shared import disclaimer, show_no_assessment_message, show_result


def render() -> None:
    if show_no_assessment_message():
        return
    result = st.session_state["assessment_result"]
    from src.explain import pretty_feature_name

    st.header("Result Summary")
    show_result(result)
    st.subheader("Key model contributors")
    contributions = pd.DataFrame(result["contributions"]).head(5)
    for _, row in contributions.iterrows():
        label = pretty_feature_name(row["feature"])
        value = float(row["shap_value"])
        direction = "Increased model contribution" if value >= 0 else "Reduced model contribution"
        st.write(f"**{label}** — {direction} ({value:+.4f})")
    st.subheader("Lifestyle suggestions")
    for recommendation in result["recommendations"]:
        st.write(f"**{recommendation['area']}**: {recommendation['text']}")
    disclaimer()
