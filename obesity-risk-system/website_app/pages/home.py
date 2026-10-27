from __future__ import annotations

import streamlit as st

from src.config import TARGET_CLASSES
from website_app.shared import disclaimer


def render(metadata: dict) -> None:
    st.header("Home")
    st.write(
        "This system estimates one of seven obesity/weight categories using demographic, "
        "physical, dietary, and lifestyle information. It also provides a local SHAP "
        "explanation for the model prediction."
    )
    columns = st.columns(3)
    columns[0].metric("Deployed model", metadata.get("best_model", "Not trained"))
    columns[1].metric("Selected features", metadata.get("n_selected_features", "—"))
    columns[2].metric("Dataset classes", len(metadata.get("classes", TARGET_CLASSES)))
    st.subheader("Assessment flow")
    st.write(
        "1. Enter personal and lifestyle information → 2. Generate prediction → "
        "3. Review the risk group → 4. View SHAP explanation → "
        "5. Review summary and suggestions"
    )
    disclaimer()
