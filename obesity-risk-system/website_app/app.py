from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit may put this entrypoint's folder, rather than the project root, on
# sys.path. Add the root so the app and its page modules can import src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_FILE, METADATA_FILE
from website_app.pages import about, assessment, explanation, home, model_comparison, summary
from website_app.shared import load_system, show_header


def main() -> None:
    st.set_page_config(
        page_title="AI Obesity Risk Assessment",
        page_icon=":material/health_and_safety:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    show_header()

    if not BEST_MODEL_FILE.exists() or not METADATA_FILE.exists():
        st.warning("The trained model has not been generated yet.")
        st.code("python download_data.py\npython -m src.train", language="bash")
        st.stop()

    model, metadata = load_system()
    page = st.navigation(
        [
            st.Page(
                lambda: home.render(metadata),
                title="Home",
                icon=":material/home:",
                url_path="home",
                default=True,
            ),
            st.Page(
                lambda: assessment.render(model),
                title="Risk assessment",
                icon=":material/monitor_heart:",
                url_path="risk-assessment",
            ),
            st.Page(
                explanation.render,
                title="Why this prediction?",
                icon=":material/lightbulb:",
                url_path="explanation",
            ),
            st.Page(
                summary.render,
                title="Summary",
                icon=":material/summarize:",
                url_path="summary",
            ),
            st.Page(
                lambda: about.render(metadata),
                title="About system",
                icon=":material/info:",
                url_path="about",
            ),
        ],
        position="sidebar",
    )
    page.run()


if __name__ == "__main__":
    main()
