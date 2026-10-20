"""Heart Disease Lab dashboard entrypoint and navigation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import streamlit as st

from heart_disease.config import MODEL_PATH, REPORT_DIR, UCI_PAGE
from heart_disease.logging_config import configure_logging, track_dashboard_errors
from heart_disease.prediction import metadata_path
from heart_disease.ui.common import json_file
from heart_disease.ui.evaluation import evaluation_page
from heart_disease.ui.pages import about_page, explainability_page, overview
from heart_disease.ui.prediction import prediction_page

st.set_page_config(
    page_title="Heart Disease Lab", page_icon="♥", layout="wide", initial_sidebar_state="auto"
)
st.markdown(
    """
<style>
.block-container {max-width: 1260px; padding-top: 3.8rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.045em; font-weight: 750 !important;}
h2, h3 {letter-spacing: -0.02em;}
[data-testid="stSidebar"] {border-right: 1px solid #E1E7F0;}
[data-testid="stMetric"] {background: white; border: 1px solid #E1E7F0;
    border-radius: 12px; padding: 18px 20px;}
[data-testid="stMetricLabel"] {color: #536987;}
[data-testid="stMetricValue"] {font-weight: 650; color: #172B4D;}
.eyebrow {color: #2463EB; font-size: 0.77rem; font-weight: 700;
    letter-spacing: 0.14em; margin-bottom: 0.5rem;}
.brand {font-size: 1.42rem; font-weight: 750; letter-spacing: -0.04em; margin: 0.5rem 0;}
.subtle {color: #64748B; font-size: 0.86rem;}
div[data-testid="stForm"] {background: white; border-color: #E1E7F0; border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)


def main():
    configure_logging()
    metadata = json_file(metadata_path(MODEL_PATH))
    report = json_file(REPORT_DIR / "evaluation_report.json")
    with st.sidebar:
        st.markdown(
            '<div class="brand">♥ &nbsp; Heart Disease Lab</div><div class="subtle">Data science · Educational research</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "Workspace",
            [
                "Project Overview",
                "Patient Input & Prediction",
                "Model Evaluation",
                "Explainability",
                "About & Limitations",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        if metadata and MODEL_PATH.exists():
            st.markdown("**Saved experiment**")
            st.caption(metadata.get("selected_model", "Unknown model"))
            st.caption(f"Seed 42 · Run {metadata.get('run_id', 'unknown')[:8]}")
        else:
            st.caption("Awaiting a real-data training run")
        st.markdown(f"[Dataset documentation]({UCI_PAGE})")
        st.caption("Local inference. No external prediction API.")
    if page == "Project Overview":
        overview(report, metadata)
    elif page == "Patient Input & Prediction":
        prediction_page()
    elif page == "Model Evaluation":
        evaluation_page(report, metadata)
    elif page == "Explainability":
        explainability_page(report, metadata)
    else:
        about_page(report, metadata)


if __name__ == "__main__":
    configure_logging()
    with track_dashboard_errors():
        main()
