import streamlit as st
from src.paths import PROJECT_ROOT, DATA_DIR, MANIFESTS_DIR, PLOTS_DIR, ARTIFACTS_DIR
from app_pages.page_01_summary import page_summary
from app_pages.page_02_visual_study import page_visual_study
from app_pages.page_03_prediction import page_prediction
from app_pages.page_04_hypotheses import page_hypotheses
from app_pages.page_05_technical import page_technical

# --- Streamlit page setup ---
st.set_page_config(
    page_title="Mildew Detection in Cherry Leaves",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Project Summary",
        "Visual Analysis",
        "Prediction (BR2)",
        "Hypotheses",
        "ML Performance Metrics",
    ],
    index=0,
)

# --- Page routing ---
if page == "Project Summary":
    page_summary()
elif page == "Visual Analysis":
    page_visual_study()
elif page == "Prediction (BR2)":
    page_prediction()
elif page == "Hypotheses":
    page_hypotheses()
elif page == "ML Performance Metrics":
    page_technical()
else:
    st.error("Unknown page selection.")

# --- Sidebar footer info ---
st.sidebar.markdown("---")
st.sidebar.caption("Cherry Leaves Project · 2025")