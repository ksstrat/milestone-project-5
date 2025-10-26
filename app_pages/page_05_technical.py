import streamlit as st

def page_technical():
    st.title("Technical Performance & Pipeline")
    st.info("This page will show learning curves, test metrics, confusion matrix, and a clear 'Meets ≥97%' statement.")
    st.caption("Artifacts will be loaded from versioned folders (e.g., artifacts/v1/plots, reports).")