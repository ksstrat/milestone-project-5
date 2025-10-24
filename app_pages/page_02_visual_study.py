import streamlit as st

def page_visual_study():
    st.title("Visual Study (BR1)")
    st.info("This page will display class-wise average, variability, mean-difference, and montage images.")
    st.caption("Plots will be loaded from the exported files (e.g., plots/v1/*.png) once generated.")