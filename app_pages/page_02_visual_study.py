import streamlit as st
from pathlib import Path
from PIL import Image
from src.paths import PLOTS_DIR

PLOTS_V1_DIR = PLOTS_DIR

def _img(p: Path, caption: str | None = None, use_container_width: bool = True):
    if not p.exists():
        st.warning(f"File not found: `{p.as_posix()}`")
        return
    try:
        st.image(Image.open(p), caption=caption, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Failed to load {p.name}: {e}")

def render():
    st.header("Visual Study (BR1)")
    st.write(
        "This page renders the exploratory visuals used to justify Business Requirement 1 "
        "(visual differentiation between **healthy** and **powdery mildew** leaves). "
        "All figures were precomputed in the notebooks and saved under `plots/v1/`."
    )

    st.subheader("Class Averages")
    c1, c2 = st.columns(2)
    with c1:
        _img(PLOTS_V1_DIR / "avg_healthy.png", "Average — Healthy")
    with c2:
        _img(PLOTS_V1_DIR / "avg_powdery_mildew.png", "Average — Powdery Mildew")

    st.divider()
    st.subheader("Per-Pixel Variability (Std)")
    c1, c2 = st.columns(2)
    with c1:
        _img(PLOTS_V1_DIR / "var_healthy.png", "Variability — Healthy")
    with c2:
        _img(PLOTS_V1_DIR / "var_powdery_mildew.png", "Variability — Powdery Mildew")

    st.divider()
    st.subheader("Class Difference Map")
    _img(PLOTS_V1_DIR / "diff_classes.png", "Absolute Mean Difference | Healthy − Mildew |")

    st.divider()
    st.subheader("Image Montages")
    c1, c2 = st.columns(2)
    with c1:
        _img(PLOTS_V1_DIR / "montage_healthy.png", "Montage — Healthy")
    with c2:
        _img(PLOTS_V1_DIR / "montage_powdery_mildew.png", "Montage — Powdery Mildew")

    st.divider()
    st.subheader("Normalized RGB Histograms")
    c1, c2 = st.columns(2)
    with c1:
        _img(PLOTS_V1_DIR / "hist_rgb_healthy.png", "RGB Histogram — Healthy")
    with c2:
        _img(PLOTS_V1_DIR / "hist_rgb_powdery_mildew.png", "RGB Histogram — Powdery Mildew")

    st.caption("If a figure is missing, run the corresponding notebook cells to regenerate it.")

def page_visual_study():
    """Alias for the page entrypoint expected by app.py."""
    render()