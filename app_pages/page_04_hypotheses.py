from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image

from src.paths import ARTIFACTS_DIR, PLOTS_DIR, PROJECT_ROOT

# --- Define directories ---
FEATURES_DIR_V1 = PROJECT_ROOT / "inputs" / "features" / "v1"

PLOTS_V2_DIR = PLOTS_DIR.parent / "v2"  # H1 plots
PLOTS_V3_DIR = PLOTS_DIR.parent / "v3"  # H2 training curves
PLOTS_V4_DIR = PLOTS_DIR.parent / "v4"  # H2 comparison/CM
PLOTS_V5_DIR = PLOTS_DIR.parent / "v5"  # H3 comparison/CM

REP_V1 = ARTIFACTS_DIR / "v1" / "reports" / "evaluation_report.json"
REP_V2 = ARTIFACTS_DIR / "v2" / "reports" / "evaluation_report.json"
REP_V3 = ARTIFACTS_DIR / "v3" / "reports" / "evaluation_report.json"

CMP_V1_V2 = ARTIFACTS_DIR / "v2" / "reports" / "v1_vs_v2_comparison.csv"
CMP_V1_V3 = ARTIFACTS_DIR / "v3" / "reports" / "v1_vs_v3_mild_comparison.csv"


# --- Safe loaders ---
def _load_csv(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load CSV file and handle missing or invalid data."""
    try:
        if path.exists():
            return pd.read_csv(path, nrows=nrows)
        st.warning(f"Missing CSV: `{path.as_posix()}`")
    except Exception as e:
        st.error(f"Failed to read CSV `{path.name}`: {e}")
    return None


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON file and handle missing or invalid data."""
    try:
        if path.exists():
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        st.warning(f"Missing JSON: `{path.as_posix()}`")
    except Exception as e:
        st.error(f"Failed to read JSON `{path.name}`: {e}")
    return None


def _img(path: Path, caption: str | None = None):
    """Load and render image with optional caption."""
    if not path.exists():
        st.warning(f"Image not found: `{path.as_posix()}`")
        return
    try:
        st.image(Image.open(path), caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load image `{path.name}`: {e}")


# --- H1 section ---
def _section_h1():
    st.subheader("H1 — Texture & Color Variability (GLCM)")

    st.write(
        "We compare GLCM texture features between **healthy** and **powdery mildew** leaves. "
        "A non-parametric test (Mann-Whitney U) and a rank-based effect size (Cliff's Delta) "
        "quantify distribution differences."
    )

    cols = st.columns(2)
    with cols[0]:
        stats = _load_csv(FEATURES_DIR_V1 / "glcm_stats.csv")
        if stats is not None and not stats.empty:
            st.markdown("**Statistical Summary (p-values & effect sizes)**")
            st.dataframe(stats, use_container_width=True, hide_index=True)
    with cols[1]:
        means = _load_csv(FEATURES_DIR_V1 / "glcm_feature_means.csv")
        if means is not None and not means.empty:
            st.markdown("**Class-wise Means/Medians**")
            st.dataframe(means, use_container_width=True, hide_index=True)

    _img(PLOTS_V2_DIR / "glcm_boxplots.png", caption="GLCM Features — Boxplots by Class")

    st.caption(
        "Interpretation: Significant p-values with meaningful Cliff's Delta magnitudes indicate "
        "texture differences consistent with H1."
    )

    st.success(
        "Result: Statistical tests and effect sizes show significant texture differences between healthy "
        "and mildew-infected leaves. H1 is therefore **supported**."
    )


# --- H2 section ---
def _section_h2():
    st.subheader("H2 — Input Size Impact (100x100 → 50x50)")

    st.write(
        "Architecture and training remain identical to the baseline (v1). Only the input resolution "
        "is reduced to **50x50** (v2) to test whether accuracy ≥ 97% is maintained."
    )

    rep_v1 = _load_json(REP_V1)
    rep_v2 = _load_json(REP_V2)
    cmp_v1_v2 = _load_csv(CMP_V1_V2)

    cols = st.columns(2)
    with cols[0]:
        if rep_v1:
            st.metric("v1 (100x100) — Test Accuracy", f"{rep_v1.get('test_accuracy', float('nan')):.4f}")
        if rep_v2:
            st.metric("v2 (50x50) — Test Accuracy", f"{rep_v2.get('test_accuracy', float('nan')):.4f}")
    with cols[1]:
        if rep_v2:
            meets = rep_v2.get("meets_target", False)
            st.markdown(f"**Target ≥ 0.97 met by v2?**  {meets}")

    if cmp_v1_v2 is not None and not cmp_v1_v2.empty:
        st.markdown("**v1 vs v2 — Comparison Table**")
        st.dataframe(cmp_v1_v2, use_container_width=True, hide_index=True)

    cols2 = st.columns(2)
    with cols2[0]:
        _img(PLOTS_V4_DIR / "h2_accuracy_v1_vs_v2.png", caption="Accuracy Comparison — v1 vs v2")
    with cols2[1]:
        _img(PLOTS_V4_DIR / "confusion_matrix_test_v2.png", caption="Confusion Matrix — v2 Test Set")
    
    st.caption(
        "Interpretation: If the accuracy drop from 100x100 to 50x50 is minimal (≤1-2%) and remains above "
        "the 97% threshold, the hypothesis is supported. Larger drops indicate sensitivity to input size, "
        "thus not supporting H2."
    )

    st.success(
        "Result: Reducing input size from 100x100 to 50x50 maintained accuracy above 97%, "
        "indicating minimal performance loss. H2 is **supported**."
    )


# --- H3 section ---
def _section_h3():
    st.subheader("H3 — Data Augmentation Impact")

    st.write(
        "We enable lightweight, seeded augmentation for training (flip/rotation/brightness) while "
        "keeping validation/test unaltered. Architecture and training regime match v1."
    )

    rep_v1 = _load_json(REP_V1)
    rep_v3 = _load_json(REP_V3)
    cmp_v1_v3 = _load_csv(CMP_V1_V3)

    cols = st.columns(2)
    with cols[0]:
        if rep_v1:
            st.metric("v1 (no aug) — Test Accuracy", f"{rep_v1.get('test_accuracy', float('nan')):.4f}")
        if rep_v3:
            st.metric("v3 (aug) — Test Accuracy", f"{rep_v3.get('test_accuracy', float('nan')):.4f}")
    with cols[1]:
        if rep_v3:
            meets = rep_v3.get("meets_target", False)
            st.markdown(f"**Target ≥ 0.97 met by v3?**  {meets}")

    if cmp_v1_v3 is not None and not cmp_v1_v3.empty:
        st.markdown("**v1 vs v3 — Comparison Table**")
        st.dataframe(cmp_v1_v3, use_container_width=True, hide_index=True)

    cols2 = st.columns(2)
    with cols2[0]:
        _img(PLOTS_V5_DIR / "h3_accuracy_v1_vs_v3_mild.png", caption="Accuracy Comparison — v1 vs v3 (mild)")
    with cols2[1]:
        _img(PLOTS_V5_DIR / "confusion_matrix_test_v3_mild.png", caption="Confusion Matrix — v3 (mild) Test Set")

    st.caption(
        "Interpretation: If v3 is within noise of v1 (or slightly better) while remaining ≥ 97%, the augmentation is "
        "considered neutral-to-beneficial. If worse, H3 is not supported and augmentation should be reduced."
    )

    st.warning(
        "Result: Mild augmentation led to slightly reduced test accuracy (~92%) compared to the 99% baseline. "
        "H3 is **not supported** under the tested conditions."
    )


# --- Training curves ---
def _section_training_curves():
    st.subheader("Training Curves (Diagnostics)")

    st.write(
        "Auxiliary learning curves for transparency. These plots visualize the training and validation "
        "dynamics for each model version and help to interpret convergence behavior."
    )

    cols = st.columns(2)
    with cols[0]:
        _img(PLOTS_V3_DIR / "training_curves_v1.png", caption="v1 — Training & Validation")
    with cols[1]:
        _img(PLOTS_V3_DIR / "training_curves_v2.png", caption="v2 — Training & Validation")

    cols2 = st.columns(2)
    with cols2[0]:
        _img(PLOTS_V5_DIR / "training_curves_v3.png", caption="v3 (strong aug) — Training & Validation")
    with cols2[1]:
        _img(PLOTS_V5_DIR / "training_curves_v3_mild.png", caption="v3 (mild aug) — Training & Validation")


# --- Page entrypoint ---
def render():
    st.header("Hypotheses")

    st.write(
        "This page summarizes the three project hypotheses (H1-H3) with their supporting tables, plots, "
        "and evaluation results. All artifacts were generated in the Jupyter notebooks and are loaded "
        "directly from the corresponding project directories."
    )

    with st.expander("H1 — Texture & Color Variability", expanded=True):
        _section_h1()

    with st.expander("H2 — Input Size Impact", expanded=True):
        _section_h2()

    with st.expander("H3 — Data Augmentation Impact", expanded=True):
        _section_h3()

    with st.expander("Training Curves", expanded=True):
        _section_training_curves()

    st.caption(
        "Note: If an element is missing, ensure the corresponding notebook has been executed to "
        "export its artifacts to the expected folders."
    )


def page_hypotheses():
    """Alias for the page entrypoint expected by app.py."""
    render()