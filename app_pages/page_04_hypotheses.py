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
    st.subheader("H1 — Texture & Colour Variability")

    st.write(
        """
        **Hypothesis:**
        Healthy and mildew-infected cherry leaves differ in their texture and
        colour distribution in a measurable and statistically verifiable way.

        This hypothesis supports **BR1** by testing whether the visual
        differences between both classes can be quantified objectively rather
        than only observed qualitatively.

        To evaluate this, we compare GLCM-based texture features extracted
        from healthy and powdery-mildew-infected leaves.
        A non-parametric Mann-Whitney U test, together with Cliff's Delta as a
        rank-based effect size, is used to assess distribution differences
        between the two groups.
        """
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
            st.markdown("**Class-wise Means / Medians (GLCM Features)**")
            st.dataframe(means, use_container_width=True, hide_index=True)

    _img(PLOTS_V2_DIR / "glcm_boxplots.png", caption="GLCM Features — Boxplots by Class")

    st.caption(
        """
        Interpretation: Multiple GLCM features—such as contrast, homogeneity,
        and energy—differ significantly between
        healthy leaves and mildew-infected leaves. This confirms that the
        disease affects leaf texture in a measurable way.
        """
    )

    st.success("H1 is **supported** — clear statistical differences exist between the two classes.")


# --- H2 section ---
def _section_h2():
    st.subheader("H2 — Input Size Impact (100x100 → 50x50)")

    st.write(
        """
        **Hypothesis:**
        Reducing the input image resolution from 100x100 pixels to 50x50
        pixels preserves classification accuracy at or above the required 97%.

        This hypothesis examines whether the model can operate efficiently on
        lower-resolution inputs without losing essential information about
        mildew-related leaf patterns. It therefore supports the long-term goal
        of enabling fast, lightweight prediction on resource-constrained
        devices such as mobile phones or tablets.

        To evaluate this, a reduced-resolution model (v2) is trained using the
        same architecture, optimizer, and training configuration as the
        baseline model (v1). Only the input size is changed. A comparison of
        test accuracy and confusion matrices across v1 and v2
        is used to assess any loss in predictive performance.
        """
    )

    rep_v1 = _load_json(REP_V1)
    rep_v2 = _load_json(REP_V2)
    cmp_v1_v2 = _load_csv(CMP_V1_V2)

    cols = st.columns(2)
    with cols[0]:
        if rep_v1:
            st.metric("v1 (100x100) — Accuracy", f"{rep_v1.get('test_accuracy', float('nan')):.4f}")
        if rep_v2:
            st.metric("v2 (50x50) — Accuracy", f"{rep_v2.get('test_accuracy', float('nan')):.4f}")
    with cols[1]:
        if rep_v2:
            meets = rep_v2.get("meets_target", False)
            st.markdown(f"**Target ≥ 0.97 met?** {meets}")

    if cmp_v1_v2 is not None and not cmp_v1_v2.empty:
        st.markdown("**Comparison Table — v1 vs. v2**")
        st.dataframe(cmp_v1_v2, use_container_width=True, hide_index=True)

    cols2 = st.columns(2)
    with cols2[0]:
        _img(PLOTS_V4_DIR / "h2_accuracy_v1_vs_v2.png", caption="Accuracy Comparison")
    with cols2[1]:
        _img(PLOTS_V4_DIR / "confusion_matrix_test_v2.png", caption="Confusion Matrix — v2")

    st.caption(
        """
        Interpretation: Accuracy remains above 97% when reducing to 50x50
        pixels, with only minor performance differences.
        This suggests that essential mildew-related patterns remain detectable
        even at lower resolution.
        """
    )

    st.success("H2 is **supported** — reduced input resolution did not compromise performance.")


# --- H3 section ---
def _section_h3():
    st.subheader("H3 — Data Augmentation Impact")

    st.write(
        """
        **Hypothesis:**
        Applying mild data augmentation helps the model generalise better by
        introducing controlled variability during training and reducing
        overfitting on the original dataset.

        This hypothesis tests whether simulated transformations — such as
        light flips, rotations, and brightness adjustments — improve
        robustness against real-world image variation, where lighting,
        orientation, and leaf presentation may differ from the curated dataset.

        To evaluate this, a mild augmentation pipeline is applied during
        training while keeping the validation and test sets unchanged.
        Model performance (v3_mild) is compared against the baseline (v1)
        using test accuracy, confusion matrices, and learning curves to
        determine whether augmentation improves or hinders generalisation.
        """
    )

    rep_v1 = _load_json(REP_V1)
    rep_v3 = _load_json(REP_V3)
    cmp_v1_v3 = _load_csv(CMP_V1_V3)

    cols = st.columns(2)
    with cols[0]:
        if rep_v1:
            st.metric("v1 (no augmentation) — Accuracy", f"{rep_v1.get('test_accuracy', float('nan')):.4f}")
        if rep_v3:
            st.metric("v3_mild (augmentation) — Accuracy", f"{rep_v3.get('test_accuracy', float('nan')):.4f}")
    with cols[1]:
        if rep_v3:
            meets = rep_v3.get("meets_target", False)
            st.markdown(f"**Target ≥ 0.97 met?** {meets}")

    if cmp_v1_v3 is not None and not cmp_v1_v3.empty:
        st.markdown("**Comparison Table — v1 vs. v3_mild**")
        st.dataframe(cmp_v1_v3, use_container_width=True, hide_index=True)

    cols2 = st.columns(2)
    with cols2[0]:
        _img(PLOTS_V5_DIR / "h3_accuracy_v1_vs_v3_mild.png", caption="Accuracy Comparison — v1 vs. v3_mild")
    with cols2[1]:
        _img(PLOTS_V5_DIR / "confusion_matrix_test_v3_mild.png", caption="Confusion Matrix — v3_mild")

    st.caption(
        """
        Interpretation: Contrary to expectations, mild augmentation reduced
        accuracy instead of improving it.
        This shows that generic transformations may distort subtle mildew
        patterns that are essential for correct classification.
        """
    )

    st.warning("H3 is **not supported** — augmentation reduced generalisation performance in this context.")


# --- Training curves ---
def _section_training_curves():
    st.subheader("Training Curves (Diagnostics)")

    st.write(
        """
        These curves illustrate how each model version behaved during training.
        They help assess convergence, overfitting tendencies, and the effect of augmentation.
        """
    )

    cols = st.columns(2)
    with cols[0]:
        _img(PLOTS_V3_DIR / "training_curves_v1.png", caption="v1 — Training & Validation")
    with cols[1]:
        _img(PLOTS_V3_DIR / "training_curves_v2.png", caption="v2 — Training & Validation")

    cols2 = st.columns(2)
    with cols2[0]:
        _img(PLOTS_V5_DIR / "training_curves_v3.png", caption="v3 (strong augmentation)")
    with cols2[1]:
        _img(PLOTS_V5_DIR / "training_curves_v3_mild.png", caption="v3_mild (mild augmentation)")

    st.caption("These diagnostics complement the hypothesis results by showing learning behaviour over time.")


# --- Page entrypoint ---
def render():
    st.header("Hypotheses & Validation")

    st.write(
        """
        This page summarises the three hypotheses developed during the
        analytical phase of the project.
        Each hypothesis examines a different aspect of the problem — from
        texture differences to input resolution
        and generalisation behaviour — and contributes to understanding how
        the model achieves its performance.
        """
    )

    with st.expander("H1 — Texture & Colour Variability", expanded=True):
        _section_h1()

    with st.expander("H2 — Input Size Impact", expanded=True):
        _section_h2()

    with st.expander("H3 — Data Augmentation Impact", expanded=True):
        _section_h3()

    with st.expander("Training Curves (Diagnostics)", expanded=True):
        _section_training_curves()

    st.caption(
        "If some tables or figures are missing, corresponding evaluation artefacts may not be available on this deployment."
    )


def page_hypotheses():
    """Alias for the page entrypoint expected by app.py."""
    render()
