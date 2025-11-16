from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
from PIL import Image

from src.paths import ARTIFACTS_DIR, PLOTS_DIR

# --- Conventions / locations used across notebooks ---
REP_V1 = ARTIFACTS_DIR / "v1" / "reports" / "evaluation_report.json"
REP_V2 = ARTIFACTS_DIR / "v2" / "reports" / "evaluation_report.json"
REP_V3 = ARTIFACTS_DIR / "v3" / "reports" / "evaluation_report.json"


# Saved in 05_modelling_and_evaluation.ipynb
CM_V1 = PLOTS_DIR.parent / "v3" / "confusion_matrix_test_v1.png"

# Saved in 06_hypothesis_h2.ipynb
CM_V2 = PLOTS_DIR.parent / "v4" / "confusion_matrix_test_v2.png"

# Saved in 07_hypothesis_h3.ipynb
CM_V3 = PLOTS_DIR.parent / "v5" / "confusion_matrix_test_v3_mild.png"

CURVES_V1 = PLOTS_DIR.parent / "v3" / "training_curves_v1.png"
CURVES_V2 = PLOTS_DIR.parent / "v3" / "training_curves_v2.png"
CURVES_V3 = PLOTS_DIR.parent / "v5" / "training_curves_v3_mild.png"


# --- Safe loaders ---
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
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
    if not path.exists():
        st.warning(f"Image not found: `{path.as_posix()}`")
        return
    try:
        st.image(Image.open(path), caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load image `{path.name}`: {e}")


# --- Helpers for compact tables ---
def _metrics_row(tag: str, rep: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not rep:
        return {
            "Model": tag,
            "Test Accuracy": None,
            "Test Loss": None,
            "Meets ≥0.97": None,
            "Samples": None
        }
    return {
        "Model": tag,
        "Test Accuracy": rep.get("test_accuracy"),
        "Test Loss": rep.get("test_loss"),
        "Meets ≥0.97": rep.get("meets_target"),
        "Samples": rep.get("total_samples"),
    }


# --- Sections ---
def _section_overview():
    st.subheader("Overview")
    st.write(
        """
        This page summarises the **machine learning performance** of the models
        used in the dashboard. It focuses on how well the models learned to
        distinguish between healthy and powdery-mildew-infected cherry leaves
        and how they performed on previously unseen test data.

        The content is intended for stakeholders who want to understand the
        predictive capability of the system behind the user-facing prediction
        page.
        """
    )


def _section_data_and_versions():
    st.markdown(
        """
        The models were trained and evaluated on a labelled
        [dataset](https://www.kaggle.com/codeinstitute/cherry-leaves) of cherry
        leaves containing healthy and powdery-mildew-infected samples. The
        data was split into **train**, **validation**, and **test** sets using
        a stratified strategy to preserve class balance across all splits.

        The following model versions are included in this analysis:

        - **v1** - 100x100 input resolution, no augmentation (baseline
        reference model)
        - **v2** - 50x50 input resolution, no augmentation (efficiency
        experiment for H2)
        - **v3-mild** - 100x100 input resolution, mild augmentation
        (generalisation experiment for H3)

        All versions share the same CNN architecture and training
        configuration so that differences in performance can be attributed to
        resolution or augmentation rather than structural changes to the model.
        """
    )


def _section_training_setup():
    st.markdown(
        """
        All three model versions were trained under a common configuration:

        - **Architecture:** two convolutional blocks followed by a dense layer
        (128 units), dropout (0.3) and a final softmax layer with two outputs
        - **Loss / optimiser / metric:** sparse categorical crossentropy, Adam,
        accuracy
        - **Batch size:** 32
        - **Epochs:** up to 20 with early stopping and model checkpointing for
        the best validation score

        This setup ensures that the models see enough data to learn the key
        mildew patterns while avoiding excessive overfitting.

        The learning curves below illustrate how training and validation
        accuracy and loss evolved over time for each model version.
        """
    )

    cols = st.columns(3)
    with cols[0]:
        _img(CURVES_V1, "Training & Validation - v1 (baseline)")
    with cols[1]:
        _img(CURVES_V2, "Training & Validation - v2 (50x50)")
    with cols[2]:
        _img(CURVES_V3, "Training & Validation - v3-mild (augmentation)")


def _section_metrics_and_artifacts():
    rep_v1 = _load_json(REP_V1)
    rep_v2 = _load_json(REP_V2)
    rep_v3 = _load_json(REP_V3)

    # Metrics table
    table = pd.DataFrame(
        [
            _metrics_row("v1 (100x100, no augmentation)", rep_v1),
            _metrics_row("v2 (50x50, no augmentation)", rep_v2),
            _metrics_row("v3-mild (100x100, mild augmentation)", rep_v3),
        ]
    )
    st.markdown(
        """
        ### Summary of Test Metrics
        The table below summarises the main test metrics for each model
        version, including whether the **97% accuracy target** defined in the
        business requirements was met.
        """
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        ### Confusion Matrices

        The confusion matrices show how well each model distinguishes between
        healthy and infected leaves on the test set:

        - The **diagonal cells** represent correct predictions.
        - Off-diagonal cells indicate misclassifications.
        - A strong diagonal with few off-diagonal values indicates robust
        performance.
        """
    )

    cols = st.columns(3)
    with cols[0]:
        _img(CM_V1, "Confusion Matrix - v1 (baseline)")
    with cols[1]:
        _img(CM_V2, "Confusion Matrix - v2 (50x50)")
    with cols[2]:
        _img(CM_V3, "Confusion Matrix - v3-mild (augmentation)")

    st.caption(
        """
        Together, the numerical test metrics and confusion matrices confirm
        that the baseline and reduced-resolution models comfortably meet the
        97% accuracy requirement, while the augmented v3-mild model
        underperforms in this setting.
        """
    )


def _section_repro_and_limits():
    st.markdown(
        """
        From a business perspective, the key outcome is that the selected
        models (v1 and v2) achieve accuracy levels well above the required 97%
        on held-out test data. This suggests that the system is suitable for
        supporting mildew screening workflows in practice.

        For reproducibility within the project environment, the following
        principles were applied:

        - Stratified splitting of data into train, validation and test sets
        - Stable preprocessing (decode → resize → normalise) across all model
        versions
        - Fixed random seeds during training and consistent label mapping

        Known limitations and future improvements include:

        - The augmentation strategy tested in v3-mild decreased performance;
        more domain-specific transformations will be explored in future
        iterations.
        - The CNN is intentionally compact; deeper architectures or transfer
        learning may further improve performance if required.
        - Future work may include probability calibration, threshold tuning and
        extended error analysis on borderline images to refine decision
        support.
        """
    )


# --- Page entrypoint ---
def render():
    st.header("ML Performance Metrics")
    _section_overview()

    with st.expander(
        "Data splits and model versions",
        expanded=True,
    ):
        _section_data_and_versions()

    with st.expander(
        "Training configuration and learning behaviour",
        expanded=True,
    ):
        _section_training_setup()

    with st.expander(
        "Test metrics and confusion matrices",
        expanded=True,
    ):
        _section_metrics_and_artifacts()

    with st.expander(
        "Interpretation, reproducibility and next steps",
        expanded=True,
    ):
        _section_repro_and_limits()


def page_technical():
    """Alias for the page entrypoint expected by app.py."""
    render()
