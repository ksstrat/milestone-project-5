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

CM_V1 = PLOTS_DIR.parent / "v3" / "confusion_matrix_test_v1.png"   # saved in 05_modelling_and_evaluation.ipynb
CM_V2 = PLOTS_DIR.parent / "v4" / "confusion_matrix_test_v2.png"   # saved in 06_hypothesis_h2.ipynb
CM_V3 = PLOTS_DIR.parent / "v5" / "confusion_matrix_test_v3_mild.png"  # saved in 07_hypothesis_h3.ipynb

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
        return {"Model": tag, "Test Accuracy": None, "Test Loss": None, "Meets ≥0.97": None, "Samples": None}
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
        "This technical page documents dataset conventions, model versions, training setup, "
        "exported metrics, and key artifacts used throughout the project."
    )

def _section_data_and_versions():
    st.subheader("Data & Versioning")
    st.markdown(
        "- **Dataset**: `inputs/cherry_leaves_dataset/{healthy, powdery_mildew}`\n"
        "- **Splits**: Manifests under `inputs/manifests/v1/{train,val,test}.csv`\n"
        "- **Model versions**:\n"
        "  - **v1** — 100x100, no augmentation (baseline)\n"
        "  - **v2** — 50x50, no augmentation (H2)\n"
        "  - **v3-mild** — 100x100, mild augmentation (H3)"
    )

def _section_training_setup():
    st.subheader("Training Setup (shared defaults)")
    st.markdown(
        "- **Architecture**: 2x Conv blocks → Dense(128) → Dropout(0.3) → Softmax(2)\n"
        "- **Loss / Optimizer / Metrics**: Sparse Categorical Crossentropy / Adam / Accuracy\n"
        "- **Batch Size**: 32\n"
        "- **Epochs**: up to 20 with EarlyStopping (patience 3-4), ModelCheckpoint(best)\n"
        "- **Reproducibility**: seeds set for NumPy/TensorFlow; deterministic label mapping from manifests"
    )

def _section_metrics_and_artifacts():
    st.subheader("Evaluation Metrics & Artifacts")

    rep_v1 = _load_json(REP_V1)
    rep_v2 = _load_json(REP_V2)
    rep_v3 = _load_json(REP_V3)

    # Metrics table
    table = pd.DataFrame([
        _metrics_row("v1 (100x100, no aug)", rep_v1),
        _metrics_row("v2 (50x50, no aug)", rep_v2),
        _metrics_row("v3 (100x100, mild aug)", rep_v3),
    ])
    st.markdown("**Test Metrics (from exported reports)**")
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Confusion matrices and training curves
    st.markdown("**Key Plots**")
    c1, c2, c3 = st.columns(3)
    with c1:
        _img(CM_V1, "Confusion Matrix — v1")
        _img(CURVES_V1, "Training Curves — v1")
    with c2:
        _img(CM_V2, "Confusion Matrix — v2")
        _img(CURVES_V2, "Training Curves — v2")
    with c3:
        _img(CM_V3, "Confusion Matrix — v3-mild")
        _img(CURVES_V3, "Training Curves — v3-mild")

def _section_repro_and_limits():
    st.subheader("Reproducibility Checklist")
    st.markdown(
        "- Use the provided manifests for identical splits.\n"
        "- Keep the preprocessing pipeline unchanged (`tf.data` decode → resize → normalize).\n"
        "- Fix seeds (`np.random.seed`, `tf.random.set_seed`) and sorted label mapping from train manifest.\n"
        "- Export artifacts to the same directories before running the dashboard."
    )

    st.subheader("Known Limitations & Next Steps")
    st.markdown(
        "- Augmentation settings are dataset-dependent; stronger transforms reduced accuracy on the test set.\n"
        "- The baseline CNN is intentionally compact; deeper models or transfer learning may improve margins.\n"
        "- Future work: calibrated probabilities, threshold tuning, and extended error analysis."
    )

# --- Page entrypoint ---
def render():
    st.header("Technical")
    _section_overview()
    st.divider()
    _section_data_and_versions()
    st.divider()
    _section_training_setup()
    st.divider()
    _section_metrics_and_artifacts()
    st.divider()
    _section_repro_and_limits()

def page_technical():
    """Alias for the page entrypoint expected by app.py."""
    render()