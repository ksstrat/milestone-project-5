from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.paths import ARTIFACTS_DIR

# --- Configuration ---
IMG_SIZE = (100, 100)   # model v1 input size
MAX_UPLOADS = 25        # UI limit for responsiveness
LABELS = ["healthy", "powdery_mildew"]
MODEL_PATH = ARTIFACTS_DIR / "v1" / "models" / "cnn_v1_best.keras"


# --- Lazy loading (TensorFlow + model) ---
@st.cache_resource(show_spinner=False)
def _load_tf_and_model(model_path: Path):
    """Load TensorFlow and the trained model once (cached for efficiency)."""
    import tensorflow as tf
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path.as_posix()}\n"
            "Please re-run training in Notebook 05 to export the model."
        )
    model = tf.keras.models.load_model(model_path)
    return tf, model


# --- Image preprocessing ---
def _decode_resize_normalize(img: Image.Image, size=(100, 100)) -> np.ndarray:
    """Convert PIL image to float32 RGB [0..1] with target size."""
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


# --- Prediction helper ---
def _predict_batch(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    """Run batched predictions on uploaded files and return probabilities."""
    tf, model = _load_tf_and_model(MODEL_PATH)

    images, names = [], []
    for f in files:
        try:
            pil = Image.open(io.BytesIO(f.read()))
            arr = _decode_resize_normalize(pil, size=IMG_SIZE)
            images.append(arr)
            names.append(f.name)
        except Exception as e:
            st.warning(f"Skipping '{f.name}': {e}")

    if not images:
        return pd.DataFrame(columns=["filename", "pred_label", *LABELS, "confidence"])

    batch = np.stack(images, axis=0)
    probs = model.predict(batch, verbose=0)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    rows = []
    for i, fname in enumerate(names):
        rows.append({
            "filename": fname,
            "pred_label": LABELS[preds[i]],
            LABELS[0]: float(probs[i, 0]),
            LABELS[1]: float(probs[i, 1]),
            "confidence": float(confs[i]),
        })
    return pd.DataFrame(rows)


# --- Format output table for display ---
def _format_results_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Format results for user-friendly display (percentages and clean labels)."""
    required = {"filename", "pred_label", "healthy", "powdery_mildew", "confidence"}
    missing = required.difference(df_raw.columns)
    if missing:
        raise ValueError(f"Results table is missing required columns: {sorted(missing)}")

    df = df_raw.copy()
    df["Probability (Healthy)"] = (df["healthy"] * 100).round(1).astype(str) + "%"
    df["Probability (Powdery Mildew)"] = (df["powdery_mildew"] * 100).round(1).astype(str) + "%"
    df["Confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"

    df = df.rename(columns={
        "filename": "Filename",
        "pred_label": "Prediction",
    })

    cols_view = [
        "Filename",
        "Prediction",
        "Probability (Healthy)",
        "Probability (Powdery Mildew)",
        "Confidence",
    ]
    return df[cols_view]


# --- Streamlit page layout ---
def render():
    st.header("Prediction (BR2)")
    st.write(
        "Upload one or more images of cherry leaves. The model will predict whether the leaves in "
        "each image are healthy or infested with powdery mildew."
    )
    st.caption(
        "Supported formats: JPG, JPEG, PNG. "
        f"You can upload up to {MAX_UPLOADS} images at once."
    )

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("No files selected yet.")
        return

    if len(uploaded_files) > MAX_UPLOADS:
        st.warning(
            f"You selected {len(uploaded_files)} files. "
            f"Only the first {MAX_UPLOADS} will be processed for responsiveness."
        )
        uploaded_files = uploaded_files[:MAX_UPLOADS]

    try:
        with st.spinner("Running predictions..."):
            df_results = _predict_batch(uploaded_files)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    st.subheader("Results")
    if df_results is None or df_results.empty:
        st.info("No predictions to display. Please check your files and try again.")
        return

    df_display = _format_results_table(df_results)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    csv_bytes = df_results.to_csv(
        index=False,
        float_format="%.6f"   # prevents scientific notation like 9.99E-01
    ).encode("utf-8")
    
    st.download_button(
        label="Download results (CSV)",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv"
    )


def page_prediction():
    """Alias for the page entrypoint expected by app.py."""
    render()