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
    """
    Load TensorFlow and the trained classification model (cached for
    efficiency).
    """
    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(
            "The model file required for prediction is not available on this "
            "deployment. Please contact the project owner or "
            "system maintainer."
        )

    model = tf.keras.models.load_model(model_path)
    return tf, model


# --- Image preprocessing ---
def _decode_resize_normalize(img: Image.Image, size=(100, 100)) -> np.ndarray:
    """
    Convert a PIL image to a float32 RGB array in [0..1] with the target size.
    """
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


# --- Prediction helper ---
def _predict_batch(
        files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> pd.DataFrame:
    """
    Run batched predictions on uploaded files and return probabilities.
    """
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
        return pd.DataFrame(
            columns=[
                "filename",
                "pred_label",
                *LABELS,
                "confidence",
            ]
        )

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
    """
    Format prediction results for user-friendly display (percentages and clean
    labels).
    """
    required = {
        "filename",
        "pred_label",
        "healthy",
        "powdery_mildew",
        "confidence"
    }
    missing = required.difference(df_raw.columns)
    if missing:
        raise ValueError(
            f"Results table is missing required columns: {sorted(missing)}"
        )

    df = df_raw.copy()
    df["Probability (Healthy)"] = (
        df["healthy"] * 100
        ).round(1).astype(str) + "%"
    df["Probability (Powdery Mildew)"] = (
        df["powdery_mildew"] * 100
        ).round(1).astype(str) + "%"
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
    st.header("Leaf Health Detector")

    st.write(
        """
        This page implements **Business Requirement 2 (BR2)** by providing an
        operational prediction interface. You can upload one or more images of
        cherry leaves, and the model will classify each image as **healthy** or
        **powdery_mildew**.
        """
    )

    with st.expander("How the classification works", expanded=False):
        st.markdown(
            """
            The prediction engine is based on a convolutional neural network
            (model v1) trained on the Cherry Leaves Dataset.

            - Input images are resized to **100x100 pixels** and normalised.
            - The model outputs probabilities for the two classes (*healthy*
            and *powdery_mildew*).
            - The class with the higher probability is shown as the final
            prediction.
            - Confidence values indicate how strongly the model supports its
            prediction.

            The deployed model has been evaluated on a held-out test set and
            meets the project target of at least **97% test accuracy**.

            You can use your own images or download example images of healthy
            and infected cherry leaves from the
            Cherry Leaves Dataset on
            [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves?resource=download).
            """
        )

    st.subheader("Upload Images")
    st.caption(
        "Supported formats: JPG, JPEG, PNG. "
        f"You can upload up to {MAX_UPLOADS} images at once."
    )

    uploaded_files = st.file_uploader(
        "Choose one or more leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(
            "No files selected yet. Please upload at least one image to start "
            "the analysis."
        )
        return

    if len(uploaded_files) > MAX_UPLOADS:
        st.warning(
            f"You selected {len(uploaded_files)} files. "
            f"For performance reasons, only the first {MAX_UPLOADS} will be "
            f"processed."
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
        st.info(
            "No predictions to display. Please check your files and try again."
        )
        return

    df_display = _format_results_table(df_results)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    csv_bytes = df_results.to_csv(
        index=False,
        float_format="%.6f",  # prevents scientific notation like 9.99E-01
    ).encode("utf-8")

    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )

    with st.expander("How to interpret the results", expanded=False):
        st.markdown(
            """
            - **Prediction** shows the most likely class according to the
            model.
            - **Probability columns** indicate how likely each class is, based
            on the model's output.
            - **Confidence** reflects the highest class probability and gives
            a quick indication of how decisive the model was.

            This tool is intended to support inspection workflows and should
            be used alongside domain knowledge and good agricultural practice.
            """
        )


def page_prediction():
    """Alias for the page entrypoint expected by app.py."""
    render()
