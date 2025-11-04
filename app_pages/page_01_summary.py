import json
from pathlib import Path
import streamlit as st
from src.paths import DATA_DIR, MANIFESTS_DIR, PLOTS_DIR, ARTIFACTS_DIR


def _file_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def page_project_summary():
    st.header("Project Summary")

    st.write(
        """
A compact machine learning project to detect **powdery mildew** on cherry leaves.

This dashboard consolidates the full workflow:
- **Visual Study (BR1):** demonstrate that *healthy* vs. *mildew* leaves are visually distinguishable.
- **Prediction (BR2):** train a CNN and verify whether the **≥ 97% test accuracy** target is met.
        """
    )

    st.subheader("Dataset")
    st.markdown(
        """
Dataset: *Cherry Leaves* (Healthy vs. Powdery Mildew).  
Local layout:

- `inputs/cherry_leaves_dataset/healthy/`
- `inputs/cherry_leaves_dataset/powdery_mildew/`

Split manifests (deterministic, stratified) are stored under:  
`inputs/manifests/v1/{train,val,test}.csv`
        """
    )

    st.subheader("Business Requirements")
    st.markdown(
        """
- **BR1 — Visual Differentiation:** provide plots that make the two classes visibly distinguishable.  
- **BR2 — Prediction:** develop a model that predicts *Healthy* vs *Mildew* with at least **97% accuracy** on the test set.
        """
    )

    st.subheader("Pages Overview")
    st.markdown(
        """
- **Project Summary:** General project context and artifact status overview.
- **Visual Study (BR1):** Average/variability/difference images, montages, RGB histograms.
- **Prediction (BR2):** File uploader and prediction interface.
- **Hypotheses:** H1 (texture), H2 (input size), H3 (augmentation) — analytical results and visuals.
- **Technical:** Confusion matrix, learning curves, and evaluation reports.
        """
    )

    st.subheader("Current Artifacts (auto-detected)")
    st.markdown("Data and model availability checks:")

    checks = {
        "Dataset folder": DATA_DIR,
        "Train manifest": MANIFESTS_DIR / "train.csv",
        "Validation manifest": MANIFESTS_DIR / "val.csv",
        "Test manifest": MANIFESTS_DIR / "test.csv",
        "BR1 plot example": PLOTS_DIR / "avg_healthy.png",
        "v1 model": ARTIFACTS_DIR / "v1" / "models" / "cnn_v1_best.keras",
        "v1 report": ARTIFACTS_DIR / "v1" / "reports" / "evaluation_report.json",
    }

    for label, path in checks.items():
        ok = _file_exists(path)
        st.write(f"- {label}: {'available' if ok else 'missing'}")

    st.divider()

    # Optional: show v1 accuracy metric if report exists
    eval_json = ARTIFACTS_DIR / "v1" / "reports" / "evaluation_report.json"
    if _file_exists(eval_json):
        try:
            data = json.loads(eval_json.read_text())
            acc = float(data.get("test_accuracy", float("nan")))
            meets = bool(data.get("meets_target", False))
            st.metric(
                label="Latest v1 Test Accuracy",
                value=f"{acc:.2%}",
                delta="meets target" if meets else "below 97%",
            )
        except Exception as e:
            st.warning(f"Could not read evaluation report: {e}")
    else:
        st.info("No v1 evaluation report found yet. Train the model in notebook 05 to populate reports.")

    st.caption("Cherry Leaves Mildew Detection Dashboard — 2025")


# Backwards-compatible alias expected by app.py
def page_summary():
    page_project_summary()