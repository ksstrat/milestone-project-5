import json
from pathlib import Path
import streamlit as st
from src.paths import DATA_DIR, MANIFESTS_DIR, PLOTS_DIR, ARTIFACTS_DIR


def page_project_summary():
    st.header("Project Summary")

    st.write(
        """
        This dashboard presents an end-to-end machine-learning system
        developed to detect **powdery mildew** on cherry leaves.
        It consolidates the full analytical workflow-from dataset exploration
        and hypothesis validation to model evaluation and an operational
        prediction interface for analysing new leaf images.
        """
    )

    # What is powdery mildew?
    st.markdown(
        """
        ### What is Powdery Mildew?
        Powdery mildew is a common fungal infection that produces white,
        powder-like structures on the upper surface of cherry leaves.
        Early detection is essential, as the disease spreads rapidly under
        humid conditions and can weaken leaf health, reduce photosynthetic
        efficiency, and negatively impact fruit development.
        This dashboard aims to support early recognition and improve the
        consistency of leaf health assessments.
        """
    )

    # Dataset section
    st.subheader("Dataset")
    st.markdown(
        """
        The system is based on the **Cherry Leaves Dataset**, containing
        labeled images of both *healthy* and *powdery-mildew-infected* leaves.
        These images support the visual study, hypothesis testing, and model
        training stages.

        **Kaggle Dataset Link:**
        https://www.kaggle.com/codeinstitute/cherry-leaves

        A deterministic train/validation/test split was created during
        development to ensure reproducible evaluation results.
        """
    )

    # Business requirements
    st.subheader("Business Requirements")
    st.markdown(
        """
        The dashboard and analytical workflow were designed to address two key
        business requirements:

        **BR1 - Visual Study**
        Before deploying a classification model, it must be demonstrated that
        healthy and mildew-infected leaves show meaningful and measurable
        visual differences.
        This dashboard therefore includes a dedicated visual analysis section,
        allowing users to review class averages, texture variability, RGB
        histograms, and image montages to understand the foundations of the
        modelling approach.

        **BR2 - High-Accuracy Prediction**
        To support consistent and fast assessments, the system must
        automatically classify new cherry leaf images with a minimum test
        accuracy of **97%**.
        Users can upload individual or multiple images and receive a clear,
        immediate prediction, enabling efficient screening and diagnostic
        workflows in field or laboratory settings.
        """
    )

    # Model performance metric
    st.subheader("Latest Model Performance")

    eval_json = ARTIFACTS_DIR / "v1" / "reports" / "evaluation_report.json"
    if eval_json.exists():
        try:
            data = json.loads(eval_json.read_text())
            acc = float(data.get("test_accuracy", float("nan")))
            meets = bool(data.get("meets_target", False))
            st.metric(
                label="v1 Test Accuracy",
                value=f"{acc:.2%}",
                delta="meets target" if meets else "below target",
            )
        except Exception as e:
            st.warning(f"Could not read evaluation report: {e}")
    else:
        st.info(
            "The evaluation report for model v1 is not available on this "
            "deployment."
        )

    # Reference to README for more info
    st.markdown(
        """
        For a comprehensive description of the full analytical
        process-including data exploration, hypothesis validation, modelling
        decisions, and deployment setup-please refer to the project's detailed
        **[README.md](https://github.com/ksstrat/milestone-project-5/blob/main/README.md)**.
        """
    )


# Backwards-compatible alias expected by app.py
def page_summary():
    page_project_summary()
