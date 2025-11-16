import streamlit as st
from pathlib import Path
from PIL import Image
from src.paths import PLOTS_DIR

PLOTS_V1_DIR = PLOTS_DIR


def _img(
    p: Path,
    caption: str | None = None,
    use_container_width: bool = True
):
    if not p.exists():
        st.warning(f"File not found: `{p.as_posix()}`")
        return
    try:
        st.image(
            Image.open(p),
            caption=caption,
            use_container_width=use_container_width,
        )
    except Exception as e:
        st.error(f"Failed to load {p.name}: {e}")


def render():
    st.header("Visual Analysis")

    st.write(
        """
        This section presents the core exploratory visuals supporting
        **Business Requirement 1 (BR1)**:
        to determine whether healthy cherry leaves and those affected by
        powdery mildew exhibit distinct and measurable visual patterns.

        The figures summarise the main findings from the exploratory data
        analysis and form the analytical basis for subsequent hypothesis
        testing and model development.
        """
    )

    # Class averages
    with st.expander("Class Averages", expanded=True):
        st.markdown(
            """
            Class-average images provide a high-level representation of each
            category.

            - Healthy leaves tend to show smoother colour gradients and more
            uniform surfaces.
            - Mildew-infected leaves display brighter, irregular patterns
            caused by the fungal growth.

            These averages help confirm that the two classes differ in their
            overall appearance.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            _img(PLOTS_V1_DIR / "avg_healthy.png", "Average - Healthy")
        with c2:
            _img(
                PLOTS_V1_DIR / "avg_powdery_mildew.png",
                "Average - Powdery Mildew"
            )

    # Variability
    with st.expander("Per-Pixel Variability", expanded=True):
        st.markdown(
            """
            Variability maps highlight how much pixel intensities vary across
            images within each class.

            - Healthy leaves show relatively moderate variation.
            - Infected leaves exhibit stronger variability due to uneven
            surface patterns introduced by powdery mildew.

            This supports the idea that the disease changes not only colour
            but also the underlying texture structure.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            _img(PLOTS_V1_DIR / "var_healthy.png", "Variability - Healthy")
        with c2:
            _img(
                PLOTS_V1_DIR / "var_powdery_mildew.png",
                "Variability - Powdery Mildew"
            )

    # Difference map
    with st.expander("Class Difference Map", expanded=True):
        st.markdown(
            """
            The difference map visualises the absolute pixel-level deviation
            between the average healthy leaf and the average mildew-infected
            leaf.

            Areas with high intensity indicate regions where the two classes
            diverge most strongly, providing a compact view of disease-related
            differences across the leaf surface.
            """
        )
        _img(
            PLOTS_V1_DIR / "diff_classes.png",
            "Absolute Mean Difference | Healthy - Mildew |"
        )

    # Montages
    with st.expander("Image Montages", expanded=True):
        st.markdown(
            """
            Montages display representative samples from each class side by
            side.

            They provide an intuitive impression of intra-class variability
            and show:

            - what a typical healthy leaf looks like across the dataset,
            - how powdery mildew manifests in different leaves and stages.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            _img(PLOTS_V1_DIR / "montage_healthy.png", "Montage - Healthy")
        with c2:
            _img(
                PLOTS_V1_DIR / "montage_powdery_mildew.png",
                "Montage - Powdery Mildew"
            )

    # RGB Histograms
    with st.expander("Normalized RGB Histograms", expanded=True):
        st.markdown(
            """
            RGB histograms show how colour intensities are distributed for
            healthy and infected leaves.

            Differences in these distributions indicate that the disease not
            only alters texture but also affects colour balance and brightness,
            which can later be exploited by the model during training.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            _img(
                PLOTS_V1_DIR / "hist_rgb_healthy.png",
                "RGB Histogram - Healthy"
            )
        with c2:
            _img(
                PLOTS_V1_DIR / "hist_rgb_powdery_mildew.png",
                "RGB Histogram - Powdery Mildew"
            )

    st.caption(
        "These visuals summarise the key characteristics that distinguish "
        "healthy from powdery-mildew-infected cherry leaves."
    )


def page_visual_study():
    """Alias for the page entrypoint expected by app.py."""
    render()
