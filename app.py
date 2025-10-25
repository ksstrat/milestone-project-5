import streamlit as st
from app_pages.page_01_summary import page_summary
from app_pages.page_02_visual_study import page_visual_study
from app_pages.page_03_prediction import page_prediction

def main():
    st.set_page_config(page_title="Mildew Detection in Cherry Leaves", layout="wide")
    st.sidebar.title("Navigation")

    pages = {
        "Project Summary": page_summary,
        "Visual Study (BR1)": page_visual_study,
        "Prediction (BR2)": page_prediction,
    }

    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()