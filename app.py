import streamlit as st
from app_pages.page_01_summary import page_summary
from app_pages.page_02_visual_study import page_visual_study

def main():
    st.set_page_config(page_title="Mildew Detection in Cherry Leaves", layout="wide")
    st.sidebar.title("Navigation")

    pages = {
        "Project Summary": page_summary,
        "Visual Study (BR1)": page_visual_study,
    }

    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()