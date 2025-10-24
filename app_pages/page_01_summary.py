import streamlit as st

def page_summary():
    st.title("Project Summary")
    st.markdown("""
**Dataset:** Kaggle Cherry Leaves (Healthy vs. Mildew)  
**Business Requirements:**  
- **BR1:** Visual study to distinguish healthy vs. mildew leaves  
- **BR2:** Predict label (healthy/mildew) with associated probability

**Success Metric (BR2):** Test accuracy ≥ **97%**
""")