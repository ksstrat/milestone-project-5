import streamlit as st

def page_prediction():
    st.title("Leaf Health Prediction (BR2)")
    st.info("This page will allow image upload and run the trained model to predict if a leaf is healthy or has mildew.")
    st.caption("The upload and prediction pipeline will be implemented after model training.")