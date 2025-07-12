# app.py
import streamlit as st

st.set_page_config(
    page_title="Starlink Adoption Prediction in Sri Lanka",
    layout="wide",
)

st.title("Starlink Adoption Prediction in Sri Lanka")
st.write("This application provides an interactive exploration of the Starlink adoption prediction model for Sri Lanka.")

st.markdown(
    """
    **Objective:** To build a machine‑learning model to predict the likelihood of Starlink internet adoption across Sri Lanka’s districts.

    **Contents:**
    - Exploratory Data Analysis
    - Modeling and Evaluation
    - Inference: Predict Starlink Adoption for a Household
    """
)
