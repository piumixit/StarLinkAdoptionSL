import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or use pickle
import os

# --- Load model and column list (ensure these files exist) ---
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("starlink_final_model.pkl")  # Replace with your actual path
    df1 = pd.read_csv("starlink_household_synthetic.csv")     # Must contain all expected columns including 'starlink_proxy_adoption'
    all_cols = df1.columns.drop("starlink_proxy_adoption").tolist()
    return model, all_cols

final_model, all_cols = load_model_and_columns()

# --- Raw household data ---
row_data = {
    "household_id": "NEW-0001",
    "district": "Galle",
    "province": "Southern",
    "urbanization_level": "Rural",
    "members_in_house": 4,
    "roofless_persons": 0,
    "roof_persons": 4,
    "income_lkr": 120_000.0,
    "smartphone_count": 3,
    "computers_owned": 1,
    "electricity_available": 1,
    "has_4g_coverage": 1,
    "has_fiber_coverage": 0,
    "avg_monthly_broadband_bill_lkr": 3500.0,
    "downlink_speed_mbps": 6.2,
    "uplink_speed_mbps": 1.1,
    "overall_bb_satisfaction_score": 2.8,
    "dialog_subscriptions": 1,
    "slt_subscriptions": 0,
    "hutch_subscriptions": 0,
    "other_subscriptions": 0,
    "digital_literacy_score": 0.42,
    "ecommerce_usage_score": 0.18,
    "dialog_sentiment": 0.15,
    "slt_sentiment": -0.05,
    "hutch_sentiment": 0.10,
    "starlink_sentiment_score": 0.35,
    "starlink_awareness": 1,
    "starlink_prob_proxy": 0.42,
}

# Fill missing columns
for col in all_cols:
    row_data.setdefault(col, np.nan)

# Format into DataFrame
new_row = pd.DataFrame({k: [v] for k, v in row_data.items()}) \
            .reindex(columns=all_cols)

# Ensure scalar values
assert all(np.ndim(new_row.iloc[0][c]) == 0 for c in new_row.columns)

# --- Streamlit UI ---
st.set_page_config(page_title="Starlink Adoption Predictor", layout="wide")

st.title("🌐 Starlink Adoption Probability Predictor")
st.markdown("This app uses a trained ML model to predict the **probability of adopting Starlink broadband** for a given household.")

# Show input data
with st.expander("📋 View Household Data", expanded=False):
    st.dataframe(new_row.transpose(), use_container_width=True)

# Make prediction
pred_prob = final_model.predict_proba(new_row)[0, 1]
st.metric("✅ Predicted Starlink Adoption Probability", f"{pred_prob:.2%}")

