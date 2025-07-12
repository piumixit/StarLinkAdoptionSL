import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model and input columns ---
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("final_model.pkl")  # Your trained model
    df1 = pd.read_csv("df1_sample.csv")     # Must include 'starlink_proxy_adoption' column
    all_cols = df1.columns.drop("starlink_proxy_adoption").tolist()
    return model, all_cols

final_model, all_cols = load_model_and_columns()

# --- UI Header ---
st.set_page_config(page_title="Starlink Predictor", layout="wide")
st.title("🌐 Starlink Adoption Probability")
st.markdown("Enter household data below to estimate the likelihood of adopting Starlink.")

# --- User Input Form ---
with st.form("user_input_form"):
    st.subheader("🏠 Household & Location Info")
    household_id = st.text_input("Household ID", "NEW-0001")
    district = st.selectbox("District", ["Galle", "Colombo", "Kandy", "Jaffna"])  # Modify as needed
    province = st.selectbox("Province", ["Southern", "Western", "Central", "Northern"])
    urbanization_level = st.selectbox("Urbanization Level", ["Urban", "Suburban", "Rural"])
    members_in_house = st.number_input("Household Members", min_value=1, step=1, value=4)
    roofless_persons = st.number_input("Roofless Persons", min_value=0, step=1, value=0)
    roof_persons = st.number_input("Roofed Persons", min_value=0, step=1, value=4)

    st.subheader("💰 Income & Devices")
    income_lkr = st.number_input("Monthly Income (LKR)", min_value=0.0, value=120_000.0)
    smartphone_count = st.number_input("Smartphones Owned", min_value=0, step=1, value=3)
    computers_owned = st.number_input("Computers Owned", min_value=0, step=1, value=1)

    st.subheader("🔌 Connectivity & Usage")
    electricity_available = st.checkbox("Electricity Available", value=True)
    has_4g_coverage = st.checkbox("4G Coverage", value=True)
    has_fiber_coverage = st.checkbox("Fiber Coverage", value=False)

    avg_monthly_broadband_bill_lkr = st.number_input("Monthly Broadband Bill (LKR)", min_value=0.0, value=3500.0)
    downlink_speed_mbps = st.number_input("Downlink Speed (Mbps)", min_value=0.0, value=6.2)
    uplink_speed_mbps = st.number_input("Uplink Speed (Mbps)", min_value=0.0, value=1.1)
    overall_bb_satisfaction_score = st.slider("Broadband Satisfaction (0-5)", 0.0, 5.0, 2.8)

    st.subheader("📶 Subscriptions")
    dialog_subscriptions = st.number_input("Dialog Subscriptions", min_value=0, step=1, value=1)
    slt_subscriptions = st.number_input("SLT Subscriptions", min_value=0, step=1, value=0)
    hutch_subscriptions = st.number_input("Hutch Subscriptions", min_value=0, step=1, value=0)
    other_subscriptions = st.number_input("Other Subscriptions", min_value=0, step=1, value=0)

    st.subheader("🧠 Behaviour & Sentiment")
    digital_literacy_score = st.slider("Digital Literacy (0-1)", 0.0, 1.0, 0.42)
    ecommerce_usage_score = st.slider("E-commerce Usage (0-1)", 0.0, 1.0, 0.18)

    dialog_sentiment = st.slider("Dialog Sentiment (-1 to 1)", -1.0, 1.0, 0.15)
    slt_sentiment = st.slider("SLT Sentiment (-1 to 1)", -1.0, 1.0, -0.05)
    hutch_sentiment = st.slider("Hutch Sentiment (-1 to 1)", -1.0, 1.0, 0.10)
    starlink_sentiment_score = st.slider("Starlink Sentiment (0-1)", 0.0, 1.0, 0.35)

    st.subheader("📢 Awareness & Proxy")
    starlink_awareness = st.checkbox("Aware of Starlink?", value=True)
    starlink_prob_proxy = st.slider("Starlink Proxy Probability (0-1)", 0.0, 1.0, 0.42)

    submitted = st.form_submit_button("🔮 Predict")

# --- Process Input and Predict ---
if submitted:
    # Construct data dict
    input_data = {
        "household_id": household_id,
        "district": district,
        "province": province,
        "urbanization_level": urbanization_level,
        "members_in_house": members_in_house,
        "roofless_persons": roofless_persons,
        "roof_persons": roof_persons,
        "income_lkr": income_lkr,
        "smartphone_count": smartphone_count,
        "computers_owned": computers_owned,
        "electricity_available": int(electricity_available),
        "has_4g_coverage": int(has_4g_coverage),
        "has_fiber_coverage": int(has_fiber_coverage),
        "avg_monthly_broadband_bill_lkr": avg_monthly_broadband_bill_lkr,
        "downlink_speed_mbps": downlink_speed_mbps,
        "uplink_speed_mbps": uplink_speed_mbps,
        "overall_bb_satisfaction_score": overall_bb_satisfaction_score,
        "dialog_subscriptions": dialog_subscriptions,
        "slt_subscriptions": slt_subscriptions,
        "hutch_subscriptions": hutch_subscriptions,
        "other_subscriptions": other_subscriptions,
        "digital_literacy_score": digital_literacy_score,
        "ecommerce_usage_score": ecommerce_usage_score,
        "dialog_sentiment": dialog_sentiment,
        "slt_sentiment": slt_sentiment,
        "hutch_sentiment": hutch_sentiment,
        "starlink_sentiment_score": starlink_sentiment_score,
        "starlink_awareness": int(starlink_awareness),
        "starlink_prob_proxy": starlink_prob_proxy,
    }

    # Fill missing model columns
    for col in all_cols:
        input_data.setdefault(col, np.nan)

    # Create DataFrame
    new_row = pd.DataFrame({k: [v] for k, v in input_data.items()}) \
                .reindex(columns=all_cols)

    # Predict
    pred_prob = final_model.predict_proba(new_row)[0, 1]

    st.success(f"✅ **Predicted Starlink Adoption Probability: {pred_prob:.2%}**")
