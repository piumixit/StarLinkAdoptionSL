import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load data and model (adjust paths as needed)
@st.cache_data
def load_data():
    # Replace with your actual CSV path
    csv_path = "starlink_household_synthetic.csv"
    if Path(csv_path).exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_model():
    # Replace with your actual model path
    model_path = "starlink_final_model.pkl"
    if Path(model_path).exists():
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

def get_unique_values(column):
    if df is not None and column in df.columns:
        return sorted(df[column].dropna().unique())
    return []

def main():
    st.title("Household Starlink Adoption Predictor")
    st.subheader("Enter Household Details")
    
    if df is None:
        st.error("Data file not found. Please ensure 'starlink_household_synthetic.csv' exists.")
        return
    if model is None:
        st.error("Model file not found. Please ensure 'starlink_final_model.pkl' exists.")
        return

    # Create form for user input
    with st.form("household_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Demographics")
            household_id = st.text_input("Household ID", "NEW-0001")
            district = st.selectbox("District", get_unique_values("district"), index=0)
            province = st.selectbox("Province", get_unique_values("province"), index=0)
            urbanization_level = st.selectbox("Urbanization Level", 
                                           get_unique_values("urbanization_level"), 
                                           index=0)
            members_in_house = st.number_input("Members in Household", min_value=1, max_value=20, value=4)
            roofless_persons = st.number_input("Roofless Persons", min_value=0, max_value=20, value=0)
            roof_persons = st.number_input("Roof Persons", min_value=0, max_value=20, value=4)
            
            st.markdown("### Income & Assets")
            income_lkr = st.number_input("Monthly Income (LKR)", min_value=0, value=120000, step=1000)
            smartphone_count = st.number_input("Smartphones Owned", min_value=0, max_value=20, value=3)
            computers_owned = st.number_input("Computers Owned", min_value=0, max_value=10, value=1)
        
        with col2:
            st.markdown("### Connectivity")
            electricity_available = st.selectbox("Electricity Available", [1, 0], format_func=lambda x: "Yes" if x else "No", index=0)
            has_4g_coverage = st.selectbox("4G Coverage", [1, 0], format_func=lambda x: "Yes" if x else "No", index=0)
            has_fiber_coverage = st.selectbox("Fiber Coverage", [1, 0], format_func=lambda x: "Yes" if x else "No", index=1)
            
            st.markdown("### Broadband Usage")
            avg_broadband_bill = st.number_input("Avg Monthly Broadband Bill (LKR)", min_value=0, value=3500, step=100)
            downlink_speed = st.number_input("Download Speed (Mbps)", min_value=0.0, value=6.2, step=0.1)
            uplink_speed = st.number_input("Upload Speed (Mbps)", min_value=0.0, value=1.1, step=0.1)
            bb_satisfaction = st.slider("Broadband Satisfaction (1-5)", 1.0, 5.0, 2.8, 0.1)
            
            st.markdown("### Subscriptions")
            dialog_subs = st.number_input("Dialog Subscriptions", min_value=0, max_value=5, value=1)
            slt_subs = st.number_input("SLT Subscriptions", min_value=0, max_value=5, value=0)
            hutch_subs = st.number_input("Hutch Subscriptions", min_value=0, max_value=5, value=0)
            other_subs = st.number_input("Other Subscriptions", min_value=0, max_value=5, value=0)
        
        # Behavioral and sentiment scores
        st.markdown("### Behavioral & Sentiment Scores")
        bcol1, bcol2, bcol3 = st.columns(3)
        
        with bcol1:
            digital_literacy = st.slider("Digital Literacy Score", 0.0, 1.0, 0.42, 0.01)
            ecommerce_usage = st.slider("E-commerce Usage Score", 0.0, 1.0, 0.18, 0.01)
        
        with bcol2:
            dialog_sentiment = st.slider("Dialog Sentiment", -1.0, 1.0, 0.15, 0.01)
            slt_sentiment = st.slider("SLT Sentiment", -1.0, 1.0, -0.05, 0.01)
        
        with bcol3:
            hutch_sentiment = st.slider("Hutch Sentiment", -1.0, 1.0, 0.10, 0.01)
            starlink_sentiment = st.slider("Starlink Sentiment", -1.0, 1.0, 0.35, 0.01)
        
        # Starlink awareness
        st.markdown("### Starlink Awareness")
        starlink_awareness = st.selectbox("Aware of Starlink", [1, 0], format_func=lambda x: "Yes" if x else "No", index=0)
        starlink_prob_proxy = st.slider("Starlink Proxy Probability", 0.0, 1.0, 0.42, 0.01)
        
        submitted = st.form_submit_button("Predict Starlink Adoption")

    if submitted:
        # Prepare input data
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
            "electricity_available": electricity_available,
            "has_4g_coverage": has_4g_coverage,
            "has_fiber_coverage": has_fiber_coverage,
            "avg_monthly_broadband_bill_lkr": avg_broadband_bill,
            "downlink_speed_mbps": downlink_speed,
            "uplink_speed_mbps": uplink_speed,
            "overall_bb_satisfaction_score": bb_satisfaction,
            "dialog_subscriptions": dialog_subs,
            "slt_subscriptions": slt_subs,
            "hutch_subscriptions": hutch_subs,
            "other_subscriptions": other_subs,
            "digital_literacy_score": digital_literacy,
            "ecommerce_usage_score": ecommerce_usage,
            "dialog_sentiment": dialog_sentiment,
            "slt_sentiment": slt_sentiment,
            "hutch_sentiment": hutch_sentiment,
            "starlink_sentiment_score": starlink_sentiment,
            "starlink_awareness": starlink_awareness,
            "starlink_prob_proxy": starlink_prob_proxy
        }

        # Create DataFrame with all expected columns
        all_cols = df.columns.drop("starlink_proxy_adoption").tolist()
        new_row = pd.DataFrame({k: [input_data.get(k, np.nan)] for k in all_cols})
        
        # Make prediction
        try:
            pred_prob = model.predict_proba(new_row)[0, 1]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Predicted Adoption Probability", 
                          value=f"{pred_prob:.1%}",
                          delta=f"{(pred_prob - starlink_prob_proxy):+.1%}" if 'starlink_prob_proxy' in input_data else None,
                          delta_color="normal",
                          help="Model's prediction compared to proxy probability")
            
            with col2:
                # Interpretation
                if pred_prob > 0.7:
                    st.success("**High likelihood** of Starlink adoption")
                    st.write("This household has strong indicators for potential Starlink adoption:")
                    st.write("- High income and technology assets")
                    st.write("- Positive sentiment toward Starlink")
                    st.write("- Dissatisfaction with current broadband")
                elif pred_prob > 0.4:
                    st.warning("**Moderate likelihood** of Starlink adoption")
                    st.write("This household shows some interest but may need more convincing:")
                    st.write("- Consider targeted marketing")
                    st.write("- Highlight Starlink advantages over current options")
                else:
                    st.info("**Low likelihood** of Starlink adoption")
                    st.write("This household is unlikely to adopt Starlink currently:")
                    st.write("- May be satisfied with existing services")
                    st.write("- May lack awareness or need for satellite internet")
                
                # Show key factors
                st.write("**Key influencing factors:**")
                factors = {
                    "Starlink Sentiment": starlink_sentiment,
                    "Income": income_lkr,
                    "Digital Literacy": digital_literacy,
                    "Broadband Satisfaction": bb_satisfaction
                }
                st.write(pd.DataFrame.from_dict(factors, orient='index', columns=['Value']))
            # Clear prediction button
            if st.button("Clear Prediction"):
                reset_form()
                st.rerun()
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
