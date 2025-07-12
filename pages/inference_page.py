# pages/inference_page.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load model with caching
@st.cache_resource
def load_model(model_path="starlink_final_model.pkl"):
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'named_steps'):  # Add imputer if not in pipeline
            model = make_pipeline(SimpleImputer(strategy='median'), model)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def get_input_form(sample_df):
    """Create input form with all raw data fields"""
    st.header("Household Information")
    
    input_data = {}
    
    with st.form("prediction_form"):
        # Section 1: Location & Demographics
        st.subheader("Location & Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['district'] = st.selectbox(
                "District", 
                sorted(sample_df['district'].unique())
            )
            input_data['province'] = st.selectbox(
                "Province",
                sorted(sample_df['province'].unique())
            )
            input_data['urbanization_level'] = st.selectbox(
                "Urbanization Level",
                sorted(sample_df['urbanization_level'].unique())
            )
            
        with col2:
            input_data['members_in_house'] = st.number_input(
                "Household Members",
                min_value=1, max_value=20, value=4
            )
            input_data['roofless_persons'] = st.number_input(
                "Roofless Persons", 
                min_value=0, max_value=20, value=0
            )
            input_data['income_lkr'] = st.number_input(
                "Monthly Income (LKR)",
                min_value=0.0, value=120000.0, step=1000.0
            )
        
        # Section 2: Assets & Connectivity
        st.subheader("Assets & Connectivity")
        col3, col4 = st.columns(2)
        
        with col3:
            input_data['smartphone_count'] = st.number_input(
                "Smartphones Owned",
                min_value=0, value=3
            )
            input_data['computers_owned'] = st.number_input(
                "Computers Owned",
                min_value=0, value=1
            )
            input_data['electricity_available'] = st.selectbox(
                "Electricity Available",
                options=[1, 0], format_func=lambda x: "Yes" if x else "No"
            )
            
        with col4:
            input_data['has_4g_coverage'] = st.selectbox(
                "4G Coverage Available",
                options=[1, 0], format_func=lambda x: "Yes" if x else "No"
            )
            input_data['has_fiber_coverage'] = st.selectbox(
                "Fiber Coverage Available",
                options=[1, 0], format_func=lambda x: "Yes" if x else "No"
            )
            input_data['avg_monthly_broadband_bill_lkr'] = st.number_input(
                "Monthly Broadband Bill (LKR)",
                min_value=0.0, value=3500.0, step=100.0
            )
        
        # Section 3: Internet Performance
        st.subheader("Internet Performance")
        col5, col6 = st.columns(2)
        
        with col5:
            input_data['downlink_speed_mbps'] = st.number_input(
                "Download Speed (Mbps)",
                min_value=0.0, value=6.2, step=0.1
            )
            input_data['uplink_speed_mbps'] = st.number_input(
                "Upload Speed (Mbps)",
                min_value=0.0, value=1.1, step=0.1
            )
            
        with col6:
            input_data['overall_bb_satisfaction_score'] = st.slider(
                "Broadband Satisfaction (1-5)",
                min_value=1, max_value=5, value=3
            )
            input_data['digital_literacy_score'] = st.slider(
                "Digital Literacy (0-1)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01
            )
        
        # Section 4: Subscriptions & Sentiment
        st.subheader("Subscriptions & Sentiment")
        col7, col8 = st.columns(2)
        
        with col7:
            input_data['dialog_subscriptions'] = st.number_input(
                "Dialog Subscriptions",
                min_value=0, value=1
            )
            input_data['slt_subscriptions'] = st.number_input(
                "SLT Subscriptions",
                min_value=0, value=0
            )
            input_data['dialog_sentiment'] = st.slider(
                "Dialog Sentiment (-1 to +1)",
                min_value=-1.0, max_value=1.0, value=0.0, step=0.01
            )
            
        with col8:
            input_data['hutch_subscriptions'] = st.number_input(
                "Hutch Subscriptions",
                min_value=0, value=0
            )
            input_data['other_subscriptions'] = st.number_input(
                "Other Subscriptions",
                min_value=0, value=0
            )
            input_data['starlink_sentiment_score'] = st.slider(
                "Starlink Sentiment (-1 to +1)",
                min_value=-1.0, max_value=1.0, value=0.0, step=0.01
            )
        
        # Section 5: Awareness & Behavior
        st.subheader("Awareness & Behavior")
        input_data['starlink_awareness'] = st.selectbox(
            "Aware of Starlink",
            options=[1, 0], format_func=lambda x: "Yes" if x else "No"
        )
        input_data['ecommerce_usage_score'] = st.slider(
            "E-commerce Usage (0-1)",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01
        )
        
        submitted = st.form_submit_button("Predict Adoption")
    
    return input_data, submitted

def prepare_inference_data(input_data, all_columns, sample_df):
    """Convert raw input to properly formatted DataFrame"""
    # Calculate derived fields
    if 'roof_persons' in all_columns:
        input_data['roof_persons'] = input_data['members_in_house'] - input_data.get('roofless_persons', 0)
    
    # Create DataFrame with all expected columns
    row_data = {col: np.nan for col in all_columns}
    row_data.update(input_data)
    
    # Convert to DataFrame with correct dtypes
    inference_df = pd.DataFrame([row_data])
    
    # Enforce correct data types
    for col in all_columns:
        if col in sample_df.columns:
            try:
                if pd.api.types.is_numeric_dtype(sample_df[col]):
                    inference_df[col] = pd.to_numeric(inference_df[col], errors='coerce')
                else:
                    inference_df[col] = inference_df[col].astype(str)
            except Exception as e:
                st.warning(f"Type conversion failed for {col}: {str(e)}")
    
    return inference_df[all_columns]

def show_results(prediction, pred_prob, model, inference_df):
    """Display prediction results and explanations"""
    st.success(f"Predicted Adoption Probability: {pred_prob:.1%}")
    st.write(f"Recommended Action: {'High Potential' if prediction else 'Low Priority'}")
    
    # SHAP Explanation
    try:
        if hasattr(model, 'named_steps'):
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            prep_data = model.named_steps['preprocessor'].transform(inference_df)
            shap_values = explainer.shap_values(prep_data)
            
            st.subheader("Feature Importance")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], prep_data, 
                            feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
                            plot_type="bar", show=False)
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate explanation: {str(e)}")

def run_inference_page():
    st.title("Starlink Adoption Predictor")
    
    # Load model and sample data
    model = load_model()
    if model is None:
        return
    
    try:
        sample_df = pd.read_csv('starlink_household_synthetic.csv')
        all_columns = sample_df.drop(columns=['starlink_proxy_adoption'], errors='ignore').columns.tolist()
    except Exception as e:
        st.error(f"Failed to load sample data: {str(e)}")
        return
    
    # Get user inputs
    input_data, submitted = get_input_form(sample_df)
    
    if submitted:
        with st.spinner("Making prediction..."):
            try:
                # Prepare data
                inference_df = prepare_inference_data(input_data, all_columns, sample_df)
                
                # Debug output
                st.write("Processed Input Data:")
                st.dataframe(inference_df)
                
                # Make prediction
                pred_prob = model.predict_proba(inference_df)[0, 1]
                prediction = model.predict(inference_df)[0]
                
                # Show results
                show_results(prediction, pred_prob, model, inference_df)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Problematic data:", inference_df.dtypes)

if __name__ == "__main__":
    run_inference_page()
