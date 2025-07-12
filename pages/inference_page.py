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
    """Create input form with all raw data fields including household_id"""
    st.header("Household Information")
    
    input_data = {}
    
    with st.form("prediction_form"):
        # Section 0: Household Identification
        st.subheader("Household Identification")
        input_data['household_id'] = st.text_input(
            "Household ID",
            value="NEW-0001",
            help="Unique identifier for the household"
        )
        
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
        
        # Rest of your form sections remain the same...
        # [Previous code for other sections continues here...]
        
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
    
    # Special handling for household_id to prevent numeric conversion
    if 'household_id' in inference_df.columns:
        inference_df['household_id'] = inference_df['household_id'].astype(str)
    
    # Enforce correct data types for other columns
    for col in all_columns:
        if col in sample_df.columns and col != 'household_id':  # Skip household_id as we already handled it
            try:
                if pd.api.types.is_numeric_dtype(sample_df[col]):
                    inference_df[col] = pd.to_numeric(inference_df[col], errors='coerce')
                else:
                    inference_df[col] = inference_df[col].astype(str)
            except Exception as e:
                st.warning(f"Type conversion failed for {col}: {str(e)}")
    
    return inference_df[all_columns]

# [Rest of your existing functions (show_results, run_inference_page) remain the same...]

def run_inference_page():
    st.title("Starlink Adoption Predictor")
    
    # Load model and sample data
    model = load_model()
    if model is None:
        return
    
    try:
        sample_df = pd.read_csv('starlink_household_synthetic.csv')
        # Ensure household_id is in the expected columns if it exists in sample data
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
                
                # Make prediction (excluding household_id if model doesn't need it)
                prediction_df = inference_df.drop(columns=['household_id'], errors='ignore')
                pred_prob = model.predict_proba(prediction_df)[0, 1]
                prediction = model.predict(prediction_df)[0]
                
                # Show results with household_id
                st.success(f"Household ID: {input_data['household_id']}")
                show_results(prediction, pred_prob, model, prediction_df)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Problematic data:", inference_df.dtypes)

if __name__ == "__main__":
    run_inference_page()
