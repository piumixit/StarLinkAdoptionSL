# pages/inference_page.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.impute import SimpleImputer

# Load the pre-trained model (cached for efficiency)
@st.cache_resource
def load_model(model_path="starlink_final_model.pkl"):
    """Loads the final model with imputation fallback."""
    try:
        model = joblib.load(model_path)
        
        # If model isn't already a pipeline with imputation, wrap it
        if not hasattr(model, 'named_steps'):
            model = make_pipeline(
                SimpleImputer(strategy='median'),
                model
            )
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None

def create_inference_row(input_data, all_columns):
    """Creates a properly formatted inference row with all expected columns."""
    # Start with default NaN values for all columns
    row_data = {col: np.nan for col in all_columns}
    
    # Update with user-provided values
    row_data.update(input_data)
    
    # Calculate derived fields
    if 'roof_persons' in all_columns and 'members_in_house' in row_data and 'roofless_persons' in row_data:
        row_data['roof_persons'] = row_data['members_in_house'] - row_data['roofless_persons']
    
    # Create DataFrame with correct column order
    new_row = pd.DataFrame({k: [v] for k, v in row_data.items()}).reindex(columns=all_columns)
    
    # Ensure no array-like values
    for col in new_row.columns:
        if np.ndim(new_row[col].iloc[0]) > 0:
            new_row[col] = new_row[col].iloc[0].item() if hasattr(new_row[col].iloc[0], 'item') else new_row[col].iloc[0][0]
    
    return new_row

def run_inference_page():
    st.title("Starlink Adoption Prediction")
    
    # Load model and sample data
    final_model = load_model()
    if final_model is None:
        return

    try:
        sample_df = pd.read_csv('starlink_household_synthetic.csv')
        all_columns = sample_df.drop(columns=['starlink_proxy_adoption'], errors='ignore').columns.tolist()
    except FileNotFoundError:
        st.error("Could not load sample data for column reference")
        return

    # Input form
    with st.form("prediction_form"):
        st.subheader("Household Details")
        
        # Organized input sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Location & Demographics**")
            district = st.selectbox("District", sorted(sample_df['district'].unique()))
            province = st.selectbox("Province", sorted(sample_df['province'].unique()))
            urbanization_level = st.selectbox("Urbanization Level", sorted(sample_df['urbanization_level'].unique()))
            members_in_house = st.number_input("Household Members", min_value=1, value=4)
            roofless_persons = st.number_input("Roofless Persons", min_value=0, value=0)
            income_lkr = st.number_input("Monthly Income (LKR)", min_value=0.0, value=120000.0, step=1000.0)
        
        with col2:
            st.write("**Connectivity & Assets**")
            electricity_available = st.selectbox("Electricity", [1, 0], format_func=lambda x: "Yes" if x else "No")
            has_4g_coverage = st.selectbox("4G Coverage", [1, 0], format_func=lambda x: "Yes" if x else "No")
            has_fiber_coverage = st.selectbox("Fiber Coverage", [1, 0], format_func=lambda x: "Yes" if x else "No")
            smartphone_count = st.number_input("Smartphones", min_value=0, value=3)
            computers_owned = st.number_input("Computers", min_value=0, value=1)
        
        # Additional sections...
        
        submitted = st.form_submit_button("Predict Adoption")

    if submitted:
        # Prepare input data
        input_data = {
            "district": district,
            "province": province,
            "urbanization_level": urbanization_level,
            "members_in_house": members_in_house,
            "roofless_persons": roofless_persons,
            "income_lkr": income_lkr,
            "smartphone_count": smartphone_count,
            "computers_owned": computers_owned,
            "electricity_available": electricity_available,
            "has_4g_coverage": has_4g_coverage,
            "has_fiber_coverage": has_fiber_coverage,
            # Add all other fields...
        }

        # Create inference row
        inference_df = create_inference_row(input_data, all_columns)
        
        try:
            # Make prediction
            pred_prob = final_model.predict_proba(inference_df)[0, 1]
            prediction = final_model.predict(inference_df)[0]
            
            # Display results
            st.success(f"Predicted Adoption Probability: {pred_prob:.1%}")
            st.write(f"Recommended Action: {'Target this household' if prediction else 'Low priority'}")
            
            # SHAP explanation
            if hasattr(final_model, 'named_steps'):
                explainer = shap.TreeExplainer(final_model.named_steps['classifier'])
                prep_data = final_model.named_steps['preprocessor'].transform(inference_df)
                shap_values = explainer.shap_values(prep_data)
                
                st.subheader("Feature Impact")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values[1], prep_data, 
                                feature_names=final_model.named_steps['preprocessor'].get_feature_names_out(),
                                plot_type="bar", show=False)
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_inference_page()
