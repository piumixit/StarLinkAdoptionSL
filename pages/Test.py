# pages/inference_page.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# Load the pre-trained model and explainer (cached for efficiency)
@st.cache_resource
def load_model_and_explainer(model_path="starlink_final_model.pkl"):
    """Loads the final model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    # explainer_path="starlink_explainer.pkl" # Explainer loading is handled in modeling page if needed

# Function to create a DataFrame for a single row with all necessary columns
def create_inference_dataframe(input_data, all_columns):
    """Creates a DataFrame for a single inference request."""
    # Initialize with NaN for all expected columns
    row_data = {col: np.nan for col in all_columns}
    # Update with provided input data
    row_data.update(input_data)
    # Ensure the order of columns matches the training data
    new_row = pd.DataFrame({k: [v] for k, v in row_data.items()}).reindex(columns=all_columns)
    return new_row

def run_inference_page():
    """Code for the Inference page."""
    st.title("Starlink Adoption Prediction")
    st.write("Enter household details to predict the likelihood of Starlink adoption.")

    # Load the model
    final_model = load_model_and_explainer()

    if final_model is None:
        st.warning("Model not loaded. Please ensure 'starlink_final_model.pkl' exists.")
        return

    # Define the columns expected by the model's preprocessor
    # We need access to the original training data columns before preprocessing
    # A robust way is to save the list of columns during training or load a sample
    # of the training data to get the column names and dtypes.

    # For simplicity, let's assume we can load the original data file to get column names
    try:
        sample_df = pd.read_csv('starlink_household_synthetic.csv')
        # Drop the target and the column dropped during EDA
        all_columns = sample_df.drop(columns=['starlink_proxy_adoption', 'roof_persons'], errors='ignore').columns.tolist()
    except FileNotFoundError:
        st.error("Could not load sample data to determine expected columns.")
        return

    st.subheader("Enter Household Details")

    # Create input fields for key features identified from EDA/Modeling
    # Add input fields for the most important features based on SHAP or feature importance
    # You can add more fields as needed. Using selectbox for categorical features.

    # Example input fields (add more based on your feature importance analysis)
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Location & Demographics")
        district = st.selectbox("District", sorted(sample_df['district'].unique()))
        province = st.selectbox("Province", sorted(sample_df['province'].unique()))
        urbanization_level = st.selectbox("Urbanization Level", sorted(sample_df['urbanization_level'].unique()))
        members_in_house = st.number_input("Members in Household", min_value=1, value=4, step=1)
        income_lkr = st.number_input("Average Monthly Income (LKR)", min_value=0.0, value=120000.0, step=1000.0)
        roofless_persons = st.number_input("Roofless Persons", min_value=0, value=0, step=1)
        # roof_persons is dropped, calculate it internally if needed but not as input
        roof_persons = members_in_house - roofless_persons

    with col2:
        st.write("#### Connectivity & Tech")
        electricity_available = st.selectbox("Electricity Available", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        has_4g_coverage = st.selectbox("Has 4G Coverage", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        has_fiber_coverage = st.selectbox("Has Fiber Coverage", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        downlink_speed_mbps = st.number_input("Average Downlink Speed (Mbps)", min_value=0.0, value=6.2, step=0.1)
        uplink_speed_mbps = st.number_input("Average Uplink Speed (Mbps)", min_value=0.0, value=1.1, step=0.1)
        avg_monthly_broadband_bill_lkr = st.number_input("Avg Monthly Broadband Bill (LKR)", min_value=0.0, value=3500.0, step=100.0)
        computers_owned = st.number_input("Computers Owned", min_value=0, value=1, step=1)
        smartphone_count = st.number_input("Smartphone Count", min_value=0, value=3, step=1)


    st.write("#### Sentiment & Digital Adoption")
    col3, col4 = st.columns(2)
    with col3:
        overall_bb_satisfaction_score = st.slider("Overall BB Satisfaction Score (-1 to +1)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        dialog_sentiment = st.slider("Dialog Sentiment (-1 to +1)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        slt_sentiment = st.slider("SLT Sentiment (-1 to +1)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        hutch_sentiment = st.slider("Hutch Sentiment (-1 to +1)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    with col4:
        digital_literacy_score = st.slider("Digital Literacy Score (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        ecommerce_usage_score = st.slider("E-commerce Usage Score (0 to 1)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        starlink_awareness = st.selectbox("Starlink Awareness", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        starlink_sentiment_score = st.slider("Starlink Sentiment Score (-1 to +1)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)


    # Subscription counts (using number inputs)
    st.write("#### Subscription Counts")
    col5, col6 = st.columns(2)
    with col5:
        dialog_subscriptions = st.number_input("Dialog Subscriptions", min_value=0, value=1, step=1)
        slt_subscriptions = st.number_input("SLT Subscriptions", min_value=0, value=0, step=1)
    with col6:
        hutch_subscriptions = st.number_input("Hutch Subscriptions", min_value=0, value=0, step=1)
        other_subscriptions = st.number_input("Other Subscriptions", min_value=0, value=0, step=1)

    # Add other potentially relevant features from the original df if desired
    # Based on the original df columns, we have:
    # 'household_id' (not needed for inference)
    # 'starlink_prob_proxy' (this is the target proxy, not an input feature for the final model)

    # Create a dictionary with the input data
    input_data = {
        "district": district,
        "province": province,
        "urbanization_level": urbanization_level,
        "members_in_house": members_in_house,
        "roofless_persons": roofless_persons,
        "roof_persons": roof_persons, # Included for completeness, but dropped in EDA
        "income_lkr": income_lkr,
        "smartphone_count": smartphone_count,
        "computers_owned": computers_owned,
        "electricity_available": electricity_available,
        "has_4g_coverage": has_4g_coverage,
        "has_fiber_coverage": has_fiber_coverage,
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
        "starlink_awareness": starlink_awareness,
        "starlink_sentiment_score": starlink_sentiment_score,
        # Add any other necessary features here based on the training data columns
    }

    # Create the DataFrame for prediction
    inference_df = create_inference_dataframe(input_data, all_columns)


    if st.button("Predict Starlink Adoption"):
        # Perform inference
        try:
            pred_prob = final_model.predict_proba(inference_df)[0, 1]
            prediction = final_model.predict(inference_df)[0]
            prediction_label = "Yes" if prediction == 1 else "No"

            st.subheader("Prediction Results")
            st.write(f"Predicted Starlink adoption probability: **{pred_prob:.3f}**")
            st.write(f"Predicted Adoption: **{prediction_label}**")

            # Optional: Add SHAP explanation for the single instance
            st.subheader("Prediction Explanation (SHAP)")
            st.write("How individual features influenced this prediction:")

            # Need to regenerate explainer if it's not saved or if model type requires specific explainer
            # For TreeExplainer, we can regenerate it from the model
            model_step = final_model.named_steps["model"]
            try:
                explainer = shap.TreeExplainer(model_step)
                st.write("Using TreeExplainer for explanation.")
            except Exception as e:
                 st.warning(f"Could not use TreeExplainer for explanation: {e}. SHAP explanation skipped.")
                 explainer = None # Do not proceed with SHAP if explainer fails


            if explainer:
                # Preprocess the single instance
                inference_prep = final_model.named_steps["prep"].transform(inference_df)

                # Convert to dense numeric array if sparse
                if hasattr(inference_prep, 'toarray'):
                    inference_prep_dense = inference_prep.toarray().astype(np.float32)
                else:
                    inference_prep_dense = inference_prep.astype(np.float32)

                # Calculate SHAP values for the single instance
                if hasattr(model_step, "predict_proba"):
                    shap_values_instance = explainer.shap_values(inference_prep_dense)[1] # Get values for positive class
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                     shap_values_instance = explainer.shap_values(inference_prep_dense)
                     expected_value = explainer.expected_value


                # Get feature names after preprocessing
                feature_names = final_model.named_steps["prep"].get_feature_names_out()
                 # Create a DataFrame for the instance with correct column names
                inference_plot_df = pd.DataFrame(inference_prep_dense, columns=feature_names)


                # Generate the HTML for the force plot
                # Use shap.display.force_plot which is designed for display environments
                # We need to capture its output or use components.html
                # Another approach is to save to HTML and load

                # Let's use the matplotlib version for simplicity in Streamlit
                fig_instance_force, ax_instance_force = plt.subplots(figsize=(10, 3))
                shap.force_plot(expected_value, shap_values_instance, inference_plot_df.iloc[0], matplotlib=True, show=False, ax=ax_instance_force)
                plt.tight_layout()
                st.pyplot(fig_instance_force)
                plt.close(fig_instance_force)


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    run_inference_page()
