import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Data Preprocessing")
    # Your preprocessing code here

# This makes the page work both as standalone and as module
if __name__ == "__main__":
    main()
    
# Page configuration
st.set_page_config(
    page_title="Preprocessing",
    layout="wide"
)

st.title("Data Preprocessing")
st.write("Analyze and handle missing values in your dataset")

# Session state to store the processed dataframe
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Store in session state
    if 'df_original' not in st.session_state:
        st.session_state.df_original = df.copy()
    
    # Section 1: Missing Value Analysis
    st.header("Missing Value Analysis")
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    # Create a DataFrame for display
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage (%)': missing_percent.values.round(2)
    }).sort_values('Percentage (%)', ascending=False)
    
    # Display missing value info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Missing Values Summary:")
        st.dataframe(missing_df.style.background_gradient(cmap='Reds'))
    
    with col2:
        st.write("Missing Values Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)
    
    # Section 2: Preprocessing Options
    st.header("Preprocessing Methods")
    
    # Select columns with missing values
    cols_with_missing = missing_df[missing_df['Missing Values'] > 0]['Column'].tolist()
    
    if cols_with_missing:
        selected_col = st.selectbox("Select column to process:", cols_with_missing)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Drop Column"):
                st.session_state.df_processed = df.drop(columns=[selected_col])
                st.success(f"Dropped column: {selected_col}")
        
        with col2:
            if st.button("Fill with Mean"):
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    st.session_state.df_processed = df.fillna({selected_col: df[selected_col].mean()})
                    st.success(f"Filled {selected_col} with mean value")
                else:
                    st.error("Cannot fill non-numeric column with mean")
        
        with col3:
            if st.button("Fill with Mode"):
                st.session_state.df_processed = df.fillna({selected_col: df[selected_col].mode()[0]})
                st.success(f"Filled {selected_col} with mode value")
        
        # Show processed data
        if st.session_state.df_processed is not None:
            st.header("Processed Data Preview")
            st.dataframe(st.session_state.df_processed.head())
            
            # Download button
            csv = st.session_state.df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv'
            )
    else:
        st.success("No missing values found in the dataset!")
else:
    st.info("Please upload a CSV file to begin preprocessing")
