import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "EDA", "Data Preprocessing"])

if page == "Home":
    st.title("Main App")
    st.write("Welcome to the main page!")
    
elif page == "EDA":
    st.title("Starlink Exploratory Data Analysis")
    st.write("Simple EDA with Streamlit")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Basic info section
        with st.expander("Basic Information"):
            st.write(f"Shape: {df.shape}")
            st.write("First 5 rows:")
            st.dataframe(df.head())
        
        # Data types section
        with st.expander("Data Types & Missing Values"):
            st.dataframe(pd.DataFrame({
                'Data Type': df.dtypes,
                'Missing Values': df.isnull().sum(),
                'Unique Values': df.nunique()
            }))
        
        # Numeric analysis section
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            with st.expander("Numeric Analysis"):
                selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Summary Statistics:")
                    st.dataframe(df[selected_num_col].describe().to_frame())
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_num_col], kde=True, ax=ax)
                    st.pyplot(fig)
        
        # Categorical analysis section
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            with st.expander("Categorical Analysis"):
                selected_cat_col = st.selectbox("Select categorical column:", cat_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Value Counts:")
                    st.dataframe(df[selected_cat_col].value_counts())
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.countplot(x=selected_cat_col, data=df, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
    else:
        st.info("Please upload a CSV file to begin EDA")

elif page == "Data Preprocessing":
    st.switch_page("pages/Preprocessing.py")
