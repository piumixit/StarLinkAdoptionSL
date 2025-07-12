# pages/eda_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def run_eda_page():
    st.title("Exploratory Data Analysis")

    # Load the dataset
    file_path = 'starlink_household_synthetic.csv' 
    try:
        df1 = pd.read_csv(file_path)
        st.success("Dataset loaded successfully.")
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {file_path}")
        return

    st.subheader("Dataset Information")
    #st.write("Dataset shape:", df1.shape)

    # Capture df.info() output
    #buffer = io.StringIO()
    #df1.info(buf=buffer)
    #info_str = buffer.getvalue()
    #st.write("\nColumn data types and non-null counts:")
    #st.text(info_str)

    st.subheader("Data")
    st.dataframe(df1.describe())

    st.subheader("Numeric Summary")
    st.dataframe(df1.describe().T)

    st.subheader("Target Variable Distribution")
    target_col = "starlink_proxy_adoption"
    st.write(df1[target_col].value_counts(dropna=False))

    st.subheader("Missing Value Heatmap")
    st.write("Observation - No missing values since data for final analysis was synthetically generated.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df1.isna(), cbar=False, ax=ax)
    ax.set_title("Missing-value map")
    st.pyplot(fig)
    plt.close(fig) # Close the figure to prevent it from displaying automatically

    st.subheader("Correlation Matrix for Numerical Features")
    numeric_cols = df1.select_dtypes(include=["int64", "float64"]).columns
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df1[numeric_cols].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlations")
    st.pyplot(fig)
    plt.close(fig) # Close the figure

    st.subheader("Dropping Unnecessary Columns")
    df1 = df1.drop(columns=['roof_persons'], errors='ignore') # Use errors='ignore' in case it was already dropped
    st.write("Dropped 'roof_persons' column.")
    st.dataframe(df1.head(2)) # Display head after dropping

    st.subheader("Target-Conditioned Box/Violin Plots")
    numeric_after_drop = df1.select_dtypes(include=["int64", "float64"])
    # Select a subset of numeric columns for plotting to avoid too many plots
    plot_cols = numeric_after_drop.columns[:12] if len(numeric_after_drop.columns) > 12 else numeric_after_drop.columns

    num_cols = len(plot_cols)
    num_rows = (num_cols + 3) // 4 # Arrange plots in a grid
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, num_rows * 4))
    axes = axes.ravel() if num_rows > 1 else [axes] # Flatten axes array for easy iteration if multiple rows

    for i, col in enumerate(plot_cols):
        sns.violinplot(data=df1, x="starlink_proxy_adoption",
                       y=col, ax=axes[i], inner="quart")
        axes[i].set_xlabel("")
        axes[i].set_ylabel(col)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Numeric distributions by adoption status")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for title
    st.pyplot(fig)
    plt.close(fig) # Close the figure

if __name__ == "__main__":
    run_eda_page()
