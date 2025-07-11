# pages/eda_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Define feature columns
    feature_columns = [
        'avg_monthly_broadband_bill_lkr',
        'income_lkr',
        'downlink_speed_mbps',
        'overall_bb_satisfaction_score',
        'starlink_awareness',
        'digital_literacy_score'
    ]
    adoption_col = "starlink_proxy_adoption"

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Distributions", 
        "Relationships", 
        "Urbanization", 
        "Correlations"
    ])

    with tab1:  # Overview tab
        st.subheader("Dataset Overview")
        st.write(f"Dataset shape: {df1.shape}")
        
        # Target Variable Distribution
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax = sns.countplot(data=df1, x=adoption_col, palette="viridis")
        plt.title("Starlink Adoption Distribution")
        plt.xlabel("Adopted (1 = Yes)")
        plt.ylabel("Household Count")
        
        # Add count labels on bars
        for p in ax.patches:
            count = int(p.get_height())
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 50,
                    str(count), ha='center', va='bottom', fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:  # Distributions tab
        # Feature Distributions
        # st.subheader("Feature Distributions")
        # fig = plt.figure(figsize=(15, 10))
        # df1[feature_columns].hist(bins=30, color="skyblue", edgecolor='black')
        # plt.suptitle("Feature Distributions", fontsize=16)
        # plt.tight_layout()
        # st.pyplot(fig)
        # plt.close(fig)

        # Feature Distributions - Fixed and Enhanced Version
        st.subheader("Feature Distributions")
        
        # Create figure and axes explicitly
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()  # Flatten for easy iteration
        
        # Plot each feature's histogram
        for i, col in enumerate(feature_columns):
            axes[i].hist(df1[col], bins=30, color="skyblue", edgecolor='black')
            axes[i].set_title(col, fontsize=12)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            
            # Add grid for better readability
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels if needed
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide any empty subplots if we have fewer than 6 features
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle("Feature Distributions", fontsize=16, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Boxplots by Adoption Status
        st.subheader("Feature Distributions by Adoption Status")
        fig = plt.figure(figsize=(18, 10))
        for i, col in enumerate(feature_columns):
            plt.subplot(2, 3, i + 1)
            sns.boxplot(data=df1, x=adoption_col, y=col, palette="Set2")
            plt.title(f"{col} by Adoption Status")
            plt.xlabel("")
            plt.ylabel(col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:  # Relationships tab
        # Pairplot
        st.subheader("Pairwise Feature Relationships")
        pair_grid = sns.pairplot(df1[feature_columns + [adoption_col]],
                         hue=adoption_col,
                         palette="husl", plot_kws={"alpha": 0.6, "s": 30})
        pair_grid.fig.suptitle("Pairwise Feature Relationships", y=1.02)
        st.pyplot(pair_grid.fig)
        plt.close(pair_grid.fig)

    with tab4:  # Urbanization tab
        # Urbanization Analysis
        st.subheader("Adoption by Urbanization Level")
        fig = plt.figure(figsize=(7, 4))
        sns.countplot(data=df1, x="urbanization_level", hue=adoption_col, palette="Set1")
        plt.title("Adoption by Urbanization Level")
        plt.xlabel("Urbanization")
        plt.ylabel("Household Count")
        plt.legend(title="Adopted Starlink")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab5:  # Correlations tab
        # Correlation Matrix
        st.subheader("Correlation Matrix for Numerical Features")
        numeric_cols = df1.select_dtypes(include=["int64", "float64"]).columns
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df1[numeric_cols].corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature correlations")
        st.pyplot(fig)
        plt.close(fig)

        # Missing Value Heatmap
        st.subheader("Missing Value Heatmap")
        st.write("Observation - No missing values since data was synthetically generated.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df1.isna(), cbar=False, ax=ax)
        ax.set_title("Missing-value map")
        st.pyplot(fig)
        plt.close(fig)

if __name__ == "__main__":
    run_eda_page()
