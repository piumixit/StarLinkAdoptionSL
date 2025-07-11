import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(
    page_title="Starlink Sri Lanka Adoption Analysis",
    page_icon="🛰️",
    layout="wide"
)

# Generate synthetic data for demonstration
def generate_data():
    np.random.seed(42)
    months = pd.date_range(start='2023-01', end='2025-12', freq='M')
    n = len(months)
    
    # Create three distinct groups in the data
    data = {
        'Month': months,
        'Adoption': np.concatenate([
            np.round(np.random.lognormal(mean=3, sigma=0.2, size=n//3)).astype(int),
            np.round(np.random.lognormal(mean=4, sigma=0.2, size=n//3)).astype(int),
            np.round(np.random.lognormal(mean=5, sigma=0.2, size=n//3)).astype(int)
        ]),
        'GDP_Growth': np.concatenate([
            np.random.normal(loc=3.0, scale=0.5, size=n//3),
            np.random.normal(loc=4.0, scale=0.5, size=n//3),
            np.random.normal(loc=5.0, scale=0.5, size=n//3)
        ]),
        'Urban_Population': np.random.normal(loc=18.5, scale=0.2, size=n),
        'Internet_Penetration': np.concatenate([
            np.random.normal(loc=45, scale=1, size=n//3),
            np.random.normal(loc=50, scale=1, size=n//3),
            np.random.normal(loc=55, scale=1, size=n//3)
        ]),
        'Marketing_Spend': np.concatenate([
            np.random.lognormal(mean=2, sigma=0.3, size=n//3)*10000,
            np.random.lognormal(mean=2.5, sigma=0.3, size=n//3)*10000,
            np.random.lognormal(mean=3, sigma=0.3, size=n//3)*10000
        ]),
        'Competitor_Price': np.random.normal(loc=4500, scale=300, size=n)
    }
    
    # Add some seasonality
    data['Adoption'] = data['Adoption'] * (1 + 0.2*np.sin(np.arange(n)/12*2*np.pi))
    data['Adoption'] = np.round(data['Adoption']).astype(int)
    
    df = pd.DataFrame(data)
    df['Month'] = pd.to_datetime(df['Month'])
    return df

# Load data
@st.cache_data
def load_data():
    return generate_data()

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "DBSCAN Clustering Analysis"])

# Common styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛰️ Starlink Adoption Analysis for Sri Lanka")
st.markdown("""
This application explores patterns in Starlink internet service adoption in Sri Lanka 
using DBSCAN clustering to identify distinct groups in the data.
""")

if page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.write("Explore the synthetic dataset representing potential factors affecting Starlink adoption in Sri Lanka.")
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(df)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Time series plot
    st.subheader("Starlink Adoption Over Time")
    fig = px.line(df, x='Month', y='Adoption', 
                  title='Monthly Starlink Adoption in Sri Lanka',
                  labels={'Adoption': 'Number of Subscribers'},
                  template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr_matrix = df.corr(numeric_only=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature to visualize", 
                          ['GDP_Growth', 'Urban_Population', 'Internet_Penetration', 
                           'Marketing_Spend', 'Competitor_Price'])
    
    fig = px.histogram(df, x=feature, nbins=30, 
                      title=f'Distribution of {feature}',
                      template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature vs Adoption
    st.subheader("Feature vs Adoption")
    feature_scatter = st.selectbox("Select feature to compare with adoption", 
                                  ['GDP_Growth', 'Urban_Population', 'Internet_Penetration', 
                                   'Marketing_Spend', 'Competitor_Price'])
    
    fig = px.scatter(df, x=feature_scatter, y='Adoption',
                    title=f'{feature_scatter} vs Starlink Adoption',
                    labels={'Adoption': 'Number of Subscribers'},
                    template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

elif page == "DBSCAN Clustering Analysis":
    st.header("DBSCAN Clustering Analysis")
    st.write("""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is used to identify 
    dense clusters in the data. This can help discover different adoption patterns or regimes.
    """)
    
    # Parameter selection
    st.subheader("DBSCAN Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        eps = st.slider("EPS (neighborhood radius)", 
                       min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    with col2:
        min_samples = st.slider("Minimum samples in neighborhood", 
                              min_value=1, max_value=20, value=5, step=1)
    
    # Feature selection
    st.subheader("Feature Selection for Clustering")
    features = st.multiselect("Select features to include in clustering",
                             ['GDP_Growth', 'Urban_Population', 'Internet_Penetration',
                              'Marketing_Spend', 'Competitor_Price', 'Adoption'],
                             default=['GDP_Growth', 'Internet_Penetration', 'Adoption'])
    
    if not features:
        st.warning("Please select at least one feature for clustering.")
        st.stop()
    
    # Prepare data
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Add clusters to dataframe
    df['Cluster'] = clusters
    
    # Cluster statistics
    st.subheader("Cluster Information")
    st.write(f"Number of clusters found: {len(np.unique(clusters)) - (1 if -1 in clusters else 0)}")
    st.write(f"Number of noise points: {np.sum(clusters == -1)}")
    
    if len(np.unique(clusters)) > 1:
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        st.write("""
        - Scores near +1 indicate dense, well-separated clusters
        - Scores near 0 indicate overlapping clusters
        - Scores near -1 indicate incorrect clustering
        """)
    
    # Cluster visualization
    st.subheader("Cluster Visualization")
    
    # Use PCA for dimensionality reduction if more than 2 features
    if len(features) > 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        x_axis, y_axis = 0, 1
        x_label = f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        y_label = f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        X_reduced = X_scaled
        x_axis, y_axis = 0, 1
        x_label = features[0]
        y_label = features[1] if len(features) > 1 else ""
    
    # Create cluster plot
    cluster_df = pd.DataFrame({
        'x': X_reduced[:, x_axis],
        'y': X_reduced[:, y_axis],
        'Cluster': clusters,
        'Adoption': df['Adoption'],
        'Month': df['Month'].dt.strftime('%Y-%m')
    })
    
    fig = px.scatter(cluster_df, x='x', y='y', color='Cluster',
                    hover_data=['Month', 'Adoption'],
                    title='DBSCAN Clustering Results',
                    labels={'x': x_label, 'y': y_label},
                    color_continuous_scale=px.colors.qualitative.Plotly,
                    template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    
    if len(np.unique(clusters)) > 1:
        # Show mean values for each cluster
        cluster_stats = df.groupby('Cluster')[features + ['Adoption']].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        
        # Time series by cluster
        st.write("Adoption Over Time by Cluster")
        fig = px.line(df, x='Month', y='Adoption', color='Cluster',
                     title='Starlink Adoption by Cluster',
                     template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots for each feature by cluster
        st.write("Feature Distributions by Cluster")
        feature_box = st.selectbox("Select feature to view by cluster", features + ['Adoption'])
        
        fig = px.box(df, x='Cluster', y=feature_box,
                    title=f'{feature_box} Distribution by Cluster',
                    template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Only one cluster found. Try adjusting the parameters to find more clusters.")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This application uses synthetic data for demonstration purposes. 
DBSCAN is particularly useful for identifying dense clusters and outliers in the data.
""")