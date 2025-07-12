import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

# Set page config
st.set_page_config(page_title="Sri Lanka Telecom Analysis", layout="wide")

# Navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "EDA", "Telecom Analysis", "Advanced Metrics"])

# Initialize random number generator
rng = np.random.default_rng(42)

# Constants
AVG_HH_SIZE = 3.8
WET_PROVINCES = {'Western','Central','Sabaragamuwa','Southern'}
HIGHLAND_DISTRICTS = {'Kandy','Matale','Nuwara Eliya','Badulla','Monaragala','Ratnapura','Kegalle'}
URB_MAP = {'Urban': 'urban', 'Semi-Urban': 'semi', 'Rural': 'rural'}

CENTROIDS = {
    'Colombo': (6.9271, 79.8612), 'Gampaha': (7.0873, 79.9970),
    'Kalutara': (6.5880, 79.9590), 'Kandy': (7.2906, 80.6337),
    'Matale': (7.4717, 80.6234), 'Nuwara Eliya': (6.9497, 80.7890),
    'Galle': (6.0535, 80.2200), 'Matara': (5.9496, 80.5481),
    'Hambantota': (6.1240, 81.1185), 'Jaffna': (9.6685, 80.0074),
    'Kilinochchi': (9.3890, 80.3866), 'Mannar': (8.9778, 79.9044),
    'Vavuniya': (8.7511, 80.4982), 'Mullaitivu': (9.2670, 80.8105),
    'Batticaloa': (7.7102, 81.6932), 'Ampara': (7.2999, 81.6779),
    'Trincomalee': (8.5875, 81.2152), 'Kurunegala': (7.4865, 80.3649),
    'Puttalam': (8.0335, 79.8428), 'Anuradhapura': (8.3114, 80.4037),
    'Polonnaruwa': (7.9403, 81.0188), 'Badulla': (6.9846, 81.0561),
    'Monaragala': (6.8904, 81.3448), 'Ratnapura': (6.7056, 80.3847),
    'Kegalle': (7.2512, 80.3464)
}

def create_district_data():
    districts = [
        "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
        "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar", 
        "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee", 
        "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla", 
        "Monaragala", "Ratnapura", "Kegalle"
    ]
    
    df = pd.DataFrame(districts, columns=["district"])
    
    # Basic demographics
    df["population"] = [2374461, 2433685, 1305552, 1461269, 526578, 724957, 
                       1096585, 837884, 671037, 594333, 136434, 123674, 
                       172257, 122542, 595435, 744150, 442465, 1760829, 
                       818065, 959552, 447338, 871763, 527286, 1145138, 869901]
    df["roofless_persons"] = [536, 229, 76, 139, 21, 13, 139, 133, 35, 52, 
                              0, 4, 32, 2, 29, 20, 24, 135, 113, 180, 27, 
                              72, 53, 160, 57]
    df["populationDensity_people_km2"] = [3549, 1774, 804, 779, 267, 427, 
                                        694, 648, 272, 630, 115, 66, 96, 50, 
                                        251, 178, 180, 379, 278, 147, 137, 
                                        314, 96, 355, 529]
    
    # Housing metrics
    df["roof_persons"] = df["population"] - df["roofless_persons"]
    df["roofless_percentage"] = (df["roofless_persons"]/df["population"]*100).round(2)
    df["roof_percentage"] = (df["roof_persons"]/df["population"]*100).round(2)
    
    # Economic metrics
    df["avg_income_lkr_2019"] = [132433, 100455, 84887, 74821, 54910, 54504, 
                                70681, 65323, 68528, 55380, 44004, 50978, 
                                68859, 48835, 44686, 60474, 46341, 70049, 
                                85897, 64409, 65180, 66413, 55221, 52956, 60828]
    
    # Inflation adjustment
    inflation_factors = [1.046, 1.060, 1.464, 1.159, 1.10, 1.05]
    cumulative_inflation = round(reduce(lambda x, y: x * y, inflation_factors), 3)
    df["avg_income_lkr_2025"] = (df["avg_income_lkr_2019"] * cumulative_inflation).round(0).astype(int)
    
    # Province information
    province_to_districts = {
        "Western": ["Colombo", "Gampaha", "Kalutara"],
        "Central": ["Kandy", "Matale", "Nuwara Eliya"],
        "Southern": ["Galle", "Matara", "Hambantota"],
        "Northern": ["Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"],
        "Eastern": ["Batticaloa", "Ampara", "Trincomalee"],
        "North-western": ["Kurunegala", "Puttalam"],
        "North-central": ["Anuradhapura", "Polonnaruwa"],
        "Uva": ["Badulla", "Monaragala"],
        "Sabaragamuwa": ["Ratnapura", "Kegalle"]
    }
    
    district_to_province = {}
    for province, districts in province_to_districts.items():
        for district in districts:
            district_to_province[district] = province
    
    df["province"] = df["district"].map(district_to_province)
    
    # Computer ownership
    province_ownership = {
        "Western": 31.1,
        "Central": 16.6,
        "Southern": 15.9,
        "Northern": 12.6,
        "Eastern": 12.7,
        "North-western": 16.7,
        "North-central": 11.5,
        "Uva": 13.1,
        "Sabaragamuwa": 17.4
    }
    
    province_pop = df.groupby("province")["population"].sum().to_dict()
    df["pop_ratio_in_province"] = df.apply(
        lambda x: x["population"] / province_pop[x["province"]] if x["province"] in province_pop else None,
        axis=1
    )
    
    df["computer_ownership_pct_2024_weighted"] = df.apply(
        lambda x: x["pop_ratio_in_province"] * province_ownership[x["province"]] if x["province"] in province_ownership else None,
        axis=1
    ).round(2)
    
    # ISP market share
    total_subscribers = 21569582  # Q1 2025
    isp_weights = {
        "Dialog": 0.57,
        "SLT_Mobitel": 0.26,
        "Hutch": 0.12,
        "Other": 0.05
    }
    
    df["district_pop_pct"] = df["population"] / df["population"].sum()
    for isp, share in isp_weights.items():
        col_name = f"{isp.lower().replace('-', '_')}_subs_est"
        df[col_name] = (df["district_pop_pct"] * total_subscribers * share).round().astype(int)
    
    # Urbanization classification
    def classify_urbanization(density):
        if density >= 1500:
            return "Urban"
        elif density >= 600:
            return "Semi-Urban"
        else:
            return "Rural"
    
    df["urbanization_level"] = df["populationDensity_people_km2"].apply(classify_urbanization)
    
    # Household metrics
    df["avg_household_size"] = AVG_HH_SIZE
    df["households_est"] = (df["population"] / AVG_HH_SIZE).round().astype(int)
    
    # Power availability
    if "power_availability_pct" not in df.columns:
        df["power_availability_pct"] = rng.normal(98.0, 0.5, size=len(df)).round(1)
    
    df["pct_households_with_electricity"] = df["power_availability_pct"]
    
    # Internet speeds
    def draw_dl_speed(level):
        return max(4, rng.normal({"urban": 26, "semi": 21, "rural": 16}[level], 3))
    
    df["avg_dl_speed_mbps"] = (df["urbanization_level"]
                               .map(URB_MAP)
                               .apply(draw_dl_speed)
                               .round(2))
    
    df["avg_ul_speed_mbps"] = (df["avg_dl_speed_mbps"] *
                               rng.uniform(0.25, 0.40, size=len(df))).round(2)
    
    # Broadband penetration
    def bb_pen(level):
        base = {"urban": 55, "semi": 35, "rural": 20}[level]
        return np.clip(rng.normal(base, 5), 5, 95)
    
    df["fixed_bb_penetration_pct"] = (df["urbanization_level"]
                                      .map(URB_MAP)
                                      .apply(bb_pen).round(1))
    
    # Coverage metrics
    df["4g_coverage_pct"] = rng.normal(96, 1.5, len(df)).round(1)
    
    def fiber_cov(level):
        return np.clip(rng.normal({"urban": 60, "semi": 35, "rural": 15}[level], 10), 0, 100)
    
    df["fiber_coverage_pct"] = (df["urbanization_level"]
                                 .map(URB_MAP)
                                 .apply(fiber_cov).round(1))
    
    # Pricing
    def bb_price(level):
        return max(2500, rng.normal({"urban": 5000, "semi": 4500, "rural": 4000}[level], 300))
    
    df["avg_monthly_bb_price_lkr"] = (df["urbanization_level"]
                                      .map(URB_MAP)
                                      .apply(bb_price).round().astype(int))
    
    # Sentiment analysis
    def dialog_sent(level):  return rng.normal(0.05 if level=="urban" else 0.00, 0.15)
    def slt_sent(level):     return rng.normal(0.10 if level!="rural" else 0.02, 0.15)
    def hutch_sent(level):   return rng.normal(-0.05, 0.15)
    
    level_series = df["urbanization_level"].map(URB_MAP)
    
    df["dialog_sentiment"] = level_series.apply(dialog_sent).round(3)
    df["slt_mobitel_sentiment"] = level_series.apply(slt_sent).round(3)
    df["hutch_sentiment"] = level_series.apply(hutch_sent).round(3)
    
    df["overall_bb_satisfaction_score"] = (df[["dialog_sentiment", "slt_mobitel_sentiment", "hutch_sentiment"]]
                                           .mean(axis=1).round(3))
    
    # Starlink metrics
    def awareness(level):
        low, high = {"urban": (35, 50), "semi": (25, 40), "rural": (15, 30)}[level]
        return rng.uniform(low, high)
    
    df["starlink_awareness_pct"] = level_series.apply(awareness).round(1)
    df["starlink_sentiment_score"] = rng.normal(0.25, 0.20, len(df)).round(3)
    
    # Digital adoption
    def smartphone(level):
        return np.clip(rng.normal({"urban": 85, "semi": 75, "rural": 65}[level], 4), 40, 95)
    
    df["smartphone_penetration_pct"] = level_series.apply(smartphone).round(1)
    
    df["digital_literacy_index"] = (level_series
                                    .apply(lambda l: rng.normal({"urban": 78, "semi": 68, "rural": 58}[l], 3))
                                    .clip(40, 90).round(1))
    
    df["ecommerce_usage_pct"] = (level_series
                                 .apply(lambda l: rng.normal({"urban": 40, "semi": 25, "rural": 15}[l], 5))
                                 .clip(5, 60).round(1))
    
    # Economic indicators
    def poverty(level):
        return max(3, rng.normal({"urban": 5, "semi": 10, "rural": 15}[level], 3))
    
    df["poverty_rate_pct"] = level_series.apply(poverty).round(1)
    df["inflation_24_pct"] = 7.2  # Central Bank YoY, 2024
    df["fuel_price_index"] = 109   # Jan-2025 baseline = 100
    
    # Environmental factors
    def rainfall(prov):
        wet = prov in WET_PROVINCES
        return rng.normal(2300 if wet else 1200, 200)
    
    df["avg_annual_rainfall_mm"] = df["province"].apply(rainfall).round().astype(int)
    
    df["thunderstorm_days"] = (df["province"]
                               .apply(lambda p: rng.normal(95 if p in WET_PROVINCES else 65, 8))
                               .round().astype(int))
    
    df["cloud_cover_pct"] = (df["province"]
                             .apply(lambda p: rng.normal(65 if p in WET_PROVINCES else 55, 5))
                             .round().astype(int))
    
    df["forest_cover_pct"] = (df["district"]
                              .apply(lambda d: rng.normal(35 if d in HIGHLAND_DISTRICTS else 20, 10))
                              .clip(5, 65).round().astype(int))
    
    # Geographic coordinates
    df[["district_centroid_lat","district_centroid_lon"]] = (
        df["district"]
          .apply(lambda d: CENTROIDS.get(d, (rng.uniform(5.8,9.8), rng.uniform(79.5,81.9))))
          .apply(pd.Series)
          .round(4)
    )
    
    # Infrastructure metrics
    def fibre_dist(level):
        return max(0.5, rng.normal({"urban": 2, "semi": 10, "rural": 25}[level], 3))
    
    df["distance_to_nearest_fibre_pop_km"] = level_series.apply(fibre_dist).round(1)
    
    df["terrain_elevation_mean_m"] = (df["district"]
                                      .apply(lambda d: rng.normal(1200, 200) if d in HIGHLAND_DISTRICTS
                                             else rng.normal(150, 80))
                                      .clip(0)
                                      .round().astype(int))
    
    def blackout(level):
        return max(1, rng.normal({"urban": 6, "semi": 10, "rural": 15}[level], 3))
    
    df["avg_blackout_hours_month"] = level_series.apply(blackout).round(1)
    
    # Categorical sentiment representation
    NUMERIC_SENTIMENTS = {
        "dialog_sentiment": "dialog_sent_cat",
        "slt_mobitel_sentiment": "slt_mobitel_sent_cat",
        "hutch_sentiment": "hutch_sent_cat",
        "starlink_sentiment_score": "starlink_sent_cat"
    }
    
    def to_cat(score):
        if score < -0.1: return "neg"
        if score > 0.1: return "pos"
        return "neu"
    
    for num_col, cat_col in NUMERIC_SENTIMENTS.items():
        df[cat_col] = df[num_col].apply(to_cat).astype("category")
    
    return df

# Load data
district_df = create_district_data()

if page == "Home":
    st.title("Sri Lanka Telecom & Infrastructure Dashboard")
    st.write("Comprehensive analysis of telecommunications and infrastructure across Sri Lankan districts.")
    
    # Key Metrics
    st.subheader("Key National Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Download Speed", f"{district_df['avg_dl_speed_mbps'].mean():.1f} Mbps")
    with col2:
        st.metric("Fixed Broadband Penetration", f"{district_df['fixed_bb_penetration_pct'].mean():.1f}%")
    with col3:
        st.metric("Fiber Coverage", f"{district_df['fiber_coverage_pct'].mean():.1f}%")
    with col4:
        st.metric("Avg Monthly BB Price", f"₨{district_df['avg_monthly_bb_price_lkr'].mean():,.0f}")
    
    # Top Districts
    st.subheader("Top Performing Districts")
    tab1, tab2, tab3 = st.tabs(["By Speed", "By Coverage", "By Satisfaction"])
    with tab1:
        st.dataframe(district_df.sort_values("avg_dl_speed_mbps", ascending=False).head(5))
    with tab2:
        st.dataframe(district_df.sort_values("fiber_coverage_pct", ascending=False).head(5))
    with tab3:
        st.dataframe(district_df.sort_values("overall_bb_satisfaction_score", ascending=False).head(5))
    
    # Map Visualization
    st.subheader("Broadband Infrastructure Overview")
    st.write("District centroids with broadband penetration:")
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='district_centroid_lon', 
        y='district_centroid_lat', 
        size='fixed_bb_penetration_pct', 
        hue='urbanization_level', 
        data=district_df, 
        ax=ax, 
        sizes=(50, 300),
        palette="viridis"
    )
    ax.set_title("District Broadband Penetration by Urbanization Level")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    analysis_type = st.selectbox("Select Analysis Focus:", 
                               ["Basic Demographics", "Economic Indicators", "Digital Infrastructure"])
    
    if analysis_type == "Basic Demographics":
        st.subheader("Population and Urbanization")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='urbanization_level', data=district_df, ax=ax)
            ax.set_title("District Count by Urbanization Level")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='population', y='district', 
                       data=district_df.sort_values('population', ascending=False).head(10), 
                       ax=ax)
            ax.set_title("Top 10 Districts by Population")
            st.pyplot(fig)
        
        st.subheader("Household Characteristics")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='households_est', y='avg_household_size', 
                       hue='province', data=district_df, ax=ax, s=100)
        ax.set_title("Household Size vs Number of Households")
        st.pyplot(fig)
    
    elif analysis_type == "Economic Indicators":
        st.subheader("Income Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='province', y='avg_income_lkr_2025', data=district_df, ax=ax)
            ax.set_title("Income Distribution by Province")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='avg_income_lkr_2025', y='poverty_rate_pct', 
                           hue='urbanization_level', data=district_df, ax=ax, s=100)
            ax.set_title("Income vs Poverty Rate")
            st.pyplot(fig)
    
    elif analysis_type == "Digital Infrastructure":
        st.subheader("Connectivity Metrics")
        tab1, tab2 = st.tabs(["Speed Analysis", "Coverage Analysis"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='urbanization_level', y='avg_dl_speed_mbps', data=district_df, ax=ax)
            ax.set_title("Download Speed by Urbanization Level")
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='avg_dl_speed_mbps', y='avg_monthly_bb_price_lkr', 
                           hue='province', data=district_df, ax=ax, s=100)
            ax.set_title("Speed vs Price by Province")
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='province', y='fiber_coverage_pct', data=district_df, ax=ax)
            ax.set_title("Fiber Coverage by Province")
            plt.xticks(rotation=45)
            st.pyplot(fig)

elif page == "Telecom Analysis":
    st.title("Telecommunications Deep Dive")
    
    analysis_type = st.radio("Select Analysis Type:", 
                            ["ISP Market Share", "Service Quality", "Customer Sentiment"])
    
    if analysis_type == "ISP Market Share":
        st.subheader("Internet Service Provider Market Share")
        
        # ISP subscriber estimates
        isp_cols = [col for col in district_df.columns if '_subs_est' in col]
        isp_data = district_df[['district', 'province', 'urbanization_level'] + isp_cols]
        
        selected_district = st.selectbox("Select District:", district_df["district"])
        district_data = isp_data[isp_data["district"] == selected_district]
        
        # Melt for visualization
        melted = district_data.melt(id_vars=['district', 'province', 'urbanization_level'], 
                                   var_name='ISP', value_name='Subscribers')
        melted['ISP'] = melted['ISP'].str.replace('_subs_est', '').str.title()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### {selected_district} District")
            st.dataframe(district_data.drop(columns=['district', 'province', 'urbanization_level']))
        
        with col2:
            fig, ax = plt.subplots()
            melted.plot.pie(y='Subscribers', labels=melted['ISP'], autopct='%1.1f%%', ax=ax)
            ax.set_title(f"ISP Market Share in {selected_district}")
            st.pyplot(fig)
        
        # Province-level comparison
        st.subheader("Province-Level ISP Distribution")
        province_isp = isp_data.groupby('province')[isp_cols].sum().reset_index()
        melted_province = province_isp.melt(id_vars=['province'], var_name='ISP', value_name='Subscribers')
        melted_province['ISP'] = melted_province['ISP'].str.replace('_subs_est', '').str.title()
        
        selected_province = st.selectbox("Select Province:", province_isp["province"])
        province_data = melted_province[melted_province["province"] == selected_province]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ISP', y='Subscribers', data=province_data, ax=ax)
        ax.set_title(f"ISP Subscribers in {selected_province} Province")
        st.pyplot(fig)
    
    elif analysis_type == "Service Quality":
        st.subheader("Telecom Service Quality Metrics")
        
        metrics = st.multiselect("Select metrics to compare:", 
                               ['avg_dl_speed_mbps', 'avg_ul_speed_mbps', 
                                'fixed_bb_penetration_pct', 'fiber_coverage_pct',
                                '4g_coverage_pct', 'avg_blackout_hours_month'],
                               default=['avg_dl_speed_mbps', 'fixed_bb_penetration_pct'])
        
        if metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            melted = district_df.melt(id_vars=['district', 'urbanization_level'], 
                                    value_vars=metrics, 
                                    var_name='Metric', 
                                    value_name='Value')
            sns.boxplot(x='Metric', y='Value', hue='urbanization_level', data=melted, ax=ax)
            ax.set_title("Service Quality Metrics by Urbanization Level")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Speed vs Coverage
        st.subheader("Speed vs Infrastructure Coverage")
        x_axis = st.selectbox("X-axis:", ['fiber_coverage_pct', '4g_coverage_pct', 'distance_to_nearest_fibre_pop_km'])
        y_axis = st.selectbox("Y-axis:", ['avg_dl_speed_mbps', 'avg_ul_speed_mbps'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_axis, y=y_axis, hue='urbanization_level', 
                       size='population', data=district_df, ax=ax, sizes=(50, 300))
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)
    
    elif analysis_type == "Customer Sentiment":
        st.subheader("ISP Customer Sentiment Analysis")
        
        # Sentiment by ISP
        sentiment_cols = ['dialog_sent_cat', 'slt_mobitel_sent_cat', 'hutch_sent_cat', 'starlink_sent_cat']
        sentiment_data = district_df[['district', 'urbanization_level'] + sentiment_cols]
        
        # Count sentiment categories
        sentiment_counts = []
        for col in sentiment_cols:
            counts = sentiment_data.groupby(['urbanization_level', col]).size().reset_index(name='count')
            counts['ISP'] = col.replace('_sent_cat', '').replace('_', ' ').title()
            sentiment_counts.append(counts)
        sentiment_counts = pd.concat(sentiment_counts)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='ISP', y='count', hue='dialog_sent_cat', 
                   data=sentiment_counts, ax=ax)
        ax.set_title("Customer Sentiment by ISP and Urbanization Level")
        st.pyplot(fig)
        
        # Overall satisfaction
        st.subheader("Overall Broadband Satisfaction")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.boxplot(x='urbanization_level', y='overall_bb_satisfaction_score', 
                       data=district_df, ax=ax)
            ax.set_title("Satisfaction by Urbanization Level")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x='avg_monthly_bb_price_lkr', y='overall_bb_satisfaction_score', 
                           hue='avg_dl_speed_mbps', data=district_df, ax=ax)
            ax.set_title("Satisfaction vs Price (colored by speed)")
            st.pyplot(fig)

elif page == "Advanced Metrics":
    st.title("Advanced Infrastructure Metrics")
    
    tab1, tab2, tab3 = st.tabs(["Environmental Factors", "Geospatial Analysis", "Composite Indicators"])
    
    with tab1:
        st.subheader("Environmental Impact on Infrastructure")
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='avg_annual_rainfall_mm', y='avg_blackout_hours_month', 
                           hue='urbanization_level', data=district_df, ax=ax)
            ax.set_title("Rainfall vs Power Outages")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='thunderstorm_days', y='distance_to_nearest_fibre_pop_km', 
                           hue='province', data=district_df, ax=ax)
            ax.set_title("Storm Frequency vs Fiber Access")
            st.pyplot(fig)
        
        st.subheader("Terrain Impact")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='terrain_elevation_mean_m', y='fiber_coverage_pct', 
                       hue='forest_cover_pct', data=district_df, ax=ax)
        ax.set_title("Elevation vs Fiber Coverage (colored by forest cover)")
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Geospatial Patterns")
        
        map_metric = st.selectbox("Select metric to visualize:", 
                                ['fixed_bb_penetration_pct', 'fiber_coverage_pct', 
                                 'avg_dl_speed_mbps', 'avg_monthly_bb_price_lkr'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = sns.scatterplot(
            x='district_centroid_lon', 
            y='district_centroid_lat', 
            size=map_metric, 
            hue='province', 
            data=district_df, 
            ax=ax, 
            sizes=(50, 300),
            palette="viridis"
        )
        ax.set_title(f"District {map_metric.replace('_', ' ').title()} by Province")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Composite Digital Readiness Index")
        
        # Calculate a simple composite index (for demonstration)
        district_df['digital_readiness_index'] = (
            0.3 * district_df['fixed_bb_penetration_pct'] +
            0.2 * district_df['avg_dl_speed_mbps'] +
            0.2 * district_df['computer_ownership_pct_2024_weighted'] +
            0.15 * district_df['smartphone_penetration_pct'] +
            0.15 * district_df['digital_literacy_index']
        )
        
        st.write("Top Districts by Digital Readiness:")
        st.dataframe(
            district_df.sort_values('digital_readiness_index', ascending=False)
            [['district', 'province', 'digital_readiness_index', 
              'fixed_bb_penetration_pct', 'avg_dl_speed_mbps']]
            .head(10)
            .style.format({
                'digital_readiness_index': '{:.1f}',
                'fixed_bb_penetration_pct': '{:.1f}%',
                'avg_dl_speed_mbps': '{:.1f}'
            })
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='digital_readiness_index', y='district', 
                   data=district_df.sort_values('digital_readiness_index', ascending=False).head(10), 
                   ax=ax)
        ax.set_title("Top 10 Districts by Digital Readiness")
        st.pyplot(fig)
