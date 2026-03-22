import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats

# ==========================================
# 1. PAGE CONFIGURATION & ULTRA-MODERN CSS
# ==========================================
st.set_page_config(
    page_title="InsightMart | Black Friday AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #090a0f, #1a1c29, #0f172a, #090a0f);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 229, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
    }
    
    /* Glowing Metric Typography */
    [data-testid="stMetricValue"] { 
        color: #00e5ff !important; 
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 36px !important; 
        font-weight: 800 !important; 
        text-shadow: 0 0 10px rgba(0,229,255,0.3); 
    }
    [data-testid="stMetricLabel"] { 
        color: #8b949e !important; 
        font-size: 14px !important; 
        font-weight: 600 !important; 
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
    }

    /* Gradient Headers */
    .gradient-header {
        background: -webkit-linear-gradient(45deg, #00e5ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3em;
        padding-bottom: 10px;
        line-height: 1.2;
    }
    .sub-header { color: #a1a1aa; font-weight: 300; font-size: 1.1em; margin-bottom: 20px; }

    /* Sidebar Glass UI */
    section[data-testid="stSidebar"] {
        background: rgba(9, 10, 15, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Dataframes styling */
    .stDataFrame { border-radius: 15px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS & DATA LOADING
# ==========================================
def apply_transparent_theme(fig):
    """Strips white backgrounds from Plotly charts to make them float seamlessly"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.02)',
        font=dict(color='#a1a1aa', family="Inter"),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False, zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False, zeroline=False)
    )
    return fig

@st.cache_data
def load_data():
    df = pd.read_csv('BlackFriday.csv')
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    df = df.drop_duplicates()
    
    df_ml = df.copy()
    df_ml['Gender_Bin'] = df_ml['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17':1, '18-25':2, '26-35':3, '36-45':4, '46-50':5, '51-55':6, '55+':7}
    df_ml['Age_Level'] = df_ml['Age'].map(age_map)
    city_map = {'A': 1, 'B': 2, 'C': 3}
    df_ml['City_Code'] = df_ml['City_Category'].map(city_map)
    
    scaler = MinMaxScaler()
    df_ml['Purchase_Scaled'] = scaler.fit_transform(df_ml[['Purchase']])
    return df, df_ml

raw_data, ml_data = load_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00e5ff; font-weight: 800; text-shadow: 0 0 10px rgba(0,229,255,0.5);'>InsightMart AI</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    menu = st.radio("System Modules", [
        "🌐 Executive Dashboard", 
        "🧠 Customer Segmentation", 
        "🔗 Market Basket Analysis", 
        "🚨 Anomaly Intelligence",
        "📂 Project Documentation"
    ])
    st.markdown("---")
    st.info("Current Dataset: Black Friday Sales")

# ==========================================
# 4. MAIN APPLICATION LOGIC
# ==========================================
if menu == "🌐 Executive Dashboard":
    st.markdown("<div class='gradient-header'>Sales Intelligence Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time macro analysis of the Black Friday mega sale</div>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${raw_data['Purchase'].sum():,.0f}")
    m2.metric("Avg Transaction", f"${raw_data['Purchase'].mean():.2f}")
    m3.metric("Unique Customers", f"{raw_data['User_ID'].nunique():,}")
    m4.metric("Product Count", f"{raw_data['Product_ID'].nunique():,}")
    
    st.markdown("<br>### 📊 Demographic Spending Patterns", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        fig_age = px.histogram(raw_data, x="Age", y="Purchase", color="Gender", 
                               barmode="group", title="Revenue by Age & Gender",
                               color_discrete_map={'M': '#00e5ff', 'F': '#ff007f'})
        st.plotly_chart(apply_transparent_theme(fig_age), use_container_width=True)
        
    with c2:
        top_cats = raw_data.groupby('Product_Category_1')['Purchase'].sum().reset_index()
        fig_cat = px.pie(top_cats, values='Purchase', names='Product_Category_1', 
                         title="Revenue Distribution by Category", hole=0.5,
                         color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(apply_transparent_theme(fig_cat), use_container_width=True)

    st.markdown("### 🔬 Statistical Correlations")
    corr = ml_data[['Age_Level', 'Occupation', 'Marital_Status', 'Purchase_Scaled']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale='magma')
    st.plotly_chart(apply_transparent_theme(fig_corr), use_container_width=True)

elif menu == "🧠 Customer Segmentation":
    st.markdown("<div class='gradient-header'>Advanced Clustering (Stage 4)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Grouping customers based on behavioral and demographic features via K-Means.</div>", unsafe_allow_html=True)
    
    subset = ml_data.sample(min(10000, len(ml_data)))
    cluster_features = subset[['Age_Level', 'Occupation', 'Purchase_Scaled']]
    
    st.markdown("### 1. Optimal Cluster Determination (Elbow Method)")
    distortions = []
    for k in range(1, 11):
        tmp_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cluster_features)
        distortions.append(tmp_km.inertia_)
    
    fig_elb = go.Figure()
    fig_elb.add_trace(go.Scatter(x=list(range(1, 11)), y=distortions, mode='lines+markers',
                                 line=dict(color='#00e5ff', width=3), marker=dict(size=8, color='#ff007f')))
    fig_elb.update_layout(title="Elbow Method Analysis", xaxis_title="K", yaxis_title="Inertia")
    st.plotly_chart(apply_transparent_theme(fig_elb), use_container_width=True)
    
    st.divider()
    
    st.markdown("### 2. Cluster Deployment")
    k_input = st.select_slider("Select Target Segments (K)", options=range(2, 7), value=4)
    
    km_final = KMeans(n_clusters=k_input, random_state=42, n_init=10)
    subset['Cluster'] = km_final.fit_predict(cluster_features).astype(str)
    
    fig_3d = px.scatter_3d(subset, x='Age_Level', y='Occupation', z='Purchase_Scaled',
                           color='Cluster', opacity=0.8, title=f"3D Customer Segments (K={k_input})",
                           color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_3d.update_traces(marker=dict(size=4, line=dict(width=0)))
    st.plotly_chart(apply_transparent_theme(fig_3d), use_container_width=True)
    
    st.markdown("### Segment Characteristics")
    stats_df = subset.groupby('Cluster')[['Age_Level', 'Purchase']].mean()
    st.table(stats_df)

elif menu == "🔗 Market Basket Analysis":
    st.markdown("<div class='gradient-header'>Association Rule Mining (Stage 5)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Discovering cross-selling opportunities via Apriori Algorithm.</div>", unsafe_allow_html=True)
    
    min_sup = st.slider("Minimum Support", 0.01, 0.20, 0.05)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3)
    
    with st.spinner("Mining frequent itemsets..."):
        basket = raw_data.sample(5000).groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules[rules['confidence'] >= min_conf]
            
            st.markdown("### 🔥 Generated Strategic Rules")
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False), use_container_width=True)
            
            fig_rules = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                                   color_continuous_scale='Agsunset', hover_data=['antecedents', 'consequents'], title="Rules Strength Map")
            st.plotly_chart(apply_transparent_theme(fig_rules), use_container_width=True)
        else:
            st.warning("No patterns found. Try lowering the Support threshold.")

elif menu == "🚨 Anomaly Intelligence":
    st.markdown("<div class='gradient-header' style='background: -webkit-linear-gradient(45deg, #ff007f, #ff7e5f); -webkit-background-clip: text;'>Anomaly Detection (Stage 6)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Identifying extreme spending behaviors that deviate from the norm.</div>", unsafe_allow_html=True)
    
    method = st.selectbox("Detection Method", ["Interquartile Range (IQR)", "Z-Score Analysis"])
    
    if method == "Interquartile Range (IQR)":
        Q1 = raw_data['Purchase'].quantile(0.25)
        Q3 = raw_data['Purchase'].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + (1.5 * IQR)
        anomalies = raw_data[raw_data['Purchase'] > upper_limit]
    else:
        z_scores = np.abs(stats.zscore(raw_data['Purchase']))
        anomalies = raw_data[z_scores > 3]
        upper_limit = raw_data['Purchase'].mean() + (3 * raw_data['Purchase'].std())

    st.metric("Detected Anomalous Transactions", len(anomalies))
    
    fig_anom = px.box(raw_data, y="Purchase", title="Purchase Outlier Visualization", points="outliers", color_discrete_sequence=['#ff007f'])
    fig_anom.add_hline(y=upper_limit, line_dash="dash", line_color="#00e5ff", annotation_text="Anomaly Threshold", annotation_font_color="white")
    st.plotly_chart(apply_transparent_theme(fig_anom), use_container_width=True)
    
    st.markdown("### Anomalous Shopper Profiles")
    st.dataframe(anomalies.head(100), use_container_width=True)

elif menu == "📂 Project Documentation":
    st.markdown("<div class='gradient-header'>Project Scope & Insights (Stage 1 & 7)</div>", unsafe_allow_html=True)
    
    with st.expander("1. Project Definition", expanded=True):
        st.write("""
        - **Goal**: Analyze Black Friday data to optimize retail strategies.
        - **Scope**: Includes EDA, Clustering, Association Mining, and Anomaly detection.
        """)
        
    with st.expander("2. Preprocessing Steps"):
        st.write("""
        - Imputed missing values in Product Category 2 and 3 with '0'.
        - Encoded Gender (M/F -> 0/1) and Age groups into ordinal ranks.
        - Scaled 'Purchase' using MinMaxScaler for clustering accuracy.
        """)
        
    with st.expander("3. Final Business Insights", expanded=True):
        st.success("**Key Finding 1**: Male customers in the 26-35 age bracket are the primary revenue drivers.")
        st.success("**Key Finding 2**: Strong associations exist between Product Category 1 and 5, suggesting bundle offers.")
        st.success("**Key Finding 3**: High-spending anomalies are predominantly from City Category 'C'.")

st.divider()
st.caption("Developed for Data Mining Summative Assessment | Artificial Intelligence Course")
