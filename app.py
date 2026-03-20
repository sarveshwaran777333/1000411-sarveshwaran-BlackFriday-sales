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

st.set_page_config(
    page_title="InsightMart | Black Friday AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e1e4e8; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    </style>
    """, unsafe_allow_html=True)

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

with st.sidebar:
    st.title("InsightMart AI")
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    menu = st.radio("System Modules", [
        "Executive Dashboard", 
        "Customer Segmentation", 
        "Market Basket Analysis", 
        "Anomaly Intelligence",
        "Project Documentation"
    ])
    st.divider()
    st.info("Current Dataset: Black Friday Sales")

if menu == "Executive Dashboard":
    st.header("📊 Sales Intelligence Overview")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${raw_data['Purchase'].sum():,.0f}")
    m2.metric("Avg Transaction", f"${raw_data['Purchase'].mean():.2f}")
    m3.metric("Unique Customers", f"{raw_data['User_ID'].nunique():,}")
    m4.metric("Product Count", f"{raw_data['Product_ID'].nunique():,}")
    
    st.subheader("Demographic Spending Patterns")
    c1, c2 = st.columns(2)
    
    with c1:
        fig_age = px.histogram(raw_data, x="Age", y="Purchase", color="Gender", 
                               barmode="group", title="Revenue by Age & Gender",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_age, use_container_width=True)
        
    with c2:
        top_cats = raw_data.groupby('Product_Category_1')['Purchase'].sum().reset_index()
        fig_cat = px.pie(top_cats, values='Purchase', names='Product_Category_1', 
                         title="Revenue Distribution by Category", hole=0.4)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Statistical Correlations")
    corr = ml_data[['Age_Level', 'Occupation', 'Marital_Status', 'Purchase_Scaled']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

elif menu == "Customer Segmentation":
    st.header("👥 Advanced Clustering (Stage 4)")
    st.write("Grouping customers based on behavioral and demographic features.")
    
    subset = ml_data.sample(min(10000, len(ml_data)))
    cluster_features = subset[['Age_Level', 'Occupation', 'Purchase_Scaled']]
    
    st.subheader("1. Optimal Cluster Determination")
    distortions = []
    for k in range(1, 11):
        tmp_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cluster_features)
        distortions.append(tmp_km.inertia_)
    
    fig_elb = go.Figure()
    fig_elb.add_trace(go.Scatter(x=list(range(1, 11)), y=distortions, mode='lines+markers'))
    fig_elb.update_layout(title="Elbow Method Analysis", xaxis_title="K", yaxis_title="Inertia")
    st.plotly_chart(fig_elb, use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Cluster Deployment")
    k_input = st.select_slider("Select Target Segments (K)", options=range(2, 7), value=4)
    
    km_final = KMeans(n_clusters=k_input, random_state=42, n_init=10)
    subset['Cluster'] = km_final.fit_predict(cluster_features)
    
    fig_3d = px.scatter_3d(subset, x='Age_Level', y='Occupation', z='Purchase_Scaled',
                           color='Cluster', opacity=0.7, title=f"3D Customer Segments (K={k_input})")
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.subheader("Segment Characteristics")
    stats_df = subset.groupby('Cluster')[['Age_Level', 'Purchase']].mean()
    st.table(stats_df)

elif menu == "Market Basket Analysis":
    st.header("🔗 Association Rule Mining (Stage 5)")
    st.write("Discovering cross-selling opportunities via Apriori Algorithm.")
    
    min_sup = st.slider("Minimum Support", 0.01, 0.20, 0.05)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3)
    
    basket = raw_data.sample(5000).groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        rules = rules[rules['confidence'] >= min_conf]
        
        st.subheader("Generated Strategic Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
        
        fig_rules = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                               hover_data=['antecedents', 'consequents'], title="Rules Strength Map")
        st.plotly_chart(fig_rules, use_container_width=True)
    else:
        st.warning("No patterns found. Try lowering the Support threshold.")

elif menu == "Anomaly Intelligence":
    st.header("🚨 Anomaly Detection (Stage 6)")
    st.write("Identifying extreme spending behaviors that deviate from the norm.")
    
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
    
    fig_anom = px.box(raw_data, y="Purchase", title="Purchase Outlier Visualization")
    fig_anom.add_hline(y=upper_limit, line_dash="dot", line_color="red", annotation_text="Anomaly Threshold")
    st.plotly_chart(fig_anom, use_container_width=True)
    
    st.subheader("Anomalous Shopper Profiles")
    st.dataframe(anomalies.head(100))

elif menu == "Project Documentation":
    st.header("📝 Project Scope & Insights (Stage 1 & 7)")
    
    with st.expander("1. Project Definition"):
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
        
    with st.expander("3. Final Business Insights"):
        st.success("**Key Finding 1**: Male customers in the 26-35 age bracket are the primary revenue drivers.")
        st.success("**Key Finding 2**: Strong associations exist between Product Category 1 and 5, suggesting bundle offers.")
        st.success("**Key Finding 3**: High-spending anomalies are predominantly from City Category 'C'.")

st.divider()
st.caption("Developed for Data Mining Summative Assessment | Artificial Intelligence Course ")
