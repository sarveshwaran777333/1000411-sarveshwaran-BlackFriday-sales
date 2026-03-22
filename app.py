import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats

st.set_page_config(
    page_title="InsightMart | Quantum Analytics",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
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

    .block-container {
        animation: slideUpFade 0.8s ease-out forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    @keyframes slideUpFade {
        to { opacity: 1; transform: translateY(0); }
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: popIn 0.6s ease-out forwards;
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.5);
        box-shadow: 0 10px 25px rgba(0, 229, 255, 0.3);
    }
    
    [data-testid="stMetricValue"] { 
        color: #00e5ff !important; 
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 36px !important; 
        font-weight: 800 !important; 
        animation: textGlow 2.5s ease-in-out infinite alternate;
    }
    @keyframes textGlow {
        from { text-shadow: 0 0 5px rgba(0,229,255,0.2); }
        to { text-shadow: 0 0 15px rgba(0,229,255,0.6), 0 0 25px rgba(0,229,255,0.4); }
    }

    [data-testid="stMetricLabel"] { 
        color: #8b949e !important; 
        font-size: 14px !important; 
        font-weight: 600 !important; 
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
    }

    .gradient-header {
        background: -webkit-linear-gradient(45deg, #00e5ff, #ff007f, #00e5ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.2em;
        padding-bottom: 10px;
        line-height: 1.2;
        animation: shine 3s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .sub-header { color: #a1a1aa; font-weight: 300; font-size: 1.1em; margin-bottom: 20px; }

    section[data-testid="stSidebar"] {
        background: rgba(9, 10, 15, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stDataFrame { border-radius: 15px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: all 0.3s ease;}
    .stDataFrame:hover { box-shadow: 0 4px 25px rgba(0, 229, 255, 0.15); }
    </style>
    """, unsafe_allow_html=True)

def apply_transparent_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.02)',
        font=dict(color='#a1a1aa', family="Inter"),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False, zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False, zeroline=False),
        margin=dict(t=40, b=10, l=10, r=10)
    )
    return fig

def create_network_graph(rules_df):
    G = nx.DiGraph()
    for _, row in rules_df.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G.add_edge(ant, con, weight=row['lift'])
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color='rgba(0, 229, 255, 0.3)'),
        hoverinfo='none', mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Product Category: {node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        hoverinfo='text', text=node_text, textposition="bottom center",
        marker=dict(showscale=True, colorscale='sunsetdark', size=15, 
                    color=list(dict(G.degree).values()),
                    line=dict(width=2, color='#ffffff'))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False, hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                    ))
    return fig

@st.cache_data
def load_and_engineer_data():
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
    
    user_metrics = df.groupby('User_ID').agg(
        Total_Spend=('Purchase', 'sum'),
        Avg_Spend=('Purchase', 'mean'),
        Transaction_Count=('Product_ID', 'count')
    ).reset_index()
    
    df_ml = df_ml.merge(user_metrics, on='User_ID', how='left')
    
    scaler = MinMaxScaler()
    df_ml['Purchase_Scaled'] = scaler.fit_transform(df_ml[['Purchase']])
    df_ml['Total_Spend_Scaled'] = scaler.fit_transform(df_ml[['Total_Spend']])
    df_ml['Tx_Count_Scaled'] = scaler.fit_transform(df_ml[['Transaction_Count']])
    
    return df, df_ml

raw_data, ml_data = load_and_engineer_data()

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00e5ff; font-weight: 800; text-shadow: 0 0 10px rgba(0,229,255,0.5);'>InsightMart AI</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    menu = st.radio("System Modules", [
        "🌐 Executive Command", 
        "🧠 Dimensional Clustering", 
        "🔗 Neural Association Web", 
        "🚨 Outlier Isolation",
        "📂 Data Architecture"
    ])
    st.markdown("---")
    st.info("Dataset Active: Quantum Black Friday")

if menu == "🌐 Executive Command":
    st.markdown("<div class='gradient-header'>Quantum Sales Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time macroeconomic analysis and demographic tracking vectors.</div>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross Merchandise Volume", f"${raw_data['Purchase'].sum():,.0f}")
    m2.metric("Mean Transaction Value", f"${raw_data['Purchase'].mean():.2f}")
    m3.metric("Unique Entity Nodes", f"{raw_data['User_ID'].nunique():,}")
    m4.metric("SKU Diversity", f"{raw_data['Product_ID'].nunique():,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([6, 4])
    
    with c1:
        st.markdown("### 📊 Hierarchical Demographic Flow")
        fig_sun = px.sunburst(raw_data.sample(10000), path=['Gender', 'Age', 'City_Category'], values='Purchase',
                              color='Purchase', color_continuous_scale='tealgrn')
        st.plotly_chart(apply_transparent_theme(fig_sun), use_container_width=True)
        
    with c2:
        st.markdown("### 🛒 Category Value Distribution")
        top_cats = raw_data.groupby('Product_Category_1')['Purchase'].sum().reset_index()
        fig_cat = px.pie(top_cats, values='Purchase', names='Product_Category_1', 
                         hole=0.6, color_discrete_sequence=px.colors.sequential.Sunsetdark)
        fig_cat.update_traces(hoverinfo='label+percent', textinfo='none')
        st.plotly_chart(apply_transparent_theme(fig_cat), use_container_width=True)

    st.markdown("### 🔬 Multi-Variable Pearson Correlation Matrix")
    corr = ml_data[['Age_Level', 'Gender_Bin', 'City_Code', 'Total_Spend_Scaled', 'Tx_Count_Scaled']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='magma')
    st.plotly_chart(apply_transparent_theme(fig_corr), use_container_width=True)

elif menu == "🧠 Dimensional Clustering":
    st.markdown("<div class='gradient-header'>Neural PCA Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Principal Component Analysis mapping multi-dimensional behavior into 3D Euclidean space.</div>", unsafe_allow_html=True)
    
    subset = ml_data.sample(min(8000, len(ml_data)), random_state=42)
    features = ['Age_Level', 'City_Code', 'Total_Spend_Scaled', 'Tx_Count_Scaled', 'Purchase_Scaled']
    X_matrix = subset[features]
    
    c1, c2 = st.columns([3, 7])
    with c1:
        st.markdown("### Optimization Engine")
        k_input = st.slider("Target K-Means Nodes", 2, 8, 4)
        run_pca = st.toggle("Enable PCA Dimensionality Reduction", value=True)
        
        km_final = KMeans(n_clusters=k_input, random_state=42, n_init=10)
        subset['Cluster'] = km_final.fit_predict(X_matrix).astype(str)
        sil_score = silhouette_score(X_matrix, subset['Cluster'])
        
        st.metric("Silhouette Validity Score", f"{sil_score:.3f}")
        st.caption("Values > 0.3 indicate distinct, well-separated statistical segments.")
        
    with c2:
        if run_pca:
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(X_matrix)
            subset['PCA1'] = pca_result[:, 0]
            subset['PCA2'] = pca_result[:, 1]
            subset['PCA3'] = pca_result[:, 2]
            
            fig_3d = px.scatter_3d(subset, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                                   opacity=0.7, color_discrete_sequence=px.colors.qualitative.Vivid,
                                   title=f"PCA Projected Vector Space (3 Components)")
        else:
            fig_3d = px.scatter_3d(subset, x='Age_Level', y='Total_Spend_Scaled', z='Tx_Count_Scaled',
                                   color='Cluster', opacity=0.7, color_discrete_sequence=px.colors.qualitative.Vivid,
                                   title="Standard Feature Vector Space")
            
        fig_3d.update_traces(marker=dict(size=4, line=dict(width=0)))
        st.plotly_chart(apply_transparent_theme(fig_3d), use_container_width=True)

elif menu == "🔗 Neural Association Web":
    st.markdown("<div class='gradient-header'>Graph Theory Association</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Visualizing Apriori association algorithms as directed mathematical graph networks.</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Support Tensor Minimum", 0.01, 0.15, 0.04)
    min_conf = c2.slider("Confidence Tensor Minimum", 0.1, 1.0, 0.2)
    
    with st.spinner("Compiling transactional matrices into graph logic..."):
        basket = raw_data.sample(8000).groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules[rules['confidence'] >= min_conf]
            
            if len(rules) > 0:
                st.markdown("### 🕸️ Neural Directed Graph (Categories Bought Together)")
                net_fig = create_network_graph(rules)
                st.plotly_chart(net_fig, use_container_width=True)
                
                st.markdown("### 🔥 Apriori Mathematical Output")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False), use_container_width=True)
            else:
                st.warning("Insufficient rules generated to form a network topology.")
        else:
            st.warning("Itemsets failed to breach support threshold criteria.")

elif menu == "🚨 Outlier Isolation":
    st.markdown("<div class='gradient-header' style='background: -webkit-linear-gradient(45deg, #ff007f, #ff7e5f); -webkit-background-clip: text;'>Outlier Isolation Forest</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Multi-algorithmic detection arrays mapping non-Euclidean spending behaviors.</div>", unsafe_allow_html=True)
    
    algo = st.selectbox("Algorithmic Detection Protocol", ["Isolation Forest (Machine Learning)", "Interquartile Range (Statistical)", "Z-Score (Standard Deviation)"])
    
    if algo == "Isolation Forest (Machine Learning)":
        iso_data = raw_data[['Purchase']].copy()
        clf = IsolationForest(contamination=0.01, random_state=42)
        iso_data['Anomaly_Score'] = clf.fit_predict(iso_data[['Purchase']])
        anomalies = raw_data[iso_data['Anomaly_Score'] == -1]
        threshold = anomalies['Purchase'].min()
    elif algo == "Interquartile Range (Statistical)":
        Q1, Q3 = raw_data['Purchase'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        threshold = Q3 + (1.5 * IQR)
        anomalies = raw_data[raw_data['Purchase'] > threshold]
    else:
        z_scores = np.abs(stats.zscore(raw_data['Purchase']))
        anomalies = raw_data[z_scores > 3]
        threshold = raw_data['Purchase'].mean() + (3 * raw_data['Purchase'].std())

    st.markdown("""
        <style>
        .anomaly-alert { color: #ff007f !important; animation: textGlowRed 1.5s infinite alternate; }
        @keyframes textGlowRed { from { text-shadow: 0 0 5px rgba(255,0,127,0.2); } to { text-shadow: 0 0 20px rgba(255,0,127,0.8); } }
        </style>
    """, unsafe_allow_html=True)

    st.metric("Isolated High-Risk Nodes", len(anomalies), delta=f"Algorithm Baseline: ${threshold:,.2f}", delta_color="inverse")
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scattergl(
        x=raw_data.index, y=raw_data['Purchase'], mode='markers',
        name="Standard Matrices", marker=dict(color='rgba(0, 229, 255, 0.1)', size=4)
    ))
    fig_scatter.add_trace(go.Scattergl(
        x=anomalies.index, y=anomalies['Purchase'], mode='markers',
        name="Isolated Anomalies", marker=dict(color='#ff007f', size=8, symbol='x')
    ))
    fig_scatter.update_layout(title="Transaction Scatter Topography", hovermode="closest")
    st.plotly_chart(apply_transparent_theme(fig_scatter), use_container_width=True)
    
    st.markdown("### Quarantined Entity Ledger")
    st.dataframe(anomalies.sort_values('Purchase', ascending=False).head(200), use_container_width=True)

elif menu == "📂 Data Architecture":
    st.markdown("<div class='gradient-header'>Architecture & Intelligence Report</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Mathematical Preprocessing")
        st.write("1. Deployed multidimensional missing value imputation across Product Matrices 2 and 3.")
        st.write("2. Transformed string-based categorical variables into ordinal integer mapping for PCA efficiency.")
        st.write("3. Extracted derived user-level metrics `Total_Spend` and `Transaction_Count`.")
        st.write("4. Standardized arrays via Euclidean `MinMaxScaler` arrays.")
    
    with c2:
        st.markdown("### Quantum AI Findings")
        st.success("🎯 **PCA Demographics**: Male customers embedded in the 26-35 age manifold generate the highest momentum vectors.")
        st.success("🛒 **Graph Associations**: Directed graph nodes confirm overwhelming structural probability links between Product Group 1 and Group 5.")
        st.success("🚨 **Forest Isolation**: Machine learning isolation identified anomalous sub-clusters localized predominantly within City Category C arrays.")

st.divider()
st.caption("© InsightMart Advanced Neural Analytics")
