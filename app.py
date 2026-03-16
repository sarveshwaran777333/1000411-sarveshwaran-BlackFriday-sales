import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Black Friday Insights", page_icon="🛒", layout="wide")

st.title("🛒 Mining the Future: Black Friday Sales Insights")
st.markdown("Welcome to the interactive dashboard for exploring customer behavior, segmentation, and sales anomalies during the Black Friday mega sale.")

# --- DATA LOADING & CACHING ---
# We cache the data so it doesn't reload every time you click a button!
@st.cache_data
def load_and_clean_data():
    # Load raw data
    df = pd.read_csv('BlackFriday.csv')
    
    # Clean data (Stage 2)
    df = df.drop_duplicates()
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    
    # Keep a raw copy for display, but encode a copy for ML
    df_ml = df.copy()
    df_ml['Gender'] = df_ml['Gender'].map({'M': 0, 'F': 1})
    age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df_ml['Age_Encoded'] = df_ml['Age'].map(age_mapping)
    
    scaler = MinMaxScaler()
    df_ml['Purchase_Normalized'] = scaler.fit_transform(df_ml[['Purchase']])
    
    return df, df_ml

# Load the data
df_raw, df_ml = load_and_clean_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Overview & EDA", "Customer Segmentation", "Anomaly Detection"])

# --- PAGE 1: OVERVIEW & EDA ---
if page == "Overview & EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("A quick look at the cleaned dataset.")
    st.dataframe(df_raw.head(10))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Purchase Amount by Gender")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Gender', y='Purchase', data=df_raw, palette='pastel', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Purchase Distribution by Age")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Age', y='Purchase', data=df_raw, palette='Set2', order=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], ax=ax)
        st.pyplot(fig)

# --- PAGE 2: CUSTOMER SEGMENTATION ---
elif page == "Customer Segmentation":
    st.header("Customer Clustering (K-Means)")
    st.write("Grouping customers based on their Age, Occupation, and Purchase behavior.")
    
    # We sample the data to make the web app run faster
    sample_df = df_ml.sample(10000, random_state=42)
    features = sample_df[['Age_Encoded', 'Occupation', 'Marital_Status', 'Purchase_Normalized']]
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    sample_df['Cluster'] = kmeans.fit_predict(features)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Age_Encoded', y='Purchase', hue='Cluster', data=sample_df, palette='viridis', alpha=0.6, ax=ax)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
    plt.title("Customer Segments")
    st.pyplot(fig)
    
    st.info("💡 **Insight:** Notice how different clusters separate distinct spending habits across age groups! You can label these groups as 'Budget Shoppers', 'Premium Buyers', etc.")

# --- PAGE 3: ANOMALY DETECTION ---
elif page == "Anomaly Detection":
    st.header("Anomaly Detection: High Spenders")
    st.write("Detecting unusually high purchase amounts using the IQR (Interquartile Range) method.")
    
    # Calculate IQR
    Q1 = df_raw['Purchase'].quantile(0.25)
    Q3 = df_raw['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = df_raw[df_raw['Purchase'] > upper_bound]
    
    st.warning(f"🚨 Detected {len(anomalies):,} anomalous transactions with unusually high purchase amounts (Threshold: ${upper_bound:,.2f})")
    
    st.write("Sample of High-Spending Anomalies:")
    st.dataframe(anomalies.head(10))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_raw['Purchase'], bins=50, color='blue', kde=False, ax=ax, label='Normal Purchases')
    sns.histplot(anomalies['Purchase'], bins=50, color='red', kde=False, ax=ax, label='Anomalies')
    plt.axvline(upper_bound, color='black', linestyle='dashed', linewidth=2, label='Anomaly Threshold')
    plt.legend()
    plt.title("Purchase Distribution Highlighting Anomalies")
    st.pyplot(fig)
