
# Customer Segmentation Web App
# Using K-Means and DBSCAN


# Fix KMeans warning on Windows
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 2: App Title 
st.title("🧠 Customer Segmentation using Unsupervised Machine Learning")
st.write("K-Means and DBSCAN Clustering with PCA Visualization")

# Step 3: Load Dataset 
st.header("📂 Load Dataset")

@st.cache_data
def load_data():
    data = pd.read_csv("Mall_Customers.csv")
    return data

dataset = load_data()

st.write("Dataset Preview:")
st.dataframe(dataset.head())

# Step 4: Data Preprocessing
st.header("🧹 Data Preprocessing")

# Drop CustomerID 
df = dataset.drop("CustomerID", axis=1)

# Encode 'Gender' column
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

st.write("Preprocessed Data:")
st.dataframe(df.head())

# Step 5: Feature Scaling 
st.header("⚖️ Feature Scaling")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

st.write("Scaled Feature Sample:")
st.write(X_scaled[:5])

# Step 6: PCA for Visualization 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# K-MEANS CLUSTERING

st.header("🔵 K-Means Clustering")

# Slider to select number of clusters
k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)

# Train K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Silhouette Score
kmeans_silhouette = silhouette_score(X_scaled, kmeans_clusters)
st.write("**K-Means Silhouette Score:**", round(kmeans_silhouette, 3))

# Plot K-Means clusters
fig1, ax1 = plt.subplots()
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=kmeans_clusters,
    palette="viridis",
    ax=ax1
)
ax1.set_title("K-Means Clustering (PCA View)")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
st.pyplot(fig1)


# DBSCAN CLUSTERING

st.header("🟢 DBSCAN Clustering")

# Sliders for DBSCAN parameters
eps = st.slider("Select eps value", 0.1, 1.5, 0.5, step=0.1)
min_samples = st.slider("Select min_samples", 3, 10, 5)

# Train DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_clusters = dbscan.fit_predict(X_scaled)

# Count clusters (excluding noise)
unique_clusters = set(dbscan_clusters)
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

st.write("**Number of clusters (excluding noise):**", num_clusters)
st.write("**Noise points detected:**", list(dbscan_clusters).count(-1))

# Silhouette Score 
mask = dbscan_clusters != -1

if len(set(dbscan_clusters[mask])) > 1:
    dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_clusters[mask])
    st.write("**DBSCAN Silhouette Score:**", round(dbscan_silhouette, 3))
else:
    st.write("Silhouette Score not available (only one cluster detected)")

# Plot DBSCAN clusters
fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=dbscan_clusters,
    palette="tab10",
    ax=ax2
)
ax2.set_title("DBSCAN Clustering (PCA View)")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
st.pyplot(fig2)

st.header("🔵 K-Means Clustering Vs DBSCAN")

st.write("**K-Means Silhouette Score:**", round(kmeans_silhouette, 3))
st.write("**DBSCAN Silhouette Score:**", round(dbscan_silhouette, 3))
