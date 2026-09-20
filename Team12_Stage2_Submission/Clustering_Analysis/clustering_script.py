"""
Stage 2 - Clustering Analysis
Customer Churn Analysis - Team 12

Determines the optimal number of clusters via the elbow method,
trains a K-Means model, and produces labelled visualisations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import joblib


# ---------------------------------------------------------------
# SETTINGS AND FILE PATHS
# ---------------------------------------------------------------

RANDOM_STATE = 42

plt.rcParams["figure.dpi"] = 120

# Automatically find the project root based on this script's location
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data_Preparation"
CLUSTER_DIR = BASE_DIR / "Clustering_Analysis"

# Make sure the clustering folder exists
CLUSTER_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# 1. LOAD PREPROCESSED FEATURES
# ---------------------------------------------------------------

features_file = CLUSTER_DIR / "_features_for_clustering.csv"

X = pd.read_csv(features_file)

print(f"Clustering feature set shape: {X.shape}")
print(f"Features used: {list(X.columns)}")


# ---------------------------------------------------------------
# 2. ELBOW METHOD AND SILHOUETTE SCORE
# ---------------------------------------------------------------

inertias = []
silhouettes = []

k_range = range(1, 11)

for k in k_range:

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    km.fit(X)

    # Inertia for elbow method
    inertias.append(km.inertia_)

    # Silhouette score requires at least 2 clusters
    if k > 1:
        score = silhouette_score(X, km.labels_)
        silhouettes.append(score)
    else:
        silhouettes.append(np.nan)


# ---------------------------------------------------------------
# 2A. ELBOW METHOD PLOT
# ---------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    list(k_range),
    inertias,
    marker="o"
)

ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax.set_title("Elbow Method for Optimal k")

ax.set_xticks(list(k_range))
ax.grid(alpha=0.3)

plt.tight_layout()

elbow_file = CLUSTER_DIR / "elbow_method.png"

plt.savefig(elbow_file)
plt.close()

print(f"Saved: {elbow_file}")


# ---------------------------------------------------------------
# 2B. SILHOUETTE SCORE PLOT
# ---------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    list(k_range)[1:],
    silhouettes[1:],
    marker="o"
)

ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score by Number of Clusters")

ax.set_xticks(list(k_range)[1:])
ax.grid(alpha=0.3)

plt.tight_layout()

silhouette_file = CLUSTER_DIR / "silhouette_scores.png"

plt.savefig(silhouette_file)
plt.close()

print(f"Saved: {silhouette_file}")


# ---------------------------------------------------------------
# 2C. PRINT CLUSTER EVALUATION RESULTS
# ---------------------------------------------------------------

print("\nInertia by k:")

print(
    dict(
        zip(
            k_range,
            [round(i, 1) for i in inertias]
        )
    )
)


print("\nSilhouette by k:")

print(
    dict(
        zip(
            k_range,
            [
                round(s, 3) if not np.isnan(s) else None
                for s in silhouettes
            ]
        )
    )
)


# ---------------------------------------------------------------
# 3. SELECT NUMBER OF CLUSTERS
# ---------------------------------------------------------------

# Based on the elbow analysis and cluster interpretability
OPTIMAL_K = 4

print(
    f"\nOptimal number of clusters selected: {OPTIMAL_K}"
)


# ---------------------------------------------------------------
# 4. TRAIN FINAL K-MEANS MODEL
# ---------------------------------------------------------------

kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X)


# Save K-Means model
model_file = CLUSTER_DIR / "kmeans_model.pkl"

joblib.dump(
    kmeans,
    model_file
)

print(f"Saved: {model_file}")


# ---------------------------------------------------------------
# 5. PCA VISUALISATION
# ---------------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(X)

# Transform cluster centres into PCA space
centers_pca = pca.transform(
    kmeans.cluster_centers_
)


fig, ax = plt.subplots(figsize=(8, 6))


scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="tab10",
    s=12,
    alpha=0.6
)


# Plot cluster centres
ax.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Cluster Centres"
)


ax.set_xlabel(
    f"PCA Component 1 "
    f"({pca.explained_variance_ratio_[0]:.1%} variance)"
)

ax.set_ylabel(
    f"PCA Component 2 "
    f"({pca.explained_variance_ratio_[1]:.1%} variance)"
)

ax.set_title(
    f"Customer Segments "
    f"(K-Means, k={OPTIMAL_K}) - PCA Projection"
)


# Cluster legend
legend1 = ax.legend(
    *scatter.legend_elements(),
    title="Cluster"
)

ax.add_artist(legend1)

ax.legend(
    loc="upper right"
)

plt.tight_layout()


pca_file = CLUSTER_DIR / "cluster_visualisation_pca.png"

plt.savefig(pca_file)

plt.close()

print(f"Saved: {pca_file}")


# ---------------------------------------------------------------
# 6. CLUSTER PROFILING
# ---------------------------------------------------------------

# Load original dataset for readable interpretation
original_file = DATA_DIR / "Dataset_ATS_v2.csv"

df_original = pd.read_csv(original_file)


# Remove duplicates in the same way as preprocessing
df_original = (
    df_original
    .drop_duplicates()
    .reset_index(drop=True)
)


# Fix logical inconsistency
inconsistent_mask = (
    (df_original["PhoneService"] == "No")
    &
    (df_original["MultipleLines"] == "Yes")
)

df_original.loc[
    inconsistent_mask,
    "MultipleLines"
] = "No"


# Make sure number of rows matches clustering data
if len(df_original) != len(cluster_labels):

    raise ValueError(
        "The number of rows in the cleaned original dataset "
        "does not match the number of clustering records."
    )


# Add cluster number
df_original["Cluster"] = cluster_labels


# ---------------------------------------------------------------
# 6A. NUMERIC PROFILE
# ---------------------------------------------------------------

profile_numeric = (
    df_original
    .groupby("Cluster")
    [["tenure", "MonthlyCharges", "SeniorCitizen"]]
    .mean()
    .round(1)
)


# ---------------------------------------------------------------
# 6B. CHURN RATE
# ---------------------------------------------------------------

profile_churn = (
    df_original
    .groupby("Cluster")["Churn"]
    .apply(lambda s: (s == "Yes").mean())
    .round(3)
)


# ---------------------------------------------------------------
# 6C. MOST COMMON CONTRACT
# ---------------------------------------------------------------

profile_contract = (
    df_original
    .groupby("Cluster")["Contract"]
    .agg(lambda s: s.mode()[0])
)


# ---------------------------------------------------------------
# 6D. MOST COMMON INTERNET SERVICE
# ---------------------------------------------------------------

profile_internet = (
    df_original
    .groupby("Cluster")["InternetService"]
    .agg(lambda s: s.mode()[0])
)


# ---------------------------------------------------------------
# 6E. CLUSTER SIZE
# ---------------------------------------------------------------

profile_size = (
    df_original
    .groupby("Cluster")
    .size()
)


# ---------------------------------------------------------------
# 6F. COMBINE PROFILE INFORMATION
# ---------------------------------------------------------------

profile = profile_numeric.copy()

profile["ChurnRate"] = profile_churn

profile["MostCommonContract"] = profile_contract

profile["MostCommonInternet"] = profile_internet

profile["ClusterSize"] = profile_size


# ---------------------------------------------------------------
# PRINT CLUSTER PROFILE
# ---------------------------------------------------------------

print("\nCluster profile summary:")

print(profile)


# ---------------------------------------------------------------
# SAVE CLUSTER PROFILE
# ---------------------------------------------------------------

profile_file = (
    CLUSTER_DIR /
    "cluster_profile_summary.csv"
)

profile.to_csv(profile_file)

print(
    f"\nSaved: {profile_file}"
)


# ---------------------------------------------------------------
# 7. CHURN RATE BY CLUSTER
# ---------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))


bars = ax.bar(
    profile.index.astype(str),
    profile["ChurnRate"]
)


ax.set_xlabel("Cluster")

ax.set_ylabel("Churn Rate")

ax.set_title(
    "Churn Rate by Customer Cluster"
)


# Add percentage labels
for bar, val in zip(
    bars,
    profile["ChurnRate"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val:.0%}",
        ha="center"
    )


plt.tight_layout()


churn_chart_file = (
    CLUSTER_DIR /
    "churn_rate_by_cluster.png"
)

plt.savefig(
    churn_chart_file
)

plt.close()

print(
    f"Saved: {churn_chart_file}"
)


# ---------------------------------------------------------------
# 8. SAVE DATASET WITH CLUSTER LABELS
# ---------------------------------------------------------------

cluster_dataset_file = (
    CLUSTER_DIR /
    "dataset_with_cluster_labels.csv"
)

df_original.to_csv(
    cluster_dataset_file,
    index=False
)

print(
    f"Saved: {cluster_dataset_file}"
)


# ---------------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------------

print("\nClustering analysis complete.")