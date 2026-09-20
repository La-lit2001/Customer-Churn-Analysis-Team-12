# Customer Churn Analysis — Team 12 (Stage 2)

Advanced Telecommunications Solutions customer churn analysis project (ACS WIL Program).
This repository contains the Stage 2 deliverables: Data Preparation and Clustering Analysis.

## Repository Structure

```
├── Data_Preparation/
│   ├── Preprocessed_Dataset.csv              # Cleaned, encoded, scaled dataset (6,741 rows)
│   ├── Training_Set.csv                      # 80% stratified training split
│   ├── Testing_Set.csv                       # 20% stratified testing split
│   ├── Scaling_Techniques_Documentation.docx # Explanation of cleaning/encoding/scaling with code
│   └── preprocessing_script.py               # Full preprocessing pipeline
│
├── Clustering_Analysis/
│   ├── elbow_method.png                      # Elbow chart used to select k
│   ├── silhouette_scores.png                 # Secondary validation of k
│   ├── cluster_visualisation_labelled.png    # PCA-projected, labelled customer segments
│   ├── churn_rate_by_labelled_cluster.png    # Key business insight chart
│   ├── cluster_profile_summary.csv           # Per-cluster averages (tenure, charges, churn)
│   ├── cluster_labels_and_insights.csv       # Final labels + summary stats
│   ├── dataset_with_cluster_labels.csv       # Original data with assigned cluster per row
│   ├── kmeans_model.pkl                      # Trained K-Means model (k=4)
│   ├── Optimal_Clusters_Analysis.docx        # Elbow method write-up + cluster interpretation
│   └── clustering_script.py                  # Full clustering pipeline
│
└── README.md
```

## Summary of Approach

1. **Data Preparation**: Cleaned 7,043-row dataset (removed 302 duplicates, fixed 256 logically
   inconsistent records), encoded categorical variables, applied StandardScaler to numeric
   fields, and split into training/testing sets (80/20, stratified by churn).

2. **Clustering Analysis**: Used the elbow method (validated with silhouette score) to select
   k=4. Trained a K-Means model that separates customers into four segments primarily along
   tenure and monthly spend.

## Key Finding

| Segment | Avg Tenure | Avg Monthly Charges | Churn Rate |
|---|---|---|---|
| Loyal Low-Spend | 54.3 mo | $35.03 | 5.0% |
| **New High-Spend (High Risk)** | 15.1 mo | $81.77 | **49.0%** |
| Loyal High-Spend | 58.6 mo | $93.18 | 16.0% |
| New Low-Spend | 11.3 mo | $33.88 | 25.0% |

Customers who are both **new and high-spending churn at ~10x the rate** of loyal, low-spend
customers — the clearest actionable target for retention efforts identified in Stage 2.

## Team 12

| Name | Role |
|---|---|
| Lalit Dilip Gonji | Project Manager |
| Harshrajsinh Gohil | Data Engineer — Preprocessing and Feature Engineering |
| Neel Patel | Data Analyst — Clustering Analysis |
| Dishant Patel | Data Analyst — Predictive Modeling |
| Rushi Patel | Business Analyst |
