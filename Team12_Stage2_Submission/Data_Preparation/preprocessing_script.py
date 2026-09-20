"""
Stage 2 - Data Preparation
Customer Churn Analysis - Team 12

Handles: missing data check, duplicate removal, logical-consistency check,
categorical encoding, feature scaling, and train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("Data_Preparation/Dataset_ATS_v2.csv")
print(f"Original shape: {df.shape}")

# ---------------------------------------------------------------
# 2. Handle missing data
# ---------------------------------------------------------------
missing_before = df.isnull().sum().sum()
print(f"Missing values found: {missing_before}")
# No literal missing values in this dataset, but we still run a defensive
# fill in case future data drops introduce any, so the pipeline is robust.
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------------------------------------------------------
# 3. Remove duplicate records (data integrity)
# ---------------------------------------------------------------
dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {dupes}")

# ---------------------------------------------------------------
# 4. Fix logical inconsistency: MultipleLines requires PhoneService
# ---------------------------------------------------------------
inconsistent_mask = (df["PhoneService"] == "No") & (df["MultipleLines"] == "Yes")
print(f"Logically inconsistent rows fixed (PhoneService=No but MultipleLines=Yes): {inconsistent_mask.sum()}")
df.loc[inconsistent_mask, "MultipleLines"] = "No"

print(f"Shape after cleaning: {df.shape}")

# ---------------------------------------------------------------
# 5. Encode categorical variables
# ---------------------------------------------------------------
df_encoded = df.copy()

binary_cols = ["gender", "Dependents", "PhoneService", "MultipleLines", "InternetService", "Churn"]
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Contract has 3 categories -> one-hot encode
df_encoded = pd.get_dummies(df_encoded, columns=["Contract"], prefix="Contract")
contract_dummy_cols = [c for c in df_encoded.columns if c.startswith("Contract_")]
df_encoded[contract_dummy_cols] = df_encoded[contract_dummy_cols].astype(int)

print("\nLabel encoding mapping:")
for col, mapping in label_encoders.items():
    print(f"  {col}: {mapping}")
print(f"  Contract: one-hot encoded into {contract_dummy_cols}")

# ---------------------------------------------------------------
# 6. Feature scaling (numeric features)
# ---------------------------------------------------------------
numeric_cols = ["tenure", "MonthlyCharges"]
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

print("\nScaled numeric feature summary (should be ~0 mean, ~1 std):")
print(df_encoded[numeric_cols].describe().loc[["mean", "std"]])

# ---------------------------------------------------------------
# 7. Save the fully preprocessed dataset
# ---------------------------------------------------------------
df_encoded.to_csv("Data_Preparation/Preprocessed_Dataset.csv", index=False)
print("\nSaved: Data_Preparation/Preprocessed_Dataset.csv")

# ---------------------------------------------------------------
# 8. Train / test split
# ---------------------------------------------------------------
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

train_df = X_train.copy()
train_df["Churn"] = y_train
test_df = X_test.copy()
test_df["Churn"] = y_test

train_df.to_csv("Data_Preparation/Training_Set.csv", index=False)
test_df.to_csv("Data_Preparation/Testing_Set.csv", index=False)

print(f"\nTraining set: {train_df.shape[0]} rows ({train_df.shape[0]/len(df_encoded):.0%})")
print(f"Testing set:  {test_df.shape[0]} rows ({test_df.shape[0]/len(df_encoded):.0%})")
print(f"Churn rate - train: {y_train.mean():.3f}, test: {y_test.mean():.3f} (stratified split preserves class balance)")

# Save a version WITHOUT the target column too, for clustering (unsupervised)
df_encoded.drop(columns=["Churn"]).to_csv("Clustering_Analysis/_features_for_clustering.csv", index=False)

print("\nPreprocessing complete.")