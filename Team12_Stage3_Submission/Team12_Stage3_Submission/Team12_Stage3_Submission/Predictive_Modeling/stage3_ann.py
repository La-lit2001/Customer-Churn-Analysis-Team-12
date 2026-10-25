
"""
Team 12 - Stage 3: ANN Churn Model

Generates:
    confusion_matrix.png
    roc_curve.png
    loss_curve.png
    feature_importance.png
    results.txt
    churn_ann_model.joblib
"""

import os
import sys
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ==========================================================
# 1. SETUP
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Save all output files in the same folder as this script.
OUT = SCRIPT_DIR

# Read the dataset path supplied in the terminal.
if len(sys.argv) > 1:
    CSV_PATH = sys.argv[1]

    if not os.path.isabs(CSV_PATH):
        CSV_PATH = os.path.abspath(CSV_PATH)
else:
    CSV_PATH = os.path.join(
        SCRIPT_DIR,
        "Dataset_ATS_v2.csv"
    )

SEED = 42

os.makedirs(OUT, exist_ok=True)


def output_path(filename):
    return os.path.join(OUT, filename)


print("=" * 60)
print("TEAM 12 - STAGE 3: ANN CHURN MODEL")
print("=" * 60)

print("Python script:", os.path.abspath(__file__))
print("Dataset:", CSV_PATH)
print("Output folder:", os.path.abspath(OUT))


# ==========================================================
# 2. LOAD AND CLEAN DATA
# ==========================================================

print("\n[1/7] Loading and cleaning dataset...")

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}\n"
        "Check the CSV file location."
    )

df = pd.read_csv(CSV_PATH)

required_columns = [
    "gender",
    "SeniorCitizen",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "MonthlyCharges",
    "InternetService",
    "Contract",
    "Churn"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.drop_duplicates().copy()

# Correct inconsistent phone-service values.
df.loc[
    (df["PhoneService"] == "No") &
    (df["MultipleLines"] == "Yes"),
    "MultipleLines"
] = "No"

# Convert binary categorical features.
mapping = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}

for col in [
    "gender",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "Churn"
]:
    df[col] = df[col].replace(mapping)

# Ensure the target contains numeric binary values.
df["Churn"] = pd.to_numeric(
    df["Churn"],
    errors="coerce"
)

df = df.dropna(subset=["Churn"]).copy()

if not set(df["Churn"].unique()).issubset({0, 1}):
    raise ValueError(
        "Churn must contain Yes/No or 0/1 values."
    )

df["Churn"] = df["Churn"].astype(int)

# Encode InternetService and Contract.
df = pd.get_dummies(
    df,
    columns=["InternetService", "Contract"],
    dtype=int
)

# Prepare input features and target.
y = df["Churn"].copy()

X = df.drop(columns=["Churn"]).copy()

# Keep numeric features only.
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values and fill missing numeric values.
X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median())

# Remove columns with no usable values.
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise ValueError("No usable numeric input features found.")

if y.nunique() != 2:
    raise ValueError(
        "The dataset must contain both churn classes."
    )

feature_columns = list(X.columns)

print("Dataset rows:", len(df))
print("Input features:", len(feature_columns))


# ==========================================================
# 3. SPLIT AND SCALE DATA
# ==========================================================

print("\n[2/7] Splitting and scaling data...")

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# FIX: Convert integer columns to floats before scaling.
# StandardScaler generates decimal values.
X_tr = X_tr.astype(float).copy()
X_te = X_te.astype(float).copy()

scale_columns = [
    col for col in ["tenure", "MonthlyCharges"]
    if col in X_tr.columns
]

sc = StandardScaler()

if scale_columns:
    sc.fit(X_tr[scale_columns])

    X_tr.loc[:, scale_columns] = sc.transform(
        X_tr[scale_columns]
    )

    X_te.loc[:, scale_columns] = sc.transform(
        X_te[scale_columns]
    )

print("Scaling completed successfully.")
print("Training records:", len(X_tr))
print("Testing records:", len(X_te))
print("Scaled columns:", scale_columns)


# ==========================================================
# 4. BALANCE TRAINING DATA
# ==========================================================

print("\n[3/7] Balancing training data...")

training_data = pd.concat(
    [
        X_tr.reset_index(drop=True),
        y_tr.reset_index(drop=True).rename("Churn")
    ],
    axis=1
)

majority = training_data[
    training_data["Churn"] == 0
]

minority = training_data[
    training_data["Churn"] == 1
]

if len(majority) == 0 or len(minority) == 0:
    raise ValueError(
        "Both churn classes must exist in training data."
    )

minority_up = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=SEED
)

balanced_data = pd.concat(
    [majority, minority_up],
    ignore_index=True
)

balanced_data = balanced_data.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

Xb = balanced_data.drop(columns=["Churn"])
yb = balanced_data["Churn"]

print("Balanced training records:", len(Xb))


# ==========================================================
# 5. TRAIN ANN MODEL
# ==========================================================

print("\n[4/7] Training ANN model...")

ann = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    learning_rate_init=1e-3,
    batch_size=64,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=SEED
)

ann.fit(Xb, yb)

model_path = output_path("churn_ann_model.joblib")

joblib.dump(
    {
        "model": ann,
        "scaler": sc,
        "scale_columns": scale_columns,
        "columns": feature_columns
    },
    model_path
)

print("Training completed.")
print("Training iterations:", ann.n_iter_)
print("Model saved:", model_path)


# ==========================================================
# 6. EVALUATE MODEL AND GENERATE CHARTS
# ==========================================================

print("\n[5/7] Evaluating ANN model...")

prob = ann.predict_proba(X_te)[:, 1]

pred = (prob >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_te, pred),
    "precision": precision_score(
        y_te, pred, zero_division=0
    ),
    "recall": recall_score(
        y_te, pred, zero_division=0
    ),
    "f1": f1_score(
        y_te, pred, zero_division=0
    ),
    "auc": roc_auc_score(y_te, prob)
}

cm = confusion_matrix(
    y_te,
    pred,
    labels=[0, 1]
)


# ----------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------

print("Generating confusion matrix...")

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("Confusion Matrix - ANN Test Set")

fig.tight_layout()

fig.savefig(
    output_path("confusion_matrix.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------
# ROC curve
# ----------------------------------------------------------

print("Generating ROC curve...")

fpr, tpr, _ = roc_curve(y_te, prob)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    fpr,
    tpr,
    label=f"ANN (AUC = {metrics['auc']:.3f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="grey",
    label="Random classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve - ANN")

ax.legend()

fig.tight_layout()

fig.savefig(
    output_path("roc_curve.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------
# Training loss curve
# ----------------------------------------------------------

print("Generating training loss curve...")

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    range(1, len(ann.loss_curve_) + 1),
    ann.loss_curve_
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Training Loss")
ax.set_title("ANN Training Loss Curve")

fig.tight_layout()

fig.savefig(
    output_path("loss_curve.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------
# Feature importance
# ----------------------------------------------------------

print("Calculating feature importance...")

pi = permutation_importance(
    ann,
    X_te,
    y_te,
    scoring="recall",
    n_repeats=10,
    random_state=SEED
)

importance = pd.Series(
    pi.importances_mean,
    index=X_te.columns
).sort_values()

print("Generating feature importance chart...")

fig, ax = plt.subplots(figsize=(9, 6))

importance.plot.barh(ax=ax)

ax.set_xlabel("Decrease in Recall When Feature Is Shuffled")
ax.set_title("Factors That Matter Most for Predicting Churn")

fig.tight_layout()

fig.savefig(
    output_path("feature_importance.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ==========================================================
# 7. SAVE RESULTS
# ==========================================================

print("\n[6/7] Saving results.txt...")

architecture = (
    f"Input layer: {X.shape[1]} features "
    f"({', '.join(X.columns)})\n"
    "Hidden layer 1: 32 neurons, ReLU\n"
    "Hidden layer 2: 16 neurons, ReLU\n"
    "Output layer: 1 neuron, churn probability\n"
    "Optimiser: Adam (lr=0.001, L2 alpha=0.001), "
    "batch size 64, early stopping "
    "(10% validation, patience 15)"
)

results_path = output_path("results.txt")

with open(results_path, "w", encoding="utf-8") as f:

    f.write("ARCHITECTURE\n")
    f.write(architecture + "\n\n")

    f.write(
        f"TRAIN/TEST: {len(X_tr)} train "
        f"(balanced to {len(Xb)}) / {len(X_te)} test\n\n"
    )

    f.write("METRICS (test set)\n")

    f.write(
        json.dumps(
            {
                key: round(float(value), 3)
                for key, value in metrics.items()
            },
            indent=2
        )
    )

    f.write("\n\nCONFUSION MATRIX [[TN FP],[FN TP]]\n")
    f.write(str(cm.tolist()))

    f.write("\n\nTOP FACTORS\n")

    f.write(
        importance.sort_values(ascending=False)
        .head(6)
        .round(3)
        .to_string()
    )

    f.write("\n")


# Print the complete results to the terminal.
print("\n" + "=" * 60)
print("ANN MODEL RESULTS")
print("=" * 60)

with open(results_path, "r", encoding="utf-8") as f:
    print(f.read())


# ==========================================================
# FINAL FILE VERIFICATION
# ==========================================================

print("\n[7/7] Verifying generated files...")

expected_files = [
    "confusion_matrix.png",
    "roc_curve.png",
    "loss_curve.png",
    "feature_importance.png",
    "results.txt",
    "churn_ann_model.joblib"
]

all_ok = True

print("\nAll output files are saved in:")
print(os.path.abspath(OUT))

for filename in expected_files:

    filepath = output_path(filename)

    if (
        os.path.isfile(filepath)
        and os.path.getsize(filepath) > 0
    ):
        print(
            f"[OK] {filename} "
            f"({os.path.getsize(filepath):,} bytes)"
        )
    else:
        print("[ERROR] File missing or empty:", filepath)
        all_ok = False

print("\n" + "=" * 60)

if all_ok:
    print("SUCCESS: All six output files were generated.")
else:
    print("ERROR: Some output files were not generated.")

print("=" * 60)