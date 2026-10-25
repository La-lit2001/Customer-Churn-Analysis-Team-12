# Customer Churn Analysis – Team 12

**Advanced Telecommunications Solutions · ACS WIL Program**

This repository contains our work on predicting and understanding customer churn for a telecom company, using the dataset `Dataset_ATS_v2.csv` (7,043 rows, 10 columns).

| Stage | Focus | Status |
|-------|-------|--------|
| Stage 1 | Planning and setup (Jira, Confluence, project charter) | Complete |
| Stage 2 | Data preparation and clustering analysis | Complete |
| Stage 3 | Predictive modelling (ANN) and final report | Complete |

## Team

| Name | Role |
|------|------|
| Lalit Dilip Gonji | Project Manager |
| Harshrajsinh Gohil | Data Engineer |
| Neel Patel | Data Analyst (Clustering) |
| Dishant Patel | Data Analyst (Predictive Modelling) |
| Rushi Patel | Business Analyst |

## Repository structure

```
.
├── README.md
├── Dataset_ATS_v2.csv            # raw dataset
├── (Stage 2 folders)             # data preparation and clustering
└── Predictive_Modeling/          # Stage 3 deliverables
    ├── Architecture_of_ANN_Model.docx
    ├── churn_ann_model.joblib    # trained model (+ scaler and column list)
    ├── stage3_ann.py             # training and evaluation code
    ├── results.txt               # all metrics from the final run
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

The Final Report (four documents) is submitted separately through the LMS.

## Data preparation (Stage 2 summary)

- 302 exact duplicate rows removed.
- 256 logically inconsistent rows corrected (`PhoneService = No` but `MultipleLines = Yes` set to `MultipleLines = No`).
- Final cleaned dataset: **6,741 customers**, overall churn rate **26.6%**.
- Binary fields label-encoded; `InternetService` and `Contract` one-hot encoded.
- `tenure` and `MonthlyCharges` standardised with `StandardScaler`.
- 80/20 train/test split, stratified by `Churn` (5,392 train / 1,349 test).
- Customer segmentation with K-means (k = 4), chosen using the elbow method and silhouette score.

## ANN model (Stage 3)

| Layer | Details |
|-------|---------|
| Input | 12 features |
| Hidden 1 | 32 neurons, ReLU |
| Hidden 2 | 16 neurons, ReLU |
| Output | 1 neuron, sigmoid (probability of churn) |

- Optimiser: Adam (learning rate 0.001), L2 regularisation (alpha 0.001), batch size 64
- Early stopping with a 10% validation split (patience 15)
- Class imbalance handled by randomly oversampling churners in the **training set only**; the test set is untouched
- Implemented with scikit-learn `MLPClassifier`

### Results (test set, 1,349 unseen customers)

| Metric | Value |
|--------|-------|
| Accuracy | 72.4% |
| Precision | 48.8% |
| Recall | 78.8% |
| F1-score | 60.3% |
| AUC | 0.817 |

Confusion matrix: 695 true negatives, 296 false positives, 76 false negatives, 282 true positives. The model identifies 282 of the 358 customers who actually churned. Recall was prioritised over accuracy because only about a quarter of customers churn, so accuracy alone is misleading.

### Key findings

- **Tenure** is the strongest predictor of churn: 48% of customers churn in their first 12 months, versus under 10% after 48 months.
- **Monthly charges** is the second most important factor.
- Customers **without dependents** churn more (31% vs 16%).
- The highest-risk segment is newer customers (about 15 months) paying about $81 per month, with roughly 49% churn.
- Contract type showed little relationship with churn in this dataset.

## How to run

Requirements: Python 3.9+

```bash
pip install pandas scikit-learn matplotlib joblib
python Predictive_Modeling/stage3_ann.py Dataset_ATS_v2.csv
```

Running the script re-trains the model and regenerates the model file, charts and `results.txt` in a `Predictive_Modeling/` folder in the current directory. A fixed random seed (42) is used, so results are reproducible with the same library versions.

### Using the trained model

```python
import joblib, pandas as pd

bundle = joblib.load("Predictive_Modeling/churn_ann_model.joblib")
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]

# new_customers: a DataFrame prepared with the same cleaning and encoding,
# with columns in the order stored in `columns`
new_customers = new_customers[columns].copy()
new_customers[["tenure", "MonthlyCharges"]] = scaler.transform(new_customers[["tenure", "MonthlyCharges"]])
churn_probability = model.predict_proba(new_customers)[:, 1]
```

## Limitations

- Only 10 input columns (no payment method, usage, complaints or billing history).
- Precision is about 49%, so roughly half of flagged customers would have stayed.
- Single train/test split; cross-validation and baseline model comparisons would strengthen the results.
- The dataset is a fictional case study, so findings should be validated on real data before use.

## Tools

Python (pandas, scikit-learn, matplotlib, joblib) · Jira and Confluence (project management and documentation) · GitHub (version control) · Zoom (recorded team meetings)
