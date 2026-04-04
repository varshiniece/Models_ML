# =========================================================
# CUSTOMER CHURN MODEL - LOGISTIC REGRESSION + XGBOOST
# =========================================================

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from xgboost import XGBClassifier

# =========================================================
# 1. DATABASE CONNECTION
# =========================================================


RDS_CONN = "postgresql+psycopg2://redash_user:#X5vf6o5Ef4lNeh&DUw#!OrV@sf-prod-replica.c5wxkh2ztikt.ap-south-1.rds.amazonaws.com:5432/postgres"
engine = create_engine(RDS_CONN)

QUERY = """
SELECT
    o.customer_id,
    o.id AS order_id,
    o.expected_delivery_date AS order_date,
    SUM(oi.quantity * oi.purchase_price) AS order_amount,
    COUNT(DISTINCT oi.variant_id) AS product_count,
    AVG(oi.rating) AS rating
FROM "order" o
JOIN order_item oi ON o.id = oi.order_id
JOIN customers c ON c.id = o.customer_id
WHERE o.expected_delivery_date BETWEEN '2025-01-01' AND '2025-10-30'
  AND oi.status IN ('DELIVERED','CONFIRMED','PARTIALLY_DELIVERED')
  AND c.type <> 'B2B'
GROUP BY 1,2,3
"""

orders_df = pd.read_sql(QUERY, engine)

# =========================================================
# 2. BASIC CLEANING
# =========================================================

orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
orders_df['rating'] = pd.to_numeric(orders_df['rating'], errors='coerce')

# =========================================================
# 3. SNAPSHOT FEATURE ENGINEERING
# =========================================================

snapshot_dates = pd.date_range(
    start='2025-04-30',
    end='2025-09-30',
    freq='ME'
)

all_snapshots = []

for snapshot_date in snapshot_dates:

    hist_df = orders_df[orders_df['order_date'] <= snapshot_date]

    # Recency
    last_order = (
        hist_df.groupby('customer_id')['order_date']
        .max()
        .reset_index(name='last_order_date')
    )
    last_order['days_since_last_order'] = (
        snapshot_date - last_order['last_order_date']
    ).dt.days

    # Frequency
    freq_30 = (
        hist_df[hist_df['order_date'] >= snapshot_date - pd.Timedelta(days=30)]
        .groupby('customer_id')
        .size()
        .reset_index(name='orders_last_30d')
    )

    freq_90 = (
        hist_df[hist_df['order_date'] >= snapshot_date - pd.Timedelta(days=90)]
        .groupby('customer_id')
        .size()
        .reset_index(name='orders_last_90d')
    )

    # Monetary
    spend_30 = (
        hist_df[hist_df['order_date'] >= snapshot_date - pd.Timedelta(days=30)]
        .groupby('customer_id')['order_amount']
        .sum()
        .reset_index(name='total_spend_last_30d')
    )

    avg_order_value = (
        hist_df.groupby('customer_id')['order_amount']
        .mean()
        .reset_index(name='avg_order_value')
    )

    # Tenure
    first_order = (
        hist_df.groupby('customer_id')['order_date']
        .min()
        .reset_index(name='first_order_date')
    )
    first_order['customer_tenure_days'] = (
        snapshot_date - first_order['first_order_date']
    ).dt.days

    dfs = [last_order, freq_30, freq_90, spend_30, avg_order_value, first_order]

    features = reduce(
        lambda l, r: pd.merge(l, r, on='customer_id', how='left'),
        dfs
    )

    features.fillna(0, inplace=True)
    features['snapshot_date'] = snapshot_date

    # CHURN LABEL (no order in next 30 days)
    future_df = orders_df[
        (orders_df['order_date'] > snapshot_date) &
        (orders_df['order_date'] <= snapshot_date + pd.Timedelta(days=30))
    ]

    active_customers = future_df['customer_id'].unique()

    features['churn'] = features['customer_id'].apply(
        lambda x: 0 if x in active_customers else 1
    )

    all_snapshots.append(features)

training_df = pd.concat(all_snapshots, ignore_index=True)

# =========================================================
# 4. FEATURE SELECTION
# =========================================================

feature_cols = [
    'days_since_last_order',
    'orders_last_30d',
    'orders_last_90d',
    'total_spend_last_30d',
    'avg_order_value',
    'customer_tenure_days'
]

X = training_df[feature_cols]
y = training_df['churn']

# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================================================
# 6. LOGISTIC REGRESSION
# =========================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:,1]

roc_lr = roc_auc_score(y_test, y_prob_lr)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print("ROC-AUC:", round(roc_lr, 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# =========================================================
# 7. XGBOOST
# =========================================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

roc_xgb = roc_auc_score(y_test, y_prob_xgb)

print("\n==============================")
print("XGBOOST RESULTS")
print("==============================")
print("ROC-AUC:", round(roc_xgb, 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

# =========================================================
# 8. MODEL COMPARISON
# =========================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(f"Logistic Regression AUC: {round(roc_lr,4)}")
print(f"XGBoost AUC: {round(roc_xgb,4)}")

# =========================================================
# 9. FEATURE IMPORTANCE (XGBOOST)
# =========================================================

importance = pd.Series(xgb.feature_importances_, index=feature_cols)
print("\nTop Features Driving Churn:")
print(importance.sort_values(ascending=False))

# ===============================
# SAVE TEST PREDICTIONS TO CSV
# ===============================

test_results = X_test.copy()

test_results["actual_churn"] = y_test.values
test_results["predicted_churn"] = y_pred_xgb
test_results["churn_probability"] = y_prob_xgb

# Optional: add customer_id if you want
test_results["customer_id"] = training_df.loc[X_test.index, "customer_id"].values

test_results = test_results[
    ["customer_id"] +
    list(X_test.columns) +
    ["actual_churn", "predicted_churn", "churn_probability"]
]

test_results.to_csv("churn_test_predictions.csv", index=False)

print("CSV saved successfully!")

