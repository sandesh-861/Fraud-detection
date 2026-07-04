"""
=====================================================================
PROJECT 2: Supervised Learning — Fraud Detection Pipeline
=====================================================================
Requirements covered:
  1. Handle class imbalance using SMOTE
  2. Train Logistic Regression + Random Forest
  3. Evaluate using Precision, Recall, ROC-AUC (NOT accuracy)
  4. Tune hyperparameters using GridSearchCV
  5. Use imblearn Pipeline (prevents data leakage)
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # IMPORTANT: imblearn, NOT sklearn

# ---------------------------------------------------------------
# STEP 0: LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv('data/fraud_data.csv')

print("=" * 60)
print("STEP 0: RAW DATA OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nFraud distribution:")
print(df['is_fraud'].value_counts())
print(f"\nFraud rate: {df['is_fraud'].mean():.2%}")
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------------
# STEP 1: PREPARE FEATURES AND TARGET
# ---------------------------------------------------------------
# X = features (everything except target and ID)
# y = target (what we want to predict: 0=legit, 1=fraud)

X = df.drop(columns=['transaction_id', 'is_fraud'])
y = df['is_fraud']

print("\n" + "=" * 60)
print("STEP 1: FEATURES AND TARGET")
print("=" * 60)
print("Feature columns:", list(X.columns))
print("Target: is_fraud (0=Legitimate, 1=Fraud)")

# ---------------------------------------------------------------
# STEP 2: SPLIT DATA FIRST (then SMOTE only on training data)
# ---------------------------------------------------------------
# GOLDEN RULE from your PDF: ALWAYS split BEFORE applying SMOTE
# stratify=y ensures both train and test have same fraud ratio

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,
    stratify=y           # keeps fraud ratio same in both splits
)

print("\n" + "=" * 60)
print("STEP 2: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set:     {X_test.shape[0]} rows")
print(f"Train fraud count: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Test fraud count:  {y_test.sum()} ({y_test.mean():.2%})")

# ---------------------------------------------------------------
# STEP 3: BUILD PIPELINES (SMOTE safely inside pipeline)
# ---------------------------------------------------------------
# Pipeline automatically applies SMOTE only on training fold
# during cross-validation — preventing data leakage

print("\n" + "=" * 60)
print("STEP 3: BUILDING PIPELINES")
print("=" * 60)

# Pipeline 1: Logistic Regression (needs StandardScaler)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),   # normalize features
    ('smote', SMOTE(random_state=42)),  # balance classes
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Pipeline 2: Random Forest (no scaler needed - tree-based)
rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),  # balance classes
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

print("Pipeline 1: StandardScaler → SMOTE → Logistic Regression")
print("Pipeline 2: SMOTE → Random Forest")

# ---------------------------------------------------------------
# STEP 4: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ---------------------------------------------------------------
# GridSearchCV tries every combination and finds the best one
# scoring='recall' because we care most about catching fraud

print("\n" + "=" * 60)
print("STEP 4: HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

# Logistic Regression parameter grid
lr_params = {
    'smote__k_neighbors': [3, 5],
    'classifier__C': [0.01, 0.1, 1.0]  # regularization strength
}

# Random Forest parameter grid
rf_params = {
    'smote__k_neighbors': [3, 5],
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20, None]
}

print("Tuning Logistic Regression... (this takes ~30 seconds)")
lr_grid = GridSearchCV(
    lr_pipeline,
    lr_params,
    cv=5,            # 5-fold cross validation
    scoring='recall', # optimize for catching fraud
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)
print(f"Best LR params: {lr_grid.best_params_}")

print("\nTuning Random Forest... (this takes ~1-2 minutes)")
rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")

# ---------------------------------------------------------------
# STEP 5: EVALUATE ON TEST DATA (untouched, still imbalanced)
# ---------------------------------------------------------------
# We evaluate on ORIGINAL imbalanced test data — real world!

print("\n" + "=" * 60)
print("STEP 5: FINAL EVALUATION ON TEST DATA")
print("=" * 60)

for name, model in [("Logistic Regression", lr_grid),
                     ("Random Forest", rf_grid)]:
    print(f"\n--- {name} ---")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC-AUC Score
    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc:.4f}  (aim for 0.85+)")

    # Precision, Recall, F1
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Legitimate', 'Fraud']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives  (Legit correctly caught):  {cm[0][0]}")
    print(f"  False Positives (Legit wrongly flagged):   {cm[0][1]}")
    print(f"  False Negatives (Fraud missed!):           {cm[1][0]}")
    print(f"  True Positives  (Fraud correctly caught):  {cm[1][1]}")

print("\n" + "=" * 60)
print("PROJECT 2 COMPLETE!")
print("=" * 60)
print("Key metrics used: Precision, Recall, ROC-AUC")
print("Accuracy was intentionally ignored (misleading on imbalanced data)")
