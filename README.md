# 🔍 Fraud Detection Pipeline — Supervised Learning

> **Project 2** | Data Science Industrial Training Kit | DecodeLabs (Batch 2026)

A complete end-to-end supervised machine learning pipeline that detects fraudulent financial transactions in a highly imbalanced dataset using SMOTE, Logistic Regression, Random Forest, and strict fraud-optimized evaluation metrics.

---

## 📌 Project Overview

In real financial datasets, fraud represents less than 2% of all transactions. A naive model that predicts "Legitimate" every time achieves 98%+ accuracy — while catching **zero fraud**. This project solves that fundamental problem using:

- **SMOTE** to synthetically generate minority class examples
- **imblearn Pipeline** to prevent data leakage during resampling
- **Precision, Recall, and ROC-AUC** instead of misleading accuracy
- **GridSearchCV** to tune both models and SMOTE parameters simultaneously

---

## 📂 Repository Structure

```
├── fraud_data.csv           # Raw imbalanced dataset (10,000 transactions, 1.70% fraud)
├── fraud_detection.py       # Complete fraud detection pipeline script
└── README.md                # Project documentation
```

---

## 🧩 Dataset Description

A simulated financial transaction dataset with **10,000 rows** and **1.70% fraud rate**, matching the structure of real-world fraud datasets:

| Column | Description |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `amount` | Transaction amount (₹) |
| `time_of_day` | Hour of transaction (0-23) |
| `location_risk` | Risk score of transaction location (0.0-1.0) |
| `is_foreign` | 1 = Foreign transaction, 0 = Domestic |
| `num_prev_transactions` | Number of prior transactions on account |
| `account_age_days` | Age of the account in days |
| `is_fraud` | Target variable — 1 = Fraud, 0 = Legitimate |

**Class Distribution:**
- Legitimate: 9,830 (98.30%)
- Fraudulent: 170 (1.70%)

---

## ⚙️ Methodology

### The Core Problem: Why Accuracy Fails Here

A model predicting "Legitimate" for every transaction achieves 98.30% accuracy while catching zero fraud — catastrophic for a financial system. This project completely discards accuracy in favor of fraud-specific metrics.

### 1️⃣ The Golden Rule: Split Before SMOTE

```
❌ WRONG:  Entire Data → SMOTE → Train/Test Split
✅ CORRECT: Entire Data → Train/Test Split → SMOTE (on training only)
```

Applying SMOTE before splitting causes **data leakage** — synthetic test samples contaminate the training process, producing falsely optimistic results.

### 2️⃣ SMOTE — Synthetic Minority Over-sampling

SMOTE doesn't clone existing fraud examples — it **interpolates** between them to create brand new, realistic synthetic fraud transactions. Formula:

```
X_new = X_i + λ × (X_nn − X_i)    where λ ~ Uniform(0, 1)
```

### 3️⃣ imblearn Pipeline (Not sklearn Pipeline)

The `imblearn.pipeline.Pipeline` safely isolates SMOTE **inside each cross-validation fold** during GridSearchCV, ensuring the validation fold never sees synthetic data.

```python
Pipeline([
    ('scaler', StandardScaler()),      # Logistic Regression only
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())
])
```

### 4️⃣ Two Models Trained

| Model | Scaler Needed | Decision Boundary |
|---|---|---|
| Logistic Regression | ✅ Yes (StandardScaler) | Linear |
| Random Forest | ❌ No (tree-based) | Non-linear, complex |

### 5️⃣ Correct Evaluation Metrics

| Metric | Formula | What it means |
|---|---|---|
| **Precision** | TP / (TP + FP) | When we flag fraud, are we right? |
| **Recall** | TP / (TP + FN) | Did we catch ALL the fraud? |
| **ROC-AUC** | Area under curve | Overall fraud vs legit separation ability |

---

## 📊 Results

| Model | ROC-AUC | Precision (Fraud) | Recall (Fraud) | Fraud Caught |
|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.00 | 1.00 | 34/34 ✅ |
| Random Forest | 1.0000 | 1.00 | 1.00 | 34/34 ✅ |

**Confusion Matrix (both models):**
```
True Negatives  (Legit correctly caught):  1966
False Positives (Legit wrongly flagged):      0
False Negatives (Fraud missed!):              0
True Positives  (Fraud correctly caught):    34
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Execution
```bash
python fraud_detection.py
```

The script will print step-by-step output for all 5 stages and display final evaluation metrics for both models.

---

## 🔑 Key Concepts Mastered

| Concept | Application |
|---|---|
| Class Imbalance | Real-world fraud datasets are 99%+ legitimate |
| SMOTE | Synthetic interpolation, not duplication |
| Data Leakage | Prevented by splitting BEFORE resampling |
| imblearn Pipeline | Isolates SMOTE within CV folds |
| GridSearchCV | Tunes SMOTE + model params simultaneously |
| Recall over Accuracy | Missing fraud = direct financial loss |

---

## 🛠️ Tech Stack

- **Python** — core scripting language
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — ML models, GridSearchCV, metrics
- **Imbalanced-learn** — SMOTE, imblearn Pipeline

---

## 🎯 Key Takeaway

> A model with 98% accuracy that catches zero fraud is not a fraud detector — it is a liability. The real metric of success in imbalanced classification is **Recall**: catching every fraudulent event before it causes financial damage.

---

## 📬 Contact

Project completed as part of the **DecodeLabs Industrial Training Kit (Batch 2026)**.
🌐 [decodelabs.tech](http://www.decodelabs.tech) | 📍 Greater Lucknow, India
