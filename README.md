# 🛡️ Financial Fraud Detection System
### End-to-End AI/ML Pipeline with Explainability, SMOTE & Real-time Prediction

> Detecting fraudulent credit card transactions using advanced machine learning, class imbalance handling, SHAP explainability, and threshold optimization — built for production readiness.
---

## 📌 Problem Statement

Credit card fraud costs the global financial industry **over $32 billion annually**. Traditional rule-based systems struggle with evolving fraud patterns and generate high false positive rates — flagging legitimate transactions and damaging customer experience.

This project builds a **production-grade ML pipeline** that:
- Detects fraudulent transactions with high precision and recall
- Handles extreme class imbalance (fraud = ~0.17% of all transactions)
- Explains **why** a transaction is flagged using SHAP values
- Optimizes decision thresholds beyond the default 0.5 to maximize F1
- Exposes a real-time prediction function ready for API integration

---

## 🗂️ Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807
- **Fraud Cases:** 492 (~0.172%)
- **Features:** 30 (V1–V28 PCA components + Time + Amount)
- **Target:** `0` = Legitimate, `1` = Fraud

> ⚠️ Dataset is highly imbalanced — a naive model predicting all-legitimate achieves 99.8% accuracy but catches 0 frauds. This is why standard accuracy is a misleading metric here.

---

## 🔄 Pipeline Architecture

```
Raw Data (284K transactions)
        ↓
  EDA & Visualization
  (Class distribution, Amount/Time patterns, Correlation)
        ↓
  Feature Engineering
  (Log Amount, Hour of Day, Night Flag, High Amount Flag,
   Round Amount Flag, V1xV2 / V3xV4 interaction terms)
        ↓
  Preprocessing
  (StandardScaler, Stratified 80/20 Split)
        ↓
  SMOTE Oversampling
  (Balance fraud:legit ratio in training set only)
        ↓
  Model Training & Comparison
  (Logistic Regression | Random Forest | XGBoost | LightGBM)
        ↓
  Evaluation
  (ROC-AUC | Avg Precision | F1 | Confusion Matrix | PR Curve)
        ↓
  SHAP Explainability
  (Feature importance + per-transaction explanations)
        ↓
  Threshold Optimization
  (Find optimal decision boundary beyond default 0.5)
        ↓
  Model Persistence & Real-time Prediction Function
```

---

## 🧹 Feature Engineering

| Feature | Description | Rationale |
|---|---|---|
| `Log_Amount` | log(1 + Amount) | Reduces right skew in transaction amounts |
| `Hour` | Hour of day (0-23) | Fraud patterns vary by time of day |
| `Is_Night` | 1 if between 10PM-5AM | Fraudsters often act during off-hours |
| `High_Amount` | 1 if amount > 95th percentile | Unusually large transactions are risky |
| `Round_Amount` | 1 if amount is a whole number | Round amounts are common in automated fraud |
| `V1_V2_interaction` | V1 x V2 | Captures non-linear PCA feature interactions |
| `V3_V4_interaction` | V3 x V4 | Captures non-linear PCA feature interactions |

---

## ⚖️ Handling Class Imbalance

Three strategies applied:

1. **SMOTE** — Generates synthetic fraud samples in training set only (no data leakage)
2. **`class_weight='balanced'`** — Applied to Logistic Regression and Random Forest
3. **`scale_pos_weight`** — Applied to XGBoost to penalize missing fraud cases more heavily

> SMOTE is applied **after** the train/test split to prevent synthetic samples from leaking into evaluation.

---

## 🤖 Models Trained

| Model | Strengths | Config |
|---|---|---|
| Logistic Regression | Fast baseline, interpretable | `C=0.1`, L2, balanced |
| Random Forest | Robust to outliers, feature importance | 200 trees, depth=10 |
| XGBoost | Gradient boosting, handles imbalance | scale_pos_weight, lr=0.05 |
| LightGBM | Fast, high performance on tabular data | 300 estimators, balanced |

---

## 📈 Evaluation Strategy

Standard accuracy is **not used** due to class imbalance. The following metrics guide model selection:

| Metric | Why It Matters |
|---|---|
| **ROC-AUC** | Overall model discrimination ability |
| **Average Precision (PR-AUC)** | Best metric for imbalanced datasets |
| **F1 Score** | Balance between precision and recall |
| **Confusion Matrix** | False positives (customer friction) vs false negatives (missed fraud) |

---

## 🔍 Explainability with SHAP

SHAP (SHapley Additive exPlanations) makes the model **interpretable for business stakeholders**:

- **Global Importance** — Which features drive fraud predictions overall?
- **Beeswarm Plot** — How does each feature's value impact fraud probability?
- **Per-transaction** — Why was this specific transaction flagged?

Critical in regulated industries where models must be **auditable and explainable** (GDPR, SR 11-7).

---

## 🎯 Threshold Optimization

The default 0.5 threshold is rarely optimal for fraud detection. This project:

1. Sweeps thresholds from 0.1 to 0.9
2. Computes F1 score at each threshold
3. Selects the threshold that **maximizes F1** on the test set
4. Saves the optimal threshold alongside the model for consistent inference

---

## 🚀 Real-time Prediction

A production-ready `predict_fraud()` function is included:

```python
result = predict_fraud({
    'V1': -1.36, 'V2': -0.07, ..., 'V28': 0.02,
    'Amount': 149.62, 'Time': 0
})

# Output:
{
    'is_fraud': True,
    'fraud_probability': 0.9341,
    'risk_level': 'CRITICAL',   # LOW | MEDIUM | HIGH | CRITICAL
    'threshold_used': 0.38
}
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
financial-fraud-detection/
│
├── fraud_detection.py          # Main ML pipeline (14 sections)
├── creditcard.csv              # Dataset (download from Kaggle)
├── fraud_detection_model.pkl   # Saved best model
├── scaler.pkl                  # Saved StandardScaler
├── model_config.pkl            # Optimal threshold + feature list
│
├── eda_analysis.png            # EDA visualizations
├── model_evaluation.png        # ROC & PR curves
├── confusion_matrix.png        # Confusion matrix
├── shap_importance.png         # SHAP feature importance
├── shap_beeswarm.png           # SHAP beeswarm plot
├── threshold_optimization.png  # F1 vs threshold curve
│
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
cd financial-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 4. Run the Pipeline
```bash
python fraud_detection.py
```

---

## 📦 requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
joblib>=1.2.0
```

---

## 🧠 Key Design Decisions

- **Why SMOTE over random oversampling?** SMOTE generates synthetic samples along feature-space boundaries rather than duplicating — producing more generalizable fraud patterns.
- **Why PR-AUC over ROC-AUC?** ROC-AUC can be misleadingly optimistic on imbalanced datasets. PR-AUC better reflects performance when the positive class is rare.
- **Why threshold optimization?** In fraud detection, false negatives (missed fraud) are far more costly than false positives. Tuning the threshold lets the business control this trade-off explicitly.
- **Why SHAP?** Financial institutions operate under regulatory frameworks (GDPR, SR 11-7) that require model decisions to be explainable. SHAP satisfies this requirement.
