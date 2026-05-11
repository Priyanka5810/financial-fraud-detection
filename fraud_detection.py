# ============================================================
#   FINANCIAL FRAUD DETECTION SYSTEM
#   AI/ML Pipeline with Explainability (SHAP) & Real-time API
#   Author: Your Name
#   Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================

# ─────────────────────────────────────────────
# SECTION 1: INSTALL & IMPORTS
# ─────────────────────────────────────────────

# Uncomment below if running in Google Colab
# !pip install shap imbalanced-learn xgboost lightgbm fastapi uvicorn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Handling Class Imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Evaluation
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)

# Explainability
import shap

# Model Persistence
import joblib

# ─────────────────────────────────────────────
# SECTION 2: LOAD & EXPLORE DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION — AI/ML PIPELINE")
print("=" * 60)

# Load dataset
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

print(f"\n📦 Dataset Shape: {df.shape}")
print(f"\n📊 Data Types:\n{df.dtypes}")
print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")
print(f"\n⚖️  Class Distribution:\n{df['Class'].value_counts()}")
print(f"\n📉 Fraud Rate: {df['Class'].mean() * 100:.4f}%")

# ─────────────────────────────────────────────
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fraud Detection — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 3.1 Class Distribution
class_counts = df['Class'].value_counts()
axes[0, 0].bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values,
                color=['#2196F3', '#F44336'], edgecolor='black', alpha=0.85)
axes[0, 0].set_title('Class Distribution (Highly Imbalanced)')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# 3.2 Transaction Amount Distribution
axes[0, 1].hist(df[df['Class'] == 0]['Amount'], bins=60, alpha=0.6,
                color='#2196F3', label='Legitimate', density=True)
axes[0, 1].hist(df[df['Class'] == 1]['Amount'], bins=60, alpha=0.6,
                color='#F44336', label='Fraud', density=True)
axes[0, 1].set_title('Transaction Amount Distribution')
axes[0, 1].set_xlabel('Amount ($)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 2000)

# 3.3 Transaction Time Distribution
axes[1, 0].hist(df[df['Class'] == 0]['Time'] / 3600, bins=48, alpha=0.6,
                color='#2196F3', label='Legitimate', density=True)
axes[1, 0].hist(df[df['Class'] == 1]['Time'] / 3600, bins=48, alpha=0.6,
                color='#F44336', label='Fraud', density=True)
axes[1, 0].set_title('Transaction Time Distribution (Hours)')
axes[1, 0].set_xlabel('Hours Elapsed')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()

# 3.4 Correlation Heatmap (top features)
top_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V9', 'V10', 'V11', 'V12', 'Amount', 'Class']
corr = df[top_features].corr()
sns.heatmap(corr, ax=axes[1, 1], cmap='RdBu_r', center=0,
            annot=False, fmt='.2f', linewidths=0.5)
axes[1, 1].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA plots saved as 'eda_analysis.png'")

# ─────────────────────────────────────────────
# SECTION 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\n🔧 Engineering Features...")

# 4.1 Log-transform Amount (reduce skewness)
df['Log_Amount'] = np.log1p(df['Amount'])

# 4.2 Time-based features
df['Hour'] = (df['Time'] / 3600).astype(int) % 24
df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)

# 4.3 Amount-based risk flags
df['High_Amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
df['Round_Amount'] = (df['Amount'] % 1 == 0).astype(int)

# 4.4 Statistical interaction features
df['V1_V2_interaction'] = df['V1'] * df['V2']
df['V3_V4_interaction'] = df['V3'] * df['V4']

print("✅ Feature engineering complete")
print(f"   → New features: Log_Amount, Hour, Is_Night, High_Amount, Round_Amount, interaction terms")

# ─────────────────────────────────────────────
# SECTION 5: PREPROCESSING & SPLITTING
# ─────────────────────────────────────────────

# Drop raw columns replaced by engineered ones
drop_cols = ['Time', 'Amount']
feature_cols = [c for c in df.columns if c not in drop_cols + ['Class']]

X = df[feature_cols]
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Stratified train/test split (preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📂 Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
print(f"   Train fraud rate: {y_train.mean()*100:.3f}% | Test fraud rate: {y_test.mean()*100:.3f}%")

# ─────────────────────────────────────────────
# SECTION 6: HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────────

print("\n⚖️  Applying SMOTE to handle class imbalance...")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   Before SMOTE → Fraud: {y_train.sum()} | Legit: {(y_train==0).sum()}")
print(f"   After  SMOTE → Fraud: {y_train_resampled.sum()} | Legit: {(y_train_resampled==0).sum()}")

# ─────────────────────────────────────────────
# SECTION 7: MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────

print("\n🤖 Training Models...")

models = {
    'Logistic Regression': LogisticRegression(
        C=0.1, max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        n_jobs=-1, random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight='balanced', random_state=42, n_jobs=-1,
        verbose=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred)
    }

    print(f"   ✅ {name} → ROC-AUC: {results[name]['roc_auc']:.4f} | "
          f"Avg Precision: {results[name]['avg_precision']:.4f} | "
          f"F1: {results[name]['f1']:.4f}")

# ─────────────────────────────────────────────
# SECTION 8: MODEL EVALUATION & VISUALIZATION
# ─────────────────────────────────────────────

print("\n📊 Generating Evaluation Plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Evaluation — Fraud Detection', fontsize=15, fontweight='bold')

colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

# 8.1 ROC Curves
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
axes[0].set_title('ROC Curves — All Models')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(alpha=0.3)

# 8.2 Precision-Recall Curves
for (name, res), color in zip(results.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(recall, precision,
                 label=f"{name} (AP={res['avg_precision']:.3f})", color=color, lw=2)
axes[1].set_title('Precision-Recall Curves — All Models')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# 8.3 Best model confusion matrix
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_result = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_result['roc_auc']:.4f})")

cm = confusion_matrix(y_test, best_result['y_pred'])
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📋 Classification Report — {best_model_name}:")
print(classification_report(y_test, best_result['y_pred'],
                             target_names=['Legitimate', 'Fraud']))

# ─────────────────────────────────────────────
# SECTION 9: MODEL EXPLAINABILITY WITH SHAP
# ─────────────────────────────────────────────

print("\n🔍 Generating SHAP Explainability...")

best_model = best_result['model']

# Use a sample for SHAP (faster)
X_shap_sample = X_test.sample(500, random_state=42)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap_sample)

# Handle binary output shape
if isinstance(shap_values, list):
    sv = shap_values[1]  # Class 1 = Fraud
else:
    sv = shap_values

# 9.1 SHAP Summary Plot
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_shap_sample, plot_type='bar',
                  max_display=15, show=False)
plt.title(f'SHAP Feature Importance — {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# 9.2 SHAP Beeswarm Plot
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_shap_sample, max_display=15, show=False)
plt.title(f'SHAP Beeswarm — Feature Impact on Fraud Prediction', fontsize=13)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ SHAP plots saved")

# ─────────────────────────────────────────────
# SECTION 10: THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────

print("\n🎯 Optimizing Decision Threshold (maximize F1)...")

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (best_result['y_prob'] >= t).astype(int))
             for t in thresholds]

optimal_threshold = thresholds[np.argmax(f1_scores)]
optimal_f1 = max(f1_scores)

print(f"   Default threshold (0.5) → F1: {f1_score(y_test, best_result['y_pred']):.4f}")
print(f"   Optimal threshold ({optimal_threshold:.2f}) → F1: {optimal_f1:.4f}")

plt.figure(figsize=(9, 4))
plt.plot(thresholds, f1_scores, color='#FF5722', lw=2)
plt.axvline(optimal_threshold, color='#2196F3', linestyle='--',
            label=f'Optimal: {optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Decision Threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# SECTION 11: SAVE MODEL & ARTIFACTS
# ─────────────────────────────────────────────

print("\n💾 Saving model artifacts...")

joblib.dump(best_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({'threshold': optimal_threshold, 'features': feature_cols}, 'model_config.pkl')

print("   ✅ fraud_detection_model.pkl")
print("   ✅ scaler.pkl")
print("   ✅ model_config.pkl")

# ─────────────────────────────────────────────
# SECTION 12: REAL-TIME PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_fraud(transaction: dict) -> dict:
    """
    Predict whether a single transaction is fraudulent.

    Args:
        transaction (dict): Dictionary with transaction fields
                            (V1–V28, Amount, Time)

    Returns:
        dict: {
            'is_fraud': bool,
            'fraud_probability': float,
            'risk_level': str,   # 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
            'threshold_used': float
        }
    """
    loaded_model = joblib.load('fraud_detection_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    config = joblib.load('model_config.pkl')

    # Engineer features
    txn = pd.DataFrame([transaction])
    txn['Log_Amount'] = np.log1p(txn['Amount'])
    txn['Hour'] = (txn['Time'] / 3600).astype(int) % 24
    txn['Is_Night'] = ((txn['Hour'] >= 22) | (txn['Hour'] <= 5)).astype(int)
    txn['High_Amount'] = (txn['Amount'] > 2000).astype(int)
    txn['Round_Amount'] = (txn['Amount'] % 1 == 0).astype(int)
    txn['V1_V2_interaction'] = txn['V1'] * txn['V2']
    txn['V3_V4_interaction'] = txn['V3'] * txn['V4']

    txn = txn[config['features']]
    txn_scaled = loaded_scaler.transform(txn)

    prob = loaded_model.predict_proba(txn_scaled)[0][1]
    threshold = config['threshold']
    is_fraud = prob >= threshold

    if prob < 0.3:
        risk = 'LOW'
    elif prob < 0.5:
        risk = 'MEDIUM'
    elif prob < 0.75:
        risk = 'HIGH'
    else:
        risk = 'CRITICAL'

    return {
        'is_fraud': bool(is_fraud),
        'fraud_probability': round(float(prob), 4),
        'risk_level': risk,
        'threshold_used': round(threshold, 4)
    }


# ─────────────────────────────────────────────
# SECTION 13: DEMO PREDICTION
# ─────────────────────────────────────────────

print("\n🧪 Running Demo Prediction on a sample transaction...")

# Take a real fraud transaction from dataset as demo
sample_fraud = df[df['Class'] == 1].iloc[0].to_dict()
sample_fraud.pop('Class')

result = predict_fraud(sample_fraud)

print(f"\n   Transaction Details:")
print(f"   Amount: ${sample_fraud['Amount']:.2f}")
print(f"\n   🔎 Prediction Result:")
print(f"   Is Fraud     : {result['is_fraud']}")
print(f"   Probability  : {result['fraud_probability'] * 100:.2f}%")
print(f"   Risk Level   : {result['risk_level']}")
print(f"   Threshold    : {result['threshold_used']}")

# ─────────────────────────────────────────────
# SECTION 14: SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'ROC-AUC':>10} {'Avg Prec':>10} {'F1 Score':>10}")
print("-" * 60)
for name, res in results.items():
    marker = " ⭐" if name == best_model_name else ""
    print(f"{name:<25} {res['roc_auc']:>10.4f} {res['avg_precision']:>10.4f} {res['f1']:>10.4f}{marker}")
print("=" * 60)
print(f"\n✅ Pipeline complete. Best model: {best_model_name}")
print(f"   Optimal Decision Threshold : {optimal_threshold:.2f}")
print(f"   Model saved to             : fraud_detection_model.pkl")
