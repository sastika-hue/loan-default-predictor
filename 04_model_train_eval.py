# ==============================
# ADVANCED LOAN DEFAULT PREDICTION - MODEL EVALUATION & TRAINING
# Production-Ready Pipeline with Maximum Accuracy Focus
# ==============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. LOAD AND PREPARE DATA
# Why: Import and clean dataset for modeling
# ==============================
df = pd.read_excel('loan.xlsx')
print("Initial data shape:", df.shape)

# Remove duplicates
# Why: Ensure data quality and prevent overfitting on repeated samples
df = df.drop_duplicates()

# Identify target column
# Why: Locate prediction variable dynamically
target_col = None
for col in df.columns:
    if col.lower().replace('_', '').replace(' ', '') in ['loanstatus']:
        target_col = col
        break
if target_col is None:
    raise Exception("Target column not found!")

# Encode target
# Why: Convert categorical target to numeric (0/1) for ML algorithms
if df[target_col].dtype == 'object':
    df[target_col] = LabelEncoder().fit_transform(df[target_col].str.strip())

# Drop ID column
# Why: ID provides no predictive power
if 'Applicant_ID' in df.columns:
    df.drop(['Applicant_ID'], axis=1, inplace=True)

# Encode categorical features
# Why: Convert text features to numeric for model compatibility
for col in df.columns:
    if col != target_col and df[col].dtype == 'object':
        vals = df[col].dropna().unique()
        if sorted([str(v).strip() for v in vals]) == ['No', 'Yes']:
            df[col] = df[col].map({'No': 0, 'Yes': 1})
        elif len(vals) <= 10:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df.drop(col, axis=1, inplace=True)

print("Preprocessed data shape:", df.shape)

# ==============================
# 2. FEATURE ENGINEERING
# Why: Create derived features that capture complex patterns
# ==============================
# Example: Create interaction features if Income and Debt columns exist
if 'Annual_Income' in df.columns and 'Outstanding_Debt' in df.columns:
    df['Income_Debt_Ratio'] = df['Annual_Income'] / (df['Outstanding_Debt'] + 1)

if 'Credit_Score' in df.columns and 'Existing_Loans' in df.columns:
    df['Credit_Loan_Product'] = df['Credit_Score'] * df['Existing_Loans']

print("Features after engineering:", df.shape[1])

# ==============================
# 3. TRAIN-TEST SPLIT
# Why: Separate data for training and unbiased evaluation
# ==============================
X = df.drop([target_col], axis=1)
y = df[target_col]

# Scale features
# Why: Normalize feature ranges for stable gradient descent
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train set:", X_train.shape, "| Test set:", X_test.shape)

# ==============================
# 4. BASELINE MODELS
# Why: Establish performance benchmarks
# ==============================
print("\n=== TRAINING BASELINE MODELS ===")

lr = LogisticRegression(max_iter=500, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

baseline_models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'Gradient Boosting': gb
}

baseline_preds = {}
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    baseline_preds[name] = model.predict(X_test)
    print(f"{name} - Accuracy: {accuracy_score(y_test, baseline_preds[name]):.4f}")

# ==============================
# 5. ADVANCED MODELS (XGBoost, LightGBM)
# Why: Leverage gradient boosting for superior performance
# ==============================
print("\n=== TRAINING ADVANCED MODELS ===")

xgb = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.05,
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
lgbm = LGBMClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.05,
    random_state=42, verbose=-1
)

advanced_models = {
    'XGBoost': xgb,
    'LightGBM': lgbm
}

advanced_preds = {}
for name, model in advanced_models.items():
    model.fit(X_train, y_train)
    advanced_preds[name] = model.predict(X_test)
    print(f"{name} - Accuracy: {accuracy_score(y_test, advanced_preds[name]):.4f}")

# ==============================
# 6. HYPERPARAMETER TUNING (OPTIMIZED)
# Why: Optimize model parameters for maximum accuracy
# FIXED: Reduced grid size and n_jobs=1 to prevent memory crashes
# ==============================
print("\n=== HYPERPARAMETER TUNING ===")

# Random Forest tuning (reduced grid)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=3, scoring='accuracy', n_jobs=1  # FIXED: n_jobs=1, cv=3
)
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF score: {rf_grid.best_score_:.4f}")

# XGBoost tuning (reduced grid)
xgb_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    xgb_param_grid, cv=3, scoring='accuracy', n_jobs=1  # FIXED: n_jobs=1, cv=3
)
xgb_grid.fit(X_train, y_train)
print(f"Best XGB params: {xgb_grid.best_params_}")
print(f"Best XGB score: {xgb_grid.best_score_:.4f}")

# ==============================
# 7. ENSEMBLE STACKING
# Why: Combine multiple models for improved robustness
# ==============================
print("\n=== ENSEMBLE STACKING ===")

voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_grid.best_estimator_),
        ('xgb', xgb_grid.best_estimator_),
        ('lgbm', lgbm)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred_ensemble = voting_clf.predict(X_test)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")

# ==============================
# 8. CROSS-VALIDATION
# Why: Validate model stability across multiple data folds
# ==============================
print("\n=== CROSS-VALIDATION RESULTS ===")

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # FIXED: 3 folds
final_models = {
    'Best RF': rf_grid.best_estimator_,
    'Best XGB': xgb_grid.best_estimator_,
    'LightGBM': lgbm,
    'Ensemble': voting_clf
}

for name, model in final_models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv_strategy, scoring='accuracy')
    print(f"{name} - CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ==============================
# 9. COMPREHENSIVE EVALUATION
# Why: Assess all performance dimensions
# ==============================
print("\n=== MODEL EVALUATION ===")

all_preds = {**baseline_preds, **advanced_preds, 'Ensemble': y_pred_ensemble}

for name, y_pred in all_preds.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")

# ==============================
# 10. RESULTS SUMMARY
# Why: Compare all models in a clear table
# ==============================
print("\n=== MODEL COMPARISON SUMMARY ===")

results = pd.DataFrame({
    'Model': list(all_preds.keys()),
    'Accuracy': [accuracy_score(y_test, all_preds[m]) for m in all_preds],
    'Precision': [precision_score(y_test, all_preds[m]) for m in all_preds],
    'Recall': [recall_score(y_test, all_preds[m]) for m in all_preds],
    'F1 Score': [f1_score(y_test, all_preds[m]) for m in all_preds],
    'ROC-AUC': [roc_auc_score(y_test, all_preds[m]) for m in all_preds]
})
results = results.sort_values('Accuracy', ascending=False)
print(results)

# ==============================
# 11. TRACK EXPERIMENTS
# Why: Log all configurations and results for reproducibility
# ==============================
exp_log = {
    'run_time': str(datetime.datetime.now()),
    'data_shape': df.shape,
    'train_test_split': {'train': X_train.shape[0], 'test': X_test.shape[0]},
    'best_rf_params': rf_grid.best_params_,
    'best_xgb_params': xgb_grid.best_params_,
    'results_summary': results.to_dict()
}
pd.DataFrame([exp_log]).to_json('experiment_log.json')
print("\nExperiment log saved to 'experiment_log.json'")

print("\n=== PIPELINE COMPLETE ===")
print(f"Best Model: {results.iloc[0]['Model']}")
print(f"Best Accuracy: {results.iloc[0]['Accuracy']:.4f}")


import joblib

# After you define X (your final feature DataFrame, BEFORE scaling)
feature_names = X.columns.tolist()

# Save feature names
joblib.dump(feature_names, 'feature_names.joblib')
print("✓ Feature names saved to 'feature_names.joblib'")

# Then save your model and scaler as usual
joblib.dump(rf, 'loan_rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("✓ Model and scaler saved")

