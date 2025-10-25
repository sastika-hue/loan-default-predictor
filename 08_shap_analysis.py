import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pandas.api.types import CategoricalDtype
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 1️⃣ LOAD AND PREPROCESS DATA
# ======================================================
df = pd.read_excel("loan.xlsx").drop_duplicates()

# Identify target column
target_col = None
for col in df.columns:
    if col.lower().replace("_", "").replace(" ", "") == "loanstatus":
        target_col = col
        break
if target_col is None:
    raise ValueError("Target column 'Loan_Status' not found!")

# Encode target if categorical
if df[target_col].dtype == "object":
    df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str).str.strip())

# Drop non-informative columns
for drop_col in ["Applicant_ID", "ID", "Serial", "S.No"]:
    if drop_col in df.columns:
        df.drop(columns=[drop_col], inplace=True)

# Encode categorical features
for col in df.columns:
    if col == target_col:
        continue
    dtype = df[col].dtype
    if dtype == "object" or isinstance(dtype, CategoricalDtype):
        vals = df[col].dropna().unique()
        if sorted([str(v).strip() for v in vals]) == ["No", "Yes"]:
            df[col] = df[col].map({"No": 0, "Yes": 1})
        elif len(vals) <= 10:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df.drop(columns=[col], inplace=True)

# Derived features
if {"Annual_Income", "Outstanding_Debt"} <= set(df.columns):
    df["Income_Debt_Ratio"] = df["Annual_Income"] / (df["Outstanding_Debt"] + 1)
if {"Credit_Score", "Existing_Loans"} <= set(df.columns):
    df["Credit_Loan_Product"] = df["Credit_Score"] * df["Existing_Loans"]

# Split data
X = df.drop(columns=[target_col])
y = df[target_col]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# 2️⃣ DEFINE HELPER FOR SAFE SHAP PLOTS
# ======================================================
def safe_summary_plot(shap_values, X, model_name):
    """Safely handle binary/multiclass and shape mismatches."""
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(shap_values_to_plot, X, show=False)
    plt.title(f"{model_name} - SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values_to_plot, X, plot_type="bar", show=False)
    plt.title(f"{model_name} - SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

# ======================================================
# 3️⃣ RANDOM FOREST
# ======================================================
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

rf_explainer = shap.TreeExplainer(rf)
rf_shap_values = rf_explainer.shap_values(X_test)
safe_summary_plot(rf_shap_values, X_test, "Random Forest")

# ======================================================
# 4️⃣ XGBOOST
# ======================================================
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")

xgb_explainer = shap.TreeExplainer(xgb)
xgb_shap_values = xgb_explainer.shap_values(X_test)
safe_summary_plot(xgb_shap_values, X_test, "XGBoost")

# ======================================================
# 5️⃣ LOGISTIC REGRESSION
# ======================================================
logreg = LogisticRegression(max_iter=2000, random_state=42)
logreg.fit(X_train, y_train)
log_pred = logreg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.4f}")

log_explainer = shap.Explainer(logreg, X_train)
log_shap_values = log_explainer(X_test)
shap.summary_plot(log_shap_values, X_test, show=False)
plt.title("Logistic Regression - SHAP Summary Plot")
plt.tight_layout()
plt.show()

# ======================================================
# 6️⃣ SAVE ARTIFACTS
# ======================================================
joblib.dump(rf, "loan_rf_model.joblib")
joblib.dump(xgb, "loan_xgb_model.joblib")
joblib.dump(logreg, "loan_logreg_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\n✅ Analysis Complete.\nModels & plots saved successfully.\n")
