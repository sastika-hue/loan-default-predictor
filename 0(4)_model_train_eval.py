import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

# 1. Load and preprocess data
df = pd.read_excel('loan.xlsx')
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])  # 1=Default, 0=Not Default

# 2. EDA (plot numeric distributions and correlations)
num_cols = df.select_dtypes(include=np.number).columns.drop('Loan_Status')
for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Only use numeric columns for correlation (exclude Applicant_ID if needed)
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x='Loan_Status', y=col, data=df)
    plt.title(f'{col} vs Loan_Status')
    plt.show()

# 3. Prepare for modeling
X = df.drop(['Loan_Status','Applicant_ID'], axis=1, errors='ignore')
y = df['Loan_Status']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Baseline and advanced model training
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 5. Model documentation/architectures
print(lr)
print(rf)
print(xgb)

# 6. Hyperparameter tuning (RandomForest example)
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_rf.fit(X_train, y_train)
print("Best RF Parameters:", grid_rf.best_params_)

# 7. Cross-validation scores
for model, name in zip([lr, rf, xgb], ['Logistic', 'RandomForest', 'XGBoost']):
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f'{name} mean CV accuracy:', np.mean(scores))

# 8. Metrics and result comparison
def print_metrics(y_true, y_pred, model_name):
    print(f"=== {model_name} Evaluation ===")
    print(classification_report(y_true, y_pred))
    print('ROC-AUC:', roc_auc_score(y_true, y_pred))

print_metrics(y_test, y_pred_lr, "Logistic Regression")
print_metrics(y_test, y_pred_rf, "Random Forest")
print_metrics(y_test, y_pred_xgb, "XGBoost")

# 9. Confusion matrices
for name, pred in zip(['Logistic', 'RandomForest', 'XGBoost'], [y_pred_lr, y_pred_rf, y_pred_xgb]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 10. Summarize results
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_xgb)],
    'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_xgb)],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_xgb)],
    'F1 Score': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_xgb)]
})
print(results)

# 11. Track experiments/code
import datetime
exp_log = {
    'run_time': datetime.datetime.now(),
    'model_params': {
        'LogisticRegression': lr.get_params(),
        'RandomForest': rf.get_params(),
        'XGB': xgb.get_params()
    },
    'results_summary': results.to_dict()
}
pd.DataFrame([exp_log]).to_json('experiment_log.json')
