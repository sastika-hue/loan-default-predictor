import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Tools
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)

# Load dataset
df = pd.read_excel(r"C:\Users\sasti\OneDrive\Desktop\fdsnew\loan.xlsx", engine='openpyxl')

# Encode target
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])  # 1=Default, 0=Not Default

# Visualize numeric feature distribution
for col in df.select_dtypes(include=np.number).columns:
    if col != 'Loan_Status':
        plt.figure(figsize=(5,3))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

# Visualize correlations (excluding ID)
plt.figure(figsize=(15,8))
sns.heatmap(df.drop(columns=['Applicant_ID'], errors='ignore').corr(numeric_only=True),
            annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap (Excluding Applicant_ID)")
plt.show()

# Feature-target relationships
for col in df.select_dtypes(include=np.number).columns:
    if col != 'Loan_Status':
        plt.figure(figsize=(6,3))
        sns.boxplot(x='Loan_Status', y=col, data=df)
        plt.title(f'{col} vs Loan_Status')
        plt.show()

# Prepare features (exclude non-predictive unique ID)
X = df.drop(['Loan_Status', 'Applicant_ID'], axis=1, errors='ignore')
y = df['Loan_Status']

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Build preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models (with preprocessing pipeline)
models = {
    "Logistic Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42))
    ])
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

# Convert results to dataframe
results_df = pd.DataFrame(results)
print("\n=== Model Performance Comparison ===")
print(results_df)

# Confusion matrices
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Log experiment
import datetime
exp_log = {
    'run_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'models_evaluated': list(models.keys()),
    'scores_summary': results_df.to_dict()
}
pd.DataFrame([exp_log]).to_json('experiment_log.json', orient='records', indent=4)
