import joblib
rf_model = joblib.load('loan_default_model.joblib')  # Your saved optimized model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=3, scoring='accuracy', n_jobs=1
)
rf_grid.fit(X_train, y_train)
