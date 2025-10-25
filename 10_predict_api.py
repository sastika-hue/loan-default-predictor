# ==============================
# PREDICTION API - LOAN DEFAULT PREDICTOR
# Load trained model and make predictions on new data
# ==============================

import joblib
import pandas as pd
import numpy as np

# ==============================
# 1. LOAD TRAINED MODEL, SCALER & FEATURE NAMES
# ==============================
try:
    rf_model = joblib.load('loan_rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    expected_features = joblib.load('feature_names.joblib')
    print("✓ Model, scaler, and feature names loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("\nMake sure these files exist:")
    print("  - loan_rf_model.joblib")
    print("  - scaler.joblib")
    print("  - feature_names.joblib")
    exit()

print(f"✓ Model expects {len(expected_features)} features")

# ==============================
# 2. CREATE FEATURE TEMPLATE
# ==============================
feature_template = pd.DataFrame(0, index=[0], columns=expected_features)

# ==============================
# 3. PREDICTION FUNCTION
# ==============================
def predict_loan_default(input_dict):
    """
    Predict loan default probability for a new applicant.
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary with feature names as keys and applicant values.
        Missing features will default to 0.
    
    Returns:
    --------
    dict : Prediction results with class and probability
    """
    df_input = feature_template.copy()
    
    for key, value in input_dict.items():
        if key in df_input.columns:
            df_input[key] = value
        else:
            print(f"⚠ Warning: '{key}' not in model features, ignoring.")
    
    df_scaled = scaler.transform(df_input)
    prediction = rf_model.predict(df_scaled)[0]
    probability = rf_model.predict_proba(df_scaled)[0]
    
    result = {
        'prediction': 'Default' if prediction == 1 else 'No Default',
        'prediction_code': int(prediction),
        'default_probability': float(probability[1]),
        'no_default_probability': float(probability[0])
    }
    
    return result

# ==============================
# 4. DISPLAY RESULTS FUNCTION
# ==============================
def display_prediction(result):
    """Display prediction results in a readable format."""
    print("\n" + "="*50)
    print("LOAN DEFAULT PREDICTION RESULTS")
    print("="*50)
    print(f"Prediction:          {result['prediction']}")
    print(f"Default Risk:        {result['default_probability']:.2%}")
    print(f"No Default Risk:     {result['no_default_probability']:.2%}")
    print("="*50 + "\n")

# ==============================
# 5. BATCH PREDICTION FROM CSV
# ==============================
def batch_predict_from_csv(csv_path):
    """Predict on multiple applicants from a CSV file."""
    df = pd.read_csv(csv_path)
    results = []
    
    for idx, row in df.iterrows():
        input_dict = row.to_dict()
        result = predict_loan_default(input_dict)
        results.append({
            'Applicant_ID': idx + 1,
            'Prediction': result['prediction'],
            'Default_Probability': result['default_probability']
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('batch_predictions.csv', index=False)
    print(f"✓ Batch predictions saved to 'batch_predictions.csv'")
    return results_df

# ==============================
# 6. EXAMPLE USAGE
# ==============================
if __name__ == "__main__":
    print("\n--- EXAMPLE: Single Prediction ---")
    sample_applicant = {
        'Age': 35,
        'Annual_Income': 60000,
        'Credit_Score': 720,
        'Outstanding_Debt': 15000,
        'Loan_Amount': 200000,
        'Down_Payment': 40000,
    }
    
    result = predict_loan_default(sample_applicant)
    display_prediction(result)
    
    print("\n--- MODEL FEATURE LIST ---")
    print("The model expects these features:")
    for i, feat in enumerate(expected_features, 1):
        print(f"{i}. {feat}")
