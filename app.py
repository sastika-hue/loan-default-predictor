import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# LOAD MODEL & RESOURCES
# ==============================
@st.cache_resource
def load_model():
    """Load trained model, scaler, and feature names"""
    try:
        model = joblib.load('loan_rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        features = joblib.load('feature_names.joblib')
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

rf_model, scaler, feature_names = load_model()

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_default(input_data):
    """Make prediction on input data"""
    # Create template with all features
    template = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Update with provided values
    for key, value in input_data.items():
        if key in template.columns:
            template[key] = value
    
    # Scale and predict
    scaled_data = scaler.transform(template)
    prediction = rf_model.predict(scaled_data)[0]
    probability = rf_model.predict_proba(scaled_data)[0]
    
    return {
        'prediction': 'Default' if prediction == 1 else 'No Default',
        'default_probability': float(probability[1]),
        'no_default_probability': float(probability[0])
    }

# ==============================
# APP HEADER
# ==============================
st.title("💰 Loan Default Risk Predictor")
st.markdown("### AI-Powered Risk Assessment System")
st.markdown("---")

# ==============================
# SIDEBAR - INPUT FORM
# ==============================
st.sidebar.header("📋 Applicant Information")

with st.sidebar:
    st.subheader("Personal Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    marital_status = st.selectbox("Marital Status", [0, 1, 2], format_func=lambda x: ["Single", "Married", "Divorced"][x])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education", [0, 1, 2], format_func=lambda x: ["High School", "Bachelor", "Master+"][x])
    
    st.subheader("Financial Details")
    annual_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
    outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0, value=15000, step=1000)
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
    
    st.subheader("Loan Details")
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=200000, step=1000)
    loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=240)
    down_payment = st.number_input("Down Payment ($)", min_value=0, value=40000, step=1000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=4.5, step=0.1)
    
    predict_button = st.sidebar.button("🔮 Predict Risk", type="primary", use_container_width=True)

# ==============================
# MAIN CONTENT
# ==============================

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Prediction Results")
    
    if predict_button:
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Marital_Status': marital_status,
            'Dependents': dependents,
            'Education': education,
            'Annual_Income': annual_income,
            'Credit_Score': credit_score,
            'Outstanding_Debt': outstanding_debt,
            'Existing_Loans': existing_loans,
            'Loan_Amount': loan_amount,
            'Loan_Term_Months': loan_term,
            'Down_Payment': down_payment,
            'Interest_Rate': interest_rate,
            'Debt_to_Income': outstanding_debt / annual_income if annual_income > 0 else 0,
            'Loan_to_Value': loan_amount / (loan_amount + down_payment) if (loan_amount + down_payment) > 0 else 0,
        }
        
        # Make prediction
        with st.spinner('Analyzing risk...'):
            result = predict_default(input_data)
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Risk indicator
        risk_level = result['default_probability']
        if risk_level < 0.3:
            risk_label = "🟢 Low Risk"
            risk_color = "green"
        elif risk_level < 0.6:
            risk_label = "🟡 Medium Risk"
            risk_color = "orange"
        else:
            risk_label = "🔴 High Risk"
            risk_color = "red"
        
        # Display prediction
        st.markdown(f"### Prediction: **{result['prediction']}**")
        st.markdown(f"### Risk Level: **{risk_label}**")
        
        # Progress bars
        st.markdown("#### Probability Breakdown")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Default Probability", f"{result['default_probability']:.1%}")
            st.progress(result['default_probability'])
        
        with col_b:
            st.metric("No Default Probability", f"{result['no_default_probability']:.1%}")
            st.progress(result['no_default_probability'])
        
        # Financial ratios
        st.markdown("---")
        st.markdown("#### Key Financial Ratios")
        ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
        
        with ratio_col1:
            dti = (outstanding_debt / annual_income * 100) if annual_income > 0 else 0
            st.metric("Debt-to-Income", f"{dti:.1f}%")
        
        with ratio_col2:
            ltv = (loan_amount / (loan_amount + down_payment) * 100) if (loan_amount + down_payment) > 0 else 0
            st.metric("Loan-to-Value", f"{ltv:.1f}%")
        
        with ratio_col3:
            dp_percent = (down_payment / loan_amount * 100) if loan_amount > 0 else 0
            st.metric("Down Payment %", f"{dp_percent:.1f}%")
    
    else:
        st.info("👈 Fill in applicant details and click **Predict Risk** to get started!")

with col2:
    st.subheader("ℹ️ About")
    st.markdown("""
    This AI-powered system predicts loan default risk using:
    
    - **Random Forest Classifier**
    - **40 financial & demographic features**
    - **SHAP explainability**
    
    **Risk Categories:**
    - 🟢 **Low Risk**: < 30% default probability
    - 🟡 **Medium Risk**: 30-60% default probability
    - 🔴 **High Risk**: > 60% default probability
    
    **Model Performance:**
    - Accuracy: ~90%
    - Trained on historical loan data
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Model Insights")
    st.markdown("""
    **Top Risk Factors:**
    1. Loan Amount
    2. Down Payment
    3. Debt-to-Income Ratio
    4. Credit Score
    5. Outstanding Debt
    """)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("**Developed by:** Sasti Muthu | **Model Version:** 1.0 | **Last Updated:** Oct 2025")
