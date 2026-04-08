import streamlit as st
import pandas as pd
import joblib

# Import your preprocessing function
from data_preprocessing import preprocess_data

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("model/churn_model.pkl")

# -------------------------------
# UI
# -------------------------------
st.title("Customer Churn Prediction")

# Basic inputs
tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

# Optional advanced inputs (recommended)
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    # Create raw input (same format as dataset)
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    try:
        # Apply SAME preprocessing as training
        processed_df = preprocess_data(input_df)

        # Remove target column if exists
        if "Churn" in processed_df.columns:
            processed_df = processed_df.drop("Churn", axis=1)

        # Predict
        prediction = model.predict(processed_df)

        # Output
        if prediction[0] == 1:
            st.error("Customer will churn ❌")
        else:
            st.success("Customer will not churn ✅")

    except Exception as e:
        st.error(f"Error: {e}")