import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("../model/model.pkl", "rb"))

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict"):
    input_data = [[tenure, monthly_charges, total_charges]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will not churn ✅")