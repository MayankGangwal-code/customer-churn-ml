import joblib
import pandas as pd

# Load trained model
model = joblib.load("../model/churn_model.pkl")

def predict_churn(input_data):
    
    data = pd.DataFrame([input_data])
    
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        return "Customer will churn"
    else:
        return "Customer will stay"