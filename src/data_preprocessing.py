import pandas as pd
def preprocess_data(path):
    df = pd.read_csv(path)
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    df = pd.get_dummies(df, drop_first=True)
    return df