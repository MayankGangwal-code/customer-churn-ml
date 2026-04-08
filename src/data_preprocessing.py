import pandas as pd # type: ignore

def preprocess_data(data):
    # If input is a file path → read CSV
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        # If already DataFrame → copy
        df = data.copy()

    # Drop customerID only if exists
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert target (only if exists)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df