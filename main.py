
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

app = FastAPI(title="Fraud Detection API")

# Train model on startup
def train_model():
    np.random.seed(42)
    n = 50000
    normal = pd.DataFrame({
        "amount":               np.random.exponential(80, n),
        "distance_from_home":   np.random.exponential(20, n),
        "hour":                 np.random.choice(range(8, 23), n),
        "frequency_last_hour":  np.random.poisson(1, n),
        "avg_spend_multiplier": np.random.uniform(0.5, 2.0, n),
        "is_new_merchant":      np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "is_foreign_country":   np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "is_weekend":           np.random.choice([0, 1], n, p=[0.70, 0.30]),
        "card_age_days":        np.random.randint(30, 2000, n),
        "Class": 0
    })
    f = 800
    fraud = pd.DataFrame({
        "amount":               np.random.exponential(400, f),
        "distance_from_home":   np.random.exponential(300, f),
        "hour":                 np.random.choice(range(0, 6), f),
        "frequency_last_hour":  np.random.poisson(5, f),
        "avg_spend_multiplier": np.random.uniform(5.0, 20.0, f),
        "is_new_merchant":      np.random.choice([0, 1], f, p=[0.20, 0.80]),
        "is_foreign_country":   np.random.choice([0, 1], f, p=[0.30, 0.70]),
        "is_weekend":           np.random.choice([0, 1], f, p=[0.50, 0.50]),
        "card_age_days":        np.random.randint(1, 60, f),
        "Class": 1
    })
    df = pd.concat([normal, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_sm, y_train_sm)
    return model, scaler

print("Training model...")
model, scaler = train_model()
print("Model ready!")

class Transaction(BaseModel):
    amount: float
    distance_from_home: float
    hour: int
    frequency_last_hour: int
    avg_spend_multiplier: float
    is_new_merchant: int
    is_foreign_country: int
    is_weekend: int
    card_age_days: int

@app.get("/")
def home():
    return {"message": "Fraud Detection API is live!"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest v1.0"}

@app.post("/predict")
def predict(txn: Transaction):
    features = [[
        txn.amount, txn.distance_from_home,
        txn.hour, txn.frequency_last_hour,
        txn.avg_spend_multiplier, txn.is_new_merchant,
        txn.is_foreign_country, txn.is_weekend,
        txn.card_age_days
    ]]
    scaled = scaler.transform(features)
    prob   = model.predict_proba(scaled)[0][1]
    risk   = round(prob * 100, 2)
    is_trusted = txn.card_age_days > 180

    if risk < 35:
        verdict = "APPROVED"
    elif risk >= 65 and not is_trusted:
        verdict = "BLOCKED"
    else:
        verdict = "OTP_REQUIRED"

    return {
        "risk_score": risk,
        "trusted_user": is_trusted,
        "verdict": verdict
    }
