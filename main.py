
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

model  = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/fraud_scaler.pkl")

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
