# Credit Card Fraud Detection API

A production-deployed machine learning API that detects credit card fraud in real-time using Random Forest classification, smart trust logic, and multi-layer security.

**Live API:** https://fraud-detection-api-production-99bf.up.railway.app  
**Live Dashboard:** https://fraud-detection-frontend-xi.vercel.app

---

## What It Does

This system goes beyond simple fraud flagging. It thinks — distinguishing between a real user in an emergency and an actual fraudster making the same transaction at 3am, 100km from home.

- Real user with trusted card? OTP sent. Transaction verified.
- Actual fraudster? Blocked instantly.

---

## Features

- Real-time risk scoring (0–100%) using machine learning
- Random Forest model trained on 50,800 transactions
- SMOTE oversampling for class imbalance handling
- Smart trust detection based on card history
- Live OTP via SMS and Phone Call using Twilio Verify
- Multi-Factor Authentication simulation
- RESTful API with auto-generated Swagger docs
- Fully Dockerized and deployed on Railway

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Framework | FastAPI |
| ML Model | Random Forest — Scikit-learn |
| Class Balancing | SMOTE (imbalanced-learn) |
| OTP Service | Twilio Verify API |
| Containerization | Docker |
| Deployment | Railway |

---

## API Endpoints

### GET /
```json
{ "message": "Fraud Detection API is live!" }
```

### GET /health
```json
{ "status": "ok", "model": "RandomForest v1.0" }
```

### POST /predict

**Request:**
```json
{
  "amount": 70000,
  "distance_from_home": 100,
  "hour": 2,
  "frequency_last_hour": 5,
  "avg_spend_multiplier": 4.0,
  "is_new_merchant": 1,
  "is_foreign_country": 0,
  "is_weekend": 0,
  "card_age_days": 365
}
```

**Response:**
```json
{
  "risk_score": 93.0,
  "trusted_user": true,
  "verdict": "OTP_REQUIRED"
}
```

**Verdict values:**

| Verdict | Meaning |
|---------|---------|
| `APPROVED` | Low risk — transaction cleared |
| `OTP_REQUIRED` | Suspicious — OTP sent to user |
| `BLOCKED` | High risk — transaction blocked |

---

## How It Works
```
Transaction Input
      ↓
ML Risk Scoring (Random Forest)
      ↓
Trust Level Check (card age)
      ↓
APPROVED  /  OTP via SMS or Call  /  BLOCKED
```

---

## Model Details

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Training Samples | 50,800 |
| Fraud Cases | 800 (1.57%) |
| Class Balancing | SMOTE |
| Accuracy | 100% |
| ROC-AUC Score | 1.0 |

---

## Local Setup
```bash
git clone https://github.com/Zaira-Shahid/fraud-detection-api.git
cd fraud-detection-api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI.

---

## Docker
```
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api

## Author
  Zaira Shahid
- LinkedIn: [linkedin.com/in/zaira-shahid-](https://linkedin.com/in/zaira-shahid-)
- GitHub: [github.com/Zaira-Shahid](https://github.com/Zaira-Shahid)
```
