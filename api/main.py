# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0")

# Load model once at startup
model = joblib.load("api/model.pkl")

# ── Input schema (19 features after dropping customerID) ──────
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# ── Health check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API is running"}

# ── Prediction endpoint ───────────────────────────────────────
@app.post("/predict")
def predict(data: CustomerData):
    features = np.array([[
        data.gender, data.SeniorCitizen, data.Partner, data.Dependents,
        data.tenure, data.PhoneService, data.MultipleLines,
        data.InternetService, data.OnlineSecurity, data.OnlineBackup,
        data.DeviceProtection, data.TechSupport, data.StreamingTV,
        data.StreamingMovies, data.Contract, data.PaperlessBilling,
        data.PaymentMethod, data.MonthlyCharges, data.TotalCharges
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "verdict": "Will churn" if prediction == 1 else "Will not churn"
    }