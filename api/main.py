# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Churn Prediction API", version="1.0")
model = joblib.load("api/model.pkl")

@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: Dict[str, Any]):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "verdict": "Will churn" if prediction == 1 else "Will not churn"
    }

@app.get("/features")
def features():
    return {"expected_features": list(model.get_booster().feature_names)}