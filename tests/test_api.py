# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_returns_valid_response():
    payload = {
        "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
        "tenure": 12, "PhoneService": 1, "MultipleLines": 0,
        "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 1,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 1,
        "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 2, "MonthlyCharges": 65.5, "TotalCharges": 786.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
    assert "verdict" in response.json()

def test_prediction_probability_range():
    payload = {
        "gender": 0, "SeniorCitizen": 1, "Partner": 0, "Dependents": 0,
        "tenure": 1, "PhoneService": 1, "MultipleLines": 1,
        "InternetService": 2, "OnlineSecurity": 0, "OnlineBackup": 0,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
        "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 3, "MonthlyCharges": 95.0, "TotalCharges": 95.0
    }
    response = client.post("/predict", json=payload)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0