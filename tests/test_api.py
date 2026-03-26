# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_features_endpoint():
    response = client.get("/features")
    assert response.status_code == 200
    assert "expected_features" in response.json()

def test_prediction_returns_valid_response():
    # Get expected features dynamically
    features_resp = client.get("/features")
    feature_names = features_resp.json()["expected_features"]
    # Send all zeros as dummy input
    payload = {f: 0 for f in feature_names}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0