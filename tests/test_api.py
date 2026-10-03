from fastapi.testclient import TestClient

from api import create_app


def test_api_returns_clear_error_without_trained_model(monkeypatch):
    import api

    def missing_model(_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(api, "load_model_bundle", missing_model)
    response = TestClient(create_app()).post("/predict", json={"financial_data": {"Revenue": 1000}})
    assert response.status_code == 503
    assert "Train a model" in response.json()["detail"]


def test_api_validates_request_shape():
    response = TestClient(create_app()).post("/predict", json={"financial_data": "not an object"})
    assert response.status_code == 422


def test_api_returns_prediction_payload(monkeypatch):
    import api

    expected = {
        "distress_probability": 0.72,
        "risk_category": "High Risk",
        "confidence_indicator": 0.44,
        "top_risk_factors": [],
        "protective_factors": [],
    }
    monkeypatch.setattr(api, "load_model_bundle", lambda _path: {})
    monkeypatch.setattr(api, "predict_financial_health", lambda _data, bundle: expected)
    response = TestClient(create_app()).post("/predict", json={"financial_data": {"Revenue": 1000}})
    assert response.status_code == 200
    assert response.json() == expected