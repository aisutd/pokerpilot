from fastapi.testclient import TestClient

from pokerpilot_model import api


def test_api_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("POKERPILOT_MODEL_PATH", str(tmp_path / "missing.joblib"))
    api.artifact.cache_clear()
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "model_missing"}


def test_prediction_requires_model(monkeypatch, tmp_path):
    monkeypatch.setenv("POKERPILOT_MODEL_PATH", str(tmp_path / "missing.joblib"))
    api.artifact.cache_clear()
    response = TestClient(api.app).post("/predict", json={"features": {}})
    assert response.status_code == 503
