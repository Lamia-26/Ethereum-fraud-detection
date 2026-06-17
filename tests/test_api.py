"""Tests de l'API."""

from fastapi.testclient import TestClient

from ethereum_fraud.api import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint_exists():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "version" in response.json()
