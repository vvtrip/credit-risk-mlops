from fastapi.testclient import TestClient

from credit_risk.api import app


def test_root_endpoint():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
    assert response.json()["app"] == "/app"


def test_dashboard_endpoint():
    response = TestClient(app).get("/app")
    assert response.status_code == 200
    assert "Credit Risk Studio" in response.text


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_application_is_rejected():
    response = TestClient(app).post("/predict", json={"duration_months": -1})
    assert response.status_code == 422
