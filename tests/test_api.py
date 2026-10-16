import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_mode"] in {"heuristic", "trained"}
    assert "adyen" in body["routes"]


def test_route_endpoint(client: TestClient):
    payload = json.loads(
        Path("examples/request.json").read_text(encoding="utf-8")
    )
    response = client.post("/v1/route", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["recommended"]
    assert body["ranked"]
    assert body["disclaimer"]


def test_compare_endpoint(client: TestClient):
    response = client.post(
        "/v1/compare",
        json={"transaction_type": "deposit", "amount": 250, "currency": "GBP", "country": "United Kingdom"},
    )
    assert response.status_code == 200
    assert "balanced" in response.json()["policies"]
