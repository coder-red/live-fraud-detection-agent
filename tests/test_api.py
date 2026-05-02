import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


SAMPLE_TRANSACTION = {
    "trans_date_trans_time": "2026-02-19 03:00:00",
    "amt": 9099.99,
    "category": "shopping_net",
    "merchant": "fraud_store_xyz",
    "lat": 40.71,
    "long": -74.0,
    "merch_lat": 40.75,
    "merch_long": -73.98,
    "city": "Lagos",
    "state": "LG",
    "city_pop": 8000000,
    "dob": "1990-01-01",
    "gender": "M",
    "job": "Engineer",
}


class StubInference:
    def __init__(self, probability: float):
        self.probability = probability

    def predict(self, raw_data: dict) -> dict:
        return {
            "is_fraud": self.probability >= 0.5221,
            "probability": self.probability,
            "threshold": 0.5221,
            "features": [],
        }


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    for module_name in [
        "app.main",
        "app.routes",
        "app.db.models",
        "app.db.connections",
    ]:
        sys.modules.pop(module_name, None)

    connections = importlib.import_module("app.db.connections")
    importlib.import_module("app.db.models")
    connections.Base.metadata.create_all(bind=connections.engine)

    routes = importlib.import_module("app.routes")
    main = importlib.import_module("app.main")

    def override_get_db():
        db = connections.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[routes.get_db] = override_get_db

    with TestClient(main.app) as client:
        yield client, routes

    main.app.dependency_overrides.clear()


def test_predict_low_risk_persists_prediction_without_case(api_client):
    client, routes = api_client
    routes.engine = StubInference(probability=0.12)

    response = client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_band"] == "LOW"
    assert payload["decision"] == "APPROVE"
    assert payload["requires_review"] is False
    assert payload["case_id"] is None

    predictions = client.get("/api/v1/predictions").json()
    assert len(predictions) == 1
    assert predictions[0]["id"] == payload["id"]


def test_predict_high_risk_creates_case_and_accepts_human_decision(api_client):
    client, routes = api_client
    routes.engine = StubInference(probability=0.93)
    routes.generate_agent_review = lambda transaction, probability, policy: SimpleNamespace(
        recommendation="BLOCK",
        confidence=0.93,
        reason_codes=["HIGH_AMOUNT", "HIGH_RISK_CATEGORY"],
        reviewer_questions=["Can the customer confirm this transaction?"],
        summary="Critical risk transaction; policy recommends BLOCK.",
    )

    response = client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_band"] == "CRITICAL"
    assert payload["decision"] == "BLOCK"
    assert payload["requires_review"] is True
    assert payload["case_id"]
    assert payload["agent_recommendation"] == "BLOCK"

    pending = client.get("/api/v1/cases/pending").json()
    assert len(pending) == 1
    assert pending[0]["case_id"] == payload["case_id"]
    assert pending[0]["status"] == "PENDING_REVIEW"

    decision = client.post(
        f"/api/v1/cases/{payload['case_id']}/decision",
        json={"decision": "BLOCK", "note": "Customer could not verify the transaction."},
    )

    assert decision.status_code == 200
    reviewed_case = decision.json()
    assert reviewed_case["status"] == "BLOCKED"
    assert reviewed_case["human_decision"] == "BLOCK"
    assert reviewed_case["human_note"] == "Customer could not verify the transaction."
    assert reviewed_case["reviewed_at"] is not None

    duplicate = client.post(
        f"/api/v1/cases/{payload['case_id']}/decision",
        json={"decision": "APPROVE"},
    )
    assert duplicate.status_code == 409
