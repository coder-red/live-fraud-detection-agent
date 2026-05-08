import importlib  # This lets the test reload app modules after changing environment variables.
import sys  # This lets the test remove already loaded modules from Python memory.
from types import SimpleNamespace  # This creates a small fake object for the agent review.

import httpx  # This lets async tests call the FastAPI app directly through ASGI.
import pytest  # This gives us pytest fixtures and test helpers.


pytestmark = pytest.mark.anyio  # This lets every test in this file use async/await.


SAMPLE_TRANSACTION = {  # 🟢 NORMAL CASE DATA: standard valid transaction used across tests
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


class StubInference:  # 🧪 TEST DOUBLE (MOCK): replaces real ML model
    def __init__(self, probability: float):
        self.probability = probability

    def predict(self, raw_data: dict) -> dict:
        return {
            "is_fraud": self.probability >= 0.5221,
            "probability": self.probability,
            "threshold": 0.5221,
            "features": [],
        }

# async is used for allowing a task to run in the background while waiting for a response, improving performance
@pytest.fixture()
def anyio_backend():  # 🧪 TEST SETUP
    return "asyncio"


@pytest.fixture()
async def api_client(tmp_path, monkeypatch):  # 🧪 TEST INFRASTRUCTURE (shared setup)

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

    state = {"probability": 0.12}

    def override_get_db():
        db = connections.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def override_inference():
        return StubInference(probability=state["probability"])

    main.app.dependency_overrides[routes.get_db] = override_get_db
    main.app.dependency_overrides[routes.get_inference_engine] = override_inference

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, routes, state

    main.app.dependency_overrides.clear()


# =========================
# 🟢 NORMAL CASE TESTS
# =========================

async def test_predict_low_risk_persists_prediction_without_case(api_client):  # 🟢 NORMAL CASE
    client, routes, state = api_client
    state["probability"] = 0.12

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_band"] == "LOW"
    assert payload["decision"] == "APPROVE"
    assert payload["requires_review"] is False
    assert payload["case_id"] is None

    predictions = (await client.get("/api/v1/predictions")).json()
    assert len(predictions) == 1
    assert predictions[0]["id"] == payload["id"]


async def test_predict_same_payload_is_idempotent(api_client):  # 🟢 DEDUPE
    client, routes, state = api_client
    state["probability"] = 0.12

    first = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()
    second = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()

    assert first["id"] == second["id"]
    predictions = (await client.get("/api/v1/predictions")).json()
    assert len(predictions) == 1


async def test_get_prediction_returns_saved_prediction(api_client):  # 🟢 NORMAL CASE
    client, routes, state = api_client
    state["probability"] = 0.12

    created = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()

    response = await client.get(f"/api/v1/predictions/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


async def test_predict_high_risk_creates_case_and_accepts_human_decision(api_client):  # 🟢 NORMAL CASE (high-risk workflow)
    client, routes, state = api_client
    state["probability"] = 0.93

    routes.generate_agent_review = lambda transaction, probability, policy, **kwargs: SimpleNamespace(
        recommendation="BLOCK",
        confidence=0.93,
        reason_codes=["HIGH_AMOUNT", "HIGH_RISK_CATEGORY"],
        reviewer_questions=["Can the customer confirm this transaction?"],
        summary="Critical risk transaction; policy recommends BLOCK.",
    )

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_band"] == "CRITICAL"
    assert payload["decision"] == "BLOCK"
    assert payload["requires_review"] is True
    assert payload["case_id"]
    # Map the aliases to the actual keys we want to test
    assert payload["agent_action"] == "BLOCK"

    pending = (await client.get("/api/v1/cases/pending")).json()
    assert len(pending) == 1

    decision = await client.post(
        f"/api/v1/cases/{payload['case_id']}/decision",
        json={"decision": "BLOCK", "note": "Customer could not verify the transaction."},
    )

    assert decision.status_code == 200
    reviewed_case = decision.json()
    assert reviewed_case["status"] == "BLOCKED"

    duplicate = await client.post(
        f"/api/v1/cases/{payload['case_id']}/decision",
        json={"decision": "APPROVE"},
    )

    assert duplicate.status_code == 409


async def test_get_case_returns_pending_case(api_client):  # 🟢 NORMAL CASE
    client, routes, state = api_client
    state["probability"] = 0.93

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)
    case_id = response.json()["case_id"]

    case_response = await client.get(f"/api/v1/cases/{case_id}")

    assert case_response.status_code == 200
    assert case_response.json()["case_id"] == case_id


# =========================
# 🟡 EDGE CASE TESTS
# =========================

async def test_predictions_limit_accepts_upper_boundary(api_client):  # 🟡 EDGE CASE (boundary value)
    client, routes, state = api_client

    response = await client.get("/api/v1/predictions?limit=100")

    assert response.status_code == 200
    assert response.json() == []


# =========================
# 🔴 INVALID CASE TESTS
# =========================

@pytest.mark.parametrize("limit", [0, 101])  # 🔴 INVALID CASE (out of allowed range)
async def test_predictions_limit_rejects_out_of_range_values(api_client, limit):
    client, routes, state = api_client

    response = await client.get(f"/api/v1/predictions?limit={limit}")

    assert response.status_code == 422


async def test_predict_rejects_missing_required_field(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client

    bad_transaction = SAMPLE_TRANSACTION.copy()
    bad_transaction.pop("merchant")

    response = await client.post("/api/v1/predict", json=bad_transaction)

    assert response.status_code == 422


async def test_predict_rejects_negative_amount(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client

    bad_transaction = {**SAMPLE_TRANSACTION, "amt": -10.0}

    response = await client.post("/api/v1/predict", json=bad_transaction)

    assert response.status_code == 422


async def test_predict_rejects_bad_transaction_datetime(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client

    bad_transaction = {**SAMPLE_TRANSACTION, "trans_date_trans_time": "not-a-date"}

    response = await client.post("/api/v1/predict", json=bad_transaction)

    assert response.status_code == 422


async def test_get_prediction_returns_404_for_unknown_id(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client

    response = await client.get("/api/v1/predictions/999999")

    assert response.status_code == 404


async def test_get_case_returns_404_for_unknown_case_id(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client

    response = await client.get("/api/v1/cases/not-a-real-case")

    assert response.status_code == 404


async def test_decide_case_rejects_invalid_human_decision(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client
    state["probability"] = 0.93

    created = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()

    response = await client.post(
        f"/api/v1/cases/{created['case_id']}/decision",
        json={"decision": "MAYBE"},
    )

    assert response.status_code == 422


async def test_decide_case_returns_404_for_unknown_case_id(api_client):  # 🔴 INVALID CASE
    client, routes, state = api_client

    response = await client.post(
        "/api/v1/cases/not-a-real-case/decision",
        json={"decision": "APPROVE"},
    )

    assert response.status_code == 404
