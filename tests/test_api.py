import importlib  # This lets the test reload app modules after changing environment variables.
import sys  # This lets the test remove already loaded modules from Python memory.
from types import SimpleNamespace  # This creates a small fake object for the agent review.

import httpx  # This lets async tests call the FastAPI app directly through ASGI.
import pytest  # This gives us pytest fixtures and test helpers.


pytestmark = pytest.mark.anyio  # This lets every test in this file use async/await.


SAMPLE_TRANSACTION = {  # This is one fake transaction used by the API tests.
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


class StubInference:  # This fake model replaces the real XGBoost model in tests.
    def __init__(self, probability: float):
        self.probability = probability  # This saves the probability the fake model should return.

    def predict(self, raw_data: dict) -> dict:  # This gives the same shape as the real model output.
        return {
            "is_fraud": self.probability >= 0.5221,  # This says if the fake probability passes the model threshold.
            "probability": self.probability,
            "threshold": 0.5221,
            "features": [],  # This is included because the real model returns a features field.
        }


@pytest.fixture()  # This creates shared setup for API tests.
def anyio_backend():  # This tells pytest-anyio to use asyncio instead of trying multiple async backends.
    return "asyncio"


@pytest.fixture()  # This creates shared setup for API tests.
async def api_client(tmp_path, monkeypatch):  # This fixture gives each test a clean API client.
    db_path = tmp_path / "test.db"  # This creates a temporary SQLite database path.
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")  # This tells the app to use SQLite for this test.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # This prevents the test from calling Groq.

    for module_name in [  # This clears modules that depend on environment variables.
        "app.main",
        "app.routes",
        "app.db.models",
        "app.db.connections",
    ]:
        sys.modules.pop(module_name, None)  # This forces Python to import the module fresh.

    connections = importlib.import_module("app.db.connections")  # This imports the DB connection using the test database URL.
    importlib.import_module("app.db.models")  # This imports the table models so SQLAlchemy knows them.
    connections.Base.metadata.create_all(bind=connections.engine)  # This creates the test tables.

    routes = importlib.import_module("app.routes")  # This imports the API routes.
    main = importlib.import_module("app.main")  # This imports the FastAPI app.
    state = {"probability": 0.12}  # This lets each test choose the fake model probability.

    def override_get_db():  # This replaces the normal database dependency.
        db = connections.SessionLocal()  # This opens a test database session.
        try:
            yield db  # This gives the session to the API route.
        finally:
            db.close()  # This closes the test database session.

    def override_inference():  # This replaces the real model dependency.
        return StubInference(probability=state["probability"])  # This returns the fake model with the current probability.

    main.app.dependency_overrides[routes.get_db] = override_get_db  # This tells FastAPI to use the test database.
    main.app.dependency_overrides[routes.get_inference_engine] = override_inference  # This tells FastAPI to use the fake model.

    transport = httpx.ASGITransport(app=main.app)  # This connects httpx to the FastAPI app without starting a server.
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:  # This creates an async API client.
        yield client, routes, state  # This gives the test client, route module, and mutable state to the test.

    main.app.dependency_overrides.clear()  # This removes test overrides after the test finishes.


async def test_predict_low_risk_persists_prediction_without_case(api_client):  # NORMAL CASE: this tests the low-risk API path.
    client, routes, state = api_client  # This gets the prepared API test tools.
    state["probability"] = 0.12  # This makes the fake model return a low fraud probability.

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)  # This sends the fake transaction to the API.

    assert response.status_code == 200  # The API request should succeed.
    payload = response.json()  # This converts the response body into a Python dictionary.
    assert payload["risk_band"] == "LOW"  # Low probability should create a low risk band.
    assert payload["decision"] == "APPROVE"  # Low risk should be approved.
    assert payload["requires_review"] is False  # Low risk should not need human review.
    assert payload["case_id"] is None  # No review case should be created.

    predictions = (await client.get("/api/v1/predictions")).json()  # This asks the API for saved predictions.
    assert len(predictions) == 1  # Exactly one prediction should have been saved.
    assert predictions[0]["id"] == payload["id"]  # The saved prediction should match the prediction response.


async def test_get_prediction_returns_saved_prediction(api_client):  # NORMAL CASE: this checks reading one saved prediction by ID.
    client, routes, state = api_client  # This gets the prepared API test tools.
    state["probability"] = 0.12  # This makes the fake model return a low fraud probability.
    created = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()  # This creates one prediction first.

    response = await client.get(f"/api/v1/predictions/{created['id']}")  # This asks for the saved prediction by ID.

    assert response.status_code == 200  # The API should find the prediction.
    assert response.json()["id"] == created["id"]  # The returned prediction should be the same one we created.


async def test_predict_high_risk_creates_case_and_accepts_human_decision(api_client):  # NORMAL CASE: this tests the high-risk review path.
    client, routes, state = api_client  # This gets the prepared API test tools.
    state["probability"] = 0.93  # This makes the fake model return a critical fraud probability.
    routes.generate_agent_review = lambda transaction, probability, policy: SimpleNamespace(  # This replaces the agent review with a fake result.
        recommendation="BLOCK",  # The fake agent recommends blocking.
        confidence=0.93,  # The fake agent confidence matches the probability.
        reason_codes=["HIGH_AMOUNT", "HIGH_RISK_CATEGORY"],  # The fake agent gives reason codes.
        reviewer_questions=["Can the customer confirm this transaction?"],  # The fake agent gives a reviewer question.
        summary="Critical risk transaction; policy recommends BLOCK.",  # The fake agent gives a short summary.
    )

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)  # This sends the fake transaction to the API.

    assert response.status_code == 200  # The API request should succeed.
    payload = response.json()  # This converts the response body into a Python dictionary.
    assert payload["risk_band"] == "CRITICAL"  # High probability should create a critical risk band.
    assert payload["decision"] == "BLOCK"  # Critical risk should be blocked by policy.
    assert payload["requires_review"] is True  # Critical risk should require human review.
    assert payload["case_id"]  # A review case ID should exist.
    assert payload["agent_recommendation"] == "BLOCK"  # The response should include the fake agent recommendation.

    pending = (await client.get("/api/v1/cases/pending")).json()  # This asks the API for pending review cases.
    assert len(pending) == 1  # Exactly one pending case should exist.
    assert pending[0]["case_id"] == payload["case_id"]  # The pending case should be the same case from the prediction response.
    assert pending[0]["status"] == "PENDING_REVIEW"  # The case should still be waiting for review.

    decision = await client.post(  # This sends the human reviewer decision.
        f"/api/v1/cases/{payload['case_id']}/decision",
        json={"decision": "BLOCK", "note": "Customer could not verify the transaction."},
    )

    assert decision.status_code == 200  # The human decision request should succeed.
    reviewed_case = decision.json()  # This converts the reviewed case response into a Python dictionary.
    assert reviewed_case["status"] == "BLOCKED"  # The case should now be blocked.
    assert reviewed_case["human_decision"] == "BLOCK"  # The human decision should be saved.
    assert reviewed_case["human_note"] == "Customer could not verify the transaction."  # The human note should be saved.
    assert reviewed_case["reviewed_at"] is not None  # The review timestamp should be saved.

    duplicate = await client.post(  # This tries to review the same case again.
        f"/api/v1/cases/{payload['case_id']}/decision",
        json={"decision": "APPROVE"},
    )
    assert duplicate.status_code == 409  # The API should reject a second review decision.


async def test_get_case_returns_pending_case(api_client):  # NORMAL CASE: this checks reading one fraud review case by case_id.
    client, routes, state = api_client  # This gets the prepared API test tools.
    state["probability"] = 0.93  # This makes the fake model create a review case.

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)  # This creates one high-risk prediction.
    case_id = response.json()["case_id"]  # This saves the generated case ID.

    case_response = await client.get(f"/api/v1/cases/{case_id}")  # This asks for that one case.

    assert case_response.status_code == 200  # The API should find the fraud case.
    assert case_response.json()["case_id"] == case_id  # The returned case should match the generated case.
    assert case_response.json()["status"] == "PENDING_REVIEW"  # A new case should still be waiting for review.


async def test_predictions_limit_accepts_upper_boundary(api_client):  # EDGE CASE: this checks the biggest allowed list limit.
    client, routes, state = api_client  # This gets the prepared API test tools.

    response = await client.get("/api/v1/predictions?limit=100")  # This uses the maximum allowed limit.

    assert response.status_code == 200  # The API should accept limit=100.
    assert response.json() == []  # The test database is empty, so the response should be an empty list.


@pytest.mark.parametrize("limit", [0, 101])  # These are just outside the allowed limit range.
async def test_predictions_limit_rejects_out_of_range_values(api_client, limit):  # INVALID CASE: limit must be from 1 to 100.
    client, routes, state = api_client  # This gets the prepared API test tools.

    response = await client.get(f"/api/v1/predictions?limit={limit}")  # This sends an invalid limit.

    assert response.status_code == 422  # FastAPI should reject the invalid query parameter.


async def test_predict_rejects_missing_required_field(api_client):  # INVALID CASE: required transaction fields cannot be missing.
    client, routes, state = api_client  # This gets the prepared API test tools.
    bad_transaction = SAMPLE_TRANSACTION.copy()  # This starts from a valid transaction.
    bad_transaction.pop("merchant")  # This removes one required field.

    response = await client.post("/api/v1/predict", json=bad_transaction)  # This sends the incomplete transaction.

    assert response.status_code == 422  # FastAPI should reject the request before calling the model.


async def test_predict_rejects_negative_amount(api_client):  # INVALID CASE: transaction amounts must be greater than zero.
    client, routes, state = api_client  # This gets the prepared API test tools.
    bad_transaction = {**SAMPLE_TRANSACTION, "amt": -10.0}  # This creates a transaction with an impossible amount.

    response = await client.post("/api/v1/predict", json=bad_transaction)  # This sends the bad amount.

    assert response.status_code == 422  # Pydantic should reject the invalid amount.


async def test_predict_rejects_bad_transaction_datetime(api_client):  # INVALID CASE: transaction datetime must be parseable.
    client, routes, state = api_client  # This gets the prepared API test tools.
    bad_transaction = {**SAMPLE_TRANSACTION, "trans_date_trans_time": "not-a-date"}  # This breaks the datetime field.

    response = await client.post("/api/v1/predict", json=bad_transaction)  # This sends the bad datetime.

    assert response.status_code == 422  # The route should return a validation-style error.
    assert response.json()["detail"] == "Invalid isoformat string: 'not-a-date'"  # The response should explain the bad value.


async def test_get_prediction_returns_404_for_unknown_id(api_client):  # INVALID CASE: asking for a prediction that does not exist.
    client, routes, state = api_client  # This gets the prepared API test tools.

    response = await client.get("/api/v1/predictions/999999")  # This asks for an ID that was never created.

    assert response.status_code == 404  # The API should say the prediction was not found.


async def test_get_case_returns_404_for_unknown_case_id(api_client):  # INVALID CASE: asking for a fraud case that does not exist.
    client, routes, state = api_client  # This gets the prepared API test tools.

    response = await client.get("/api/v1/cases/not-a-real-case")  # This asks for a fake case ID.

    assert response.status_code == 404  # The API should say the fraud case was not found.


async def test_decide_case_rejects_invalid_human_decision(api_client):  # INVALID CASE: human decision must be APPROVE or BLOCK.
    client, routes, state = api_client  # This gets the prepared API test tools.
    state["probability"] = 0.93  # This makes the fake model create a review case.
    created = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()  # This creates one pending case.

    response = await client.post(  # This sends a decision value that the API does not allow.
        f"/api/v1/cases/{created['case_id']}/decision",
        json={"decision": "MAYBE"},
    )

    assert response.status_code == 422  # FastAPI should reject the invalid decision value.


async def test_decide_case_returns_404_for_unknown_case_id(api_client):  # INVALID CASE: cannot review a case that does not exist.
    client, routes, state = api_client  # This gets the prepared API test tools.

    response = await client.post(  # This sends a valid decision to a fake case ID.
        "/api/v1/cases/not-a-real-case/decision",
        json={"decision": "APPROVE"},
    )

    assert response.status_code == 404  # The API should say the fraud case was not found.
