import importlib
import sys
from collections import defaultdict
from types import SimpleNamespace

import httpx
import pytest


pytestmark = pytest.mark.anyio


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


class FakeRedisPipeline:
    def __init__(self, redis_store):
        self.redis_store = redis_store
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def zremrangebyscore(self, key, minimum, maximum):
        self.commands.append(("zremrangebyscore", key, minimum, maximum))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    async def execute(self):
        results = []
        for command in self.commands:
            op = command[0]
            if op == "zremrangebyscore":
                _, key, minimum, maximum = command
                bucket = self.redis_store.sorted_sets[key]
                to_remove = [member for member, score in bucket.items() if minimum <= score <= maximum]
                for member in to_remove:
                    del bucket[member]
                results.append(len(to_remove))
            elif op == "zcard":
                _, key = command
                results.append(len(self.redis_store.sorted_sets[key]))
            elif op == "zadd":
                _, key, mapping = command
                for member, score in mapping.items():
                    self.redis_store.sorted_sets[key][member] = score
                results.append(len(mapping))
            elif op == "expire":
                results.append(True)
        return results


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sorted_sets = defaultdict(dict)

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = str(value)
        return True

    def pipeline(self, transaction=True):
        return FakeRedisPipeline(self)


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def api_client(tmp_path, monkeypatch):
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


async def test_predict_low_risk_persists_prediction_without_case(api_client):
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


async def test_predict_same_payload_is_idempotent(api_client):
    client, routes, state = api_client
    state["probability"] = 0.12

    first = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()
    second = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()

    assert first["id"] == second["id"]
    predictions = (await client.get("/api/v1/predictions")).json()
    assert len(predictions) == 1


async def test_get_prediction_returns_saved_prediction(api_client):
    client, routes, state = api_client
    state["probability"] = 0.12

    created = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()

    response = await client.get(f"/api/v1/predictions/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


async def test_predict_high_risk_creates_case_and_accepts_human_decision(api_client):
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
    assert payload["agent_recommendation"] == "BLOCK"

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


async def test_get_case_returns_pending_case(api_client):
    client, routes, state = api_client
    state["probability"] = 0.93
    routes.generate_agent_review = lambda transaction, probability, policy, **kwargs: SimpleNamespace(
        recommendation="BLOCK",
        confidence=0.93,
        reason_codes=["HIGH_AMOUNT"],
        reviewer_questions=["Can the customer confirm this transaction?"],
        summary="Escalated for review.",
    )

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)
    case_id = response.json()["case_id"]

    case_response = await client.get(f"/api/v1/cases/{case_id}")

    assert case_response.status_code == 200
    assert case_response.json()["case_id"] == case_id


async def test_predict_caches_fingerprint_in_redis(api_client):
    client, routes, state = api_client
    state["probability"] = 0.12

    fake_redis = FakeRedis()
    main = importlib.import_module("app.main")

    async def override_get_redis():
        return fake_redis

    main.app.dependency_overrides[routes.get_redis] = override_get_redis

    response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)

    assert response.status_code == 200
    fingerprint = routes._input_fingerprint(routes.Transaction(**SAMPLE_TRANSACTION))
    assert fake_redis.values[f"fp:{fingerprint}"] == str(response.json()["id"])


async def test_rate_limit_returns_429_after_threshold(api_client):
    client, routes, state = api_client
    state["probability"] = 0.12

    fake_redis = FakeRedis()
    main = importlib.import_module("app.main")

    async def override_get_redis():
        return fake_redis

    main.app.dependency_overrides[routes.get_redis] = override_get_redis

    for _ in range(10):
        response = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)
        assert response.status_code == 200

    blocked = await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)

    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["error"] == "Rate limit exceeded"
    assert detail["limit"] == 10
    assert detail["window_seconds"] == 60


async def test_predictions_limit_accepts_upper_boundary(api_client):
    client, routes, state = api_client

    response = await client.get("/api/v1/predictions?limit=100")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("limit", [0, 101])
async def test_predictions_limit_rejects_out_of_range_values(api_client, limit):
    client, routes, state = api_client

    response = await client.get(f"/api/v1/predictions?limit={limit}")

    assert response.status_code == 422


async def test_predict_rejects_missing_required_field(api_client):
    client, routes, state = api_client

    bad_transaction = SAMPLE_TRANSACTION.copy()
    bad_transaction.pop("merchant")

    response = await client.post("/api/v1/predict", json=bad_transaction)

    assert response.status_code == 422


async def test_predict_rejects_negative_amount(api_client):
    client, routes, state = api_client

    bad_transaction = {**SAMPLE_TRANSACTION, "amt": -10.0}

    response = await client.post("/api/v1/predict", json=bad_transaction)

    assert response.status_code == 422


async def test_predict_rejects_bad_transaction_datetime(api_client):
    client, routes, state = api_client

    bad_transaction = {**SAMPLE_TRANSACTION, "trans_date_trans_time": "not-a-date"}

    response = await client.post("/api/v1/predict", json=bad_transaction)

    assert response.status_code == 422


async def test_get_prediction_returns_404_for_unknown_id(api_client):
    client, routes, state = api_client

    response = await client.get("/api/v1/predictions/999999")

    assert response.status_code == 404


async def test_get_case_returns_404_for_unknown_case_id(api_client):
    client, routes, state = api_client

    response = await client.get("/api/v1/cases/not-a-real-case")

    assert response.status_code == 404


async def test_decide_case_rejects_invalid_human_decision(api_client):
    client, routes, state = api_client
    state["probability"] = 0.93
    routes.generate_agent_review = lambda transaction, probability, policy, **kwargs: SimpleNamespace(
        recommendation="BLOCK",
        confidence=0.93,
        reason_codes=["HIGH_AMOUNT"],
        reviewer_questions=["Can the customer confirm this transaction?"],
        summary="Escalated for review.",
    )

    created = (await client.post("/api/v1/predict", json=SAMPLE_TRANSACTION)).json()

    response = await client.post(
        f"/api/v1/cases/{created['case_id']}/decision",
        json={"decision": "MAYBE"},
    )

    assert response.status_code == 422


async def test_decide_case_returns_404_for_unknown_case_id(api_client):
    client, routes, state = api_client

    response = await client.post(
        "/api/v1/cases/not-a-real-case/decision",
        json={"decision": "APPROVE"},
    )

    assert response.status_code == 404
