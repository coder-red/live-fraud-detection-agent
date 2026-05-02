# API Reference

The FastAPI service exposes real-time fraud scoring, stored prediction history, and a human review queue for high-risk cases.

Base URL for local development:

```text
http://127.0.0.1:8000
```

API prefix:

```text
/api/v1
```

## Health Check

```http
GET /
```

Example response:

```json
{
  "status": "online",
  "model": "XGBoost Fraud Agent"
}
```

## Create Prediction

```http
POST /api/v1/predict
```

Scores a transaction with the XGBoost model, applies the deterministic risk policy, persists the prediction, and creates a fraud case when human review is required.

Example request:

```json
{
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
  "job": "Engineer"
}
```

Example response:

```json
{
  "id": 1,
  "trans_date_trans_time": "2026-02-19T03:00:00",
  "amt": 9099.99,
  "category": "shopping_net",
  "merchant": "fraud_store_xyz",
  "city": "Lagos",
  "state": "LG",
  "is_fraud": true,
  "probability": 0.9342,
  "threshold": 0.5221,
  "risk_band": "CRITICAL",
  "decision": "BLOCK",
  "requires_review": true,
  "case_id": "0f0bca59-3d35-44ce-a87e-9056f8e6b716",
  "agent_recommendation": "BLOCK",
  "agent_summary": "CRITICAL risk transaction in shopping_net for $9,099.99; policy recommends BLOCK.",
  "created_at": "2026-05-02T10:30:00"
}
```

## List Predictions

```http
GET /api/v1/predictions?limit=20
```

Returns the newest saved predictions first.

| Name | Default | Range | Description |
| --- | ---: | ---: | --- |
| `limit` | `20` | `1-100` | Maximum number of prediction records to return. |

## Get Prediction

```http
GET /api/v1/predictions/{prediction_id}
```

Returns one saved prediction by database ID.

| Status | Meaning |
| ---: | --- |
| `404` | Prediction was not found. |

## List Pending Cases

```http
GET /api/v1/cases/pending?limit=20
```

Returns fraud cases waiting for human review.

| Name | Default | Range | Description |
| --- | ---: | ---: | --- |
| `limit` | `20` | `1-100` | Maximum number of pending cases to return. |

## Get Case

```http
GET /api/v1/cases/{case_id}
```

Returns one fraud case by `case_id`.

| Status | Meaning |
| ---: | --- |
| `404` | Fraud case was not found. |

## Submit Human Decision

```http
POST /api/v1/cases/{case_id}/decision
```

Closes a pending fraud case with the reviewer verdict. The model and agent recommendation remain stored as evidence, but the human decision becomes the final operational outcome.

Example request:

```json
{
  "decision": "BLOCK",
  "note": "Customer could not verify the merchant or amount."
}
```

Allowed decisions:

```text
APPROVE
BLOCK
```

| Status | Meaning |
| ---: | --- |
| `404` | Fraud case was not found. |
| `409` | Fraud case was already reviewed. |

## Risk Policy

The model returns a fraud probability. `src/policy.py` converts that probability into an auditable business decision:

| Probability | Risk band | Decision | Human review |
| ---: | --- | --- | --- |
| `>= 0.90` | `CRITICAL` | `BLOCK` | Yes |
| `>= 0.55` | `HIGH` | `REVIEW` | Yes |
| `>= 0.25` | `MEDIUM` | `APPROVE` | No |
| `< 0.25` | `LOW` | `APPROVE` | No |

## Local Run Commands

Run the full simulation:

```bash
uv run run_all.py
```

Run the Docker stack:

```bash
docker compose up --build
```

Run with pgAdmin:

```bash
docker compose --profile devtools up --build
```
