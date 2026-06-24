<p align="center">
  <img src="assets/FRAUD.png" alt="Project Banner" width="100%">
</p>

![Python version](https://img.shields.io/badge/Python%20version-3.11%2B-lightgrey)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat&logo=graph-dot-org&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f55036?style=flat&logo=speedtest&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2ECC71?style=flat&logo=anaconda&logoColor=white)

## Live Demo

- API: https://live-fraud-detection-agent.onrender.com
- Dashboard: https://mhmdxch-fraud-dashboard.hf.space

## Author

- [@coder-red](https://www.github.com/coder-red)

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Deployment](#deployment)
- [MLOps & Observability](#mlops--observability)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Limitations & Trade-offs](#limitations--trade-offs)
- [Repository Structure](#repository-structure)

---

## Architecture

The system uses a dual-engine decision pipeline: an XGBoost classifier for fast bulk scoring, and a LangGraph-based LLM agent for contextual investigation of borderline cases. A risk policy layer translates raw probabilities into business actions (`APPROVE`, `REVIEW`, `BLOCK`), and suspicious cases enter a Human-In-The-Loop review workflow.

```
Transaction → API → XGBoost Score → Risk Policy ──┬─ Low/Medium → Auto-approve
                                                    └─ High/Critical → LLM Agent → Human Review
```

### Design Decisions

| Decision | Rationale |
|---|---|
| **XGBoost for scoring** | Provides calibrated probabilities with sub-millisecond inference. Outperforms neural nets on tabular data at this scale. |
| **LangGraph agent for grey areas** | Rule-based thresholds alone miss context (e.g., high amount at a known merchant vs. unknown). The agent queries merchant history, velocity, and geo-anomaly before recommending an action. |
| **Redis for rate limiting + dedup** | Sliding-window counter prevents API abuse. Request fingerprinting avoids re-scoring identical payloads. Both are fast-path operations that don't touch Postgres. |
| **PostgreSQL for persistence** | Predictions and review cases need ACID guarantees, foreign key relationships, and ad-hoc querying for the dashboard and agent tools. |
| **Risk policy as a configurable layer** | Keeps business rules decoupled from model logic. Thresholds and actions can change without retraining. |

### Request Lifecycle

1. `POST /api/v1/predict` receives a transaction payload
2. Redis sliding-window rate limiter checks request volume
3. Request fingerprinted against recent keys — duplicate check
4. `preprocess_features()` engineers temporal, geospatial, and categorical features
5. XGBoost returns fraud probability
6. `classify_risk()` maps probability → `APPROVE` / `REVIEW` / `BLOCK`
7. Prediction persisted to Postgres
8. If `REVIEW` or `BLOCK` → LLM agent investigates via merchant history, velocity, and geo tools
9. Agent report stored alongside the prediction in a `FraudCase`
10. Dashboard polls `/cases/pending` for human review

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ (uv for dependency management) |
| **API Framework** | FastAPI + Uvicorn (async, auto-docs via Swagger/ReDoc) |
| **ML Model** | XGBoost (trained on ~1.3M transactions) |
| **LLM Agent** | Groq (llama-3.3-70b) via LangChain / LangGraph |
| **LLM Observability** | LangSmith (tracing, evaluation) |
| **Database** | PostgreSQL 16 + SQLAlchemy (async-capable) |
| **Cache / Rate Limiting** | Redis 7 |
| **Feature Engineering** | Pandas, NumPy → stdlib `math` (optimized for inference) |
| **Dashboard** | Streamlit + Plotly |
| **Containerization** | Docker multi-stage build + Docker Compose |

## Deployment

| Service | Platform | URL | Uptime |
|---|---|---|---|
| API | Render (Docker) | https://live-fraud-detection-agent.onrender.com | Always-on via keep-alive cron |
| Dashboard | Hugging Face Spaces (Streamlit SDK) | https://mhmdxch-fraud-dashboard.hf.space | Warm via keep-alive cron |
| Database | Supabase (PostgreSQL) | Managed | 24/7 |
| Redis | Redis Cloud (Free 30MB) | Managed | 24/7 |

**Keep-alive:** GitHub Actions runs every 5 minutes (`/.github/workflows/keep-awake.yml`) to prevent cold starts on free-tier services.

## MLOps & Observability

- **CI/CD:** GitHub Actions for scheduled keep-alive and deployment triggers
- **Containerization:** Multi-stage Docker build (builder → slim runtime, non-root user)
- **Model Versioning:** XGBoost model stored in JSON format (`model/fraud_model.json`) — git-tracked and diffable
- **Feature Pipeline:** Deterministic preprocessing (`src/features.py`) — same code for training and inference
- **LLM Tracing:** LangSmith captures every agent invocation, tool call, and decision for debugging and evaluation
- **Evaluation:** LangSmith eval scripts in `evals/` for regression testing agent behavior
- **Health Checks:** API exposes `/` health endpoint consumed by Render and the dashboard
- **Secret Management:** Environment variables via `.env` locally, Render secrets in production
- **Output Guardrails:** PII redaction and prompt injection detection on agent output (`src/output_guard.py`)
- **API Documentation:** Auto-generated OpenAPI spec at `/docs`

## Quick Start

**Prerequisites:** Python 3.11+, Groq API key, `uv` installed.

```bash
git clone https://github.com/coder-red/live-fraud-detection-agent/
cp .env.example .env   # fill in your keys
uv sync
```

**Run the API:**
```bash
uv run uvicorn app.main:app --reload
```

**Run the dashboard:**
```bash
uv run streamlit run dashboard.py
```

**Run with full stack (Docker):**
```bash
docker compose up --build
```

**Run tests:**
```bash
uv run pytest tests
```

## API Reference

Full API docs at `/docs` when running, or see [`docs/API.md`](docs/API.md).

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/api/v1/predict` | POST | Score a transaction |
| `/api/v1/predictions` | GET | List predictions (paginated) |
| `/api/v1/cases/pending` | GET | Pending review cases |
| `/api/v1/cases/resolved` | GET | Resolved cases |
| `/api/v1/cases/{id}/decision` | POST | Submit human verdict |
| `/api/v1/stream` | GET | SSE live feed |
| `/api/v1/predictions/run-simulation` | POST | Generate test transactions |

## Model Performance

XGBoost classifier trained on ~1.3M transactions from the [Kaggle Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

**Top predictive signals:**
- Merchant category (`shopping_net`, `misc_net`, `grocery_pos`)
- Late-night activity (22:00–03:00)
- Geographic distance between cardholder and merchant
- Transaction amount relative to typical spending

**Decision Layer:** Raw probability is mapped through a configurable risk policy (`src/policy.py`) into business actions rather than exposing probability alone.

## Limitations & Trade-offs

- **Static model:** XGBoost is not retrained online. Fraud patterns that drift require a full retraining cycle.
- **No authentication:** API is open for demo purposes. Production would need API keys, rate limiting per user, and RBAC.
- **Dual-engine latency:** The LLM agent adds 2–5s for grey-area cases. Acceptable for review workflows, not for real-time blocking.
- **No database migrations:** Uses `create_all()` for simplicity. Alembic or similar would be needed for schema evolution.

## Repository Structure

<details>
  <summary><strong>Click to expand</strong></summary>

```text
.
├── Dockerfile              # Multi-stage production build
├── docker-compose.yml      # Local dev stack (API + Postgres + Redis)
├── pyproject.toml          # Project metadata + dependencies
├── app/
│   ├── main.py             # FastAPI entrypoint
│   ├── routes.py           # All API endpoints
│   ├── events.py           # SSE event manager
│   └── db/
│       ├── connections.py  # DB engine, Redis client, session management
│       └── models.py       # SQLAlchemy ORM (FraudPrediction, FraudCase)
├── src/
│   ├── inference.py        # XGBoost model loading and inference
│   ├── features.py         # Feature engineering
│   ├── policy.py           # Risk classification
│   ├── agent_review.py     # LLM investigator agent
│   ├── output_guard.py     # PII redaction
│   └── transaction_generator.py  # Synthetic transaction generator
├── agents/
│   └── tools.py            # DB tools (merchant history, velocity, geo)
├── dashboard.py            # Streamlit review console
├── model/
│   ├── fraud_model.json    # Trained XGBoost model
│   └── feature_list.pkl    # Feature column order
├── tests/                  # pytest suite
├── evals/                  # LangSmith evaluation scripts
└── docs/                   # API docs + roadmap
```

</details>
