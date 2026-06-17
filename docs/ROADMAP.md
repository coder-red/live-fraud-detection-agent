# Roadmap

This roadmap tracks what is already implemented and what remains for the live fraud detection system.

## Completed

### Phase 1: Data Exploration and Model Training

- Used the Kaggle credit card fraud dataset as the project data source.
- Explored fraud patterns across category, time, geography, and customer attributes.
- Trained an XGBoost fraud classifier.
- Saved the trained model to `model/fraud_model.json`.
- Saved the model feature order to `model/feature_list.pkl`.
- Added result visuals for target distribution, feature importance, and confusion matrix.

### Phase 2: Feature Engineering and Inference

- Added reusable feature preprocessing in `src/features.py`.
- Added geospatial distance calculation between customer and merchant.
- Added temporal features such as hour, day of week, weekend flag, and night transaction flag.
- Added `FraudInference` in `src/inference.py` to load the model and score new transactions.

### Phase 3: FastAPI Prediction Service

- Added the FastAPI app in `app/main.py`.
- Added `POST /api/v1/predict` for real-time fraud scoring.
- Added health check support at `GET /`.
- Added request and response schemas with Pydantic.

### Phase 4: Business Risk Policy

- Added deterministic risk classification in `src/policy.py`.
- Converted model probability into `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL` risk bands.
- Separated model scoring from operational decisions such as `APPROVE`, `REVIEW`, and `BLOCK`.

### Phase 5: Agent Review and HITL Flow

- Added structured agent review generation in `src/agent_review.py`.
- Added Groq/LangChain support for reviewer-facing explanations.
- Added deterministic fallback review when the LLM is unavailable.
- Added LangGraph orchestration in `agents/fraud_agents.py`.
- Changed the simulation flow so human review is queued through the API instead of blocking in the terminal.

### Phase 6: Persistence and Case Management

- Added SQLAlchemy database connection setup in `app/db/connections.py`.
- Added persisted prediction and fraud case models in `app/db/models.py`.
- Added endpoints to list predictions, fetch predictions, list pending cases, fetch cases, and submit human decisions.
- Added Docker Compose services for API, Postgres, Redis, and optional pgAdmin.
- **[Redis] Added request fingerprint caching to skip redundant scoring for duplicate transactions.**
- **[Redis] Added sliding-window rate limiting to protect the API from spammy traffic and carding-style abuse.**

### Phase 7: Tests and Review Console

- Added unit tests for feature preprocessing.
- Added unit tests for risk policy thresholds.
- Added API tests for prediction creation, review cases, human decisions, deduplication, and rate limiting.
- Added a Streamlit review dashboard for pending fraud cases and analyst decisions.

### Phase 8: Observability and Traceability (2026 Best Practice)

- **[LangSmith]** Integrated full end-to-end traceability for the API, Agent Reasoning, and Database Tools.
- **[LangSmith]** Implemented automated dataset exports and evaluation experiments.
- Added direct-link observability to CLI scripts for instant access to cloud traces.

### Phase 9: Model Training and Automated Reporting

- Created a reproducible training pipeline in `scripts/train_model.py`.
- Automated the generation of model artifacts (`fraud_model.json`, `feature_list.pkl`).
- Automated the generation of visual reports (Confusion Matrix, Feature Importance).
- Implemented `reports/training_metrics.json` to track model performance and serving thresholds.

## Current State

The project is now a feature-complete local-first AI application with real-time scoring, deterministic policy enforcement, PostgreSQL persistence, Redis-backed deduplication and rate limiting, human-in-the-loop case management, LangSmith tracing/evals, and a reproducible training/reporting pipeline.

The LLM judge now consumes a deterministic CoVe verification bundle, so DB-backed claims are checked against the same evidence the agent saw.

It is still not fully production-hardened. Authentication, database migrations, CI/CD, prompt-injection defenses, and deployment hardening are still open.

## Remaining Work

### Phase 10: AI Reliability & "LLM-as-a-Judge" (Partially Complete)

- **[Done]** Implemented a **CoVe verification bundle** in `src/agent_review.py` so reviews carry deterministic evidence from the transaction, policy, and DB tools.
- **[Done]** Updated the **LLM-as-a-Judge** evaluator in `evals/langsmith_eval.py` to score summaries against the shared verification bundle instead of the summary text alone.
- **[Done]** Added a GitHub Actions CI/CD workflow that runs tests, validates the Docker build, and exposes a manual LangSmith eval gate for Phase 10.
- **[Done]** Tightened the LangSmith eval gate so it can block `main` automatically when the required secrets and reviewed-case data are available in CI.
- **[Done]** Added a dependency security scan to the CI pipeline so known vulnerable packages fail the release path early.
- **[Done]** Added explicit LangSmith score thresholds so the eval job fails if human match or faithfulness falls below the gate.
- **[Done]** Added cleaner evidence formatting, including a markdown decision summary and evidence table, inside the agent's reasoning.

### Phase 11: AI Safety & Efficiency 

- **[Guardrails,Done]** Implemented **Prompt Injection Detection** to protect the Groq API from malicious transaction data by redacting suspicious instruction-like content before model invocation.
- **[Routing]** Implement **LLM Routing**: Use a fast, cheap 3B model (e.g. Llama-3.2-3B) for easy cases and escalate to 70B only for "Review" decisions.
- **[Policy Filtering]** Add an output guardrail to ensure the agent never leaks PII or violates corporate policy in its summaries.

### Phase 12: Deployment, Security, and CI/CD

- **[Hosting]** Prepare for deployment to Oracle Cloud or similar Docker-native VPS.
- **[CI/CD]** Add GitHub Actions for automated testing, linting, and Docker build validation.
- **[Security]** Add authentication for review endpoints to secure the human-in-the-loop flow.
- Optimize the Docker image size for faster CI/CD and lower bandwidth deployments.

## Future Improvements

- Add request logging and structured application logs.
- Add OpenAPI examples directly to FastAPI schemas.
- Add Kafka-based event ingestion so transaction scoring can move from direct API requests to an async pipeline.
