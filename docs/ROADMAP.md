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

## Current State

The system can score transactions, persist predictions, deduplicate repeated requests, create review cases, generate agent summaries, enforce a basic rate limiter, and accept human decisions through API endpoints and the review dashboard.

The next best work is no longer basic backend setup. The project now needs stronger evaluation, better fraud reasoning, and more useful AI-assisted review.

## Remaining Work

### Phase 8: CI/CD and Validation

- Add CI checks for tests, formatting, and Docker build validation.
- Add dependency/security checks such as `pip-audit`.
- Add regression tests around the sample transaction flow.
- Add a simple PR/commit workflow so changes to model logic or API behaviour are validated automatically.

### Phase 9: Model Evaluation and Decision Quality

- Report the actual model metrics more clearly in the README and docs.
- Evaluate threshold trade-offs between false positives, false negatives, and review volume.
- Track how often the policy layer sends cases to `APPROVE`, `REVIEW`, and `BLOCK`.
- Add slice-based evaluation across category, amount, geography, and time of day.
- Compare raw model output against final policy decisions so the decision layer is auditable.

### Phase 10: Agent Quality and HITL Usefulness

- Improve the quality and consistency of agent reasoning.
- Make reviewer questions more specific and more actionable.
- Evaluate when the LLM adds useful context versus when the fallback logic is enough.
- Improve case summaries so they read like something a real fraud analyst would use.
- Add cleaner reason codes and stronger evidence formatting for human review.

### Phase 11: Advanced Fraud Signals

- Add Redis-backed velocity counters such as transaction count per city/merchant over the last hour.
- Add behaviour-based signals for unusual merchant, geography, and repeat activity patterns.
- Feed richer stateful signals into the model and/or agent review layer.
- Compare whether these signals improve fraud capture without creating too much review noise.

### Phase 12: Dynamic Retraining and Monitoring

- Track prediction counts, review counts, approval/block rates, fraud hit rate, and agent-review usage.
- Track model quality metrics over time when labels are available.
- Monitor model drift across amount, category, geography, and hour.
- Add a sliding-window retraining pipeline.
- Compare candidate models against the current model before promotion.
- Save model versions and evaluation reports.
- Add rollback support if a new model performs worse.

### Phase 13: Schema and Deployment Hardening

- Replace `Base.metadata.create_all()` with a migration workflow such as Alembic.
- Add versioned schema changes for prediction and case tables.
- Document how to apply migrations locally and in Docker.
- Add authentication for review endpoints when the app is ready to move beyond demo mode.

## Future Improvements

- Add request logging and structured application logs.
- Add OpenAPI examples directly to FastAPI schemas.
- Add a proper deployment edge layer such as Nginx only when the app needs a more realistic hosting setup.
