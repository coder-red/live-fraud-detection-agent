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

## Current State

The system can score transactions, persist predictions, create review cases, generate agent summaries, and accept human decisions through API endpoints.

The main remaining work is production hardening and operational polish.

## Remaining Work

### Phase 7: Tests and Validation

- Add unit tests for feature preprocessing.
- Add unit tests for risk policy thresholds.
- Add API tests for prediction creation, case creation, and human decision submission.
- Add regression tests around the sample transaction flow.

### Phase 8: Database Migrations

- Replace `Base.metadata.create_all()` with a migration workflow such as Alembic.
- Add versioned schema changes for prediction and case tables.
- Document how to apply migrations locally and in Docker.

### Phase 9: Review Dashboard

- Build a small UI or admin surface for pending cases.
- Show transaction facts, model probability, risk band, agent summary, reason codes, and reviewer questions.
- Allow reviewers to approve or block a case from the dashboard.

### Phase 10: Monitoring and Evaluation

- Track prediction counts, review counts, approval/block rates, and fraud hit rate.
- Track model quality metrics over time when labels are available.
- Monitor model drift across amount, category, geography, and hour.

### Phase 11: Dynamic Retraining

- Add a sliding-window retraining pipeline.
- Compare candidate models against the current model before promotion.
- Save model versions and evaluation reports.
- Add rollback support if a new model performs worse.

## Future Improvements

- Add authentication for review endpoints.
- Add request logging and structured application logs.
- Add Redis-backed background jobs if review or retraining tasks become async.
- Add OpenAPI examples directly to FastAPI schemas.
- Add CI checks for formatting, tests, and Docker build validation.
