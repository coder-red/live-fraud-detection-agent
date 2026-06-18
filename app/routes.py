import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from redis import asyncio as redis_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.db.connections import get_db, get_redis
from app.db.models import FraudCase, FraudPrediction
from app.events import events
from src.agent_review import generate_agent_review
from src.output_guard import guard_agent_output
from src.policy import classify_risk
from langsmith import traceable


router = APIRouter()

RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in {"1", "true", "yes", "on"}


def _client_ip(request: Request) -> str:
    """
    Resolve the client IP. Proxy headers are only trusted when explicitly enabled.
    """
    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            first_hop = forwarded_for.split(",")[0].strip()
            if first_hop:
                return first_hop

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def rate_limit(request: Request, redis: redis_asyncio.Redis | None = Depends(get_redis)):
    """
    Redis-backed sliding-window rate limiting per client IP.
    """
    if not redis:
        return

    client_ip = _client_ip(request)
    key = f"rate_limit:{client_ip}"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    request_member = f"{now_ms}:{uuid.uuid4().hex}"
    window_start_ms = now_ms - (RATE_LIMIT_WINDOW_SECONDS * 1000)

    try:
        async with redis.pipeline(transaction=True) as pipe:
            # Keep only requests that are still inside the current sliding window.
            pipe.zremrangebyscore(key, 0, window_start_ms)
            pipe.zcard(key)
            pipe.zadd(key, {request_member: now_ms})
            pipe.expire(key, RATE_LIMIT_WINDOW_SECONDS + 5)
            results = await pipe.execute()

        current_count = int(results[1]) + 1
        if current_count > RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests from this IP. Please try again in {RATE_LIMIT_WINDOW_SECONDS} seconds.",
                    "ip": client_ip,
                    "limit": RATE_LIMIT_MAX_REQUESTS,
                    "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
                },
            )
    except HTTPException:
        raise
    except Exception:
        # If Redis fails during the check, allow the request rather than taking down the API.
        pass


# --- DATA MODELS (Pydantic) ---
# These classes define what "valid" data looks like for our API.
# If a user sends data that doesn't match these, FastAPI will automatically send an error.

class Transaction(BaseModel):
    """The data we expect when someone asks for a fraud prediction."""
    trans_date_trans_time: str
    amt: float = Field(gt=0) # Must be a number greater than 0
    category: str
    merchant: str
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city: str
    state: str
    city_pop: int
    dob: str
    gender: str
    job: str

class PredictionRecord(BaseModel):
    """The data we send back after a prediction is made."""
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    trans_date_trans_time: datetime
    amt: float
    category: str
    merchant: str
    city: str
    state: str
    is_fraud: bool
    probability: float
    threshold: float
    risk_band: str | None = None #none means the model didn't assign a risk band (e.g., if it failed before that step)
    decision: str | None = None
    requires_review: bool | None = None
    case_id: str | None = None
    agent_recommendation: str | None = Field(None, alias="agent_action") # We use 'alias' to map the database field 'agent_action' to the API field 'agent_recommendation' for clarity in the API response.
    agent_summary: str | None = Field(None, alias="reasoning")
    created_at: datetime


class FraudCaseRecord(BaseModel):
    """The data model for a 'Case' that needs human or AI agent review."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: str
    prediction_id: int
    risk_band: str
    model_decision: str
    # 'None' means the field is empty because it hasn't been filled yet (e.g., waiting for review).
    agent_recommendation: str | None = None 
    agent_confidence: float | None = None
    reason_codes: list[str] | None = None
    reviewer_questions: list[str] | None = None
    reasoning: str | None = None
    human_decision: str | None = None
    human_note: str | None = None
    status: str
    created_at: datetime
    reviewed_at: datetime | None = None


class HumanDecisionRequest(BaseModel):
    """What we expect when a human clicks 'Approve' or 'Block' in the dashboard."""
    decision: Literal["APPROVE", "BLOCK"]
    note: str | None = None # Optional field for the human reviewer to add notes about their decision.


class PredictionPage(BaseModel):
    items: list[PredictionRecord]
    total: int
    offset: int
    limit: int
    
# --- GLOBAL STATE ---
# We store the ML model engine here so we don't have to reload it from disk every time someone hits the API.
_engine: Any | None = None 


def get_inference_engine() -> Any:
    """Helper function to load and return the Fraud Detection AI model."""
    global _engine
    if _engine is None:
        from src.inference import FraudInference
        base_path = Path(__file__).parent.parent / "model"
        _engine = FraudInference(
            model_path=str(base_path / "fraud_model.json"),
            feature_list_path=str(base_path / "feature_list.pkl"),
        )
    return _engine


# --- HELPER FUNCTIONS ---

def _parse_datetime(value: str) -> datetime:
    """Converts a date string from the user into a Python datetime object."""
    return datetime.fromisoformat(value)


def _input_fingerprint(data: Transaction) -> str:
    """
    Creates a unique ID (hash) for a transaction based on its data.
    If two transactions have the exact same amount, merchant, etc., they get the same ID.
    This helps us detect duplicates instantly.
    """
    d = data.model_dump()
    # We round numbers and strip spaces to make sure tiny differences don't break our matching.
    for key in ("amt", "lat", "long", "merch_lat", "merch_long"):
        d[key] = round(float(d[key]), 6)
    d["city_pop"] = int(d["city_pop"])
    for key in ("trans_date_trans_time", "category", "merchant", "city", "state", "dob", "gender", "job"):
        d[key] = str(d[key]).strip()
    # Sort the keys so the order doesn't change the ID.
    ordered = {k: d[k] for k in sorted(d)} # creates an ordered dict sorted by key, sorting is crucial cos hashes will be different if the same data is in a different order. By sorting, we ensure the same data always produces the same hash.
    blob = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _latest_case_for_prediction(db: Session, prediction_id: int) -> FraudCase | None:
    """Finds the most recent review 'Case' associated with a specific prediction."""
    return (
        db.query(FraudCase)
        .filter(FraudCase.prediction_id == prediction_id)
        .order_by(FraudCase.created_at.desc())
        .first()
    )


def _prediction_payload(record: FraudPrediction, case: FraudCase | None = None) -> dict:
    """Combines a Database record and its Review Case into a single dictionary for the API response."""
    return {
        "id": record.id,
        "trans_date_trans_time": record.trans_date_trans_time,
        "amt": record.amt,
        "category": record.category,
        "merchant": record.merchant,
        "city": record.city,
        "state": record.state,
        "is_fraud": record.is_fraud,
        "probability": record.probability,
        "threshold": record.threshold,
        "risk_band": record.risk_band,
        "decision": record.decision,
        "requires_review": record.requires_review,
        "case_id": case.case_id if case else None,
        "agent_recommendation": case.agent_recommendation if case else record.agent_action,
        "agent_summary": case.reasoning if case else record.reasoning,
        "created_at": record.created_at,
    }


def _record_to_dict(record: FraudPrediction) -> dict:
    """Turns a saved database record back into a simple dictionary (used for re-running reviews)."""
    return {
        "trans_date_trans_time": record.trans_date_trans_time.isoformat(),
        "amt": record.amt,
        "category": record.category,
        "merchant": record.merchant,
        "lat": record.lat,
        "long": record.long,
        "merch_lat": record.merch_lat,
        "merch_long": record.merch_long,
        "city": record.city,
        "state": record.state,
        "city_pop": record.city_pop,
        "dob": record.dob.isoformat(),
        "gender": record.gender,
        "job": record.job,
    }


def _open_new_case(db: Session, record: FraudPrediction, policy: dict) -> FraudCase:
    """
    Triggers an automated 'Agent Review' (using the LLM) and creates a new
    entry in the FraudCase table for human review.
    """
    review = generate_agent_review(_record_to_dict(record), record.probability, policy, db=db)
    summary, questions, pii_findings = guard_agent_output(review.summary, review.reviewer_questions)
    if pii_findings:
        import logging
        logging.getLogger("uvicorn").warning("PII redacted from agent output: %s", pii_findings)
    case = FraudCase(
        case_id=str(uuid.uuid4()),
        prediction_id=record.id,
        risk_band=policy["risk_band"],
        model_decision=policy["decision"],
        agent_recommendation=review.recommendation,
        agent_confidence=review.confidence,
        reason_codes=review.reason_codes,
        reviewer_questions=questions,
        reasoning=summary,
        status="PENDING_REVIEW",
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.post("/predict", dependencies=[Depends(rate_limit)])
@traceable(name="API: Transaction Prediction")
async def predict(
    # dependency injection
    data: Transaction,
    db: Session = Depends(get_db), # Depends is how fastapi handles "dependencies": here it is for getting a db session
    redis: redis_asyncio.Redis | None = Depends(get_redis), # redis | none means the API can work even if Redis is down
    inference: Any = Depends(get_inference_engine), # Any specifies that the inference engine can be of any type, and we use Depends to load it lazily.
):
    """This is the main endpoint for getting a fraud prediction.
It first checks Redis for a recent duplicate transaction (fast), then checks Postgres (slow), and if it's new, 
it runs the ML model to get a prediction. It also handles the logic for opening review cases if needed."""
    try:
        fingerprint = _input_fingerprint(data)
        
        # 1. Check Redis for recent deduplication (Fast Path).
        # We use a hash (fingerprint) of the transaction data as a key to skip expensive ML inference
        # if the exact same transaction was processed recently.
        if redis:
            try:
                # 'fp:' prefix stands for 'fingerprint'.
                cached_id = await redis.get(f"fp:{fingerprint}")
                if cached_id:
                    # If found in Redis, fetch the full record from DB.
                    existing = db.get(FraudPrediction, int(cached_id))
                    if existing:
                        case = _latest_case_for_prediction(db, existing.id)
                        # Logic for re-opening cases for review if needed...
                        if existing.requires_review and (case is None or case.status != "PENDING_REVIEW"):
                            should_reopen = True
                            if case is not None and case.status in ("APPROVED", "BLOCKED") and case.reviewed_at:
                                from datetime import datetime, timedelta
                                if datetime.utcnow() - case.reviewed_at < timedelta(seconds=5):
                                    should_reopen = False
                            if should_reopen:
                                policy = {
                                    "risk_band": existing.risk_band,
                                    "decision": existing.decision,
                                    "requires_review": True,
                                }
                                case = _open_new_case(db, existing, policy)
                        return _prediction_payload(existing, case)
            except Exception:# Exception is generic, catches any exception that is not caught by more specific exception handlers.
                # If Redis is down or times out, we silently fail and proceed to the DB check.
                # This ensures Redis is an optimization, not a hard dependency for the API.
                pass

        # 2. Check Postgres (Slow Path/Fallback)
        existing = (
            db.query(FraudPrediction)
            .filter(FraudPrediction.input_fingerprint == fingerprint)
            .one_or_none()
        )
        if existing is not None:
            # If found in Postgres but not in Redis, populate the Redis cache for subsequent requests.
            if redis:
                try:
                    # Cache the mapping of fingerprint -> DB record ID for 1 hour (3600 seconds).
                    await redis.setex(f"fp:{fingerprint}", 3600, existing.id)
                except Exception: # Exception is generic, catches any exception that is not caught by more specific exception handlers.
                    pass
            
            case = _latest_case_for_prediction(db, existing.id)
            if existing.requires_review and (case is None or case.status != "PENDING_REVIEW"):
                should_reopen = True
                if case is not None and case.status in ("APPROVED", "BLOCKED") and case.reviewed_at:
                    from datetime import datetime, timedelta
                    if datetime.utcnow() - case.reviewed_at < timedelta(seconds=5):
                        should_reopen = False
                if should_reopen:
                    policy = {
                        "risk_band": existing.risk_band,
                        "decision": existing.decision,
                        "requires_review": True,
                    }
                    case = _open_new_case(db, existing, policy)
            return _prediction_payload(existing, case)

        transaction_dict = data.model_dump()
        result = inference.predict(transaction_dict)
        policy = classify_risk(result["probability"])

        record = FraudPrediction(
            trans_date_trans_time=_parse_datetime(data.trans_date_trans_time),
            amt=data.amt,
            category=data.category,
            merchant=data.merchant,
            lat=data.lat,
            long=data.long,
            merch_lat=data.merch_lat,
            merch_long=data.merch_long,
            city=data.city,
            state=data.state,
            city_pop=data.city_pop,
            dob=_parse_datetime(data.dob),
            gender=data.gender,
            job=data.job,
            is_fraud=result["is_fraud"],
            probability=result["probability"],
            threshold=result["threshold"],
            risk_band=policy["risk_band"],
            decision=policy["decision"],
            requires_review=policy["requires_review"],
            input_fingerprint=fingerprint,
        )

        db.add(record)
        try:
            db.commit()
            db.refresh(record)
            # 3. Add to Redis cache after successful creation
            if redis:
                try:
                    await redis.setex(f"fp:{fingerprint}", 3600, record.id)
                except Exception:
                    pass
        except IntegrityError: # IntegrityError is a subclass of Exception
            db.rollback() # rollback undoes changes in the db if integrity error occurs
            winner = (
                db.query(FraudPrediction)
                .filter(FraudPrediction.input_fingerprint == fingerprint)
                .one()
            )
            # Update Redis cache with the winner
            if redis:
                try:
                    await redis.setex(f"fp:{fingerprint}", 3600, winner.id) # setex sets a key with an expiration time
                except Exception:
                    pass
            
            case = _latest_case_for_prediction(db, winner.id)
            if winner.requires_review and (case is None or case.status != "PENDING_REVIEW"):
                should_reopen = True
                if case is not None and case.status in ("APPROVED", "BLOCKED") and case.reviewed_at:
                    from datetime import datetime, timedelta
                    if datetime.utcnow() - case.reviewed_at < timedelta(seconds=5):
                        should_reopen = False
                if should_reopen:
                    policy = {
                        "risk_band": winner.risk_band,
                        "decision": winner.decision,
                        "requires_review": True,
                    }
                    case = _open_new_case(db, winner, policy)
            payload = _prediction_payload(winner, case)
            await events.broadcast("new_prediction", payload)
            return payload

        case = None
        if policy["requires_review"]:
            case = _open_new_case(db, record, policy)
            record.agent_action = case.agent_recommendation
            record.reasoning = case.reasoning
            db.commit()
            db.refresh(record)

        payload = _prediction_payload(record, case)
        await events.broadcast("new_prediction", payload)
        return payload

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@router.get("/stream")
async def stream_events(request: Request):
    async def event_stream():
        queue = events.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'ping'})}\n\n"
        finally:
            events.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/predictions", response_model=PredictionPage)
async def list_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    total = db.query(FraudPrediction).count()
    items = (
        db.query(FraudPrediction)
        .order_by(FraudPrediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return PredictionPage(items=items, total=total, offset=offset, limit=limit)


@router.get("/predictions/{prediction_id}", response_model=PredictionRecord)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Fetches the details of a single prediction using its ID."""
    record = db.get(FraudPrediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record


@router.get("/cases/pending", response_model=list[FraudCaseRecord])
async def list_pending_cases(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Returns a list of 'Cases' that are currently waiting for a human or agent review.
    These are the ones shown in the 'Pending' tab of the dashboard.
    """
    return (
        db.query(FraudCase)
        .filter(FraudCase.status == "PENDING_REVIEW")
        .order_by(FraudCase.created_at.desc())
        .limit(limit)
        .all()
    )

@router.get("/cases/{case_id}", response_model=FraudCaseRecord)
async def get_case(case_id: str, db: Session = Depends(get_db)):
    """Fetches the details of a single review Case using its unique Case ID (UUID)."""
    fraud_case = db.query(FraudCase).filter(FraudCase.case_id == case_id).first()
    if fraud_case is None:
        raise HTTPException(status_code=404, detail="Fraud case not found")
    return fraud_case


@router.post("/cases/{case_id}/decision", response_model=FraudCaseRecord)
async def decide_case(
    case_id: str,
    decision: HumanDecisionRequest,
    db: Session = Depends(get_db),
):
    """
    This is called when a human reviewer makes a final decision (Approve or Block).
    It updates the Case status and saves the human's notes.
    """
    fraud_case = db.query(FraudCase).filter(FraudCase.case_id == case_id).first()
    if fraud_case is None:
        raise HTTPException(status_code=404, detail="Fraud case not found")
    
    # We can't decide on a case that is already closed.
    if fraud_case.status != "PENDING_REVIEW":
        raise HTTPException(status_code=409, detail="Fraud case has already been reviewed")

    # Update the case with the human's decision.
    fraud_case.human_decision = decision.decision
    fraud_case.human_note = decision.note
    fraud_case.status = "APPROVED" if decision.decision == "APPROVE" else "BLOCKED"
    fraud_case.reviewed_at = datetime.utcnow()

    # Also update the original prediction record with the final human verdict.
    prediction = db.get(FraudPrediction, fraud_case.prediction_id)
    if prediction is not None:
        prediction.human_verdict = decision.decision

    db.commit() # Save changes to the database.
    db.refresh(fraud_case)
    return fraud_case
