import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.db.connections import get_db
from app.db.models import FraudCase, FraudPrediction
from src.agent_review import generate_agent_review
from src.policy import classify_risk


router = APIRouter()

class Transaction(BaseModel):
    trans_date_trans_time: str
    amt: float = Field(gt=0)
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
    model_config = ConfigDict(from_attributes=True)

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
    risk_band: str | None = None
    decision: str | None = None
    requires_review: bool | None = None
    case_id: str | None = None
    agent_recommendation: str | None = None
    agent_summary: str | None = None
    created_at: datetime


class FraudCaseRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: str
    prediction_id: int
    risk_band: str
    model_decision: str
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
    decision: Literal["APPROVE", "BLOCK"]
    note: str | None = None


_engine: Any | None = None


def get_inference_engine() -> Any:
    global _engine
    if _engine is None:
        from src.inference import FraudInference
        base_path = Path(__file__).parent.parent / "model"
        _engine = FraudInference(
            model_path=str(base_path / "fraud_model.json"),
            feature_list_path=str(base_path / "feature_list.pkl"),
        )
    return _engine


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _input_fingerprint(data: Transaction) -> str:
    d = data.model_dump()
    for key in ("amt", "lat", "long", "merch_lat", "merch_long"):
        d[key] = round(float(d[key]), 6)
    d["city_pop"] = int(d["city_pop"])
    for key in ("trans_date_trans_time", "category", "merchant", "city", "state", "dob", "gender", "job"):
        d[key] = str(d[key]).strip()
    ordered = {k: d[k] for k in sorted(d)}
    blob = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _latest_case_for_prediction(db: Session, prediction_id: int) -> FraudCase | None:
    return (
        db.query(FraudCase)
        .filter(FraudCase.prediction_id == prediction_id)
        .order_by(FraudCase.created_at.desc())
        .first()
    )


def _prediction_payload(record: FraudPrediction, case: FraudCase | None = None) -> dict:
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
        "agent_recommendation": case.agent_recommendation if case else None,
        "agent_summary": case.reasoning if case else None,
        "created_at": record.created_at,
    }


def _record_to_dict(record: FraudPrediction) -> dict:
    """Rebuild transaction dict from a saved FraudPrediction for agent review."""
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
    """Run agent review and insert a fresh PENDING_REVIEW FraudCase."""
    review = generate_agent_review(_record_to_dict(record), record.probability, policy)
    case = FraudCase(
        case_id=str(uuid.uuid4()),
        prediction_id=record.id,
        risk_band=policy["risk_band"],
        model_decision=policy["decision"],
        agent_recommendation=review.recommendation,
        agent_confidence=review.confidence,
        reason_codes=review.reason_codes,
        reviewer_questions=review.reviewer_questions,
        reasoning=review.summary,
        status="PENDING_REVIEW",
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.post("/predict", response_model=PredictionRecord)
async def predict_fraud(
    data: Transaction,
    db: Session = Depends(get_db),
    inference: Any = Depends(get_inference_engine),
):
    try:
        fingerprint = _input_fingerprint(data)
        existing = (
            db.query(FraudPrediction)
            .filter(FraudPrediction.input_fingerprint == fingerprint)
            .one_or_none()
        )
        if existing is not None:
            case = _latest_case_for_prediction(db, existing.id)
            # Re-open a case if the previous one was already reviewed
            if existing.requires_review and (case is None or case.status != "PENDING_REVIEW"):
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
        except IntegrityError:
            db.rollback()
            winner = (
                db.query(FraudPrediction)
                .filter(FraudPrediction.input_fingerprint == fingerprint)
                .one()
            )
            case = _latest_case_for_prediction(db, winner.id)
            if winner.requires_review and (case is None or case.status != "PENDING_REVIEW"):
                policy = {
                    "risk_band": winner.risk_band,
                    "decision": winner.decision,
                    "requires_review": True,
                }
                case = _open_new_case(db, winner, policy)
            return _prediction_payload(winner, case)
        db.refresh(record)

        case = None
        if policy["requires_review"]:
            case = _open_new_case(db, record, policy)
            record.agent_action = case.agent_recommendation
            record.reasoning = case.reasoning
            db.commit()
            db.refresh(record)

        return _prediction_payload(record, case)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@router.get("/predictions", response_model=list[PredictionRecord])
async def list_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return (
        db.query(FraudPrediction)
        .order_by(FraudPrediction.created_at.desc())
        .limit(limit)
        .all()
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionRecord)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    record = db.get(FraudPrediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record


@router.get("/cases/pending", response_model=list[FraudCaseRecord])
async def list_pending_cases(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return (
        db.query(FraudCase)
        .filter(FraudCase.status == "PENDING_REVIEW")
        .order_by(FraudCase.created_at.desc())
        .limit(limit)
        .all()
    )

@router.get("/cases/{case_id}", response_model=FraudCaseRecord)
async def get_case(case_id: str, db: Session = Depends(get_db)):
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
    fraud_case = db.query(FraudCase).filter(FraudCase.case_id == case_id).first()
    if fraud_case is None:
        raise HTTPException(status_code=404, detail="Fraud case not found")
    if fraud_case.status != "PENDING_REVIEW":
        raise HTTPException(status_code=409, detail="Fraud case has already been reviewed")

    fraud_case.human_decision = decision.decision
    fraud_case.human_note = decision.note
    fraud_case.status = "APPROVED" if decision.decision == "APPROVE" else "BLOCKED"
    fraud_case.reviewed_at = datetime.utcnow()

    prediction = db.get(FraudPrediction, fraud_case.prediction_id)
    if prediction is not None:
        prediction.human_verdict = decision.decision

    db.commit()
    db.refresh(fraud_case)
    return fraud_case