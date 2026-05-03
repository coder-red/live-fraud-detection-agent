import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.db.connections import get_db
from app.db.models import FraudCase, FraudPrediction
from src.agent_review import generate_agent_review
from src.policy import classify_risk


router = APIRouter()

# this defines the request body so when someone calls POST /predict, FastAPI reads the JSON body and turns it into a Transaction object.
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

# this defines the response shape
class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True) # this tells Pydantic it can build the response from an ORM object, not just a plain dict

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
    """Load the model lazily so tests and app startup can override it cleanly."""
    global _engine
    if _engine is None:
        from src.inference import FraudInference

        base_path = Path(__file__).parent.parent / "model"
        _engine = FraudInference(
            model_path=str(base_path / "fraud_model.json"),
            feature_list_path=str(base_path / "feature_list.pkl"),
        )
    return _engine

# this function converts the ISO datetime string from the request into a Python datetime object that we can save in Postgres. FastAPI doesn't do this automatically because the input is a string, not a datetime, but our DB model expects a datetime.
def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _prediction_payload(record: FraudPrediction, case: FraudCase | None = None) -> dict:
    """Build an API response that includes case metadata when it exists."""
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


# router.post tells FastAPI: “the function below handles POST requests to /predict”, while response_model=PredictionRecord tells FastAPI how to shape the output
@router.post("/predict", response_model=PredictionRecord)
async def predict_fraud(
    data: Transaction,
    db: Session = Depends(get_db),
    inference: Any = Depends(get_inference_engine),
):
    try:
        transaction_dict = data.model_dump()
        result = inference.predict(transaction_dict)
        policy = classify_risk(result["probability"])

        # Build one DB row from the input transaction plus model output.
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
        )

        # Commit the prediction first so the review case can reference its ID.
        db.add(record)
        db.commit()
        db.refresh(record)

        case = None
        if policy["requires_review"]:
            # The deterministic policy opens the case; the LLM only summarizes
            # evidence and recommends a review action for human operators.
            review = generate_agent_review(transaction_dict, result["probability"], policy)
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
            record.agent_action = review.recommendation
            record.reasoning = review.summary
            db.add(case)
            db.commit()
            db.refresh(case)
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
    # Return the newest saved predictions first.
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
    # This is the review queue a dashboard would poll for human decisions.
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

    # Human decision closes the case. The model and agent remain as evidence,
    # but the reviewer owns the final operational verdict.
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
