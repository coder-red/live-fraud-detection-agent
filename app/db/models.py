from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.connections import Base


# ORM model = Python class mapped to one database table.
class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    trans_date_trans_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    amt: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(120), nullable=False)
    merchant: Mapped[str] = mapped_column(String(255), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    long: Mapped[float] = mapped_column(Float, nullable=False)
    merch_lat: Mapped[float] = mapped_column(Float, nullable=False)
    merch_long: Mapped[float] = mapped_column(Float, nullable=False)
    city: Mapped[str] = mapped_column(String(120), nullable=False)
    # "state" here means region/state code, like NY or CA.
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    city_pop: Mapped[int] = mapped_column(Integer, nullable=False)
    dob: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    job: Mapped[str] = mapped_column(String(255), nullable=False)
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    requires_review: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    agent_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class FraudCase(Base):
    __tablename__ = "fraud_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("fraud_predictions.id"), nullable=False)
    risk_band: Mapped[str] = mapped_column(String(32), nullable=False)
    model_decision: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_recommendation: Mapped[str | None] = mapped_column(String(32), nullable=True)
    agent_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    reviewer_questions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    human_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
