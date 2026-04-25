from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column # mapped is used for type hinting, mapped_column is used to define columns in the model

from app.db.connections import Base # Base is the shared registry for our ORM models, it is imported from connections.py to ensure all models are registered correctly with SQLAlchemy



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
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    city_pop: Mapped[int] = mapped_column(Integer, nullable=False)
    dob: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    job: Mapped[str] = mapped_column(String(255), nullable=False)
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    agent_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
