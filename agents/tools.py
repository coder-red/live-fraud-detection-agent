# agents/tools.py
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from sqlalchemy.orm import Session
from app.db.models import FraudPrediction
from langsmith import traceable

@traceable(name="DB: Merchant Fraud History")
def check_merchant_fraud_history(db: Session, merchant: str) -> dict:
    """How risky is this merchant based on past transactions?"""
    total = db.query(FraudPrediction).filter(
        FraudPrediction.merchant == merchant
    ).count()
    
    fraud_count = db.query(FraudPrediction).filter(
        FraudPrediction.merchant == merchant,
        FraudPrediction.is_fraud == True
    ).count()
    
    rate = (fraud_count / total * 100) if total > 0 else 0
    return {
        "merchant": merchant,
        "total_transactions": total,
        "prior_fraud_count": fraud_count,
        "fraud_rate_pct": round(rate, 1)
    }


@traceable(name="DB: Transaction Velocity Check")
def check_velocity(db: Session, city: str, state: str, window_minutes: int = 60) -> dict:
    """How many transactions from this location in the last N minutes?"""
    since = datetime.utcnow() - timedelta(minutes=window_minutes)
    
    recent = db.query(FraudPrediction).filter(
        FraudPrediction.city == city,
        FraudPrediction.state == state,
        FraudPrediction.trans_date_trans_time >= since
    ).count()
    
    return {
        "location": f"{city}, {state}",
        "window_minutes": window_minutes,
        "transaction_count": recent,
        "high_velocity": recent > 3
    }


@traceable(name="DB: Geographic Anomaly Check")
def check_geo_anomaly(db: Session, merchant: str, merch_lat: float, merch_long: float) -> dict:
    """Is this merchant location consistent with where it's been seen before?"""
    prior = db.query(FraudPrediction).filter(
        FraudPrediction.merchant == merchant
    ).order_by(FraudPrediction.created_at.desc()).limit(10).all()
    
    if not prior:
        return {"anomaly": False, "reason": "No prior merchant locations on record"}
    
    # Haversine distance to last known merchant location
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distances = [haversine(p.merch_lat, p.merch_long, merch_lat, merch_long) for p in prior]
    avg_distance_km = sum(distances) / len(distances)
    
    return {
        "avg_distance_from_known_location_km": round(avg_distance_km, 1),
        "anomaly": avg_distance_km > 500,
        "reason": f"Merchant seen {round(avg_distance_km)}km from its usual location" if avg_distance_km > 500 else "Location consistent with history"
    }