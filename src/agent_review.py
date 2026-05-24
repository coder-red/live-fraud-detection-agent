import json
import os
from functools import lru_cache
from typing import Any, Literal
from dotenv import load_dotenv

load_dotenv() # Ensure LangSmith/Groq env vars are loaded

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError
from langsmith import traceable


Recommendation = Literal["APPROVE", "REVIEW", "BLOCK"]


class AgentReview(BaseModel):
    recommendation: Recommendation
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str]
    summary: str
    reviewer_questions: list[str]


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


def _get_db_tools(db: Any) -> dict:
    """Get database analysis tools that the agent can use."""
    from agents.tools import check_merchant_fraud_history, check_velocity, check_geo_anomaly
    
    def merchant_fraud_tool(merchant: str) -> dict:
        """Check merchant's fraud history in the database."""
        return check_merchant_fraud_history(db, merchant)
    
    def velocity_tool(city: str, state: str, window_minutes: int = 60) -> dict:
        """Check transaction velocity from a location."""
        return check_velocity(db, city, state, window_minutes)
    
    def geo_anomaly_tool(merchant: str, merch_lat: float, merch_long: float) -> dict:
        """Check if merchant location is consistent with history."""
        return check_geo_anomaly(db, merchant, merch_lat, merch_long)
    
    return {
        "merchant_fraud_history": merchant_fraud_tool,
        "velocity_check": velocity_tool,
        "geo_anomaly_check": geo_anomaly_tool,
    }


def _reason_codes(transaction: dict, probability: float, risk_band: str) -> list[str]:
    codes: list[str] = []
    amount = float(transaction.get("amt", 0) or 0)
    category = str(transaction.get("category", "")).lower()

    if amount >= 500:
        codes.append("HIGH_AMOUNT")
    if category in {"shopping_net", "misc_net", "grocery_pos"}:
        codes.append("HIGH_RISK_CATEGORY")
    if probability >= 0.90:
        codes.append("MODEL_HIGH_CONFIDENCE")
    if risk_band in {"HIGH", "CRITICAL"}:
        codes.append(f"{risk_band}_RISK_BAND")

    return codes or ["MODEL_RISK_SIGNAL"]


def _fallback_review(transaction: dict, probability: float, policy: dict) -> AgentReview:
    codes = _reason_codes(transaction, probability, policy["risk_band"])
    amount = float(transaction.get("amt", 0) or 0)
    category = transaction.get("category", "unknown")
    decision = policy["decision"]

    return AgentReview(
        recommendation=decision,
        confidence=min(max(probability, 0.0), 1.0),
        reason_codes=codes,
        summary=(
            f"{policy['risk_band']} risk transaction in {category} for "
            f"${amount:,.2f}; policy recommends {decision}."
        ),
        reviewer_questions=[
            "Does this customer have similar recent transaction behavior?",
            "Can the customer confirm the merchant and amount?",
        ],
    )


def _query_db_tools(db: Any, transaction: dict) -> dict:
    """Query database tools to get additional context for the agent."""
    try:
        tools = _get_db_tools(db)
        
        merchant = transaction.get("merchant", "")
        city = transaction.get("city", "")
        state = transaction.get("state", "")
        merch_lat = transaction.get("merch_lat", 0.0)
        merch_long = transaction.get("merch_long", 0.0)
        
        db_context = {
            "merchant_fraud_history": tools["merchant_fraud_history"](merchant),
            "velocity_check": tools["velocity_check"](city, state),
            "geo_anomaly": tools["geo_anomaly_check"](merchant, merch_lat, merch_long),
        }
        return db_context
    except Exception as e:
        print(f"[agent_review] DB tool error: {e}")
        return {}


@traceable(name="Fraud Agent Reasoning")
def generate_agent_review(
    transaction: dict, 
    probability: float, 
    policy: dict,
    db: Any = None
) -> AgentReview:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print(f"[agent_review] GROQ_API_KEY not set, using fallback review")
        return _fallback_review(transaction, probability, policy)

    # Query database tools for additional context if DB is available
    db_context = {}
    if db is not None:
        db_context = _query_db_tools(db, transaction)

    db_context_str = json.dumps(db_context, indent=2) if db_context else "No database context available"

    prompt = f"""You are a senior fraud operations analyst with access to real-time database analytics. Your analysis must be specific, evidence-based, and actionable. Return only valid JSON matching this schema:
{{
  "recommendation": "APPROVE" | "REVIEW" | "BLOCK",
  "confidence": number between 0 and 1,
  "reason_codes": ["HIGH_AMOUNT", "NIGHT_TRANSACTION", "HIGH_RISK_CATEGORY", "LONG_DISTANCE_MERCHANT", "MODEL_HIGH_CONFIDENCE", "VELOCITY_CHECK", "GEOGRAPHIC_ANOMALY", "MERCHANT_FRAUD_HISTORY"],
  "summary": "Expert analysis referencing specific data points, policy thresholds, behavioral patterns, AND database insights. Mention which policy threshold was hit and any relevant DB findings.",
  "reviewer_questions": ["Specific next best action for human reviewer"]
}}

CONTEXT:
- Model probability: {probability:.4f} ({probability:.1%})
- Policy decision: {policy}
- Risk band thresholds: LOW (<25%), MEDIUM (25-54%), HIGH (55-89%), CRITICAL (≥90%)
- Transaction details: {transaction}

DATABASE INSIGHTS:
{db_context_str}

INSTRUCTIONS:
1. Start your summary by stating which policy threshold was triggered and why.
2. Reference specific transaction data points (amount, time, location, category).
3. **Use the DATABASE INSIGHTS** to add context about merchant fraud history, transaction velocity, and geographic anomalies.
4. If the merchant has a high fraud rate in the DB, emphasize this.
5. If there's high transaction velocity from the location, flag it.
6. If there's a geographic anomaly (merchant far from usual location), highlight it.
7. Provide actionable next steps, not generic questions.
8. Be concise but specific — you're writing for a fraud analyst, not a customer.

Example of good analysis:
"CRITICAL risk band triggered at 97% probability, exceeding 90% BLOCK threshold. $1,433 transaction at 'fraud-store-net' in Boston at 01:28 AM. DATABASE ALERT: This merchant has an 87% fraud rate across 23 prior transactions. High velocity detected: 5 transactions from this location in the last hour. Geographic anomaly: merchant is 500km from its usual location. Recommend immediate block."

Do not invent facts not in the transaction or database. Use "unknown" for missing data.
"""

    try:
        response = _get_llm().invoke(prompt).content
        
        # Robust JSON extraction from markdown if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            import re
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                
        return AgentReview.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        print(f"[agent_review] LLM failed: {type(e).__name__}: {e}")
        return _fallback_review(transaction, probability, policy)
