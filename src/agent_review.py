import json
import os
from functools import lru_cache
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError


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


def generate_agent_review(transaction: dict, probability: float, policy: dict) -> AgentReview:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print(f"[agent_review] GROQ_API_KEY not set, using fallback review")
        return _fallback_review(transaction, probability, policy)

    prompt = f"""You are a senior fraud operations analyst. Your analysis must be specific, evidence-based, and actionable. Return only valid JSON matching this schema:
{{
  "recommendation": "APPROVE" | "REVIEW" | "BLOCK",
  "confidence": number between 0 and 1,
  "reason_codes": ["HIGH_AMOUNT", "NIGHT_TRANSACTION", "HIGH_RISK_CATEGORY", "LONG_DISTANCE_MERCHANT", "MODEL_HIGH_CONFIDENCE", "VELOCITY_CHECK", "GEOGRAPHIC_ANOMALY"],
  "summary": "Expert analysis referencing specific data points, policy thresholds, and behavioral patterns. Mention which policy threshold was hit (e.g., 'CRITICAL risk band triggered at {probability:.0%} probability, exceeding 90% BLOCK threshold').",
  "reviewer_questions": ["Specific next best action for human reviewer, e.g., 'Outbound call required: Verify if customer is currently traveling' or 'Check for recent data breaches affecting this merchant category'"]
}}

CONTEXT:
- Model probability: {probability:.4f} ({probability:.1%})
- Policy decision: {policy}
- Risk band thresholds: LOW (<25%), MEDIUM (25-54%), HIGH (55-89%), CRITICAL (≥90%)
- Transaction details: {transaction}

INSTRUCTIONS:
1. Start your summary by stating which policy threshold was triggered and why.
2. Reference specific transaction data points (amount, time, location, category).
3. Compare against typical customer behavior patterns when possible.
4. Provide actionable next steps, not generic questions.
5. Be concise but specific — you're writing for a fraud analyst, not a customer.

Example of good analysis:
"CRITICAL risk band triggered at 97% probability, exceeding 90% BLOCK threshold. $1,433 transaction at 'fraud-store-net' in Boston at 01:28 AM — 4.2x above customer's 30-day average. Nighttime misc_net activity deviates from historic Miami-based profile. Velocity check: 3rd transaction in 2 hours."

Do not invent facts not in the transaction. Use "unknown" for missing data.
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