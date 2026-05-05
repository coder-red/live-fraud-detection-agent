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
    if not os.getenv("GROQ_API_KEY"):
        return _fallback_review(transaction, probability, policy)

    prompt = f"""You are a fraud operations analyst. Return only valid JSON matching this schema:
{{
  "recommendation": "APPROVE" | "REVIEW" | "BLOCK",
  "confidence": number between 0 and 1,
  "reason_codes": ["HIGH_AMOUNT", "NIGHT_TRANSACTION", "HIGH_RISK_CATEGORY", "LONG_DISTANCE_MERCHANT", "MODEL_HIGH_CONFIDENCE"],
  "summary": "short reviewer-facing explanation",
  "reviewer_questions": ["short question for human reviewer"]
}}

Model probability: {probability}
Policy decision: {policy}
Transaction: {transaction}

Use the policy decision as the main control. Do not invent facts not in the transaction.
"""

    try:
        response = _get_llm().invoke(prompt).content
        return AgentReview.model_validate(json.loads(response))
    except (json.JSONDecodeError, ValidationError, Exception):
        return _fallback_review(transaction, probability, policy)