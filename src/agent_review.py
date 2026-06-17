from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError
from langsmith import traceable

load_dotenv()  # Load LangSmith and Groq environment variables early.

Recommendation = Literal["APPROVE", "REVIEW", "BLOCK"]


class EvidenceItem(BaseModel):
    source: str
    claim: str
    evidence: str


class AgentReview(BaseModel):
    recommendation: Recommendation
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str]
    summary: str
    reviewer_questions: list[str]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    verification_context: dict[str, Any] = Field(default_factory=dict)


SUSPICIOUS_PROMPT_PATTERNS = (
    r"\bignore (all|any|the) previous instructions\b",
    r"\bdisregard (all|any|the) previous instructions\b",
    r"\b(system prompt|developer message|hidden prompt)\b",
    r"\bdo not (follow|use) the rules\b",
    r"\breturn only\b.*\bjson\b",
    r"\bprint\b.*\bsecret\b",
)


def _is_suspicious_text(value: str) -> bool:
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in SUSPICIOUS_PROMPT_PATTERNS)


def _sanitize_transaction(transaction: dict) -> tuple[dict, dict[str, Any]]:
    sanitized: dict[str, Any] = {}
    flagged_fields: list[str] = []

    for key, value in transaction.items():
        if isinstance(value, str) and _is_suspicious_text(value):
            sanitized[key] = "[redacted suspicious content]"
            flagged_fields.append(key)
        else:
            sanitized[key] = value

    return sanitized, {
        "detected": bool(flagged_fields),
        "flagged_fields": flagged_fields,
        "action": "redacted" if flagged_fields else "none",
    }


def _build_prompt_safety_note(prompt_safety: dict[str, Any]) -> str:
    if not prompt_safety.get("detected"):
        return "No suspicious instruction-like content detected in transaction fields."
    flagged = ", ".join(prompt_safety.get("flagged_fields", []))
    return (
        "Potential prompt injection detected in transaction fields: "
        f"{flagged}. Sensitive text was redacted before model invocation."
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


def _get_db_tools(db: Any) -> dict:
    """Get database analysis tools that the agent can use."""
    from agents.tools import (
        check_geo_anomaly,
        check_merchant_fraud_history,
        check_velocity,
    )

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


def _build_cove_evidence(
    transaction: dict,
    probability: float,
    policy: dict,
    db_context: dict[str, Any],
) -> list[dict[str, str]]:
    # The bundle is deterministic so the judge can validate the review against
    # concrete facts instead of inferring support from the prose summary.
    amount = float(transaction.get("amt", 0) or 0)
    category = str(transaction.get("category", "unknown"))
    merchant = str(transaction.get("merchant", "unknown"))
    city = str(transaction.get("city", "unknown"))
    state = str(transaction.get("state", "unknown"))
    trans_time = str(transaction.get("trans_date_trans_time", "unknown"))

    evidence: list[dict[str, str]] = [
        {
            "source": "policy",
            "claim": f"Policy assigned {policy['decision']} in the {policy['risk_band']} band.",
            "evidence": (
                f"probability={probability:.4f}, "
                f"requires_review={policy['requires_review']}"
            ),
        },
        {
            "source": "transaction",
            "claim": f"Transaction amount is ${amount:,.2f} in category {category}.",
            "evidence": (
                f"merchant={merchant}, city={city}, state={state}, timestamp={trans_time}"
            ),
        },
    ]

    merchant_history = db_context.get("merchant_fraud_history") or {}
    if merchant_history:
        evidence.append(
            {
                "source": "merchant_fraud_history",
                "claim": (
                    "Merchant fraud rate is "
                    f"{merchant_history.get('fraud_rate_pct', 0)}% "
                    f"across {merchant_history.get('prior_fraud_count', 0)} prior fraud cases."
                ),
                "evidence": (
                    f"{merchant_history.get('prior_fraud_count', 0)} prior fraud cases "
                    f"across {merchant_history.get('total_transactions', 0)} total transactions."
                ),
            }
        )

    velocity = db_context.get("velocity_check") or {}
    if velocity:
        evidence.append(
            {
                "source": "velocity_check",
                "claim": (
                    f"{velocity.get('transaction_count', 0)} transactions were observed "
                    f"from {velocity.get('location', 'unknown')} in the last "
                    f"{velocity.get('window_minutes', 60)} minutes."
                ),
                "evidence": f"high_velocity={velocity.get('high_velocity', False)}",
            }
        )

    geo_anomaly = db_context.get("geo_anomaly") or {}
    if geo_anomaly:
        evidence.append(
            {
                "source": "geo_anomaly_check",
                "claim": (
                    "Merchant location is "
                    f"{'anomalous' if geo_anomaly.get('anomaly') else 'consistent'} "
                    "with prior observations."
                ),
                "evidence": str(
                    geo_anomaly.get("reason", "No geographic anomaly details available")
                ),
            }
        )

    return evidence


def build_verification_context(
    transaction: dict,
    probability: float,
    policy: dict,
    db: Any = None,
    db_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Shared CoVe context for both the agent and the evaluator.
    if db_context is None and db is not None:
        db_context = _query_db_tools(db, transaction)

    db_context = db_context or {}
    sanitized_transaction, prompt_safety = _sanitize_transaction(transaction)

    return {
        "transaction": sanitized_transaction,
        "probability": probability,
        "policy": policy,
        "db_context": db_context,
        "prompt_safety": prompt_safety,
        "evidence": _build_cove_evidence(transaction, probability, policy, db_context),
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


def _format_evidence_table(evidence: list[dict[str, str]]) -> str:
    rows = ["| Source | Claim | Evidence |", "| --- | --- | --- |"]
    for item in evidence:
        source = item["source"].replace("|", "\\|")
        claim = item["claim"].replace("|", "\\|")
        details = item["evidence"].replace("|", "\\|")
        rows.append(f"| {source} | {claim} | {details} |")
    return "\n".join(rows)


def _build_summary(
    *,
    risk_band: str,
    category: str,
    amount: float,
    decision: str,
    evidence: list[dict[str, str]],
    prompt_safety: dict[str, Any] | None = None,
) -> str:
    # Keep the prose concise, but make the supporting facts easy to scan.
    evidence_table = _format_evidence_table(evidence)
    safety_note = ""
    if prompt_safety and prompt_safety.get("detected"):
        flagged = ", ".join(prompt_safety.get("flagged_fields", []))
        safety_note = (
            f"\n### Safety Note\n"
            f"- Suspicious instruction-like content was detected in: `{flagged}`\n"
            f"- The transaction text was redacted before model invocation\n"
        )
    return (
        f"### Decision Summary\n"
        f"- Risk band: `{risk_band}`\n"
        f"- Category: `{category}`\n"
        f"- Amount: `${amount:,.2f}`\n"
        f"- Recommendation: `{decision}`\n\n"
        f"{safety_note}"
        f"### Evidence\n"
        f"{evidence_table}\n"
    )


def _fallback_review(
    transaction: dict,
    probability: float,
    policy: dict,
    verification_context: dict[str, Any] | None = None,
) -> AgentReview:
    verification_context = verification_context or build_verification_context(
        transaction,
        probability,
        policy,
    )
    evidence = verification_context["evidence"]
    codes = _reason_codes(transaction, probability, policy["risk_band"])
    amount = float(transaction.get("amt", 0) or 0)
    category = transaction.get("category", "unknown")
    decision = policy["decision"]

    return AgentReview(
        recommendation=decision,
        confidence=min(max(probability, 0.0), 1.0),
        reason_codes=codes,
        summary=_build_summary(
            risk_band=policy["risk_band"],
            category=category,
            amount=amount,
            decision=decision,
            evidence=evidence,
            prompt_safety=verification_context.get("prompt_safety"),
        ),
        reviewer_questions=[
            "Does this customer have similar recent transaction behavior?",
            "Can the customer confirm the merchant and amount?",
        ],
        evidence=evidence,
        verification_context=verification_context,
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
    db: Any = None,
) -> AgentReview:
    groq_key = os.getenv("GROQ_API_KEY")
    verification_context = build_verification_context(
        transaction,
        probability,
        policy,
        db=db,
    )
    prompt_safety = verification_context["prompt_safety"]
    sanitized_transaction = verification_context["transaction"]
    if not groq_key:
        print("[agent_review] GROQ_API_KEY not set, using fallback review")
        return _fallback_review(
            sanitized_transaction,
            probability,
            policy,
            verification_context=verification_context,
        )

    db_context = verification_context["db_context"]
    db_context_str = (
        json.dumps(db_context, indent=2) if db_context else "No database context available"
    )
    evidence_str = json.dumps(verification_context["evidence"], indent=2)

    prompt = f"""You are a senior fraud operations analyst with access to real-time database analytics.
Your analysis must be specific, evidence-based, and actionable. Return only valid JSON matching this schema:
{{
  "recommendation": "APPROVE" | "REVIEW" | "BLOCK",
  "confidence": number between 0 and 1,
  "reason_codes": ["HIGH_AMOUNT", "NIGHT_TRANSACTION", "HIGH_RISK_CATEGORY", "LONG_DISTANCE_MERCHANT", "MODEL_HIGH_CONFIDENCE", "VELOCITY_CHECK", "GEOGRAPHIC_ANOMALY", "MERCHANT_FRAUD_HISTORY"],
  "summary": "Return a concise markdown summary with a short decision section and an evidence table.",
  "reviewer_questions": ["Specific next best action for human reviewer"]
}}

CONTEXT:
- Model probability: {probability:.4f} ({probability:.1%})
- Policy decision: {policy}
- Risk band thresholds: LOW (<25%), MEDIUM (25-54%), HIGH (55-89%), CRITICAL (>=90%)
- Transaction details: {sanitized_transaction}
- Prompt safety note: {_build_prompt_safety_note(prompt_safety)}

VERIFICATION BUNDLE:
{evidence_str}

DATABASE INSIGHTS:
{db_context_str}

INSTRUCTIONS:
1. Start your summary by stating which policy threshold was triggered and why.
2. Reference specific transaction data points (amount, time, location, category).
3. Use the verification bundle to support every claim.
4. Cite evidence sources inline with the exact source names, such as [policy], [transaction], [merchant_fraud_history], [velocity_check], or [geo_anomaly_check].
5. If the merchant has a high fraud rate in the DB, emphasize this.
6. If there's high transaction velocity from the location, flag it.
7. If there's a geographic anomaly (merchant far from usual location), highlight it.
8. Provide actionable next steps, not generic questions.
9. Be concise but specific - you're writing for a fraud analyst, not a customer.

Example of good analysis:
"CRITICAL risk band triggered at 97% probability, exceeding the 90% BLOCK threshold [policy]. $1,433 transaction at 'fraud-store-net' in Boston at 01:28 AM [transaction]. This merchant has an 87% fraud rate across 23 prior transactions [merchant_fraud_history]. High velocity detected: 5 transactions from this location in the last hour [velocity_check]. Geographic anomaly: merchant is 500km from its usual location [geo_anomaly_check]. Recommend immediate block."

Do not invent facts not in the transaction or database. Use "unknown" for missing data.
"""

    try:
        response = _get_llm().invoke(prompt).content

        # Robust JSON extraction from markdown if present.
        cleaned = response.strip()
        if cleaned.startswith("```"):
            import re

            match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        payload = json.loads(cleaned)
        # Attach the shared verification bundle so the judge can score groundedness.
        payload["evidence"] = verification_context["evidence"]
        payload["verification_context"] = verification_context
        return AgentReview.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        print(f"[agent_review] LLM failed: {type(e).__name__}: {e}")
        return _fallback_review(
            sanitized_transaction,
            probability,
            policy,
            verification_context=verification_context,
        )
