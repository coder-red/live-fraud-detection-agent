from typing import Literal, TypedDict


Decision = Literal["APPROVE", "REVIEW", "BLOCK"]
RiskBand = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class PolicyDecision(TypedDict):
    risk_band: RiskBand
    decision: Decision
    requires_review: bool


def classify_risk(probability: float) -> PolicyDecision:
    """Turn a model probability into an auditable business decision.

    The model estimates risk; this policy decides what the product should do.
    Keeping this deterministic makes decisions explainable and easy to tune
    without retraining the model.
    """
    if probability >= 0.90:
        return {
            "risk_band": "CRITICAL",
            "decision": "BLOCK",
            "requires_review": True,
        }

    if probability >= 0.55:
        return {
            "risk_band": "HIGH",
            "decision": "REVIEW",
            "requires_review": True,
        }

    if probability >= 0.25:
        return {
            "risk_band": "MEDIUM",
            "decision": "APPROVE",
            "requires_review": False,
        }

    return {
        "risk_band": "LOW",
        "decision": "APPROVE",
        "requires_review": False,
    }
