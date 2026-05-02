import pytest

from src.policy import classify_risk


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.91, {"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True}),
        (0.90, {"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True}),
        (0.89, {"risk_band": "HIGH", "decision": "REVIEW", "requires_review": True}),
        (0.55, {"risk_band": "HIGH", "decision": "REVIEW", "requires_review": True}),
        (0.54, {"risk_band": "MEDIUM", "decision": "APPROVE", "requires_review": False}),
        (0.25, {"risk_band": "MEDIUM", "decision": "APPROVE", "requires_review": False}),
        (0.24, {"risk_band": "LOW", "decision": "APPROVE", "requires_review": False}),
    ],
)
def test_classify_risk_thresholds(probability, expected):
    assert classify_risk(probability) == expected
