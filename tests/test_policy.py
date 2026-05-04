import pytest  # This gives us pytest tools, including running one test with many inputs.

from src.policy import classify_risk  # This is the function we want to test.


@pytest.mark.parametrize(  # This tells pytest to run the same test several times.
    ("probability", "expected"),
    [
        (0.91, {"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True}),  # 🟡 EDGE CASE (boundary above 0.90)
        (0.90, {"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True}),  # 🟡 EDGE CASE (exact boundary 0.90)
        (0.89, {"risk_band": "HIGH", "decision": "REVIEW", "requires_review": True}),      # 🟡 EDGE CASE (just below 0.90)
        (0.55, {"risk_band": "HIGH", "decision": "REVIEW", "requires_review": True}),      # 🟡 EDGE CASE (upper boundary of HIGH)
        (0.54, {"risk_band": "MEDIUM", "decision": "APPROVE", "requires_review": False}),  # 🟡 EDGE CASE (just below 0.55)
        (0.25, {"risk_band": "MEDIUM", "decision": "APPROVE", "requires_review": False}),  # 🟡 EDGE CASE (boundary of MEDIUM)
        (0.24, {"risk_band": "LOW", "decision": "APPROVE", "requires_review": False}),     # 🟡 EDGE CASE (just below 0.25 → LOW)
    ],
)
def test_classify_risk_thresholds(probability, expected):  # 🟡 EDGE CASE TEST: boundary thresholds for policy rules
    assert classify_risk(probability) == expected  # Output must match expected policy decision.


@pytest.mark.parametrize("probability", [-0.01, 1.01])  # 🔴 INVALID CASE: probabilities outside valid range [0, 1]
def test_classify_risk_rejects_invalid_probability(probability):  # 🔴 INVALID CASE TEST
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):  # Expect clean failure
        classify_risk(probability)  # Send invalid input into policy logic