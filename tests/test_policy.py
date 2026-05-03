import pytest  # This gives us pytest tools, including running one test with many inputs.

from src.policy import classify_risk  # This is the function we want to test.


@pytest.mark.parametrize(  # This tells pytest to run the same test several times.
    ("probability", "expected"),
    [
        (0.91, {"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True}),  # Above 0.90 should be critical.
        (0.90, {"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True}),  # Exactly 0.90 should also be critical.
        (0.89, {"risk_band": "HIGH", "decision": "REVIEW", "requires_review": True}),  # Just below 0.90 should be high.
        (0.55, {"risk_band": "HIGH", "decision": "REVIEW", "requires_review": True}),  # Exactly 0.55 should be high.
        (0.54, {"risk_band": "MEDIUM", "decision": "APPROVE", "requires_review": False}),  # Just below 0.55 should be medium.
        (0.25, {"risk_band": "MEDIUM", "decision": "APPROVE", "requires_review": False}),  # Exactly 0.25 should be medium.
        (0.24, {"risk_band": "LOW", "decision": "APPROVE", "requires_review": False}),  # Just below 0.25 should be low.
    ],
)
def test_classify_risk_thresholds(probability, expected):  # This checks the policy boundary values.
    assert classify_risk(probability) == expected  # The real output must match the expected output.


@pytest.mark.parametrize("probability", [-0.01, 1.01])  # This tests invalid probabilities outside the 0 to 1 range.
def test_classify_risk_rejects_invalid_probability(probability):  # INVALID CASE: fraud probabilities cannot be below 0 or above 1.
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):  # The policy should fail clearly.
        classify_risk(probability)  # This sends the bad probability into the policy function.
