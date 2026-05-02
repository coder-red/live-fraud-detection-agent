from src.agent_review import generate_agent_review


def test_generate_agent_review_falls_back_without_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    review = generate_agent_review(
        transaction={"amt": 750, "category": "shopping_net"},
        probability=0.93,
        policy={"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True},
    )

    assert review.recommendation == "BLOCK"
    assert review.confidence == 0.93
    assert "HIGH_AMOUNT" in review.reason_codes
    assert "HIGH_RISK_CATEGORY" in review.reason_codes
    assert "MODEL_HIGH_CONFIDENCE" in review.reason_codes
    assert "CRITICAL_RISK_BAND" in review.reason_codes
    assert review.reviewer_questions
