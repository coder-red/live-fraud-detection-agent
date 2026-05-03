from src.agent_review import generate_agent_review  


def test_generate_agent_review_falls_back_without_groq_key(monkeypatch):  # This checks the no-LLM fallback path.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # This deletes the Groq key only for this test, monkeypatch is pytest tool for modifying env temporarily

    review = generate_agent_review(
        transaction={"amt": 750, "category": "shopping_net"},
        probability=0.93,  # This is a high model fraud probability.
        policy={"risk_band": "CRITICAL", "decision": "BLOCK", "requires_review": True},
    )

    assert review.recommendation == "BLOCK"  # The fallback should follow the policy decision.
    assert review.confidence == 0.93  # The fallback confidence should match the model probability.
    assert "HIGH_AMOUNT" in review.reason_codes  # The fallback should notice the high amount.
    assert "HIGH_RISK_CATEGORY" in review.reason_codes 
    assert "MODEL_HIGH_CONFIDENCE" in review.reason_codes 
    assert "CRITICAL_RISK_BAND" in review.reason_codes 
    assert review.reviewer_questions  # The fallback should give the human reviewer questions to ask.
