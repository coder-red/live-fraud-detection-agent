from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.connections import SessionLocal
from app.db.models import FraudCase


def pct(numerator: int, denominator: int) -> float:
    # Small helper so we can print readable percentages safely.
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def main() -> None:
    db = SessionLocal()
    try:
        # These are the cases with final human outcomes, so they become our
        # first ground truth dataset for local evaluation.
        cases = (
            db.query(FraudCase)
            .filter(FraudCase.status.in_(["APPROVED", "BLOCKED"]))
            .filter(FraudCase.human_decision.isnot(None))
            .all()
        )

        total = len(cases)
        model_human_matches = 0
        agent_human_matches = 0
        agent_review_count = 0
        model_agent_disagreements = 0
        missing_agent_recommendation = 0

        for case in cases:
            # How often did the policy/model-backed case decision match the
            # eventual human outcome?
            if case.model_decision == case.human_decision:
                model_human_matches += 1

            if case.agent_recommendation is None:
                missing_agent_recommendation += 1
            else:
                # Exact-match is a strict metric. It will count REVIEW as a
                # miss when the human later picks APPROVE or BLOCK.
                if case.agent_recommendation == case.human_decision:
                    agent_human_matches += 1

                # This tells us how often the agent is escalating instead of
                # making a final binary call.
                if case.agent_recommendation == "REVIEW":
                    agent_review_count += 1

                # Useful for seeing whether the agent is mostly repeating the
                # model decision or diverging from it.
                if case.model_decision != case.agent_recommendation:
                    model_agent_disagreements += 1

        print("Fraud Eval Results")
        print("==================")
        print(f"Reviewed cases: {total}")
        print(f"Model-Human matches: {model_human_matches} / {total}")
        print(f"Model-Human match rate: {pct(model_human_matches, total):.2f}%")
        print(f"Agent-Human matches: {agent_human_matches} / {total}")
        print(f"Agent-Human match rate: {pct(agent_human_matches, total):.2f}%")
        print(f"Agent REVIEW count: {agent_review_count}")
        print(f"Model-Agent disagreements: {model_agent_disagreements}")
        print(f"Cases missing agent recommendation: {missing_agent_recommendation}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
