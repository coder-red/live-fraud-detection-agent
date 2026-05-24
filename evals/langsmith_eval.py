from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "false")

try:
    from langsmith import Client, traceable, tracing_context
except ImportError as exc:  # pragma: no cover - user-facing setup error
    raise SystemExit(
        "LangSmith is not installed. Run `uv add langsmith` or "
        "`pip install langsmith>=0.3.13` first."
    ) from exc

from langchain_core.tracers.langchain import wait_for_all_tracers

from app.db.connections import SessionLocal
from app.db.models import FraudCase, FraudPrediction
from src.agent_review import generate_agent_review


def parse_args() -> argparse.Namespace:
    # Keep the first interface small: choose a dataset name, a case limit,
    # and whether you want upload-only mode.
    parser = argparse.ArgumentParser(
        description="Create a LangSmith dataset from reviewed fraud cases and run an eval."
    )
    parser.add_argument(
        "--dataset-name",
        default=f"fraud-reviewed-cases-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="LangSmith dataset name to create.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of reviewed fraud cases to upload and evaluate.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only create the LangSmith dataset, do not run the eval experiment.",
    )
    return parser.parse_args()


def record_to_transaction(prediction: FraudPrediction) -> dict:
    # Rebuild the original transaction payload in the same general shape your
    # app uses when calling the agent review flow.
    return {
        "trans_date_trans_time": prediction.trans_date_trans_time.isoformat(),
        "amt": prediction.amt,
        "category": prediction.category,
        "merchant": prediction.merchant,
        "lat": prediction.lat,
        "long": prediction.long,
        "merch_lat": prediction.merch_lat,
        "merch_long": prediction.merch_long,
        "city": prediction.city,
        "state": prediction.state,
        "city_pop": prediction.city_pop,
        "dob": prediction.dob.isoformat(),
        "gender": prediction.gender,
        "job": prediction.job,
    }


def fetch_reviewed_examples(limit: int) -> list[dict]:
    db = SessionLocal()
    try:
        # Reviewed cases are the ones with final human outcomes, so they become
        # the best first labeled dataset for this repo.
        cases = (
            db.query(FraudCase)
            .filter(FraudCase.status.in_(["APPROVED", "BLOCKED"]))
            .filter(FraudCase.human_decision.isnot(None))
            .order_by(FraudCase.reviewed_at.desc(), FraudCase.created_at.desc())
            .limit(limit)
            .all()
        )

        examples: list[dict] = []
        for case in cases:
            prediction = db.get(FraudPrediction, case.prediction_id)
            if prediction is None:
                continue

            # LangSmith examples store:
            # - inputs: what we will run through the agent
            # - outputs: reference answers / labels we want to compare against
            # - metadata: extra fields that help inspect failures later
            examples.append(
                {
                    "inputs": {
                        "transaction": record_to_transaction(prediction),
                        "probability": prediction.probability,
                        "policy": {
                            "risk_band": case.risk_band,
                            "decision": case.model_decision,
                            "requires_review": True,
                        },
                    },
                    "outputs": {
                        "human_decision": case.human_decision,
                        "saved_agent_recommendation": case.agent_recommendation,
                        "model_decision": case.model_decision,
                    },
                    "metadata": {
                        "case_id": case.case_id,
                        "prediction_id": case.prediction_id,
                        "status": case.status,
                    },
                }
            )

        return examples
    finally:
        db.close()


def create_dataset(client: Client, dataset_name: str, examples: list[dict]) -> str:
    # This creates a fresh dataset every run so you can experiment without
    # mutating an old one accidentally.
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Reviewed fraud cases exported from the local app database for agent evaluation.",
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    
    # 2026 Best Practice: Direct links to created resources
    # If we can't get the org, we provide the datasets list link and the specific ID
    dataset_url = f"https://smith.langchain.com/datasets/{dataset.id}"
    print(f"\n[LangSmith] Dataset ID: {dataset.id}")
    print(f"[LangSmith] View here: {dataset_url}")
    print(f"[LangSmith] Created Name: {dataset.name}")
    print(f"[LangSmith] Uploaded examples: {len(examples)}\n")
    
    return dataset.name


@traceable
def reviewed_case_agent(inputs: dict) -> dict:
    # This is the function LangSmith will trace and evaluate.
    db = SessionLocal()
    try:
        review = generate_agent_review(
            transaction=inputs["transaction"],
            probability=inputs["probability"],
            policy=inputs["policy"],
            db=db,
        )
        return {
            "recommendation": review.recommendation,
            "confidence": review.confidence,
            "reason_codes": review.reason_codes,
            "summary": review.summary,
            "reviewer_questions": review.reviewer_questions,
        }
    finally:
        db.close()


def recommendation_matches_human(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> dict:
    # First simple evaluator: exact recommendation match vs saved human verdict.
    score = outputs["recommendation"] == reference_outputs["human_decision"]
    return {
        "key": "recommendation_matches_human",
        "score": score,
    }


def recommendation_is_review(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> dict:
    # Helpful secondary signal for this repo because REVIEW is common and can
    # explain why exact-match scores are lower than expected.
    return {
        "key": "agent_escalated_review",
        "score": outputs["recommendation"] == "REVIEW",
    }


def main() -> None:
    args = parse_args()
    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "fraudeval")

    examples = fetch_reviewed_examples(limit=args.limit)
    if not examples:
        raise SystemExit("No reviewed fraud cases found in the configured database.")

    dataset_name = create_dataset(client, args.dataset_name, examples)
    if args.upload_only:
        print("Upload-only mode enabled; skipping evaluation run.")
        return

    # Exact-match is the first useful metric for this repo. LangSmith will
    # store per-example traces so you can inspect where and why the agent fails.
    with tracing_context(enabled=True, project_name="fraudeval"):
        results = client.evaluate(
            reviewed_case_agent,
            data=dataset_name,
            evaluators=[recommendation_matches_human, recommendation_is_review],
            experiment_prefix="fraud-agent-vs-human",
            description="Rerun reviewed fraud cases and compare agent recommendation to saved human decisions.",
            max_concurrency=4,
            metadata={"models": ["groq:llama-3.3-70b-versatile"]},
        )
        wait_for_all_tracers()

    print(f"Started LangSmith eval for dataset: {dataset_name}")
    print(f"LangSmith project: {project_name}")
    experiment_name = getattr(results, "experiment_name", None)
    if experiment_name:
        print(f"Experiment: {experiment_name}")


if __name__ == "__main__":
    main()
