from typing import TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph


load_dotenv()


class FraudState(TypedDict, total=False):
    transaction: dict
    api_response: dict
    action: str
    reasoning: str
    case_id: str | None
    human_verdict: str


class HITLFraudAgent:
    def __init__(self, api_url="http://127.0.0.1:8000/api/v1/predict"):
        self.api_url = api_url
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(FraudState)

        workflow.add_node("call_api", self._call_api)
        workflow.add_node("record_decision", self._record_decision)
        workflow.add_node("queue_human_review", self._queue_human_review)

        workflow.set_entry_point("call_api")
        workflow.add_edge("call_api", "record_decision")

        # The API policy decides whether a human review case exists. LangGraph
        # only routes the simulation based on that persisted decision.
        workflow.add_conditional_edges(
            "record_decision",
            self._check_if_human_needed,
            {"human": "queue_human_review", "auto": END},
        )
        workflow.add_edge("queue_human_review", END)

        return workflow.compile()

    def _call_api(self, state: FraudState) -> dict:
        """Call the FastAPI decision endpoint and fail loudly on bad responses."""
        amount = state["transaction"].get("amt")
        print(f"Agent calling API for amount: ${amount}...")
        response = requests.post(self.api_url, json=state["transaction"], timeout=30)
        response.raise_for_status()
        return {"api_response": response.json()}

    def _record_decision(self, state: FraudState) -> dict:
        """Normalize the API response into the graph state.

        The API already persisted prediction, policy, and agent review. This
        node keeps the CLI simulation thin and avoids duplicating decision logic.
        """
        api_response = state["api_response"]
        return {
            "action": api_response.get("decision", "APPROVE"),
            "reasoning": api_response.get("agent_summary") or "Policy auto-approved the transaction.",
            "case_id": api_response.get("case_id"),
        }

    def _check_if_human_needed(self, state: FraudState) -> str:
        if state.get("api_response", {}).get("requires_review"):
            return "human"
        return "auto"

    def _queue_human_review(self, state: FraudState) -> dict:
        """Mark the graph result as waiting for dashboard/API review.

        Human verdicts are now submitted through POST /cases/{case_id}/decision,
        so the simulation does not block on terminal input.
        """
        print("\n" + "!" * 30)
        print(f"HUMAN REVIEW QUEUED (Case: {state.get('case_id')})")
        print(f"Policy Action: {state.get('action')}")
        print(f"Agent Summary: {state.get('reasoning')}")
        return {"human_verdict": "PENDING_REVIEW"}

    def run_on_transaction(self, tx: dict):
        return self.graph.invoke({"transaction": tx})


if __name__ == "__main__":
    test_tx = {
        "trans_date_trans_time": "2026-02-19 3:00:00",
        "amt": 9099.99,
        "category": "shopping_net",
        "merchant": "fraud_store_xyz",
        "lat": 40.71,
        "long": -74.00,
        "merch_lat": 40.75,
        "merch_long": -73.98,
        "city": "Lagos",
        "state": "LG",
        "city_pop": 8000000,
        "dob": "1990-01-01",
        "gender": "M",
        "job": "Engineer",
    }

    agent = HITLFraudAgent()
    print("Starting Agent Investigation...")
    result = agent.run_on_transaction(test_tx)

    print("\n" + "=" * 50)
    print("FINAL AGENT VERDICT")
    print("=" * 50)
    print(f"Case ID: {result.get('case_id')}")
    print(f"Action:  {result.get('action')}")
    print(f"Human:   {result.get('human_verdict', 'N/A (Auto-processed)')}")
    print("=" * 50)
