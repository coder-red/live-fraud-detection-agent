import requests
import uuid # For generating unique case IDs
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq


from dotenv import load_dotenv
load_dotenv()

class FraudState(TypedDict):
    transaction: dict
    api_response: dict
    action: str
    reasoning: str
    human_verdict: str  # Added for HITL
    case_id: str   # works with uuid to track cases

class HITLFraudAgent:
    def __init__(self, api_url="http://127.0.0.1:8000/api/v1/predict"):
        
        self.api_url = api_url
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(FraudState)

        workflow.add_node("call_api", self._call_api)
        workflow.add_node("ai_reasoning", self._ai_reasoning)
        workflow.add_node("human_intervention", self._human_intervention)

        workflow.set_entry_point("call_api")
        workflow.add_edge("call_api", "ai_reasoning")
        
        # Route to human only if it's risky
        workflow.add_conditional_edges(
            "ai_reasoning",
            self._check_if_human_needed,
            {"human": "human_intervention", "auto": END}
        )
        workflow.add_edge("human_intervention", END)

        return workflow.compile()

    # --- NODES ---

    def _call_api(self, state: FraudState) -> dict:
        """COMMUNICATE WITH THE API: The agent calls the FastAPI directly."""
        print(f"Agent calling API for amount: ${state['transaction']['amt']}...")
        try:
            response = requests.post(self.api_url, json=state['transaction'])
            return {"api_response": response.json(), "case_id": str(uuid.uuid4())}
        except Exception as e:
            print(f"API Connection Failed: {e}")
            return {"api_response": {"probability": 0.5, "features": {}}}

    def _ai_reasoning(self, state: FraudState) -> dict:
        """Agent interprets the API score."""
        score = state["api_response"].get("probability", 0)
        prompt = f"Analyze this fraud score: {score:.2%}. Transaction: {state['transaction']}. Action: APPROVE/BLOCK?"
        res = self.llm.invoke(prompt).content
        # Assume parsing logic exists here
        return {"action": "BLOCK" if score > 0.5221 else "APPROVE", "reasoning": res}

    def _check_if_human_needed(self, state: FraudState) -> str:
        if state["action"] == "BLOCK":
            return "human"
        return "auto"

    def _human_intervention(self, state: FraudState) -> dict:
        """THE HITL FACTOR: The system stops and waits for human intervention."""
        print("\n" + "!"*30)
        print(f"HUMAN INTERVENTION REQUIRED (Case: {state['case_id']})")
        print(f"AI Suggestion: {state['action']}")
        print(f"AI Reasoning: {state['reasoning']}")
        print(f"Score: {state['api_response']['probability']:.2%}")
        
        # In a real system, this would be a dashboard. For now, we simulate with input.
        choice = input("Accept AI Decision? (y/n) or type 'OVERRIDE_APPROVE': ")
        
        final_action = state["action"]
        if choice.lower() == 'n':
            final_action = "MANUAL_OVERRIDE_REVERSED"
        elif choice == "OVERRIDE_APPROVE":
            final_action = "APPROVE"
            
        return {"human_verdict": final_action}

    def run_on_transaction(self, tx: dict):
        return self.graph.invoke({"transaction": tx})

if __name__ == "__main__":
    # Setup the transaction
    test_tx = {
        "trans_date_trans_time": "2026-02-19 3:00:00",
        "amt": 9099.99,
        "category": "shopping_net",
        "merchant": "fraud_store_xyz",
        "lat": 40.71, "long": -74.00,
        "merch_lat": 40.75, "merch_long": -73.98,
        "city": "Lagos", "state": "LG", "city_pop": 8000000,
        "dob": "1990-01-01", "gender": "M", "job": "Engineer"
    }
    
    # Initialize and RUN
    agent = HITLFraudAgent()
    
    print("Starting Agent Investigation...")
    result = agent.run_on_transaction(test_tx)
    
    # Print the results clearly
    print("\n" + "="*50)
    print("FINAL AGENT VERDICT")
    print("="*50)
    print(f"Case ID: {result.get('case_id')}")
    print(f"Action:  {result.get('action')}")
    print(f"Human:   {result.get('human_verdict', 'N/A (Auto-processed)')}")
    # print(f"Reason:  {result.get('reasoning')}")
    print("="*50)