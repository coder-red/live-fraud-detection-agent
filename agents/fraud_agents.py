# agents/fraud_agent.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing import TypedDict
import json

from agents.tools import (
    check_geo_anomaly,
    check_merchant_fraud_history,
    check_velocity,
)

class FraudState(TypedDict, total=False):
    transaction: dict
    api_response: dict
    enrichment: dict        # NEW — tool results
    agent_reasoning: str    # NEW — LLM's own analysis
    agent_decision: str     # NEW — LLM's own verdict
    action: str
    case_id: str | None
    human_verdict: str


class HITLFraudAgent:
    def __init__(self, db_session, api_url="http://127.0.0.1:8000/api/v1/predict"):
        self.api_url = api_url
        self.db = db_session
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(FraudState)

        workflow.add_node("call_api",           self._call_api)
        workflow.add_node("enrich",             self._enrich)        # NEW
        workflow.add_node("investigate",        self._investigate)   # NEW
        workflow.add_node("record_decision",    self._record_decision)
        workflow.add_node("queue_human_review", self._queue_human_review)

        workflow.set_entry_point("call_api")
        workflow.add_edge("call_api",        "enrich")
        workflow.add_edge("enrich",          "investigate")
        workflow.add_edge("investigate",     "record_decision")
        workflow.add_conditional_edges(
            "record_decision",
            self._check_if_human_needed,
            {"human": "queue_human_review", "auto": END},
        )
        workflow.add_edge("queue_human_review", END)
        return workflow.compile()

    def _call_api(self, state: FraudState) -> dict:
        import requests
        response = requests.post(self.api_url, json=state["transaction"], timeout=30)
        response.raise_for_status()
        return {"api_response": response.json()}

    def _enrich(self, state: FraudState) -> dict:
        """Run all three investigative tools and collect results."""
        tx = state["transaction"]
        
        merchant_history = check_merchant_fraud_history(self.db, tx["merchant"])
        velocity         = check_velocity(self.db, tx["city"], tx["state"])
        geo              = check_geo_anomaly(self.db, tx["merchant"], tx["merch_lat"], tx["merch_long"])
        
        enrichment = {
            "merchant_history": merchant_history,
            "velocity":         velocity,
            "geo_anomaly":      geo,
        }
        print(f"\n[Enrichment] {json.dumps(enrichment, indent=2)}")
        return {"enrichment": enrichment}

    def _investigate(self, state: FraudState) -> dict:
        """LLM receives model score + tool results and writes its own verdict."""
        tx       = state["transaction"]
        api_resp = state["api_response"]
        enrich   = state["enrichment"]

        prompt = f"""You are a senior fraud investigator at a fintech company.

TRANSACTION:
- Amount: ${tx['amt']}
- Merchant: {tx['merchant']}
- Category: {tx['category']}
- Location: {tx['city']}, {tx['state']}
- Time: {tx['trans_date_trans_time']}

MODEL SCORE:
- Fraud probability: {api_resp.get('probability', 0)*100:.1f}%
- Risk band: {api_resp.get('risk_band')}
- Policy decision: {api_resp.get('decision')}

INVESTIGATION FINDINGS:
- Merchant fraud history: {enrich['merchant_history']}
- Location velocity (last 60 min): {enrich['velocity']}
- Geo anomaly check: {enrich['geo_anomaly']}

Based on ALL of the above, give:
1. Your own APPROVE / REVIEW / BLOCK recommendation
2. Your confidence (0-100)
3. Two or three specific reason codes (e.g. HIGH_VELOCITY, GEO_ANOMALY, CLEAN_MERCHANT)
4. One paragraph explaining your reasoning
5. Two questions a human reviewer should answer if escalated

Respond as JSON only:
{{
  "recommendation": "APPROVE|REVIEW|BLOCK",
  "confidence": 85,
  "reason_codes": ["...", "..."],
  "reasoning": "...",
  "reviewer_questions": ["...", "..."]
}}"""

        response = self.llm.invoke([
            SystemMessage(content="You are a fraud investigation agent. Respond only in valid JSON."),
            HumanMessage(content=prompt)
        ])
        
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = {
                "recommendation": api_resp.get("decision", "REVIEW"),
                "confidence": 50,
                "reason_codes": ["PARSE_ERROR"],
                "reasoning": response.content,
                "reviewer_questions": []
            }
        
        print(f"\n[Agent] {parsed['recommendation']} ({parsed['confidence']}%) — {parsed['reason_codes']}")
        return {
            "agent_decision":  parsed["recommendation"],
            "agent_reasoning": parsed["reasoning"],
        }

    def _record_decision(self, state: FraudState) -> dict:
        # Agent's decision takes priority over pure policy if they disagree
        agent_dec = state.get("agent_decision", "APPROVE")
        api_dec   = state["api_response"].get("decision", "APPROVE")
        
        # Escalate if either engine flags it
        final = "REVIEW" if "REVIEW" in (agent_dec, api_dec) or "BLOCK" in (agent_dec, api_dec) else "APPROVE"
        
        return {
            "action":   final,
            "reasoning": state.get("agent_reasoning"),
            "case_id":  state["api_response"].get("case_id"),
        }

    def _check_if_human_needed(self, state: FraudState) -> str:
        return "human" if state.get("action") in ("REVIEW", "BLOCK") else "auto"

    def _queue_human_review(self, state: FraudState) -> dict:
        print(f"\n{'!'*30}")
        print(f"HUMAN REVIEW QUEUED (Case: {state.get('case_id')})")
        print(f"Agent decision: {state.get('action')}")
        print(f"Reasoning: {state.get('reasoning')}")
        return {"human_verdict": "PENDING_REVIEW"}

    def run_on_transaction(self, tx: dict):
        return self.graph.invoke({"transaction": tx})