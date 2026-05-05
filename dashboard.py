import streamlit as st
import requests
import pandas as pd
import time
from pathlib import Path
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_ROOT_URL = API_BASE_URL.split("/api")[0]

st.set_page_config(page_title="Fraud Console", layout="wide")

st.markdown("""
    <style>
    @import url('https://rsms.me/inter/inter.css');
    .stApp { background-color: #0e1117; color: #fafafa; font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
    }
    .stTextArea textarea {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #fafafa !important;
    }
    .stButton > button { border-radius: 6px; font-weight: 500; }
    .arch-note { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---

def get_data(endpoint):
    url = f"{API_BASE_URL}/{endpoint}" if endpoint else f"{API_BASE_URL}/"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return "NOT_FOUND"
        return None
    except requests.RequestException:
        return None

def post_data(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False

def post_predict(payload: dict) -> tuple[bool, object, float]:
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            return True, r.json(), latency_ms
        return False, r.text, latency_ms
    except requests.RequestException as exc:
        return False, str(exc), 0.0

def _project_root() -> Path:
    # Try script-relative first, fall back to cwd
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "data" / "sample_transactions.csv").exists():
        return script_dir
    return Path.cwd()

st.write(f"DEBUG: looking for CSV at `{csv_path}`")

def _csv_row_to_payload(row: dict) -> dict:
    out: dict = {}
    for key, val in row.items():
        if key == "is_fraud":
            continue
        if pd.isna(val):
            continue
        if key in ("amt", "lat", "long", "merch_lat", "merch_long"):
            out[key] = float(val)
        elif key == "city_pop":
            out[key] = int(val)
        else:
            out[key] = str(val).strip()
    return out

def load_all_sample_transactions() -> list[dict]:
    csv_path = _project_root() / "data" / "sample_transactions.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    return [_csv_row_to_payload(row.to_dict()) for _, row in df.iterrows()]

# --- Cached data fetchers ---
# TTL=15s: fast for demo, fresh enough after decisions

@st.cache_data(ttl=15)
def get_pending():
    return get_data("cases/pending")

@st.cache_data(ttl=15)
def get_predictions():
    return get_data("predictions?limit=100")

@st.cache_data(ttl=60)
def get_health():
    try:
        r = requests.get(f"{API_ROOT_URL}/", timeout=3)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None

# --- Connection check (cached, doesn't re-hit on every rerun) ---

health = get_health()
if not health:
    st.error(f"Connection Error: Could not reach API at {API_BASE_URL}")
    st.stop()

# --- Header ---

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## Fraud Detection Console")
    st.caption("XGBoost · LangGraph HITL · FastAPI  —  real-time transaction scoring + human review queue")
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺ Refresh", use_container_width=True):
        st.session_state.pop("sim_results", None)
        get_pending.clear()
        get_predictions.clear()
        st.rerun()

st.divider()

# --- Simulation ---

st.markdown("### Run Simulation")
st.markdown(
    '<p class="arch-note">Production: Kafka consumer → fraud scorer → decision router. '
    'Demo: direct POST /api/v1/predict — same model, same policy.</p>',
    unsafe_allow_html=True,
)

if st.button("▶  Score all transactions", key="run_simulation"):
    payloads = load_all_sample_transactions()

    if not payloads:
        st.error("No sample data found at `data/sample_transactions.csv`.")
    else:
        results  = []
        latencies = []
        progress  = st.progress(0, text="Scoring transactions…")

        for i, payload in enumerate(payloads):
            ok, result, latency_ms = post_predict(payload)
            latencies.append(latency_ms)
            if ok and isinstance(result, dict):
                results.append({
                    "merchant":        payload.get("merchant", "—"),
                    "amount":          payload.get("amt", 0),
                    "decision":        result.get("decision", "—"),
                    "risk_band":       result.get("risk_band", "—"),
                    "requires_review": result.get("requires_review", False),
                    "probability":     result.get("probability", 0),
                })
            else:
                results.append({
                    "merchant":        payload.get("merchant", "—"),
                    "amount":          payload.get("amt", 0),
                    "decision":        "ERROR",
                    "risk_band":       "—",
                    "requires_review": False,
                    "probability":     0,
                })
            progress.progress((i + 1) / len(payloads), text=f"Scored {i + 1} / {len(payloads)}")

        progress.empty()

        df      = pd.DataFrame(results)
        total   = len(df)
        auto_ok = len(df[df["decision"] == "APPROVE"])
        flagged = len(df[df["requires_review"] == True])
        blocked = len(df[df["decision"] == "BLOCK"])
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        st.session_state["sim_results"] = {
            "df":       df,
            "total":    total,
            "auto_ok":  auto_ok,
            "flagged":  flagged,
            "blocked":  blocked,
            "avg_lat":  avg_lat,
            "total_ms": sum(latencies),
        }
        # Invalidate queue cache so new cases appear immediately
        get_pending.clear()
        get_predictions.clear()
        st.rerun()

# Render simulation results if they exist (persists across reruns)
if "sim_results" in st.session_state:
    r = st.session_state["sim_results"]
    df = r["df"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Transactions scored", r["total"])
    m2.metric("Auto-approved",       r["auto_ok"])
    m3.metric("Sent to review",      r["flagged"])
    m4.metric("Auto-blocked",        r["blocked"])
    m5.metric("Avg latency",         f"{r['avg_lat']:.0f} ms")

    st.markdown("#### Auto-approved")
    auto_df = df[df["decision"] == "APPROVE"][["merchant", "amount", "risk_band", "probability"]].copy()
    auto_df["amount"]      = auto_df["amount"].map(lambda x: f"${x:,.2f}")
    auto_df["probability"] = auto_df["probability"].map(lambda x: f"{x*100:.1f}%")
    auto_df.columns        = ["Merchant", "Amount", "Risk band", "Fraud prob."]
    if auto_df.empty:
        st.info("No auto-approved transactions.")
    else:
        st.dataframe(auto_df, use_container_width=True, hide_index=True)

    st.markdown("#### Flagged / blocked")
    flag_df = df[df["requires_review"] | (df["decision"] == "BLOCK")][
        ["merchant", "amount", "decision", "risk_band", "probability"]
    ].copy()
    flag_df["amount"]      = flag_df["amount"].map(lambda x: f"${x:,.2f}")
    flag_df["probability"] = flag_df["probability"].map(lambda x: f"{x*100:.1f}%")
    flag_df.columns        = ["Merchant", "Amount", "Decision", "Risk band", "Fraud prob."]
    if flag_df.empty:
        st.info("No flagged or blocked transactions.")
    else:
        st.dataframe(flag_df, use_container_width=True, hide_index=True)

    st.caption(f"Completed in {r['total_ms']:.0f} ms total.")

st.divider()

# --- Data (cached) ---

pending         = get_pending()
all_predictions = get_predictions()

if pending == "NOT_FOUND" or all_predictions == "NOT_FOUND":
    st.error("Endpoint Error: Required API routes not found.")
    st.warning("Your Docker image is outdated. Run: docker compose up --build")
    st.stop()

if pending is None:
    st.error("Could not load pending cases from API.")
    st.stop()

if all_predictions is None:
    all_predictions = []

# --- Navigation ---

view_mode = st.radio(
    "View",
    ["Review Queue", "Activity History"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Review Queue ──────────────────────────────────────────────────────────────

if view_mode == "Review Queue":
    st.title("REVIEW QUEUE")

    if len(pending) == 0:
        st.success("All clear — no cases require manual review.")
        c1, c2 = st.columns(2)
        c1.metric("Pending cases",       0)
        c2.metric("Transactions scored", len(all_predictions))
        st.caption("Run the simulation above to generate review cases.")
        if st.button("Refresh queue"):
            get_pending.clear()
            get_predictions.clear()
            st.rerun()
    else:
        st.caption(f"{len(pending)} case(s) awaiting reviewer action.")
        selected_id = st.selectbox(
            "Queue",
            [c["case_id"] for c in pending],
            format_func=lambda x: f"CASE {x[:8]}".upper(),
            label_visibility="collapsed",
        )
        current_case = next(c for c in pending if c["case_id"] == selected_id)
        tx = get_data(f"predictions/{current_case['prediction_id']}")

        if tx and tx != "NOT_FOUND":
            st.title(f"CASE {selected_id[:8]}".upper())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("FRAUD PROB",  f"{tx['probability']*100:.0f}%")
            m2.metric("RISK BAND",   tx["risk_band"].upper())
            m3.metric("AMOUNT",      f"${tx['amt']:,.2f}")
            m4.metric("CATEGORY",    tx["category"].replace("_", " ").upper())

            st.divider()
            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                st.markdown("### CONTEXT")
                st.json({
                    "Merchant": tx["merchant"],
                    "Location": f"{tx['city']}, {tx['state']}",
                    "Time":     tx["trans_date_trans_time"],
                })

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### AGENT ANALYSIS")
                reasoning = current_case.get("reasoning")
                if reasoning:
                    st.write(reasoning)
                else:
                    st.warning(
                        "Agent reasoning not saved — check that LangGraph enrichment "
                        "is wiring `reasoning` back to `FraudCase.reasoning` in routes.py."
                    )

                if current_case.get("reason_codes"):
                    st.markdown(" ".join([f"`{c}`" for c in current_case["reason_codes"]]))

                if current_case.get("agent_confidence") is not None:
                    st.caption(f"Agent confidence: {current_case['agent_confidence']*100:.0f}%")

            with col_right:
                st.markdown("### CHECKLIST")
                reviewer_questions = current_case.get("reviewer_questions")
                if reviewer_questions:
                    for q in reviewer_questions:
                        st.write(f"· {q}")
                else:
                    st.caption("No checklist items.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### RESOLUTION")
                note = st.text_area("NOTES", placeholder="Decision rationale…", label_visibility="collapsed")

                b1, b2 = st.columns(2)
                if b1.button("APPROVE", type="primary", use_container_width=True):
                    if post_data(f"cases/{selected_id}/decision", {"decision": "APPROVE", "note": note}):
                        get_pending.clear()
                        get_predictions.clear()
                        st.toast("APPROVED")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed. Please retry.")

                if b2.button("BLOCK", use_container_width=True):
                    if post_data(f"cases/{selected_id}/decision", {"decision": "BLOCK", "note": note}):
                        get_pending.clear()
                        get_predictions.clear()
                        st.toast("BLOCKED")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed. Please retry.")

# ── Activity History ──────────────────────────────────────────────────────────

elif view_mode == "Activity History":
    st.title("ACTIVITY HISTORY")

    if all_predictions:
        auto_approved = [p for p in all_predictions if not p.get("requires_review")]

        st.markdown(f"#### Auto-approved ({len(auto_approved)})")
        if auto_approved:
            df = pd.DataFrame(auto_approved)
            if "trans_date_trans_time" in df.columns:
                df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
                df = df.sort_values("trans_date_trans_time", ascending=False)
            display_df = df[["trans_date_trans_time", "amt", "category", "merchant", "risk_band", "decision"]].copy()
            display_df.columns = ["Time", "Amount", "Category", "Merchant", "Risk", "Outcome"]
            display_df["Amount"] = display_df["Amount"].map(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No automatic approvals yet.")

        st.markdown("#### Full audit log")
        df_full = pd.DataFrame(all_predictions)
        if "trans_date_trans_time" in df_full.columns:
            df_full["trans_date_trans_time"] = pd.to_datetime(df_full["trans_date_trans_time"], errors="coerce")
            df_full = df_full.sort_values("trans_date_trans_time", ascending=False)
        log_df = df_full[["trans_date_trans_time", "amt", "merchant", "risk_band", "decision", "requires_review"]].copy()
        log_df.columns = ["Time", "Amount", "Merchant", "Risk", "Action", "Review Req."]
        log_df["Amount"] = log_df["Amount"].map(lambda x: f"${x:,.2f}")
        st.dataframe(log_df.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("No transaction history yet. Run the simulation above.")