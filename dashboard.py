import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os

API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    try:
        r = requests.get("http://localhost:8000/", timeout=3)
        API_BASE_URL = "http://localhost:8000/api/v1" if r.ok else "https://live-fraud-detection-agent.onrender.com/api/v1"
    except requests.RequestException:
        API_BASE_URL = "https://live-fraud-detection-agent.onrender.com/api/v1"

st.set_page_config(page_title="Fraud Console", layout="wide")

_custom = st.session_state.get("_custom_api_url")
if _custom:
    API_BASE_URL = _custom

API_ROOT_URL = API_BASE_URL.split("/api")[0]

# --- Helpers ---

def get_data(endpoint):
    url = f"{API_BASE_URL}/{endpoint}" if endpoint else f"{API_BASE_URL}/"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return "NOT_FOUND"
        return None
    except requests.RequestException:
        return None

def post_data(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=15)
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

def generate_simulation_transactions(count: int = 50, fraud_ratio: float = 0.5) -> list[dict]:
    """Generate fresh random transactions for simulation.

    Each call produces unique transactions, ensuring no duplicate fingerprints
    and always creating new predictions and review cases.
    """
    from src.transaction_generator import generate_transactions
    return generate_transactions(count=count, fraud_ratio=fraud_ratio)

# --- Cached data fetchers ---

@st.cache_data(ttl=15)
def get_pending():
    return get_data("cases/pending")

@st.cache_data(ttl=15)
def get_predictions(limit: int = 100, offset: int = 0):
    data = get_data(f"predictions?limit={limit}&offset={offset}")
    if isinstance(data, dict) and "items" in data:
        return data
    return {"items": data if isinstance(data, list) else [], "total": 0, "offset": offset, "limit": limit}

@st.cache_data(ttl=60)
def get_health():
    try:
        r = requests.get(f"{API_ROOT_URL}/", timeout=30)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None

# --- Connection check (retry for cold starts) ---

health = get_health()
if not health:
    with st.spinner("Waking up the API (cold start)..."):
        for _ in range(6):
            time.sleep(5)
            health = get_health()
            if health:
                st.rerun()
    st.error(
        f"Could not reach API at {API_BASE_URL}\n\n"
        "Start the API locally:\n"
        "```\nuvicorn app.main:app --host 0.0.0.0 --port 8000\n```\n\n"
        "Or enter a custom URL in the sidebar."
    )
    st.stop()

# --- Sidebar ---

with st.sidebar:
    st.markdown("## Fraud Console")
    st.divider()
    st.markdown("### Live")
    st.checkbox(
        "Live auto-refresh (10s)",
        value=st.session_state.get("live_mode", False),
        key="live_mode",
        help="SSE live feed + periodic chart refresh.",
    )
    st.divider()
    st.markdown("### Connection")
    api_status = "Connected" if health else "Disconnected"
    st.markdown(f"**Status:** {api_status}")
    custom_url = st.text_input(
        "API URL",
        value=st.session_state.get("_custom_api_url", ""),
        placeholder="http://localhost:8000/api/v1",
        label_visibility="collapsed",
    )
    if custom_url and custom_url != st.session_state.get("_custom_api_url"):
        st.session_state["_custom_api_url"] = custom_url
        st.rerun()
    st.divider()
    if st.button("Clear Cache", use_container_width=True):
        st.session_state.pop("sim_results", None)
        get_pending.clear()
        get_predictions.clear()
        st.rerun()

st.markdown("""
<style>
@import url('https://rsms.me/inter/inter.css');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #0b0d11; color: #e6edf3; }
.stApp > header { background-color: #0b0d11; border-bottom: 1px solid #21262d; }

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #101318; border: 1px solid #21262d; border-radius: 8px;
    padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"]:hover { border-color: #30363d; }
div[data-testid="stMetric"] > div:first-child { font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }
div[data-testid="stMetric"] > div:nth-child(2) { font-size: 1.5rem; font-weight: 600; color: #e6edf3; line-height: 1.2; }
div[data-testid="stMetric"] > div:nth-child(3) { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }

/* Buttons */
.stButton > button {
    border-radius: 6px; font-weight: 500; font-size: 0.8125rem;
    border: 1px solid #30363d; background: #21262d; color: #e6edf3;
    transition: background 0.15s, border-color 0.15s;
}
.stButton > button:hover { border-color: #8b949e; background: #30363d; }
.stButton > button[kind="primary"] { background: #1f6feb; border-color: #1f6feb; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #388bfd; border-color: #388bfd; }

/* Text area / input */
.stTextArea textarea, .stTextInput input {
    background-color: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; color: #e6edf3 !important; font-size: 0.8125rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: #1f6feb !important; box-shadow: 0 0 0 2px rgba(31,111,235,0.3) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.8125rem; font-weight: 500; color: #8b949e; padding: 8px 16px;
    border-bottom: 2px solid transparent; transition: color 0.15s, border-color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #e6edf3; }
.stTabs [aria-selected="true"] { color: #e6edf3; border-bottom-color: #1f6feb; }

/* Dataframe */
div[data-testid="stDataFrame"] { font-size: 0.8125rem; }
div[data-testid="stDataFrame"] th { background: #101318; color: #8b949e; font-weight: 500; text-transform: uppercase; font-size: 0.6875rem; letter-spacing: 0.04em; border-bottom: 1px solid #21262d; }
div[data-testid="stDataFrame"] td { background: transparent; border-bottom: 1px solid #161b22; }

/* Select / dropdown */
div[data-baseweb="select"] > div { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; }
div[data-baseweb="select"] > div:hover { border-color: #8b949e; }

/* Expander */
.streamlit-expanderHeader { font-size: 0.8125rem; font-weight: 500; color: #8b949e; }
.streamlit-expanderContent { border-top: 1px solid #21262d; }

/* Divider */
.stDivider { border-color: #21262d; margin: 12px 0; }

/* Caption */
.stCaption, .arch-note { font-size: 0.75rem; color: #8b949e; }

/* Status badge */
.status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
.status-badge.live { background: #052e16; color: #22c55e; border: 1px solid #115f26; }
.status-badge.error { background: #450a0a; color: #ef4444; border: 1px solid #7f1d1d; }

/* Pending banner */
.pending-banner {
    display: flex; align-items: center; gap: 12px; padding: 10px 16px;
    background: #221703; border: 1px solid #78350f; border-radius: 8px;
    font-size: 0.875rem; color: #fbbf24; margin-bottom: 16px;
}
.pending-banner strong { font-weight: 600; }

/* Toolbar */
.toolbar { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; }
.toolbar-left { display: flex; align-items: center; gap: 16px; }
.toolbar-right { display: flex; align-items: center; gap: 12px; }
</style>
""", unsafe_allow_html=True)

# --- Header / Toolbar ---

col_tool = st.columns([1, 1])
with col_tool[0]:
    st.markdown("## Fraud Detection Console")
    st.caption("XGBoost · LangGraph HITL · FastAPI")
with col_tool[1]:
    st.markdown(
        f'<div style="display:flex; justify-content:flex-end; align-items:center; gap:12px; height:100%;">'
        f'<span class="status-badge {"live" if health else "error"}">{"&#9679; Live" if health else "&#9679; Offline"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Refresh", key="header_refresh", use_container_width=True):
        st.session_state.pop("sim_results", None)
        get_pending.clear()
        get_predictions.clear()
        st.rerun()

st.divider()

# --- Simulation ---

with st.expander("Run Simulation (50 transactions)", expanded=False):
    st.markdown(
        '<p class="arch-note">Posts 50 transactions to /api/v1/predict — same model, same policy as production.</p>',
        unsafe_allow_html=True,
    )

    if st.button("▶  Start", key="run_simulation"):
        payloads = generate_simulation_transactions(count=50, fraud_ratio=0.5)

        if not payloads:
            st.error("Failed to generate transactions.")
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
            flagged = len(df[df["requires_review"]])
            blocked = len(df[df["decision"] == "BLOCK"])
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            st.session_state["sim_results"] = {
                "df":        df,
                "latencies": latencies,
                "total":     total,
                "auto_ok":   auto_ok,
                "flagged":   flagged,
                "blocked":   blocked,
                "avg_lat":   avg_lat,
                "total_ms":  sum(latencies),
            }
            get_pending.clear()
            get_predictions.clear()
            st.rerun()

    # Render simulation results
    if "sim_results" in st.session_state:
        r = st.session_state["sim_results"]
        df = r["df"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Transactions scored", r["total"])
        m2.metric("Auto-approved",       r["auto_ok"])
        m3.metric("Sent to review",      r["flagged"])
        m4.metric("Auto-blocked",        r["blocked"])
        m5.metric("Avg latency",         f"{r['avg_lat']:.0f} ms")

        auto_df = df[df["decision"] == "APPROVE"][["merchant", "amount", "risk_band", "probability"]].copy()
        auto_df["amount"]      = auto_df["amount"].map(lambda x: f"${x:,.2f}")
        auto_df["probability"] = auto_df["probability"].map(lambda x: f"{x*100:.1f}%")
        auto_df.columns        = ["Merchant", "Amount", "Risk band", "Fraud prob."]
        if auto_df.empty:
            st.info("No auto-approved transactions.")
        else:
            st.dataframe(auto_df, width='stretch', hide_index=True)

        flag_df = df[df["requires_review"] | (df["decision"] == "BLOCK")][
            ["merchant", "amount", "decision", "risk_band", "probability"]
        ].copy()
        flag_df["amount"]      = flag_df["amount"].map(lambda x: f"${x:,.2f}")
        flag_df["probability"] = flag_df["probability"].map(lambda x: f"{x*100:.1f}%")
        flag_df.columns        = ["Merchant", "Amount", "Decision", "Risk band", "Fraud prob."]
        if flag_df.empty:
            st.info("No flagged or blocked transactions.")
        else:
            st.dataframe(flag_df, width='stretch', hide_index=True)

        st.caption(f"Completed in {r['total_ms']:.0f} ms total.")

# --- Data (cached) ---

pending          = get_pending()
predictions_page = get_predictions()
all_predictions  = predictions_page.get("items", []) if isinstance(predictions_page, dict) else (predictions_page or [])
total_preds      = predictions_page.get("total", len(all_predictions)) if isinstance(predictions_page, dict) else len(all_predictions)

# Pagination: version counter resets accumulated items on cache refresh
if "prediction_version" not in st.session_state:
    st.session_state["prediction_version"] = 0
    st.session_state["prediction_items"] = list(all_predictions)
    st.session_state["prediction_offset"] = 100
else:
    prev_items = st.session_state["prediction_items"]
    stale = len(prev_items) > len(all_predictions) or (
        len(prev_items) > 0 and len(all_predictions) > 0 and prev_items[0].get("id") != all_predictions[0].get("id")
    )
    if stale:
        st.session_state["prediction_version"] += 1
        st.session_state["prediction_items"] = list(all_predictions)
        st.session_state["prediction_offset"] = 100

if pending == "NOT_FOUND" or predictions_page == "NOT_FOUND":
    st.error("Endpoint Error: Required API routes not found.")
    st.warning("Your Docker image is outdated. Run: docker compose up --build")
    st.stop()

if pending is None:
    st.error("Could not load pending cases from API.")
    st.stop()

if all_predictions is None:
    all_predictions = []

# --- Persistent KPI bar ---

pending_count = len(pending) if isinstance(pending, list) else 0
total_txns = total_preds
fraud_volume = 0
fraud_count = 0
if all_predictions and len(all_predictions) > 0:
    flagged_or_blocked = [p for p in all_predictions if p.get("requires_review") or p.get("decision") == "BLOCK"]
    fraud_volume = sum(p.get("amt", 0) for p in flagged_or_blocked)
    fraud_count = len(flagged_or_blocked)

avg_latency = 0
sim_lat = st.session_state.get("sim_results", {}).get("latencies")
if sim_lat:
    avg_latency = sum(sim_lat) / len(sim_lat)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Transactions", f"{total_txns:,}" if total_txns else "—", help="Total predictions scored")
kpi_cols[1].metric("Fraud Volume", f"${fraud_volume:,.0f}" if fraud_volume else "—", delta=f"{fraud_count} flagged" if fraud_count else None, delta_color="inverse", help="Total $ amounts flagged or blocked")
kpi_cols[2].metric("Pending Review", f"{pending_count}", delta_color="inverse" if pending_count > 0 else "normal", help="Cases awaiting human decision")
kpi_cols[3].metric("Avg Latency", f"{avg_latency:.0f} ms" if avg_latency else "—", help="API response time")

# --- Pending banner ---

if pending_count > 0:
    st.markdown(
        f'<div class="pending-banner">'
        f'<span style="font-size:1.1rem;">&#9888;</span>'
        f'<span><strong>{pending_count}</strong> case(s) pending review</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# --- Live Feed (SSE-powered, always visible when active) ---

if st.session_state.get("live_mode"):
    from streamlit.components.v1 import html as st_html
    sse_url = f"{API_BASE_URL}/stream"
    st_html(
        f"""
        <div id="sse-feed" style="font-family: 'Inter', -apple-system, sans-serif; background:#0e1117; color:#fafafa; border-radius:8px; padding:4px 0;">
            <div style="display:flex; align-items:center; gap:8px; padding:8px 12px; border-bottom:1px solid #30363d; margin-bottom:4px;">
                <span style="display:inline-block; width:8px; height:8px; background:#22c55e; border-radius:50%; animation:pulse 2s infinite;"></span>
                <span style="font-size:13px; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">Live Feed</span>
                <span id="sse-status" style="font-size:11px; color:#8b949e; margin-left:auto;">connecting…</span>
            </div>
            <div id="sse-items" style="max-height:360px; overflow-y:auto; padding:4px 0;"></div>
        </div>
        <style>
            @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
            @keyframes slideIn {{ from {{ opacity:0; transform:translateY(-12px); }} to {{ opacity:1; transform:translateY(0); }} }}
            .sse-card {{ display:flex; align-items:center; gap:12px; padding:8px 12px; margin:2px 4px; background:#161b22; border:1px solid #21262d; border-radius:6px; animation:slideIn 0.3s ease-out; font-size:13px; }}
            .sse-card:hover {{ border-color:#30363d; }}
            .sse-card .merchant {{ flex:2; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
            .sse-card .amount {{ flex:1; text-align:right; font-variant-numeric:tabular-nums; }}
            .sse-card .band {{ flex:1; text-align:center; }}
            .sse-card .prob {{ flex:1; text-align:center; font-variant-numeric:tabular-nums; }}
            .sse-card .decision {{ flex:1; text-align:center; }}
            .badge {{ display:inline-block; padding:1px 8px; border-radius:4px; font-size:11px; font-weight:600; text-transform:uppercase; }}
            .badge-LOW {{ background:#052e16; color:#22c55e; }}
            .badge-MEDIUM {{ background:#422006; color:#eab308; }}
            .badge-HIGH {{ background:#431407; color:#f97316; }}
            .badge-CRITICAL {{ background:#450a0a; color:#ef4444; }}
        </style>
        <script>
        (function(){{
            var es = new EventSource("{sse_url}");
            var items = document.getElementById("sse-items");
            var status = document.getElementById("sse-status");
            var count = 0;

            es.onopen = function() {{
                status.textContent = "connected";
                status.style.color = "#22c55e";
            }};

            es.onmessage = function(e) {{
                try {{
                    var msg = JSON.parse(e.data);
                    if (msg.event === "ping") return;
                    var d = msg.data;
                    count++;
                    status.textContent = count + " new";

                    var band = (d.risk_band || "LOW").toUpperCase();
                    var prob = ((d.probability || 0) * 100).toFixed(0) + "%";
                    var amt = "$" + (d.amt || 0).toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}});
                    var dec = d.decision || "—";
                    var emoji = {{"APPROVE":"\u2705","REVIEW":"\uD83D\uDD36","BLOCK":"\u274C"}}[dec] || "\u2795";
                    var merchant = d.merchant || "—";

                    var card = document.createElement("div");
                    card.className = "sse-card";
                    card.innerHTML =
                        '<span class="merchant">' + merchant + '</span>' +
                        '<span class="amount">' + amt + '</span>' +
                        '<span class="band"><span class="badge badge-' + band + '">' + band + '</span></span>' +
                        '<span class="prob">' + prob + '</span>' +
                        '<span class="decision">' + emoji + ' ' + dec + '</span>';
                    items.insertBefore(card, items.firstChild);

                    // Keep at most 50 items
                    while (items.children.length > 50) {{
                        items.removeChild(items.lastChild);
                    }}
                }} catch(err) {{
                    console.warn("SSE parse error", err);
                }}
            }};

            es.onerror = function() {{
                status.textContent = "reconnecting…";
                status.style.color = "#f97316";
            }};
        }})();
        </script>
        """,
        height=420,
        scrolling=False,
    )
    st.caption("New predictions appear instantly via SSE. Charts refresh every 10s.")

# --- Navigation ---

view_mode = st.radio(
    "View",
    ["Analytics", "Review Queue", "Activity History"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Analytics ─────────────────────────────────────────────────────────────────

if view_mode == "Analytics":

    sim_df = st.session_state.get("sim_results", {}).get("df")
    sim_latencies = st.session_state.get("sim_results", {}).get("latencies")
    has_sim = sim_df is not None and len(sim_df) > 0

    if not all_predictions and not has_sim:
        st.info("No data yet. Run a simulation above to generate charts.")
        st.stop()

    chart_data = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()

    if not chart_data.empty:
        chart_data["trans_date_trans_time"] = pd.to_datetime(chart_data["trans_date_trans_time"], errors="coerce")

    tabs = st.tabs(["Risk Distribution", "Decisions", "Categories", "Amounts", "Latency"])

    # ── Tab 1: Probability Distribution ──
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            if not chart_data.empty and "probability" in chart_data.columns:
                fig = px.histogram(
                    chart_data,
                    x="probability",
                    color="risk_band",
                    nbins=40,
                    title="Fraud Probability Distribution",
                    labels={"probability": "Fraud Probability", "count": "Transactions", "risk_band": "Risk Band"},
                    color_discrete_map={
                        "LOW": "#22c55e", "MEDIUM": "#eab308",
                        "HIGH": "#f97316", "CRITICAL": "#ef4444",
                    },
                    template="plotly_dark",
                )
                fig.update_layout(
                    bargap=0.05,
                    xaxis=dict(tickformat=".0%"),
                    legend=dict(orientation="h", y=1.12),
                    height=350,
                )
                st.plotly_chart(fig, width='stretch')

        with col2:
            if not chart_data.empty and "probability" in chart_data.columns:
                fig = px.box(
                    chart_data,
                    x="risk_band",
                    y="probability",
                    color="risk_band",
                    title="Probability by Risk Band",
                    labels={"probability": "Fraud Probability", "risk_band": "Risk Band"},
                    color_discrete_map={
                        "LOW": "#22c55e", "MEDIUM": "#eab308",
                        "HIGH": "#f97316", "CRITICAL": "#ef4444",
                    },
                    template="plotly_dark",
                    category_orders={"risk_band": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                )
                fig.update_layout(
                    showlegend=False,
                    yaxis=dict(tickformat=".0%"),
                    height=350,
                )
                st.plotly_chart(fig, width='stretch')

    # ── Tab 2: Decision Breakdown ──
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            if not chart_data.empty and "decision" in chart_data.columns:
                decision_counts = chart_data["decision"].value_counts().reset_index()
                decision_counts.columns = ["decision", "count"]
                fig = go.Figure(
                    go.Pie(
                        labels=decision_counts["decision"],
                        values=decision_counts["count"],
                        hole=0.5,
                        marker=dict(colors=["#22c55e", "#eab308", "#ef4444"]),
                    )
                )
                fig.update_layout(
                    title="Decision Breakdown",
                    template="plotly_dark",
                    height=350,
                    annotations=[dict(text=f"{len(chart_data)}", x=0.5, y=0.5, font_size=24, showarrow=False)],
                )
                st.plotly_chart(fig, width='stretch')

        with col2:
            if not chart_data.empty and "risk_band" in chart_data.columns:
                risk_counts = chart_data["risk_band"].value_counts().reset_index()
                risk_counts.columns = ["risk_band", "count"]
                risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                risk_counts["risk_band"] = pd.Categorical(risk_counts["risk_band"], categories=risk_order, ordered=True)
                risk_counts = risk_counts.sort_values("risk_band")
                fig = px.bar(
                    risk_counts,
                    x="risk_band",
                    y="count",
                    color="risk_band",
                    title="Risk Band Distribution",
                    labels={"risk_band": "Risk Band", "count": "Transactions"},
                    color_discrete_map={
                        "LOW": "#22c55e", "MEDIUM": "#eab308",
                        "HIGH": "#f97316", "CRITICAL": "#ef4444",
                    },
                    template="plotly_dark",
                    category_orders={"risk_band": risk_order},
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, width='stretch')

    # ── Tab 3: Categories ──
    with tabs[2]:
        if not chart_data.empty and "category" in chart_data.columns:
            cat_counts = chart_data.groupby(["category", "risk_band"]).size().reset_index(name="count")
            fig = px.bar(
                cat_counts,
                x="category",
                y="count",
                color="risk_band",
                title="Transactions by Category and Risk Band",
                labels={"category": "Category", "count": "Transactions", "risk_band": "Risk Band"},
                color_discrete_map={
                    "LOW": "#22c55e", "MEDIUM": "#eab308",
                    "HIGH": "#f97316", "CRITICAL": "#ef4444",
                },
                template="plotly_dark",
                barmode="stack",
            )
            fig.update_layout(
                xaxis=dict(tickangle=-45),
                height=400,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, width='stretch')

            flagged = chart_data[chart_data["decision"] == "BLOCK"].groupby("category").size().reset_index(name="count")
            flagged = flagged.sort_values("count", ascending=True)
            if not flagged.empty:
                fig = px.bar(
                    flagged.tail(10),
                    x="count",
                    y="category",
                    orientation="h",
                    title="Top Categories Blocked",
                    labels={"category": "", "count": "Blocked"},
                    color="count",
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                )
                fig.update_layout(height=350, showlegend=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width='stretch')

    # ── Tab 4: Amount Analysis ──
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            if not chart_data.empty and "amt" in chart_data.columns:
                fig = px.scatter(
                    chart_data,
                    x="amt",
                    y="probability",
                    color="decision",
                    title="Amount vs Fraud Probability",
                    labels={"amt": "Transaction Amount ($)", "probability": "Fraud Probability", "decision": "Decision"},
                    color_discrete_map={
                        "APPROVE": "#22c55e", "REVIEW": "#eab308",
                        "BLOCK": "#ef4444", "ERROR": "#6b7280",
                    },
                    template="plotly_dark",
                    opacity=0.6,
                )
                fig.update_layout(
                    height=400,
                    yaxis=dict(tickformat=".0%"),
                    xaxis=dict(tickprefix="$"),
                )
                st.plotly_chart(fig, width='stretch')

        with col2:
            if not chart_data.empty and "amt" in chart_data.columns:
                fig = px.box(
                    chart_data,
                    x="decision",
                    y="amt",
                    color="decision",
                    title="Amount Distribution by Decision",
                    labels={"amt": "Amount ($)", "decision": "Decision"},
                    color_discrete_map={
                        "APPROVE": "#22c55e", "REVIEW": "#eab308",
                        "BLOCK": "#ef4444", "ERROR": "#6b7280",
                    },
                    template="plotly_dark",
                )
                fig.update_layout(showlegend=False, height=400, yaxis=dict(tickprefix="$"))
                st.plotly_chart(fig, width='stretch')

    # ── Tab 5: Latency ──
    with tabs[4]:
        if has_sim and sim_latencies:
            lat_df = pd.DataFrame({"latency_ms": sim_latencies})
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    lat_df,
                    x="latency_ms",
                    nbins=30,
                    title="Prediction Latency Distribution",
                    labels={"latency_ms": "Latency (ms)", "count": "Requests"},
                    template="plotly_dark",
                )
                fig.update_layout(bargap=0.05, height=350)
                st.plotly_chart(fig, width='stretch')

            with col2:
                fig = px.box(
                    lat_df,
                    y="latency_ms",
                    title="Latency Summary",
                    labels={"latency_ms": "Latency (ms)"},
                    template="plotly_dark",
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, width='stretch')

            avg_lat = pd.Series(sim_latencies).mean()
            max_lat = pd.Series(sim_latencies).max()
            p99_lat = pd.Series(sim_latencies).quantile(0.99)
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg latency", f"{avg_lat:.0f} ms")
            m2.metric("P99 latency", f"{p99_lat:.0f} ms")
            m3.metric("Max latency", f"{max_lat:.0f} ms")
        else:
            st.info("Run a simulation above to see latency metrics.")

# ── Review Queue ──────────────────────────────────────────────────────────────

elif view_mode == "Review Queue":

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
                if b1.button("APPROVE", type="primary", width='stretch'):
                    if post_data(f"cases/{selected_id}/decision", {"decision": "APPROVE", "note": note}):
                        get_pending.clear()
                        get_predictions.clear()
                        st.toast("APPROVED")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed. Please retry.")

                if b2.button("BLOCK", width='stretch'):
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

    all_items = st.session_state.get("prediction_items", all_predictions)

    if not all_items:
        st.info("No transaction history yet. Run the simulation above.")
    else:
        df_full = pd.DataFrame(all_items)
        if "trans_date_trans_time" in df_full.columns:
            df_full["trans_date_trans_time"] = pd.to_datetime(df_full["trans_date_trans_time"], errors="coerce")
            df_full = df_full.sort_values("trans_date_trans_time", ascending=False)

        with st.expander("Filters", expanded=False):
            f1, f2, f3, f4, f5 = st.columns(5)

            with f1:
                if "trans_date_trans_time" in df_full.columns and not df_full["trans_date_trans_time"].isna().all():
                    min_date = df_full["trans_date_trans_time"].min().date()
                    max_date = df_full["trans_date_trans_time"].max().date()
                    date_start = st.date_input("From", min_date, min_value=min_date, max_value=max_date, key="f_date_start")
                    date_end = st.date_input("To", max_date, min_value=min_date, max_value=max_date, key="f_date_end")
                else:
                    date_start = date_end = None
                    st.write("No date data")

            with f2:
                categories = sorted(df_full["category"].dropna().unique()) if "category" in df_full.columns else []
                selected_cats = st.multiselect("Category", categories, default=[], key="f_cats")

            with f3:
                bands = [b for b in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] if b in df_full["risk_band"].values]
                selected_bands = st.multiselect("Risk Band", bands, default=[], key="f_bands")

            with f4:
                if "amt" in df_full.columns and not df_full["amt"].isna().all():
                    min_amt = float(df_full["amt"].min())
                    max_amt = float(df_full["amt"].max())
                    amt_range = st.slider("Amount ($)", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt), key="f_amt")
                else:
                    amt_range = None

            with f5:
                search = st.text_input("Merchant", placeholder="Search…", key="f_merchant")

        filtered = df_full.copy()
        if date_start and date_end:
            filtered = filtered[
                (filtered["trans_date_trans_time"].dt.date >= date_start) &
                (filtered["trans_date_trans_time"].dt.date <= date_end)
            ]
        if selected_cats:
            filtered = filtered[filtered["category"].isin(selected_cats)]
        if selected_bands:
            filtered = filtered[filtered["risk_band"].isin(selected_bands)]
        if amt_range:
            filtered = filtered[(filtered["amt"] >= amt_range[0]) & (filtered["amt"] <= amt_range[1])]
        if search:
            filtered = filtered[filtered["merchant"].str.contains(search, case=False, na=False)]

        total_count = len(filtered)
        auto_approved_count = len(filtered[~filtered["requires_review"]])
        flagged_count = len(filtered[filtered["requires_review"]])

        st.markdown(f"#### Summary — {total_count} transactions ({auto_approved_count} auto-approved, {flagged_count} flagged)")

        auto_df = filtered[~filtered["requires_review"]].copy()
        if not auto_df.empty:
            display = auto_df[["trans_date_trans_time", "amt", "category", "merchant", "risk_band", "decision"]].copy()
            display.columns = ["Time", "Amount", "Category", "Merchant", "Risk", "Outcome"]
            display["Amount"] = display["Amount"].map(lambda x: f"${x:,.2f}")
            st.dataframe(display, width='stretch', hide_index=True)
        else:
            st.info("No matching auto-approved transactions.")

        st.divider()
        st.markdown(f"#### Full audit log ({total_count})")

        log_df = filtered[["trans_date_trans_time", "amt", "merchant", "risk_band", "decision", "requires_review"]].copy()
        log_df.columns = ["Time", "Amount", "Merchant", "Risk", "Action", "Review Req."]
        log_df["Amount"] = log_df["Amount"].map(lambda x: f"${x:,.2f}")
        st.dataframe(log_df.head(50), width='stretch', hide_index=True)

        loaded = len(st.session_state["prediction_items"])
        if loaded < total_preds:
            if st.button(f"Load More ({loaded} / {total_preds})", use_container_width=True):
                next_page = get_data(f"predictions?limit=100&offset={st.session_state['prediction_offset']}")
                if isinstance(next_page, dict) and "items" in next_page:
                    existing_ids = {p["id"] for p in st.session_state["prediction_items"]}
                    new_items = [p for p in next_page["items"] if p["id"] not in existing_ids]
                    st.session_state["prediction_items"].extend(new_items)
                    st.session_state["prediction_offset"] += 100
                st.rerun()

# --- Auto-refresh loop ---

if st.session_state.get("live_mode"):
    get_pending.clear()
    get_predictions.clear()
    time.sleep(10)
    st.rerun()
