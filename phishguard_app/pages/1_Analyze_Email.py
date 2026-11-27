"""
PHISHGUARD — Analyze Email page.

Accepts:
  • Manual input (form fields)
  • .eml file upload

Displays results in strict security hierarchy:
  A. Final Verdict (only when Agent 3 succeeded)
  B. Confidence / Probability (only real Agent 3 probability)
  C. Why This Decision? (Detection Indicators)
  D. Agent Breakdown (Agent 1, Agent 2, Agent 3)
  E. Email Information (Sender, Recipient, Subject, Date)
  F. Action: Analyze Another Email
"""
import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st

import config
from parser import build_manual, parse_eml_bytes
from result_presenter import (
    build_agent_breakdown,
    build_detection_indicators,
    build_verdict_summary,
)
from ui.components import (
    divider,
    inject_css,
    page_header,
    render_agent_breakdown_item,
    render_detection_indicators,
    render_email_info,
    render_verdict_banner,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Analyze — {config.APP_TITLE}",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ---------------------------------------------------------------------------
# Shared orchestrator (same cached instance as app.py)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading agents…")
def get_orchestrator():
    from orchestrator import Orchestrator
    return Orchestrator()


orch = get_orchestrator()
a1_meta = orch.agent1.status().get("meta") or {}
threshold = a1_meta.get("threshold_info", {}).get(
    "threshold_prob", config.THRESHOLD_PROB_DEFAULT
)

# ---------------------------------------------------------------------------
# Sidebar — status pills
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<h3 style='margin:0'>🛡️ {config.APP_TITLE}</h3>"
        f"<p style='color:{config.COLORS['text_muted']};margin:0 0 1rem 0'>"
        f"{config.APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    all_status = orch.get_all_status()
    for key in ("agent1", "agent2", "agent3"):
        s = all_status[key]["status"]
        icon = {"ready": "🟢", "unavailable": "🟡", "pending": "⚫", "error": "🔴"}.get(s, "⚫")
        st.markdown(f"{icon} &nbsp;{all_status[key]['name']}", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"v{config.APP_VERSION} · [GitHub]({config.PROJECT_REPO})")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
page_header("🔍 Analyze Email", "Submit an email manually or upload a .eml file.")

if "eml_uploader_key" not in st.session_state:
    st.session_state["eml_uploader_key"] = 0

# ---------------------------------------------------------------------------
# Input tabs: Manual | Upload .eml
# ---------------------------------------------------------------------------
tab_manual, tab_eml = st.tabs(["✏️ Manual Input", "📎 Upload .eml"])

with tab_manual:
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem'>"
        "Fill in as many header fields as possible. "
        "Agent 1 will flag incomplete headers with a warning.</p>",
        unsafe_allow_html=True,
    )
    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        with c1:
            sender   = st.text_input("Sender (From)",  placeholder="alice@example.com")
            date_val = st.text_input("Date",           placeholder="Mon, 01 Jan 2024 10:00:00 +0000")
        with c2:
            receiver = st.text_input("Recipient (To)", placeholder="bob@example.com")
            subject  = st.text_input("Subject",        placeholder="Important notice")
        body = st.text_area(
            "Body (optional — used by Agent 2)",
            height=160,
            placeholder="Paste the email body here…",
        )
        submitted_manual = st.form_submit_button("🔍 Analyze Email", use_container_width=True)

    if submitted_manual:
        email_dict = build_manual(sender, receiver, date_val, subject, body)
        with st.spinner("Running agents…"):
            st.session_state["analyzed_email"] = email_dict
            st.session_state["analysis_results"] = orch.analyze(email_dict)

with tab_eml:
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem'>"
        "Upload a raw <code>.eml</code> file. "
        "Headers and body are extracted automatically.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a .eml file",
        type=["eml"],
        key=f"eml_uploader_{st.session_state['eml_uploader_key']}",
    )
    parsed_eml: dict | None = None
    if uploaded is not None:
        try:
            parsed_eml = parse_eml_bytes(uploaded.read())
            with st.expander("Preview parsed fields"):
                st.json({
                    k: (v[:200] + "…" if len(v) > 200 else v)
                    for k, v in parsed_eml.items()
                })
        except Exception as exc:
            st.error(f"Could not parse .eml: {exc}")
            parsed_eml = None

    if parsed_eml is not None and st.button("🔍 Analyze .eml File", use_container_width=True):
        with st.spinner("Running agents…"):
            st.session_state["analyzed_email"] = parsed_eml
            st.session_state["analysis_results"] = orch.analyze(parsed_eml)

# ---------------------------------------------------------------------------
# Results hierarchy
# ---------------------------------------------------------------------------
if "analysis_results" in st.session_state and "analyzed_email" in st.session_state:
    results = st.session_state["analysis_results"]
    email_dict = st.session_state["analyzed_email"]

    divider()

    # A & B. Final Verdict + Confidence / Probability
    verdict = build_verdict_summary(results, a1_threshold=threshold)
    render_verdict_banner(verdict)

    # C. Why This Decision? — Detection Indicators
    indicators_data = build_detection_indicators(
        results, email_dict, a1_threshold=threshold
    )
    render_detection_indicators(indicators_data)

    divider()

    # D. Agent Breakdown
    st.subheader("Agent Breakdown")
    for item in build_agent_breakdown(results, a1_threshold=threshold):
        render_agent_breakdown_item(item)

    divider()

    # E. Email Information (secondary)
    st.subheader("Email Information")
    render_email_info(email_dict)

    with st.expander("Raw agent outputs (JSON)"):
        st.json(results)

    divider()

    # F. Actions
    if st.button("🔄 Analyze Another Email", use_container_width=True):
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("analyzed_email", None)
        st.session_state["eml_uploader_key"] += 1
        st.rerun()
