"""
PHISHGUARD — Analyze Email page.

Accepts:
  • Manual input (form fields + optional real sample prefill)
  • .eml file upload

Displays results in strict security hierarchy:
  A. Final Verdict (only when Agent 3 succeeded)
  B. Confidence / Probability (only real Agent 3 probability)
  C. Score Comparison Chart & Detection Indicators
  D. Agent Breakdown (Agent 1, Agent 2, Agent 3)
  E. Email Information (Sender, Recipient, Subject, Date)
  F. Action: Analyze Another Email
"""
import json
import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st

import config
from link_advisory import analyze_links_advisory
from parser import build_manual, extract_html_from_eml_bytes, parse_eml_bytes
from result_presenter import (
    append_session_history,
    build_agent_breakdown,
    build_detection_indicators,
    build_disagreement_summary,
    build_fusion_explanation,
    build_verdict_summary,
    resolve_agent_thresholds,
)
from ui.components import (
    divider,
    inject_css,
    page_header,
    render_agent_breakdown_item,
    render_detection_indicators,
    render_disagreement_banner,
    render_email_info,
    render_link_advisory,
    render_score_comparison_chart,
    render_sidebar,
    render_verdict_banner,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Analyze Email — {config.APP_TITLE}",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_resource(show_spinner="Loading agents…")
def get_orchestrator():
    from orchestrator import Orchestrator
    return Orchestrator()


orch = get_orchestrator()
all_status = orch.get_all_status()
render_sidebar(all_status)

thresholds = resolve_agent_thresholds(all_status)
a1_threshold = thresholds["agent1"]
a2_threshold = thresholds["agent2"]
a3_threshold = thresholds["agent3"]
a3_meta = all_status["agent3"].get("meta")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "eml_uploader_key" not in st.session_state:
    st.session_state["eml_uploader_key"] = 0
if "session_history" not in st.session_state:
    st.session_state["session_history"] = []


def _record_run(email_dict: dict, results: dict, html_body: str = ""):
    st.session_state["analyzed_email"] = email_dict
    st.session_state["analysis_results"] = results
    st.session_state["analysis_html"] = html_body
    st.session_state["last_analysis"] = results
    st.session_state["session_history"] = append_session_history(
        st.session_state["session_history"], email_dict, results
    )


def _load_sample_into_form(kind: str):
    ref_path = config.AGENT1_DIR / "reference_examples.json"
    if not ref_path.is_file():
        return
    try:
        examples = json.loads(ref_path.read_text(encoding="utf-8"))
        target = 0 if kind == "safe" else 1
        ex = next((e for e in examples if e.get("true_label") == target), None)
        if ex:
            st.session_state["manual_sender"] = ex.get("sender", "")
            st.session_state["manual_receiver"] = ex.get("receiver", "")
            st.session_state["manual_date"] = ex.get("date", "")
            st.session_state["manual_subject"] = ex.get("subject", "")
            st.session_state["manual_body"] = (
                "Hi team, please review the attached notes before tomorrow's sync."
                if target == 0 else
                "URGENT: Verify your account details immediately at http://verify-login.example/auth."
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
page_header("🔍 Analyze Email", "Run live 3-agent inspection via manual entry or raw .eml upload.")

tab_manual, tab_eml = st.tabs(["✏️ Manual Entry", "📎 Upload .eml"])

with tab_manual:
    p_col1, p_col2, _ = st.columns([1, 1, 2])
    with p_col1:
        if st.button("Prefill Safe Example", use_container_width=True):
            _load_sample_into_form("safe")
    with p_col2:
        if st.button("Prefill Suspicious Example", use_container_width=True):
            _load_sample_into_form("suspicious")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        with c1:
            sender   = st.text_input("Sender (From)", key="manual_sender", placeholder="alice@example.com")
            date_val = st.text_input("Date", key="manual_date", placeholder="Mon, 01 Jan 2024 10:00:00 +0000")
        with c2:
            receiver = st.text_input("Recipient (To)", key="manual_receiver", placeholder="bob@example.com")
            subject  = st.text_input("Subject", key="manual_subject", placeholder="Important notice")
        body = st.text_area(
            "Email Body (used by Agent 2)",
            key="manual_body",
            height=150,
            placeholder="Paste the email body here…",
        )
        submitted_manual = st.form_submit_button("🔍 Analyze Email", use_container_width=True)

    if submitted_manual:
        email_dict = build_manual(sender, receiver, date_val, subject, body)
        with st.spinner("Running Agent 1, Agent 2, and Agent 3…"):
            _record_run(email_dict, orch.analyze(email_dict))

with tab_eml:
    st.caption("Upload a raw `.eml` file. Headers and body are extracted automatically in memory.")
    uploaded = st.file_uploader(
        "Choose a .eml file",
        type=["eml"],
        key=f"eml_uploader_{st.session_state['eml_uploader_key']}",
    )
    parsed_eml: dict | None = None
    parsed_html: str = ""
    if uploaded is not None:
        try:
            raw_eml_bytes = uploaded.read()
            parsed_eml = parse_eml_bytes(raw_eml_bytes)
            parsed_html = extract_html_from_eml_bytes(raw_eml_bytes)
            with st.expander("Preview parsed fields"):
                st.json({
                    k: (v[:200] + "…" if len(v) > 200 else v)
                    for k, v in parsed_eml.items()
                })
        except Exception as exc:
            st.error(f"Could not parse .eml: {exc}")
            parsed_eml = None
            parsed_html = ""

    if parsed_eml is not None and st.button("🔍 Analyze .eml File", type="primary", use_container_width=True):
        with st.spinner("Running Agent 1, Agent 2, and Agent 3…"):
            _record_run(parsed_eml, orch.analyze(parsed_eml), html_body=parsed_html)

# ---------------------------------------------------------------------------
# Results hierarchy
# ---------------------------------------------------------------------------
if "analysis_results" in st.session_state and "analyzed_email" in st.session_state:
    results    = st.session_state["analysis_results"]
    email_dict = st.session_state["analyzed_email"]

    divider()

    # A & B. Final Verdict + Probability
    verdict = build_verdict_summary(results, a1_threshold=a1_threshold)
    render_verdict_banner(verdict)

    # Disagreement banner (only shown when both agents & thresholds are valid and disagree)
    disagreement = build_disagreement_summary(results, thresholds, a3_meta=a3_meta)
    render_disagreement_banner(disagreement)

    # C. Score Chart + Detection Indicators
    col_chart, col_ind = st.columns([1, 1])
    with col_chart:
        render_score_comparison_chart(results)
    with col_ind:
        indicators_data = build_detection_indicators(
            results, email_dict, a1_threshold=a1_threshold, a2_threshold=a2_threshold
        )
        render_detection_indicators(indicators_data)
        fusion_info = build_fusion_explanation(a3_meta)
        with st.expander("How Agent 3 combines the two scores"):
            st.write(fusion_info["summary"])

    divider()

    # D. Agent Breakdown (3-column cards)
    st.subheader("Agent Breakdown")
    breakdown = build_agent_breakdown(
        results,
        a1_threshold=a1_threshold,
        a2_threshold=a2_threshold,
        a3_threshold=a3_threshold,
    )
    b1, b2, b3 = st.columns(3)
    for col, item in zip((b1, b2, b3), breakdown):
        with col:
            render_agent_breakdown_item(item)

    divider()

    # D2. Offline Link-Safety Advisory (strictly non-model, display-only)
    advisory_data = analyze_links_advisory(
        email_dict, html_body=st.session_state.get("analysis_html") or None
    )
    render_link_advisory(advisory_data)

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
        st.session_state.pop("analysis_html", None)
        st.session_state["eml_uploader_key"] += 1
        st.rerun()
