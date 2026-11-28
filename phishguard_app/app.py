"""
PHISHGUARD — Main Cybersecurity Dashboard (Entry Point).
Run from the REPO ROOT:  streamlit run phishguard_app/app.py
"""
import os
import sys

_APP_DIR = os.path.dirname(__file__)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import pandas as pd
import streamlit as st

import config
from formatting import format_probability
from result_presenter import compute_session_kpis
from ui.components import (
    divider,
    inject_css,
    kpi_tile,
    render_confusion_matrix_chart,
    render_overview_agent_card,
    render_pipeline_flow,
    render_shared_test_comparison_chart,
    render_sidebar,
    status_badge,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{config.APP_TITLE} — Email Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_resource(show_spinner="Loading PHISHGUARD agents…")
def get_orchestrator():
    from orchestrator import Orchestrator
    return Orchestrator()


orch = get_orchestrator()
all_status = orch.get_all_status()
render_sidebar(all_status)

c = config.COLORS
session_history = st.session_state.get("session_history", [])
last_analysis   = st.session_state.get("last_analysis")
kpis            = compute_session_kpis(session_history)

ready_count = sum(
    1 for k in ("agent1", "agent2", "agent3")
    if all_status[k]["status"] == "ready"
)
health_badge = status_badge("ready" if ready_count == 3 else ("unavailable" if ready_count > 0 else "error"))

# ---------------------------------------------------------------------------
# 1. Welcome Section
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="pg-hero">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.75rem">
        <div>
          <div style="font-size:0.74rem;font-weight:700;letter-spacing:0.1em;
                      text-transform:uppercase;color:{c['primary']}">
            Multi-Agent AI Email Security Platform
          </div>
          <h1 style="margin:0.15rem 0 0.35rem 0;font-size:1.95rem;font-weight:800;color:{c['text']}">
            🛡️ {config.APP_TITLE}
          </h1>
          <p style="margin:0;color:{c['text_muted']};max-width:740px;font-size:0.94rem">
            Real-time binary email threat detection (<strong>0 = Safe</strong>, <strong>1 = Not Safe</strong>)
            combining header Naive Bayes, DistilBERT text analysis, and Logistic Regression fusion.
          </p>
        </div>
        <div style="text-align:right">
          {health_badge}
          <div style="font-size:0.78rem;color:{c['text_muted']};margin-top:0.35rem">
            {ready_count} of 3 agents operational
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cta_col, flow_col = st.columns([1, 3])
with cta_col:
    if st.button("🔍 Analyze Email →", type="primary", use_container_width=True):
        st.switch_page("pages/1_Analyze_Email.py")
    st.caption("Submit manual headers/body or upload a `.eml` file.")
with flow_col:
    render_pipeline_flow(all_status)

divider()

# ---------------------------------------------------------------------------
# 2. System Overview Cards (Live Session Metrics Only)
# ---------------------------------------------------------------------------
st.subheader("Session Overview")
st.caption("Live counts from the current browser session only (not offline dataset totals).")

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_tile("Emails Analyzed", str(kpis["total"]), "Current session", c["secondary"])
with k2:
    kpi_tile("Safe (0)", str(kpis["safe"]), "Verified Safe in session", c["safe"])
with k3:
    kpi_tile("Not Safe (1)", str(kpis["not_safe"]), "Phishing / spam flagged", c["not_safe"])
with k4:
    kpi_tile("Analysis Failures", str(kpis["failures"]), "Incomplete or failed runs", c["warning"])

if kpis["total"] == 0:
    st.markdown(
        f"<div class='pg-card' style='border-left:4px solid {c['primary']}'>"
        f"<strong>No emails analyzed in this session yet.</strong> "
        f"<span style='color:{c['text_muted']}'>Click <em>Analyze Email</em> above to run a live 3-agent scan.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

divider()

# ---------------------------------------------------------------------------
# 3. Three-Agent Status Section
# ---------------------------------------------------------------------------
st.subheader("Three-Agent Architecture Status")

a1_st = all_status["agent1"]
a2_st = all_status["agent2"]
a3_st = all_status["agent3"]
a1_meta = a1_st.get("meta") or {}
a2_meta = a2_st.get("meta") or {}
a3_meta = a3_st.get("meta") or {}

last_a1 = (last_analysis or {}).get("agent1", {})
last_a2 = (last_analysis or {}).get("agent2", {})
last_a3 = (last_analysis or {}).get("agent3", {})

col1, col2, col3 = st.columns(3)

with col1:
    a1_specs = [
        ("Algorithm", "Naive Bayes"),
        ("Modality", "Headers only"),
        ("Last nb_score", format_probability(last_a1["nb_score"]) if last_a1.get("available") else "—"),
        (
            "Evidence Flag",
            ("Insufficient ⚠" if last_a1.get("insufficient_evidence") else "Sufficient")
            if last_a1.get("available") else "—",
        ),
        (
            "Last Latency",
            f"{last_a1['duration_ms']:.1f} ms" if isinstance(last_a1.get("duration_ms"), (int, float)) else "—",
        ),
    ]
    render_overview_agent_card(
        "Agent 1 — Header Analysis",
        "Smoothed Categorical NB · sender, receiver, date, subject",
        c["agent1"],
        a1_st,
        a1_specs,
    )
    with st.expander("Agent 1 Technical Details"):
        st.markdown(
            f"- **Loading status:** `{a1_st['status']}`\n"
            f"- **Version:** `{a1_meta.get('version', '—')}`\n"
            f"- **Calibration temperature:** `{a1_meta.get('temperature', 0.0):.4f}`\n"
            f"- **Agent 1 operating threshold:** `prob ≈ {a1_meta.get('threshold_info', {}).get('threshold_prob', 0.525):.4f}` (for reference only; Agent 3 performs final classification)\n"
            f"- **Feature set:** `{a1_meta.get('selected_config', {}).get('feature_set', '—')}`"
        )
        if last_a1.get("error"):
            st.error(last_a1["error"])

with col2:
    tok_ok = "Loaded" if a2_st.get("tokenizer_loaded") else "Not loaded"
    dev_str = str(a2_st.get("device") or "—").upper()
    a2_specs = [
        ("Algorithm", "DistilBERT"),
        ("Tokenizer", tok_ok),
        ("Device / MaxLen", f"{dev_str} / {a2_st.get('max_length', 256)}"),
        ("Last text_score", format_probability(last_a2["text_score"]) if last_a2.get("available") else "—"),
        (
            "Last Latency",
            f"{last_a2['duration_ms']:.1f} ms" if isinstance(last_a2.get("duration_ms"), (int, float)) else "—",
        ),
    ]
    render_overview_agent_card(
        "Agent 2 — Content Analysis",
        "DistilBERT Transformer · subject + body (max 256 tokens)",
        c["agent2"],
        a2_st,
        a2_specs,
    )
    with st.expander("Agent 2 Technical Details"):
        st.markdown(
            f"- **Model loaded:** `{a2_st.get('model_loaded', False)}`\n"
            f"- **Tokenizer loaded:** `{a2_st.get('tokenizer_loaded', False)}`\n"
            f"- **Device:** `{a2_st.get('device') or '—'}`\n"
            f"- **Max sequence length:** `{a2_st.get('max_length', 256)}` tokens\n"
            f"- **Checkpoint:** `{a2_meta.get('model_checkpoint', 'distilbert-base-uncased')}`"
        )
        if last_a2.get("error"):
            st.error(last_a2["error"])

with col3:
    upstream_ok = "Ready (A1+A2)" if (a1_st["status"] == "ready" and a2_st["status"] == "ready") else "Blocked"
    a3_cls_str = (
        ("Not Safe (1)" if last_a3["final_class"] == 1 else "Safe (0)")
        if last_a3.get("available") else "—"
    )
    a3_specs = [
        ("Algorithm", "Logistic Reg."),
        ("Upstream Inputs", upstream_ok),
        ("Threshold", f"{(a3_st.get('config') or {}).get('decision_threshold', 0.5):.2f}"),
        ("Last Probability", format_probability(last_a3["final_probability"]) if last_a3.get("available") else "—"),
        ("Last Class", a3_cls_str),
    ]
    render_overview_agent_card(
        "Agent 3 — Decision Engine",
        "Logistic Regression Fusion · clipped logits [nb_logit, text_logit]",
        c["agent3"],
        a3_st,
        a3_specs,
    )
    with st.expander("Agent 3 Technical Details"):
        coeffs = a3_meta.get("coefficients", {})
        cfg3   = a3_st.get("config") or {}
        st.markdown(
            f"- **Loading status:** `{a3_st['status']}`\n"
            f"- **Feature order:** `{cfg3.get('model_features', ['nb_logit', 'text_logit'])}`\n"
            f"- **Logit transform:** `eps={cfg3.get('logit_transform', {}).get('eps', 1e-6)}`, `clip={cfg3.get('logit_transform', {}).get('clip', 10.0)}`\n"
            f"- **Regularization:** `C={a3_meta.get('best_C', '—')}` · **Threshold:** `{cfg3.get('decision_threshold', 0.5)}`\n"
            + (
                f"- **Weights:** `nb_logit={coeffs.get('nb_logit', 0):.4f}`, "
                f"`text_logit={coeffs.get('text_logit', 0):.4f}`, `intercept={coeffs.get('intercept', 0):.4f}`"
                if coeffs else ""
            )
        )
        if last_a3.get("error"):
            st.error(last_a3["error"])

divider()

# ---------------------------------------------------------------------------
# 4. Model Performance Section (Offline Evaluation Only — Splits Separated)
# ---------------------------------------------------------------------------
st.subheader("Model Performance (Offline Evaluation)")
st.caption(
    "Offline evaluation metrics read from saved metadata files. "
    "Validation (7,340 emails) and Test (7,343 emails) splits are kept strictly separate."
)

tab_test, tab_val = st.tabs([
    "Shared Test Split (7,343 Emails)",
    "Validation Split (7,340 Emails)",
])

with tab_test:
    tm = a3_meta.get("test_metrics", {})
    if tm:
        def _fpr_from_cm(cm):
            if not cm:
                return None
            tn, fp = cm[0][0], cm[0][1]
            return fp / (fp + tn) if (fp + tn) > 0 else 0.0

        t_rows = []
        for k, label in [
            ("agent1_alone", "Agent 1 (Header NB)"),
            ("agent2_alone", "Agent 2 (DistilBERT)"),
            ("agent3_fusion", "Agent 3 (LogReg Fusion)"),
        ]:
            m = tm.get(k)
            if m:
                fpr = _fpr_from_cm(m.get("confusion_matrix"))
                t_rows.append({
                    "Agent": label,
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1": f"{m['f1']:.4f}",
                    "ROC-AUC": f"{m['roc_auc']:.4f}",
                    "FPR": f"{fpr:.4f}" if fpr is not None else "—",
                })
        st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

        c_chart, c_cm = st.columns([3, 2])
        with c_chart:
            render_shared_test_comparison_chart(tm)
        with c_cm:
            cm3 = tm.get("agent3_fusion", {}).get("confusion_matrix")
            if cm3:
                render_confusion_matrix_chart(cm3, "Agent 3 Fusion — Test Confusion Matrix (7,343)")
    else:
        st.info("Shared test metrics (`agent3_meta.json`) are not available.")

with tab_val:
    v1 = a1_meta.get("validation_metrics", {})
    v2 = a2_meta.get("validation", {})
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("**Agent 1 — Validation Split (7,340 emails)**")
        if v1:
            st.dataframe(
                pd.DataFrame([{
                    "Accuracy": f"{v1['accuracy']:.4f}",
                    "Precision": f"{v1['precision']:.4f}",
                    "Recall": f"{v1['recall']:.4f}",
                    "F1": f"{v1['f1']:.4f}",
                    "ROC-AUC": f"{v1['roc_auc']:.4f}",
                    "FPR": f"{v1['fpr']:.4f}",
                    "Log-Loss": f"{v1['log_loss']:.4f}",
                }]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Agent 1 validation metrics unavailable.")

    with col_v2:
        st.markdown("**Agent 2 — Validation Split (7,340 emails)**")
        if v2:
            st.dataframe(
                pd.DataFrame([{
                    "Accuracy": f"{v2['accuracy']:.4f}",
                    "Precision": f"{v2['precision']:.4f}",
                    "Recall": f"{v2['recall']:.4f}",
                    "F1": f"{v2['f1']:.4f}",
                    "ROC-AUC": f"{v2['roc_auc']:.4f}",
                    "FPR": f"{v2['fpr']:.4f}",
                }]),
                use_container_width=True,
                hide_index=True,
            )
            cm2_dict = v2.get("confusion_matrix")
            if cm2_dict:
                cm2 = [
                    [cm2_dict["tn"], cm2_dict["fp"]],
                    [cm2_dict["fn"], cm2_dict["tp"]],
                ]
                render_confusion_matrix_chart(cm2, "Agent 2 — Validation Confusion Matrix (7,340)")
        else:
            st.info("Agent 2 validation metrics (`agent2_metrics.json`) unavailable.")

divider()

# ---------------------------------------------------------------------------
# 5. Recent Activity (Session-Based Only)
# ---------------------------------------------------------------------------
st.subheader("Recent Session Activity")
if not session_history:
    st.markdown(
        f"<div class='pg-card'><span style='color:{c['text_muted']}'>"
        f"No emails have been analyzed in this session yet. History is stored in memory only "
        f"for the active browser session and is never saved to disk.</span></div>",
        unsafe_allow_html=True,
    )
else:
    hist_df = pd.DataFrame([
        {
            "Time": h["timestamp"],
            "Subject": h["subject"],
            "Verdict": h["verdict"],
            "P(Not Safe)": format_probability(h["final_probability"]),
            "Status": h["status"],
            "Latency (ms)": f"{h['duration_ms']:.1f}",
        }
        for h in session_history
    ])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
