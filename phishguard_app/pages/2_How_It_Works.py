"""
PHISHGUARD — How It Works page.
Explains the multi-agent architecture, Agent 1 calibration vs Agent 3 threshold,
fusion weights, offline metrics, and known limitations.
"""
import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import pandas as pd
import streamlit as st

import config
from ui.components import (
    divider,
    info_banner,
    inject_css,
    page_header,
    render_confusion_matrix_chart,
    render_fusion_weights_chart,
    render_pipeline_flow,
    render_shared_test_comparison_chart,
    render_sidebar,
)

st.set_page_config(
    page_title=f"How It Works — {config.APP_TITLE}",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_resource(show_spinner=False)
def get_orchestrator():
    from orchestrator import Orchestrator
    return Orchestrator()


orch       = get_orchestrator()
all_status = orch.get_all_status()
render_sidebar(all_status)

a1_meta    = all_status["agent1"].get("meta") or {}
a3_meta    = all_status["agent3"].get("meta") or {}
a3_cfg     = all_status["agent3"].get("config") or {}
val_m      = a1_meta.get("validation_metrics", {})
cv_m       = a1_meta.get("nested_sender_disjoint_metrics", {})
thresh_inf = a1_meta.get("threshold_info", {})
libs       = a1_meta.get("libraries", {})

page_header("📖 How It Works", "Multi-agent architecture, calibration, fusion weights, and limitations.")

# ---------------------------------------------------------------------------
# Visual Pipeline Flow
# ---------------------------------------------------------------------------
st.subheader("System Architecture")
render_pipeline_flow(all_status)

st.markdown(
    "| Binary Label | Meaning |\n"
    "|---|---|\n"
    "| **0** | **Safe** |\n"
    "| **1** | **Not Safe** (phishing and spam combined) |\n\n"
    "Every agent outputs a continuous probability **P(Not Safe) ∈ [0, 1]**. "
    "Agent 3 converts `nb_score` and `text_score` to clipped logits (`nb_logit`, `text_logit`) "
    "and applies a fixed **0.50** decision threshold."
)

divider()

# ---------------------------------------------------------------------------
# Calibration vs. Final Threshold
# ---------------------------------------------------------------------------
st.subheader("Agent 1 Calibration vs. Agent 3 Decision Threshold")
col_c1, col_c3 = st.columns(2)

with col_c1:
    st.markdown("**Agent 1 — Log-Odds Temperature Calibration**")
    if thresh_inf:
        prob  = thresh_inf.get("threshold_prob", config.THRESHOLD_PROB_DEFAULT)
        logit = a1_meta.get("threshold_logit", config.THRESHOLD_LOGIT_DEFAULT)
        rule  = a1_meta.get("decision_rule", "f1_opt")
        temp  = a1_meta.get("temperature", 0.0)
        ver   = a1_meta.get("version", "—")
        st.markdown(
            f"- **Model version:** `{ver}`\n"
            f"- **Calibration temperature:** `T = {temp:.4f}` (fitted on inner OOF logits to soften Naive Bayes overconfidence)\n"
            f"- **Agent 1 reference threshold:** `{rule}` (`logit = {logit:.4f}`, `prob ≈ {prob:.4f}`)\n"
            f"- **Downstream role:** Passes calibrated `nb_score` to Agent 3; Agent 1's own `pred_class` is **not** used as the final verdict."
        )
    else:
        info_banner("Agent 1 meta file not loaded.")

with col_c3:
    st.markdown("**Agent 3 — Logistic Regression Fusion**")
    if a3_meta:
        coeffs = a3_meta.get("coefficients", {})
        lt     = a3_cfg.get("logit_transform", {})
        st.markdown(
            f"- **Training split:** `{a3_meta.get('training_split', 'val')}` · **Evaluation split:** `{a3_meta.get('evaluation_split', 'test')}`\n"
            f"- **Preprocessing:** Clipped logit (`eps={lt.get('eps', 1e-6)}`, `clip={lt.get('clip', 10.0)}`) in order `['nb_logit', 'text_logit']`\n"
            f"- **Final decision threshold:** Fixed `{a3_cfg.get('decision_threshold', 0.5):.2f}` (`threshold_selection_method = {a3_meta.get('threshold_selection_method', 'fixed_0.5')}`)\n"
            f"- **Regularization:** `C = {a3_meta.get('best_C', 0.1)}`"
        )
        if coeffs:
            render_fusion_weights_chart(coeffs)
    else:
        info_banner("Agent 3 metadata not loaded.")

divider()

# ---------------------------------------------------------------------------
# Offline Evaluation Charts
# ---------------------------------------------------------------------------
st.subheader("Offline Evaluation Metrics")
tm = a3_meta.get("test_metrics", {})
if tm:
    c_test, c_cm = st.columns([3, 2])
    with c_test:
        render_shared_test_comparison_chart(tm)
    with c_cm:
        cm3 = tm.get("agent3_fusion", {}).get("confusion_matrix")
        if cm3:
            render_confusion_matrix_chart(cm3, "Agent 3 Fusion — Test Confusion Matrix (7,343)")

if val_m:
    st.markdown("**Agent 1 — Validation Split (7,340) vs. Nested Sender-Disjoint CV**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Hold-out validation set (7,340 rows)**")
        st.dataframe(
            pd.DataFrame([{
                "Accuracy": f"{val_m.get('accuracy', 0):.4f}",
                "Precision": f"{val_m.get('precision', 0):.4f}",
                "Recall": f"{val_m.get('recall', 0):.4f}",
                "F1": f"{val_m.get('f1', 0):.4f}",
                "ROC-AUC": f"{val_m.get('roc_auc', 0):.4f}",
                "Log-Loss": f"{val_m.get('log_loss', 0):.4f}",
                "ECE": f"{val_m.get('ece', 0):.4f}",
            }]),
            use_container_width=True,
            hide_index=True,
        )
    with col_b:
        st.markdown("**Nested sender-disjoint CV**")
        st.dataframe(
            pd.DataFrame([{
                "Accuracy": f"{cv_m.get('accuracy', 0):.4f}",
                "Precision": f"{cv_m.get('precision', 0):.4f}",
                "Recall": f"{cv_m.get('recall', 0):.4f}",
                "F1": f"{cv_m.get('f1', 0):.4f}",
                "ROC-AUC": f"{cv_m.get('roc_auc', 0):.4f}",
                "Log-Loss": f"{cv_m.get('log_loss', 0):.4f}",
                "ECE": f"{cv_m.get('ece', 0):.4f}",
            }]),
            use_container_width=True,
            hide_index=True,
        )

divider()

# ---------------------------------------------------------------------------
# Known Limitations & Environment
# ---------------------------------------------------------------------------
st.subheader("Known Limitations")
st.markdown(
    """
- **Agent 1 (Headers only):** Sees only `sender`, `receiver`, `date`, `subject`. Compromised legitimate accounts can have clean headers, and empty headers yield `insufficient_evidence = 1` (`nb_score ≈ 0.98` marked unreliable).
- **Corpus confounding:** Safe and Spam come from `CEAS_08` / `SpamAssasin`, while Phishing comes from `Nazario` / `Nigerian_Fraud`.
- **Agent 2 (DistilBERT):** Truncates inputs to **256 tokens** (`subject + body`).
- **Explainability:** Current models output calibrated probabilities without token-level or header-field attributions.
- **Experimental Notice:** PHISHGUARD is an experimental academic system and may produce false positives or false negatives.
"""
)

if libs:
    with st.expander("Agent 1 & Agent 3 Serialization Environment"):
        st.markdown(
            f"Python `{libs.get('python', '—')}` · numpy `{libs.get('numpy', '—')}` · "
            f"pandas `{libs.get('pandas', '—')}` · scikit-learn `{libs.get('sklearn', '—')}` · "
            f"scipy `{libs.get('scipy', '—')}`"
        )
