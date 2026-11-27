"""
PHISHGUARD — How It Works page.
Explains the system design, decision threshold, label convention, and known limitations.
All metrics are read from real model files — nothing is hardcoded here.
"""
import os, sys
_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st
import config
from ui.components import inject_css, page_header, divider, info_banner, warning_banner

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
a1_status  = orch.agent1.status()
a1_meta    = a1_status.get("meta") or {}
val_m      = a1_meta.get("validation_metrics", {})
cv_m       = a1_meta.get("nested_sender_disjoint_metrics", {})
thresh_inf = a1_meta.get("threshold_info", {})
libs       = a1_meta.get("libraries", {})

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
        s    = all_status[key]["status"]
        icon = {"ready": "🟢", "unavailable": "🟡", "pending": "⚫", "error": "🔴"}.get(s, "⚫")
        st.markdown(f"{icon} &nbsp;{all_status[key]['name']}", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"v{config.APP_VERSION} · [GitHub]({config.PROJECT_REPO})")

page_header("📖 How It Works", "Architecture, decision logic, and known limitations.")

# ---------------------------------------------------------------------------
# System diagram (text-based, reflects live status)
# ---------------------------------------------------------------------------
a2_tag = "READY" if all_status["agent2"]["status"] == "ready" else "UNAVAILABLE"
a3_tag = "READY" if all_status["agent3"]["status"] == "ready" else "UNAVAILABLE"

st.subheader("System Architecture")
st.markdown(
    f"""
    ```
    Email Input (headers + body)
         │
         ├──► Agent 1 (Header Naive Bayes)     [{all_status["agent1"]["status"].upper()}]
         │        • Uses: sender, receiver, date, subject
         │        • Output: nb_score = P(Not Safe)   ← calibrated
         │
         ├──► Agent 2 (DistilBERT)             [{a2_tag}]
         │        • Uses: subject + body (max 256 tokens)
         │        • Output: text_score = P(Not Safe)
         │
         └──► Agent 3 (Logistic Regression)    [{a3_tag}]
                  • Uses: clipped logits [nb_logit, text_logit]
                  • Output: final_probability = P(Not Safe), final_class (0=Safe, 1=Not Safe)
    ```
    """
)

divider()

# ---------------------------------------------------------------------------
# Label convention
# ---------------------------------------------------------------------------
st.subheader("Label Convention")
st.markdown(
    "| Value | Meaning |\n"
    "|-------|---------|\n"
    "| **0** | Safe |\n"
    "| **1** | Not Safe (phishing / spam) |\n\n"
    "Every agent score is **P(Not Safe)**, a float in [0, 1]. "
    "Higher = more likely malicious."
)

divider()

# ---------------------------------------------------------------------------
# Agent 1 — decision threshold (read from real meta)
# ---------------------------------------------------------------------------
st.subheader("Agent 1 — Decision Threshold")

if thresh_inf:
    prob   = thresh_inf.get("threshold_prob", config.THRESHOLD_PROB_DEFAULT)
    logit  = a1_meta.get("threshold_logit", config.THRESHOLD_LOGIT_DEFAULT)
    rule   = a1_meta.get("decision_rule", "f1_opt")
    temp   = a1_meta.get("temperature", "N/A")
    ver    = a1_meta.get("version", "N/A")
    st.markdown(
        f"- **Model version:** `{ver}`\n"
        f"- **Calibration temperature:** {temp:.4f}\n"
        f"- **Decision rule:** `{rule}` — threshold chosen to maximise F1 on the validation set\n"
        f"- **Threshold (log-odds):** {logit:.4f}\n"
        f"- **Threshold (probability):** {prob:.4f} ≈ {prob:.1%}\n\n"
        f"Agent 3 (the meta-classifier) has its own threshold and should be used "
        f"for final verdicts once it is delivered — do not threshold Agent 1 or 2 scores alone."
    )
else:
    info_banner("Agent 1 meta file not loaded — threshold info unavailable.")

divider()

# ---------------------------------------------------------------------------
# Agent 1 — real metrics (read from meta)
# ---------------------------------------------------------------------------
st.subheader("Agent 1 — Validation Metrics (from agent1_binary_meta.json)")

if val_m:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Hold-out validation set**")
        st.markdown(
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Accuracy  | {val_m.get('accuracy', '—'):.4f} |\n"
            f"| Precision | {val_m.get('precision', '—'):.4f} |\n"
            f"| Recall    | {val_m.get('recall', '—'):.4f} |\n"
            f"| F1        | {val_m.get('f1', '—'):.4f} |\n"
            f"| ROC-AUC   | {val_m.get('roc_auc', '—'):.4f} |\n"
            f"| Log-Loss  | {val_m.get('log_loss', '—'):.4f} |\n"
            f"| ECE       | {val_m.get('ece', '—'):.4f} |"
        )
    with col_b:
        st.markdown("**Nested sender-disjoint CV**")
        st.markdown(
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Accuracy  | {cv_m.get('accuracy', '—'):.4f} |\n"
            f"| Precision | {cv_m.get('precision', '—'):.4f} |\n"
            f"| Recall    | {cv_m.get('recall', '—'):.4f} |\n"
            f"| F1        | {cv_m.get('f1', '—'):.4f} |\n"
            f"| ROC-AUC   | {cv_m.get('roc_auc', '—'):.4f} |\n"
            f"| Log-Loss  | {cv_m.get('log_loss', '—'):.4f} |\n"
            f"| ECE       | {cv_m.get('ece', '—'):.4f} |"
        )
else:
    info_banner("Metrics not available — Agent 1 meta file not loaded.")

divider()

# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------
st.subheader("Known Limitations")

st.markdown(
    """
**Agent 1 (Header Naive Bayes)**
- **Headers only.** The model sees sender, recipient, date and subject. It has no access to the email body.
- **Corpus habits.** "Safe" and "Not Safe" examples come partly from different corpora. Some header features
  may reflect corpus conventions rather than genuine phishing signals.
- **Compromised accounts.** A malicious email sent from a legitimate, hijacked account will have clean headers
  and may score low (Safe). Header analysis cannot detect this.
- **Empty headers → unreliable score.** When all four headers are blank, the model returns nb_score ≈ 0.98
  with `insufficient_evidence = 1`. The UI displays a warning in this case.

**Agent 2 (DistilBERT)**
- Truncates input to **256 tokens**; very long emails are truncated.
- CPU inference is supported without a GPU.

**Agent 3 (Meta-Classifier)**
- Combines `[nb_score, text_score]` via clipped logits (`eps = 1e-6`, `clip = 10.0`) using Logistic Regression (`C = 0.1`, threshold `0.5`).
- Requires valid continuous probabilities from both Agent 1 and Agent 2; if either upstream agent fails or is unavailable, no final verdict is produced.

**Experimental System Notice**
- PHISHGUARD is an experimental academic detection system and may produce false positives or false negatives.
"""
)

divider()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
st.subheader("Agent 1 Environment")
if libs:
    st.markdown(
        f"| Library | Version |\n"
        f"|---------|--------|\n"
        f"| Python     | {libs.get('python', '—')} |\n"
        f"| numpy      | {libs.get('numpy', '—')} |\n"
        f"| pandas     | {libs.get('pandas', '—')} |\n"
        f"| scikit-learn | {libs.get('sklearn', '—')} |\n"
        f"| scipy      | {libs.get('scipy', '—')} |"
    )
else:
    info_banner("Library versions not available.")
