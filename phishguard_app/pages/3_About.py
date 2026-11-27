"""
PHISHGUARD — About page.
Project description, dataset info (no fabricated stats), team.
"""
import os, sys
_APP_DIR = os.path.dirname(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st
import config
from ui.components import inject_css, page_header, divider

st.set_page_config(
    page_title=f"About — {config.APP_TITLE}",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

@st.cache_resource(show_spinner=False)
def get_orchestrator():
    from orchestrator import Orchestrator
    return Orchestrator()

orch = get_orchestrator()

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

page_header("ℹ️ About PHISHGUARD")

st.markdown(
    f"""
**PHISHGUARD** is a university project implementing a Multi-Agent Anti-Phishing Firewall.
Three independent ML agents analyse an email and report their probability that it is "Not Safe".

- **Repository:** [{config.PROJECT_REPO}]({config.PROJECT_REPO})
- **Version:** {config.APP_VERSION}
- **Stack:** Python 3.13, Streamlit, scikit-learn, (PyTorch / 🤗 Transformers — Agent 2)
"""
)

divider()

st.subheader("Dataset")
st.markdown(
    """
The model was trained and evaluated on a combined dataset of:
- **CEAS 2008 Challenge** spam/ham corpus
- **SpamAssassin** public corpus

| Split | Rows |
|-------|------|
| Train | 34,256 |
| Val   | 7,340  |
| Test  | 7,343  |
| **Total** | **48,939** |

SHA-256 of `emails_clean.csv`:
`95155f7d9ab86220f08a444e37ec2d7b8965ec88ab51834a03544c8894b67d0f`

The dataset CSV is **not** included in the repository (it is in `.gitignore`).
Only trained model artifacts are committed.
"""
)

divider()

st.subheader("Agent Summary")
a1_s = "✅ Ready" if all_status["agent1"]["status"] == "ready" else "⚠️ Unavailable"
a2_s = "✅ Ready" if all_status["agent2"]["status"] == "ready" else "⚠️ Unavailable"
a3_s = "✅ Ready" if all_status["agent3"]["status"] == "ready" else "⚠️ Unavailable"
st.markdown(
    f"""
| Agent | Method | Input | Status |
|-------|--------|-------|--------|
| Agent 1 | Smoothed Categorical Naive Bayes (binary, temperature-calibrated) | Headers (sender, receiver, date, subject) | {a1_s} |
| Agent 2 | DistilBERT fine-tuned sequence classifier | Subject + body (max 256 tokens) | {a2_s} |
| Agent 3 | Logistic Regression meta-classifier | Clipped logits `[nb_logit, text_logit]` | {a3_s} |
"""
)

divider()

st.subheader("How to Run")
st.code(
    "# From the repo root:\n"
    "pip install -r phishguard_app/requirements_app.txt\n"
    "streamlit run phishguard_app/app.py",
    language="bash",
)
