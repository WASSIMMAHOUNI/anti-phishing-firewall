"""
PHISHGUARD — Main Streamlit entry point.
Run from the REPO ROOT:  streamlit run phishguard_app/app.py

Homepage shows the agent status dashboard.
"""
import os
import sys

# Make phishguard_app/ importable regardless of where Streamlit is invoked
_APP_DIR = os.path.dirname(__file__)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st

import config
from ui.components import inject_css, agent_status_card, divider, page_header

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ---------------------------------------------------------------------------
# Load orchestrator once (cached across reruns and pages)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading agents…")
def get_orchestrator():
    from orchestrator import Orchestrator
    return Orchestrator()


orch = get_orchestrator()
all_status = orch.get_all_status()

# ---------------------------------------------------------------------------
# Sidebar — persistent across pages
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<h3 style='margin:0'>🛡️ {config.APP_TITLE}</h3>"
        f"<p style='color:{config.COLORS['text_muted']};margin:0 0 1rem 0'>"
        f"{config.APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Agent Status**")

    for key, color_key in [
        ("agent1", "agent1"),
        ("agent2", "agent2"),
        ("agent3", "agent3"),
    ]:
        s = all_status[key]["status"]
        icon = {"ready": "🟢", "unavailable": "🟡", "pending": "⚫", "error": "🔴"}.get(s, "⚫")
        st.markdown(f"{icon} &nbsp;{all_status[key]['name']}", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"v{config.APP_VERSION} · [GitHub]({config.PROJECT_REPO})")

# ---------------------------------------------------------------------------
# Homepage content
# ---------------------------------------------------------------------------
page_header("🛡️ PHISHGUARD", "Multi-Agent Anti-Phishing Firewall — System Overview")

st.markdown(
    "This interface runs real machine-learning agents against an email's headers "
    "and body to estimate the probability it is **Not Safe** (phishing / spam). "
    "Scores are P(Not Safe) ∈ [0, 1]. Each agent is independent; "
    "Agent 3 combines the two when both are available.",
)

divider()
st.subheader("Agent Status")

col1, col2, col3 = st.columns(3)
with col1:
    agent_status_card(
        "Agent 1 — Header Naive Bayes",
        config.COLORS["agent1"],
        all_status["agent1"],
    )
with col2:
    agent_status_card(
        "Agent 2 — DistilBERT Text",
        config.COLORS["agent2"],
        all_status["agent2"],
    )
with col3:
    agent_status_card(
        "Agent 3 — Meta-Classifier (LR)",
        config.COLORS["agent3"],
        all_status["agent3"],
    )

divider()

# Quick-start instructions
st.subheader("Quick Start")
st.markdown(
    "1. Go to **Analyze Email** in the sidebar to submit an email manually or upload a `.eml` file.\n"
    "2. PHISHGUARD will run all available agents and show individual scores.\n"
    "3. See **How It Works** for the decision threshold, known limitations, and label convention."
)

# Label convention reminder
st.info(
    "**Label convention:** 0 = Safe &nbsp;|&nbsp; 1 = Not Safe &nbsp;|&nbsp; "
    "All scores are P(Not Safe), already calibrated.",
)
