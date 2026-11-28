"""
PHISHGUARD — About page.
Project overview, dataset split visualization, live agent inventory, and run guide.
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
    inject_css,
    page_header,
    render_dataset_split_chart,
    render_sidebar,
)

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
all_status = orch.get_all_status()
render_sidebar(all_status)

page_header("ℹ️ About PHISHGUARD", "Project scope, shared dataset, and live agent inventory.")

st.markdown(
    f"""
**PHISHGUARD** is a university machine learning project implementing a **Multi-Agent Anti-Phishing Firewall**
for binary email classification (**`0 = Safe`**, **`1 = Not Safe`**).

- **Repository:** [{config.PROJECT_REPO}]({config.PROJECT_REPO})
- **Version:** `{config.APP_VERSION}`
- **Stack:** Python `3.13`, Streamlit, scikit-learn `1.6.1`, PyTorch CPU + 🤗 Transformers (DistilBERT)
"""
)

divider()

st.subheader("Shared Dataset (`emails_clean.csv`)")
col_ds1, col_ds2 = st.columns([3, 2])

with col_ds1:
    render_dataset_split_chart()

with col_ds2:
    st.markdown(
        """
**Source Corpora (48,939 deduplicated emails):**
- `CEAS_08` (Safe & Spam)
- `SpamAssasin` (Safe & Spam)
- `Nazario` (Phishing)
- `Nigerian_Fraud` (Phishing)

**SHA-256:**
`95155f7d9ab86220f08a444e37ec2d7b8965ec88ab51834a03544c8894b67d0f`

*(Raw CSVs in `data/` are excluded via `.gitignore`; only trained artifacts are loaded.)*
"""
    )

divider()

st.subheader("Live Agent Inventory")
rows = []
for key, method, inp in [
    ("agent1", "Smoothed Categorical Naive Bayes (calibrated)", "Headers: sender, receiver, date, subject"),
    ("agent2", "DistilBERT sequence classifier", "Text: subject + body (max 256 tokens)"),
    ("agent3", "Logistic Regression meta-classifier", "Clipped logits [nb_logit, text_logit]"),
]:
    st_val = all_status[key]["status"]
    rows.append({
        "Agent": all_status[key]["name"],
        "Method": method,
        "Input": inp,
        "Status": "Ready" if st_val == "ready" else st_val.capitalize(),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

divider()

st.subheader("How to Run Locally")
st.code(
    "# From the repo root:\n"
    "pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu\n"
    "pip install -r phishguard_app/requirements.txt\n"
    "streamlit run phishguard_app/app.py",
    language="bash",
)
