# PHISHGUARD App — Setup & Run Guide

## Overview

PHISHGUARD is a Streamlit web interface for the Multi-Agent Anti-Phishing Firewall.
It runs three independent ML agents in sequence to classify an email as **Safe (`0`)** or **Not Safe (`1`)**:

1. **Agent 1** — Header-based binary Naive Bayes (`agent1_binary_outputs/`)
2. **Agent 2** — DistilBERT text classifier (`text_agent.py` + `models/distilbert_phishing_model/`)
3. **Agent 3** — Logistic Regression fusion meta-classifier (`agent3_outputs/agent3.py` + `agent3_logreg.joblib`)

---

## Requirements

- Python **3.13** (matches the serialized Agent 1 and Agent 3 scikit-learn `1.6.1` artifacts)
- CPU-only PyTorch supported (suitable for low-RAM / non-GPU machines; models are loaded once via `@st.cache_resource`)

---

## Installation

```bash
# 1. Install CPU PyTorch first (verified version on Windows / non-GPU machines)
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining pinned dependencies from the repo root
pip install -r phishguard_app/requirements_app.txt
```

### Agent 2 Weights Placement (`models/distilbert_phishing_model/`)

Agent 2's weights (`268 MB` unpacked) are excluded from Git via `.gitignore` (`models/`).
Download `distilbert_phishing_model.zip` from Release `v1.0-agent2-model` and extract it so the folder contains:

```text
models/distilbert_phishing_model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

---

## Running the App

```bash
# From the repo root
streamlit run phishguard_app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Running Tests

```bash
# From the repo root
python -m pytest phishguard_app/tests/ -v
```

---

## Environment Variables (Optional Path Overrides)

| Variable | Default | Description |
|---|---|---|
| `AGENT1_DIR` | `../agent1_binary_outputs` | Path to Agent 1 artifacts |
| `AGENT2_MODEL_DIR` | `../models/distilbert_phishing_model` | Path to Agent 2 weights |
| `AGENT2_SCRIPT_DIR` | `../` (repo root) | Directory containing `text_agent.py` |
| `AGENT3_DIR` | `../agent3_outputs` | Path to Agent 3 artifacts |
