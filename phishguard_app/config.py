"""
PHISHGUARD — Central configuration.
All paths are relative to the REPO ROOT (one level above phishguard_app/).
Override any path with the corresponding environment variable.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root (parent of this file's directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent          # phishguard_app/
REPO_ROOT = _HERE.parent               # anti-phishing-firewall/

# ---------------------------------------------------------------------------
# Agent paths  (env-var overridable)
# ---------------------------------------------------------------------------
AGENT1_DIR = Path(
    os.getenv("AGENT1_DIR", str(REPO_ROOT / "agent1_binary_outputs"))
)
AGENT2_MODEL_DIR = Path(
    os.getenv("AGENT2_MODEL_DIR", str(REPO_ROOT / "models" / "distilbert_phishing_model"))
)
AGENT2_SCRIPT_DIR = Path(
    os.getenv("AGENT2_SCRIPT_DIR", str(REPO_ROOT))   # location of text_agent.py
)
AGENT3_DIR = Path(
    os.getenv("AGENT3_DIR", str(REPO_ROOT / "agent3_outputs"))
)

# ---------------------------------------------------------------------------
# Decision threshold (read from Agent 1 meta at runtime; these are fallbacks)
# ---------------------------------------------------------------------------
THRESHOLD_LOGIT_DEFAULT: float = 0.100
THRESHOLD_PROB_DEFAULT: float = 0.525

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg":           "#0d1117",
    "card_bg":      "#161b22",
    "border":       "#30363d",
    "text":         "#c9d1d9",
    "text_muted":   "#8b949e",
    "agent1":       "#1f8ef1",   # electric blue
    "agent2":       "#9b59b6",   # purple
    "agent3":       "#f39c12",   # amber
    "safe":         "#2ecc71",   # green
    "not_safe":     "#e74c3c",   # red
    "warning":      "#f1c40f",   # yellow
    "pending":      "#8b949e",   # grey
}

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "PHISHGUARD"
APP_SUBTITLE = "Multi-Agent Anti-Phishing Firewall"
APP_VERSION = "1.0.0"
PROJECT_REPO = "https://github.com/WASSIMMAHOUNI/anti-phishing-firewall"
