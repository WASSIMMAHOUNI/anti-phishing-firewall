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
    os.getenv("AGENT2_SCRIPT_DIR", str(REPO_ROOT))   # location of text_agent.py & agent2_metrics.json
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
# Design tokens — Dark Violet / Deep Navy Cybersecurity SaaS Theme
# ---------------------------------------------------------------------------
COLORS = {
    "bg":            "#07080f",   # deep black-navy canvas
    "bg_secondary":  "#0e0d1a",   # sidebar / secondary panel
    "card_bg":       "#141224",   # dark violet card surface
    "card_elevated": "#1b1830",   # elevated inner tile
    "border":        "#2a2448",   # subtle violet-slate border
    "text":          "#e6e8f0",   # primary text
    "text_muted":    "#9499b3",   # secondary muted text
    "primary":       "#8b5cf6",   # primary purple accent
    "secondary":     "#3b82f6",   # secondary electric blue accent
    "agent1":        "#3b82f6",   # electric blue (Agent 1)
    "agent2":        "#8b5cf6",   # purple (Agent 2)
    "agent3":        "#a855f7",   # violet-magenta (Agent 3)
    "safe":          "#22c55e",   # green
    "safe_bg":       "#0d281b",
    "not_safe":      "#ef4444",   # red
    "not_safe_bg":   "#2a1215",
    "warning":       "#f59e0b",   # amber
    "warning_bg":    "#271f0e",
    "pending":       "#64748b",   # slate grey
}

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "PHISHGUARD"
APP_SUBTITLE = "Multi-Agent Email Security Platform"
APP_VERSION = "1.1.0"
PROJECT_REPO = "https://github.com/WASSIMMAHOUNI/anti-phishing-firewall"
