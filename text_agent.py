"""
Agent 2: Semantic Text Classification Inference Module (DistilBERT).

Preprocessing contract (identical to training in notebooks/train_transformer.ipynb):
- Input concatenation: f"{subject or ''} {body or ''}".strip()
- Tokenizer: WordPiece (distilbert-base-uncased), truncation=True, max_length=256
- Output class mapping: index 0 = Safe (0), index 1 = Not Safe / Phishing (1)
- Return value: float probability P(Not Safe) in [0.0, 1.0]
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "distilbert_phishing_model"
MAX_LENGTH = 256

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None
_CACHED_DIR = None


def _load_artifacts(model_dir: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE, _CACHED_DIR
    resolved_dir = str(Path(model_dir) if model_dir else DEFAULT_MODEL_DIR)

    if _CACHED_MODEL is not None and _CACHED_DIR == resolved_dir:
        return _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE

    if not Path(resolved_dir).is_dir():
        raise FileNotFoundError(
            f"Model directory not found at {resolved_dir!r}. "
            "Download distilbert_phishing_model.zip and extract it into models/distilbert_phishing_model/."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(resolved_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_dir, local_files_only=True).to(device)
    model.eval()

    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL = model
    _CACHED_DEVICE = device
    _CACHED_DIR = resolved_dir
    return tokenizer, model, device


def predict_email(subject: str, body: str, model_dir: Optional[str] = None) -> float:
    """Return text_score = P(Not Safe), a float between 0 and 1 (0 = Safe, 1 = Not Safe)."""
    tokenizer, model, device = _load_artifacts(model_dir)
    clean_subject = "" if subject is None else str(subject)
    clean_body = "" if body is None else str(body)
    text = f"{clean_subject} {clean_body}".strip()

    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        prob_not_safe = torch.softmax(logits, dim=-1)[0, 1].item()

    return float(prob_not_safe)


class TextAgent:
    """Backward-compatible wrapper around predict_email()."""

    def __init__(self, model_dir: Optional[str] = None, max_length: int = MAX_LENGTH):
        self.model_dir = model_dir
        self.max_length = max_length
        _, _, self.device = _load_artifacts(model_dir)

    def score_email(self, subject: str, body: str) -> float:
        return predict_email(subject, body, model_dir=self.model_dir)


if __name__ == "__main__":
    import pandas as pd

    examples = [
        ("Empty subject / short safe note", "", "Team meeting moved to Thursday at 10 AM in Room B2. Please bring your laptop."),
        ("Urgent credential lure", "URGENT: Account Suspended", "URGENT: Your bank account has been suspended due to unusual login. Click here immediately to verify your credentials."),
        ("Routine project sync", "Project sync", "Hi team, let us meet tomorrow at 10 AM in room B2."),
        ("Invoice lure", "Overdue Invoice #4821", "Dear customer, your account will be locked unless you open the attached payment portal and confirm your billing details."),
    ]
    print("=== REFERENCE EXAMPLES ===")
    for label, subj, bdy in examples:
        print(f"{label}: {predict_email(subj, bdy):.6f}")

    df = pd.read_csv("data/processed/emails_clean.csv")
    saved = pd.read_csv("data/processed/transformer_scores.csv").head(3)
    merged = saved.merge(df[["email_id", "subject", "body"]], on="email_id")
    print("=== SAVED CSV MATCH CHECK ===")
    for _, r in merged.iterrows():
        live_score = predict_email("" if pd.isna(r["subject"]) else r["subject"], "" if pd.isna(r["body"]) else r["body"])
        eid, sc = r["email_id"], r["text_score"]
        print(f"email_id={eid} | saved={sc:.6f} | live={live_score:.6f}")
