"""
Agent 2: Semantic Text Classification Inference Module (DistilBERT).

Preprocessing & Classification Contract (identical to notebooks/train_transformer.ipynb):
- Input concatenation: f"{subject or ''} {body or ''}".strip()
- Tokenizer: WordPiece (distilbert-base-uncased), truncation=True, max_length=256
- Binary label mapping: 0 = Safe, 1 = Not Safe (softmax output index 1)
- Return value: finite float probability P(Not Safe) in [0.0, 1.0]
"""

import math
from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "distilbert_phishing_model"
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
MAX_LENGTH = 256

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None
_CACHED_DIR = None


def _has_complete_model_files(directory: Path) -> bool:
    return directory.is_dir() and all(
        (directory / fname).is_file() and (directory / fname).stat().st_size > 0
        for fname in REQUIRED_MODEL_FILES
    )


def _load_artifacts(model_dir: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE, _CACHED_DIR
    resolved_path = Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR.resolve()
    resolved_dir = str(resolved_path)

    if _CACHED_MODEL is not None and _CACHED_DIR == resolved_dir:
        return _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE

    if not _has_complete_model_files(resolved_path):
        missing = [f for f in REQUIRED_MODEL_FILES if not (resolved_path / f).is_file() or (resolved_path / f).stat().st_size == 0]
        raise FileNotFoundError(
            f"Incomplete or missing Agent 2 model directory at {resolved_dir!r} (missing/empty: {missing}). "
            "Download distilbert_phishing_model.zip (release v1.0-agent2-model) and extract it there."
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
    """Return text_score = P(Not Safe), a finite float in [0.0, 1.0] (0 = Safe, 1 = Not Safe)."""
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
        prob_not_safe = float(torch.softmax(logits, dim=-1)[0, 1].item())

    if not math.isfinite(prob_not_safe):
        raise ValueError(f"Non-finite P(Not Safe) produced by Agent 2: {prob_not_safe!r}")

    return min(1.0, max(0.0, prob_not_safe))


class TextAgent:
    """Backward-compatible wrapper around predict_email() returning P(Not Safe)."""

    def __init__(self, model_dir: Optional[str] = None, max_length: int = MAX_LENGTH):
        self.model_dir = model_dir
        self.max_length = max_length
        _, _, self.device = _load_artifacts(model_dir)

    def score_email(self, subject: str, body: str) -> float:
        return predict_email(subject, body, model_dir=self.model_dir)
