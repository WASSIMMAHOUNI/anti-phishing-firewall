"""
Agent 2 adapter — DistilBERT text classifier.

Supports the latest text_agent.py contract (commit 878c25b on feature/transformer-model):
  predict_email(subject: str, body: str, model_dir: Optional[str] = None) -> float
and falls back to TextAgent(model_dir=...).score_email(subject, body) if needed.

Status: UNAVAILABLE until text_agent.py, torch/transformers, and trained weights
at models/distilbert_phishing_model/ (or AGENT2_MODEL_DIR) are all present.
Never fabricates scores.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_STATUS_READY       = "ready"
_STATUS_UNAVAILABLE = "unavailable"


class Agent2Adapter:
    """
    Wraps Agent 2's predict_email() / TextAgent with graceful degradation.
    Returns detailed unavailability or failure reasons.
    """

    def __init__(self, model_dir: str | Path, script_dir: str | Path):
        self._model_dir  = Path(model_dir).resolve()
        self._script_dir = Path(script_dir).resolve()
        self._predict_fn = None
        self._agent      = None
        self._status     = _STATUS_UNAVAILABLE
        self._error: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load(self):
        # 1. Check torch + transformers importable
        try:
            import torch               # noqa: F401
            import transformers        # noqa: F401
        except ImportError as exc:
            self._error = (
                f"Missing dependency — {exc}. "
                "Install torch and transformers to enable Agent 2."
            )
            return

        # 2. Check model directory exists
        if not self._model_dir.exists():
            self._error = (
                f"Model weights not found at '{self._model_dir}'. "
                "Download distilbert_phishing_model.zip (release v1.0-agent2-model) "
                "and extract it there."
            )
            return

        # 3. Check for minimum expected files
        required = ["config.json"]
        missing  = [f for f in required if not (self._model_dir / f).exists()]
        if missing:
            self._error = (
                f"Incomplete model directory — missing: {', '.join(missing)}."
            )
            return

        # 4. Check text_agent.py exists in script_dir
        script_path = self._script_dir / "text_agent.py"
        if not script_path.exists():
            self._error = (
                f"Inference module not found at '{script_path}'. "
                "Bring text_agent.py from branch feature/transformer-model."
            )
            return

        # 5. Import and verify predict_email (or backward-compatible TextAgent)
        try:
            sd = str(self._script_dir)
            if sd not in sys.path:
                sys.path.insert(0, sd)
            import text_agent  # noqa: PLC0415

            if hasattr(text_agent, "predict_email"):
                # Warm-load artifacts once via predict_email / _load_artifacts
                if hasattr(text_agent, "_load_artifacts"):
                    text_agent._load_artifacts(str(self._model_dir))
                self._predict_fn = text_agent.predict_email
                self._status = _STATUS_READY
            elif hasattr(text_agent, "TextAgent"):
                self._agent = text_agent.TextAgent(model_dir=str(self._model_dir))
                self._status = _STATUS_READY
            else:
                self._error = "text_agent.py exposes neither predict_email nor TextAgent."
        except Exception as exc:
            self._error = f"Agent 2 load error: {exc}"
            logger.exception("Agent 2 load failed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "name":   "Agent 2 — DistilBERT Text Classifier",
            "status": self._status,
            "error":  self._error,
        }

    def run(self, subject: str, body: str) -> Dict[str, Any]:
        """
        Score one email's text content.

        Returns
        -------
        dict
          available   bool
          status      'completed' | 'unavailable' | 'failed'
          text_score  float in [0,1], P(Not Safe) — present if completed
          error       str — present if not completed
        """
        if self._status != _STATUS_READY:
            return {
                "available": False,
                "status": "unavailable",
                "error": self._error,
            }

        try:
            if self._predict_fn is not None:
                score = self._predict_fn(
                    subject or "",
                    body or "",
                    model_dir=str(self._model_dir),
                )
            else:
                score = self._agent.score_email(subject or "", body or "")
            return {
                "available": True,
                "status": "completed",
                "text_score": float(score),
            }
        except Exception as exc:
            logger.exception("Agent 2 inference error")
            return {
                "available": False,
                "status": "failed",
                "error": str(exc),
            }
