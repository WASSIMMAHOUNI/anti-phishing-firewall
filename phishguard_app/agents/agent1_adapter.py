"""
Agent 1 adapter — Header-based binary Naive Bayes.

Loads HeaderAgent from agent1_binary_outputs/ and exposes:
  status() -> dict
  run(email_dict) -> dict
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_STATUS_READY = "ready"
_STATUS_ERROR = "error"


class Agent1Adapter:
    """
    Thin wrapper around HeaderAgent.
    Thread-safe after __init__ (Streamlit cache_resource calls this once).
    """

    def __init__(self, agent1_dir: str | Path):
        self._dir = Path(agent1_dir).resolve()
        self._agent = None
        self._score_email_fn = None
        self._meta: Optional[Dict] = None
        self._status = _STATUS_ERROR
        self._error: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load(self):
        # 1. Make the module importable
        dir_str = str(self._dir)
        if dir_str not in sys.path:
            sys.path.insert(0, dir_str)

        # 2. Check required files exist
        model_path  = self._dir / "agent1_binary_model.joblib"
        module_path = self._dir / "agent1_header.py"
        meta_path   = self._dir / "agent1_binary_meta.json"

        if not module_path.exists():
            self._error = f"Module not found: {module_path}"
            return
        if not model_path.exists():
            self._error = f"Model file not found: {model_path}"
            return

        # 3. Import and load
        try:
            from agent1_header import HeaderAgent, score_email  # noqa: PLC0415
            self._agent = HeaderAgent.load(str(model_path))
            self._score_email_fn = score_email
        except Exception as exc:
            self._error = f"Import/load error: {exc}"
            logger.exception("Agent 1 load failed")
            return

        # 4. Read meta (non-fatal if missing)
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    self._meta = json.load(fh)
            except Exception as exc:
                logger.warning("Could not read meta JSON: %s", exc)

        self._status = _STATUS_READY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "name":   "Agent 1 — Header Naive Bayes",
            "status": self._status,
            "error":  self._error,
            "meta":   self._meta,
        }

    def run(self, email_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Score one email.

        Parameters
        ----------
        email_dict : dict with keys sender, receiver, date, subject
                     (body is ignored — Agent 1 is header-only)

        Returns
        -------
        dict
          available            bool
          nb_score             float in [0,1], P(Not Safe) — present if available
          header_completeness  float in [0,1] — present if available
          insufficient_evidence bool — if True, score is unreliable
          error                str — present if not available
        """
        if self._status != _STATUS_READY:
            return {
                "available": False,
                "status": "unavailable",
                "error": self._error,
            }

        # Agent 1 uses only the four header fields
        header_input = {
            k: email_dict.get(k, "")
            for k in ("sender", "receiver", "date", "subject")
        }

        try:
            nb_score = self._score_email_fn(self._agent, header_input)

            detail_df = self._agent.predict_detailed([header_input])
            row = detail_df.iloc[0]

            return {
                "available":             True,
                "status":                "completed",
                "nb_score":              float(nb_score),
                "header_completeness":   float(row.get("header_completeness", 0.0)),
                "insufficient_evidence": bool(row.get("insufficient_evidence", False)),
            }
        except Exception as exc:
            logger.exception("Agent 1 inference error")
            return {
                "available": False,
                "status": "failed",
                "error": str(exc),
            }
