"""
Agent 3 adapter — Logistic Regression meta-classifier.

Loads agent3_outputs/agent3.py and agent3_logreg.joblib and exposes:
  status() -> dict
  run(nb_score, text_score) -> dict

Expected contract (from agent3_outputs/agent3.py):
  predict(nb_score: float, text_score: float) -> {"final_probability": float, "final_class": int}
  where 0 = Safe, 1 = Not Safe.
"""
import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STATUS_READY       = "ready"
_STATUS_UNAVAILABLE = "unavailable"
_STATUS_ERROR       = "error"


class Agent3Adapter:
    """
    Wraps Agent 3's Logistic Regression meta-classifier.
    Never fabricates predictions; returns unavailable/failed when unusable.
    """

    def __init__(self, agent3_dir: str | Path):
        self._dir        = Path(agent3_dir).resolve()
        self._predict_fn = None
        self._meta:   Optional[Dict[str, Any]] = None
        self._config: Optional[Dict[str, Any]] = None
        self._status     = _STATUS_UNAVAILABLE
        self._error:  Optional[str] = None
        self._load()

    def _load(self):
        module_path = self._dir / "agent3.py"
        model_path  = self._dir / "agent3_logreg.joblib"
        meta_path   = self._dir / "agent3_meta.json"
        config_path = self._dir / "agent3_config.json"

        if not module_path.exists():
            self._status = _STATUS_UNAVAILABLE
            self._error = f"Inference module not found: {module_path}"
            return
        if not model_path.exists():
            self._status = _STATUS_UNAVAILABLE
            self._error = f"Model artifact not found: {model_path}"
            return

        try:
            spec = importlib.util.spec_from_file_location("agent3_runtime", str(module_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {module_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Ensure model loads from this adapter's directory and cache once
            mod.DEFAULT_MODEL_PATH = model_path
            mod._MODEL = None
            mod._load_model()

            if not hasattr(mod, "predict"):
                raise AttributeError("agent3.py does not define predict(nb_score, text_score)")

            self._predict_fn = mod.predict
        except Exception as exc:
            self._status = _STATUS_ERROR
            self._error = f"Agent 3 load error: {exc}"
            logger.exception("Agent 3 load failed")
            return

        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    self._meta = json.load(fh)
            except Exception as exc:
                logger.warning("Could not read Agent 3 meta JSON: %s", exc)

        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as fh:
                    self._config = json.load(fh)
            except Exception as exc:
                logger.warning("Could not read Agent 3 config JSON: %s", exc)

        self._status = _STATUS_READY

    def status(self) -> Dict[str, Any]:
        return {
            "name":   "Agent 3 — Meta-Classifier (LR)",
            "status": self._status,
            "error":  self._error,
            "meta":   self._meta,
            "config": self._config,
        }

    def run(self, nb_score: float, text_score: float) -> Dict[str, Any]:
        """
        Combine continuous Agent 1 and Agent 2 probabilities via Logistic Regression.
        """
        if self._status != _STATUS_READY or self._predict_fn is None:
            return {
                "available": False,
                "status":    "unavailable",
                "error":     self._error or "Agent 3 is unavailable.",
            }

        try:
            out = self._predict_fn(nb_score, text_score)
            prob = out.get("final_probability")
            cls  = out.get("final_class")

            if (
                isinstance(prob, bool)
                or not isinstance(prob, (int, float))
                or math.isnan(float(prob))
                or math.isinf(float(prob))
                or not (0.0 <= float(prob) <= 1.0)
            ):
                raise ValueError(f"Invalid final_probability returned: {prob!r}")

            if isinstance(cls, bool) or cls not in (0, 1):
                raise ValueError(f"Invalid final_class returned: {cls!r}")

            prob_f = float(prob)
            return {
                "available":         True,
                "status":            "completed",
                "final_probability": prob_f,
                "final_score":       prob_f,
                "final_class":       int(cls),
            }
        except Exception as exc:
            logger.exception("Agent 3 inference error")
            return {
                "available": False,
                "status":    "failed",
                "error":     str(exc),
            }
