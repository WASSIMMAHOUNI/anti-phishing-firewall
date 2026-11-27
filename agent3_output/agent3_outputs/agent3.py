import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "agent3_logreg.joblib"

_MODEL = None
_THRESHOLD = 0.5
_EPS = 1e-6
_CLIP = 10.0
_FEATURES = ["nb_logit", "text_logit"]


def _validate_score(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite float in [0, 1], got {value!r}.")
    val = float(value)
    if math.isnan(val) or math.isinf(val) or val < 0.0 or val > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {val!r}.")
    return val


def _to_logit(prob: float, eps: float, clip: float) -> float:
    clipped_p = max(eps, min(1.0 - eps, prob))
    raw_logit = math.log(clipped_p / (1.0 - clipped_p))
    return max(-clip, min(clip, raw_logit))


def _load_model():
    global _MODEL, _THRESHOLD, _EPS, _CLIP, _FEATURES
    if _MODEL is None:
        bundle = joblib.load(DEFAULT_MODEL_PATH)
        if isinstance(bundle, dict):
            _MODEL = bundle["model"]
            _THRESHOLD = float(bundle.get("threshold", 0.5))
            _EPS = float(bundle.get("eps", 1e-6))
            _CLIP = float(bundle.get("clip", 10.0))
            _FEATURES = list(bundle.get("features", ["nb_logit", "text_logit"]))
        else:
            _MODEL = bundle
    return _MODEL


def predict(nb_score: float, text_score: float) -> dict:
    """Predict final phishing probability (P(Not Safe)) and class (0=Safe, 1=Not Safe)."""
    nb_clean = _validate_score(nb_score, "nb_score")
    text_clean = _validate_score(text_score, "text_score")

    model = _load_model()
    x_df = pd.DataFrame(
        [[_to_logit(nb_clean, _EPS, _CLIP), _to_logit(text_clean, _EPS, _CLIP)]],
        columns=_FEATURES,
    )
    final_prob = float(model.predict_proba(x_df)[0, 1])
    final_class = int(final_prob >= _THRESHOLD)

    return {
        "final_probability": final_prob,
        "final_class": final_class,
    }
