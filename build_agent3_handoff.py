import hashlib
import json
import platform
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

EXPECTED_SHA256 = "95155F7D9AB86220F08A444E37EC2D7B8965EC88AB51834A03544C8894B67D0F"
EPS = 1e-6
CLIP = 10.0
THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def to_logit_array(probs, eps: float = EPS, clip: float = CLIP) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)


def compute_metrics(y_true, probs, threshold: float = THRESHOLD) -> dict:
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "agent3_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. التحقق من ملف البيانات المشترك و SHA-256
    dataset_path = root / "data" / "processed" / "emails_clean.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    actual_sha = sha256_file(dataset_path)
    print(f"emails_clean.csv SHA-256: {actual_sha}")
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}"
        )

    full_data = pd.read_csv(dataset_path)
    assert len(full_data) == 48939, f"Expected 48939 rows, got {len(full_data)}"
    data = full_data[["email_id", "split", "label"]]

    # 2. دمج Agent 1 و Agent 2 و التحقق من التطابق 1-إلى-1
    a1_path = root / "agent1_binary_outputs" / "agent1_scores.csv"
    a2_path = root / "transformer_scores.csv"
    a1 = pd.read_csv(a1_path)
    a2 = pd.read_csv(a2_path)

    assert len(a1) == 14683 and len(a2) == 14683
    assert not a1["email_id"].duplicated().any()
    assert not a2["email_id"].duplicated().any()
    assert set(a1["email_id"]) == set(a2["email_id"])

    df = (
        a1.merge(a2, on=["email_id", "split"], validate="one_to_one")
        .merge(data, on=["email_id", "split"], validate="one_to_one")
    )
    assert len(df) == 14683
    assert not df.isna().any().any()

    df["nb_logit"] = to_logit_array(df["nb_score"], EPS, CLIP)
    df["text_logit"] = to_logit_array(df["text_score"], EPS, CLIP)

    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert len(val_df) == 7340
    assert len(test_df) == 7343

    features = ["nb_logit", "text_logit"]
    X_val = val_df[features]
    y_val = val_df["label"].astype(int)
    X_test = test_df[features]
    y_test = test_df["label"].astype(int)

    # 3. اختيار C على val فقط عبر 5-fold CV
    best_c = 1.0
    best_auc = -1.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, random_state=42, max_iter=1000)
        auc = cross_val_score(clf, X_val, y_val, cv=cv, scoring="roc_auc").mean()
        if auc > best_auc:
            best_auc = auc
            best_c = c

    model = LogisticRegression(C=best_c, random_state=42, max_iter=1000)
    model.fit(X_val, y_val)

    # 4. حفظ النموذج داخل agent3_outputs/
    model_path = out_dir / "agent3_logreg.joblib"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "threshold": THRESHOLD,
            "eps": EPS,
            "clip": CLIP,
        },
        model_path,
    )

    # 5. حفظ agent3_config.json
    config = {
        "feature_order": ["nb_score", "text_score"],
        "model_features": features,
        "logit_transform": {
            "type": "clipped_logit",
            "eps": EPS,
            "clip": CLIP,
        },
        "decision_threshold": THRESHOLD,
        "label_convention": {
            "0": "Safe",
            "1": "Not Safe",
        },
        "not_safe_proba_column_index": 1,
    }
    with (out_dir / "agent3_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 6. حفظ agent3_meta.json بمقاييس test لـ Agent 1 و Agent 2 و Agent 3
    p_test_a3 = model.predict_proba(X_test)[:, 1]
    meta = {
        "model_name": "Agent 3 Meta-Classifier (Logistic Regression)",
        "training_split": "val",
        "threshold_selection_split": "val",
        "evaluation_split": "test",
        "best_C": best_c,
        "coefficients": {
            "nb_logit": float(model.coef_[0][0]),
            "text_logit": float(model.coef_[0][1]),
            "intercept": float(model.intercept_[0]),
        },
        "test_metrics": {
            "agent1_alone": compute_metrics(y_test, test_df["nb_score"], THRESHOLD),
            "agent2_alone": compute_metrics(y_test, test_df["text_score"], THRESHOLD),
            "agent3_fusion": compute_metrics(y_test, p_test_a3, THRESHOLD),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (out_dir / "agent3_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 7. توليد 6 أمثلة حقيقية من val تشمل حالة اختلاف بين Agent 1 و Agent 2
    val_probs = model.predict_proba(X_val)[:, 1]
    val_classes = (val_probs >= THRESHOLD).astype(int)

    a1_bin = (val_df["nb_score"] >= THRESHOLD).astype(int)
    a2_bin = (val_df["text_score"] >= THRESHOLD).astype(int)
    disagree_indices = val_df.index[a1_bin != a2_bin].tolist()

    selected = []
    if disagree_indices:
        selected.append(disagree_indices[0])
    for i in range(len(val_df)):
        if i not in selected:
            selected.append(i)
        if len(selected) == 6:
            break

    ref_examples = []
    for idx in selected:
        ref_examples.append(
            {
                "email_id": int(val_df.loc[idx, "email_id"]),
                "nb_score": float(val_df.loc[idx, "nb_score"]),
                "text_score": float(val_df.loc[idx, "text_score"]),
                "agents_disagree": bool(a1_bin.iloc[idx] != a2_bin.iloc[idx]),
                "final_probability": float(val_probs[idx]),
                "final_class": int(val_classes[idx]),
            }
        )

    with (out_dir / "reference_examples.json").open("w", encoding="utf-8") as f:
        json.dump(ref_examples, f, indent=2)

    # 8. إنشاء agent3_outputs/agent3.py مع أسماء الأعمدة لتفادي أي تحذير
    agent3_code = '''import math
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
'''
    (out_dir / "agent3.py").write_text(agent3_code, encoding="utf-8")

    # حذف مجلدات __pycache__
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    print("Generated agent3.py, agent3_config.json, agent3_meta.json, reference_examples.json, and agent3_logreg.joblib.")


if __name__ == "__main__":
    main()