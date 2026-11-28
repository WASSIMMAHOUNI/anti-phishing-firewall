"""
PHISHGUARD — Result presentation helpers.

Builds structured, honest summaries for the Analyze Email results view:
  A. Final Verdict (ONLY when Agent 3 succeeded; never fabricated)
  B. Confidence / Probability (ONLY real continuous probability from Agent 3)
  C. Detection Indicators ("Why this decision?") grounded in real outputs
  D. Agent Breakdown (Agent 1, Agent 2, Agent 3 with completed/unavailable/failed)
  E. Threshold resolution, Agent 1/Agent 2 disagreement & Agent 3 fusion explanation
"""
from datetime import datetime
import math
from typing import Any, Dict, List, Optional
import config
from formatting import format_probability, format_probability_full


def _normalize_status(agent_result: Dict[str, Any]) -> str:
    """Return 'completed', 'failed', or 'unavailable'."""
    if agent_result.get("available"):
        return "completed"
    raw = str(agent_result.get("status", "")).lower()
    if raw in ("failed", "error"):
        return "failed"
    return "unavailable"


def _valid_threshold(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if math.isnan(val) or math.isinf(val) or not (0.0 <= val <= 1.0):
        return None
    return val


def resolve_agent_thresholds(all_status: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Read classification thresholds strictly from loaded metadata/config files:
      - Agent 1: agent1_binary_meta.json -> threshold_info.threshold_prob (None if missing)
      - Agent 2: 0.5 unless an explicit threshold is provided by its metadata
      - Agent 3: agent3_config.json -> decision_threshold (None if missing)
    Never silently invents a threshold for Agent 1 or Agent 3 if unreadable.
    """
    a1_meta = (all_status.get("agent1") or {}).get("meta")
    a1_thr: Optional[float] = None
    if isinstance(a1_meta, dict):
        thr_info = a1_meta.get("threshold_info")
        if isinstance(thr_info, dict):
            a1_thr = _valid_threshold(thr_info.get("threshold_prob"))

    a2_meta = (all_status.get("agent2") or {}).get("meta")
    a2_thr: Optional[float] = 0.5
    if isinstance(a2_meta, dict):
        explicit_a2 = a2_meta.get("decision_threshold", a2_meta.get("threshold"))
        if explicit_a2 is not None:
            a2_thr = _valid_threshold(explicit_a2)

    a3_cfg = (all_status.get("agent3") or {}).get("config")
    a3_thr: Optional[float] = None
    if isinstance(a3_cfg, dict):
        a3_thr = _valid_threshold(a3_cfg.get("decision_threshold"))

    return {
        "agent1": a1_thr,
        "agent2": a2_thr,
        "agent3": a3_thr,
    }


def build_fusion_explanation(a3_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a dynamic explanation of how Agent 3 fuses nb_logit and text_logit
    using the actual coefficients from agent3_meta.json.
    Falls back to neutral wording if metadata is missing or unreadable.
    """
    neutral = "Agent 3 combines both scores to produce the final decision."
    if not isinstance(a3_meta, dict):
        return {"has_coefficients": False, "summary": neutral, "coefficients": None}

    coeffs = a3_meta.get("coefficients")
    if not isinstance(coeffs, dict):
        return {"has_coefficients": False, "summary": neutral, "coefficients": None}

    raw_nb = coeffs.get("nb_logit")
    raw_txt = coeffs.get("text_logit")
    raw_int = coeffs.get("intercept")
    if (
        isinstance(raw_nb, bool) or not isinstance(raw_nb, (int, float))
        or isinstance(raw_txt, bool) or not isinstance(raw_txt, (int, float))
        or math.isnan(float(raw_nb)) or math.isinf(float(raw_nb))
        or math.isnan(float(raw_txt)) or math.isinf(float(raw_txt))
    ):
        return {"has_coefficients": False, "summary": neutral, "coefficients": None}

    nb_coef = float(raw_nb)
    txt_coef = float(raw_txt)
    intercept = (
        float(raw_int)
        if not isinstance(raw_int, bool) and isinstance(raw_int, (int, float))
        and not math.isnan(float(raw_int)) and not math.isinf(float(raw_int))
        else None
    )

    abs_nb = abs(nb_coef)
    abs_txt = abs(txt_coef)

    if abs_txt > abs_nb:
        weight_clause = (
            f"In the loaded meta-classifier (`agent3_meta.json`), `text_logit` (Agent 2, "
            f"coefficient {txt_coef:.4f}) carries a larger weight than `nb_logit` (Agent 1, "
            f"coefficient {nb_coef:.4f})."
        )
    elif abs_nb > abs_txt:
        weight_clause = (
            f"In the loaded meta-classifier (`agent3_meta.json`), `nb_logit` (Agent 1, "
            f"coefficient {nb_coef:.4f}) carries a larger weight than `text_logit` (Agent 2, "
            f"coefficient {txt_coef:.4f})."
        )
    else:
        weight_clause = (
            f"In the loaded meta-classifier (`agent3_meta.json`), `nb_logit` ({nb_coef:.4f}) "
            f"and `text_logit` ({txt_coef:.4f}) carry equal coefficient magnitude."
        )

    intercept_clause = f" (intercept {intercept:.4f})" if intercept is not None else ""
    caveat = (
        f" The final verdict still depends on both transformed log-odds values and the model"
        f"{intercept_clause}, so the larger-weight signal does not always determine the outcome alone."
    )

    return {
        "has_coefficients": True,
        "summary": weight_clause + caveat,
        "coefficients": {
            "nb_logit": nb_coef,
            "text_logit": txt_coef,
            "intercept": intercept,
        },
    }


def build_disagreement_summary(
    results: Dict[str, Any],
    thresholds: Dict[str, Optional[float]],
    a3_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare Agent 1 and Agent 2 using unrounded probabilities and loaded thresholds.
    Suppresses the disagreement banner if either agent or its threshold is unavailable.
    """
    a1 = results.get("agent1", {})
    a2 = results.get("agent2", {})
    a3 = results.get("agent3", {})

    a1_thr = thresholds.get("agent1")
    a2_thr = thresholds.get("agent2")
    a3_thr = thresholds.get("agent3")

    a1_ok = bool(a1.get("available")) and isinstance(a1.get("nb_score"), (int, float))
    a2_ok = bool(a2.get("available")) and isinstance(a2.get("text_score"), (int, float))

    unavailable_reasons: List[str] = []
    if not a1_ok:
        unavailable_reasons.append("Agent 1 output is unavailable")
    elif a1_thr is None:
        unavailable_reasons.append("Agent 1 threshold could not be loaded from agent1_binary_meta.json")

    if not a2_ok:
        unavailable_reasons.append("Agent 2 output is unavailable")
    elif a2_thr is None:
        unavailable_reasons.append("Agent 2 threshold is unavailable")

    fusion = build_fusion_explanation(a3_meta)

    if unavailable_reasons:
        return {
            "comparable": False,
            "suppress_banner": True,
            "disagree": False,
            "unavailable_reasons": unavailable_reasons,
            "a1_threshold": a1_thr,
            "a2_threshold": a2_thr,
            "a3_threshold": a3_thr,
            "fusion": fusion,
        }

    # Compare using original, unrounded probabilities
    nb_score = float(a1["nb_score"])
    text_score = float(a2["text_score"])
    a1_is_not_safe = nb_score >= float(a1_thr)
    a2_is_not_safe = text_score >= float(a2_thr)

    a1_side = "Not Safe" if a1_is_not_safe else "Safe"
    a2_side = "Not Safe" if a2_is_not_safe else "Safe"
    disagree = a1_is_not_safe != a2_is_not_safe

    a3_ok = bool(a3.get("available")) and a3.get("final_class") in (0, 1)
    a3_label = (
        ("Not Safe" if a3["final_class"] == 1 else "Safe")
        if a3_ok
        else None
    )
    a3_prob = (
        float(a3["final_probability"])
        if a3_ok and isinstance(a3.get("final_probability"), (int, float))
        else None
    )

    return {
        "comparable": True,
        "suppress_banner": not disagree,
        "disagree": disagree,
        "a1_side": a1_side,
        "a2_side": a2_side,
        "a1_score": nb_score,
        "a2_score": text_score,
        "a1_score_fmt": format_probability(nb_score),
        "a2_score_fmt": format_probability(text_score),
        "a1_threshold": float(a1_thr),
        "a2_threshold": float(a2_thr),
        "a3_threshold": a3_thr,
        "a1_insufficient_evidence": bool(a1.get("insufficient_evidence", False)),
        "a3_label": a3_label,
        "a3_prob": a3_prob,
        "a3_prob_fmt": format_probability(a3_prob) if a3_prob is not None else None,
        "fusion": fusion,
    }


def build_verdict_summary(
    results: Dict[str, Any],
    a1_threshold: Optional[float] = config.THRESHOLD_PROB_DEFAULT,
) -> Dict[str, Any]:
    """
    Build the top-level Final Verdict summary.
    Never returns SAFE or NOT SAFE unless Agent 3 completed successfully.
    Never invents a probability or derives confidence from a hard class.
    """
    a1 = results.get("agent1", {})
    a3 = results.get("agent3", {})
    a3_status = _normalize_status(a3)

    if a3_status != "completed":
        reason = a3.get("error") or "Agent 3 (Decision Fusion) is not available."
        return {
            "verdict_available": False,
            "verdict_label":     "Final Decision Unavailable",
            "verdict_class":     None,
            "final_score":       None,
            "status":            a3_status,
            "explanation": (
                "The final verdict cannot yet be generated because Agent 3 "
                "(Decision Fusion) has not completed successfully. "
                f"Reason: {reason}"
            ),
        }

    raw_score = a3.get("final_probability", a3.get("final_score"))
    final_score: Optional[float] = (
        float(raw_score)
        if not isinstance(raw_score, bool) and isinstance(raw_score, (int, float))
        else None
    )

    if "final_class" in a3 and a3["final_class"] in (0, 1):
        is_not_safe = a3["final_class"] == 1
    elif final_score is not None and a3.get("threshold") is not None:
        thresh = float(a3["threshold"])
        is_not_safe = final_score >= thresh
    else:
        return {
            "verdict_available": False,
            "verdict_label":     "Final Decision Unavailable",
            "verdict_class":     None,
            "final_score":       None,
            "status":            "failed",
            "explanation": (
                "Agent 3 completed without returning a valid score or binary class."
            ),
        }

    verdict_class = 1 if is_not_safe else 0
    verdict_label = "NOT SAFE" if is_not_safe else "SAFE"

    if is_not_safe:
        explanation = (
            "This email contains suspicious characteristics across the analyzed "
            "signals that may indicate phishing or spam."
        )
    else:
        explanation = (
            "No significant suspicious indicators were detected across the "
            "analyzed header and content signals."
        )

    if a1.get("available") and a1.get("insufficient_evidence"):
        explanation += (
            " Note: header evidence was incomplete, so Agent 1 flagged its "
            "header score as unreliable."
        )

    return {
        "verdict_available": True,
        "verdict_label":     verdict_label,
        "verdict_class":     verdict_class,
        "final_score":       final_score,
        "status":            "completed",
        "explanation":       explanation,
    }


def build_detection_indicators(
    results: Dict[str, Any],
    email_dict: Dict[str, str],
    a1_threshold: Optional[float] = config.THRESHOLD_PROB_DEFAULT,
    a2_threshold: Optional[float] = 0.5,
) -> Dict[str, Any]:
    """
    Build 'Detection Indicators' / 'Why this decision?' strictly from real
    model outputs and observed input completeness. Never fabricates feature
    attributions.
    """
    indicators: List[Dict[str, str]] = []
    limitations: List[str] = []

    a1 = results.get("agent1", {})
    a2 = results.get("agent2", {})
    a3 = results.get("agent3", {})

    # --- Agent 1 real signals ---
    a1_status = _normalize_status(a1)
    if a1_status == "completed":
        nb_score = float(a1["nb_score"])
        nb_fmt = format_probability(nb_score)
        comp = float(a1.get("header_completeness", 0.0))
        insuff = bool(a1.get("insufficient_evidence", False))

        if insuff:
            indicators.append({
                "level": "warning",
                "text": (
                    f"Incomplete header fields (header completeness: {comp:.0%}). "
                    "Agent 1 flagged its score as unreliable due to insufficient header evidence."
                ),
            })
        elif a1_threshold is None:
            indicators.append({
                "level": "warning",
                "text": (
                    f"Agent 1 (Header Analysis) scored P(Not Safe) = {nb_fmt}, "
                    "but its decision threshold could not be loaded."
                ),
            })
        elif nb_score >= a1_threshold:
            indicators.append({
                "level": "risk",
                "text": (
                    f"Agent 1 (Header Analysis) flagged elevated risk in the email headers "
                    f"(P(Not Safe) = {nb_fmt}, threshold ≈ {a1_threshold:.3f})."
                ),
            })
        else:
            indicators.append({
                "level": "safe",
                "text": (
                    f"Agent 1 (Header Analysis) found low header-based risk "
                    f"(P(Not Safe) = {nb_fmt}, threshold ≈ {a1_threshold:.3f})."
                ),
            })
    elif a1_status == "failed":
        indicators.append({
            "level": "warning",
            "text": f"Agent 1 (Header Analysis) failed during inference: {a1.get('error', 'unknown error')}",
        })
    else:
        indicators.append({
            "level": "muted",
            "text": f"Agent 1 (Header Analysis) unavailable: {a1.get('error', 'not loaded')}",
        })

    # --- Agent 2 real signals ---
    a2_status = _normalize_status(a2)
    if a2_status == "completed":
        text_score = float(a2["text_score"])
        txt_fmt = format_probability(text_score)
        if a2_threshold is None:
            indicators.append({
                "level": "warning",
                "text": (
                    f"Agent 2 (Content Analysis) scored P(Not Safe) = {txt_fmt}, "
                    "but its decision threshold is unavailable."
                ),
            })
        elif text_score >= a2_threshold:
            indicators.append({
                "level": "risk",
                "text": (
                    f"Agent 2 (Content Analysis) detected elevated risk in subject/body text "
                    f"(P(Not Safe) = {txt_fmt})."
                ),
            })
        else:
            indicators.append({
                "level": "safe",
                "text": (
                    f"Agent 2 (Content Analysis) found low risk in subject/body text "
                    f"(P(Not Safe) = {txt_fmt})."
                ),
            })
    elif a2_status == "failed":
        indicators.append({
            "level": "warning",
            "text": f"Agent 2 (Content Analysis) failed during inference: {a2.get('error', 'unknown error')}",
        })
    else:
        limitations.append(
            "Subject and body text were not analyzed because Agent 2 (DistilBERT) is unavailable."
        )

    # --- Agent 3 real signals ---
    a3_status = _normalize_status(a3)
    if a3_status != "completed":
        limitations.append(
            "Combined decision fusion did not run because Agent 3 is unavailable."
        )

    # --- Honest explainability limitation ---
    limitations.append(
        "The current models output calibrated probability scores without per-feature or "
        "token-level attributions, so individual suspicious words, links, or sender tokens "
        "are not isolated automatically."
    )

    return {
        "indicators":  indicators,
        "limitations": limitations,
    }


def build_agent_breakdown(
    results: Dict[str, Any],
    a1_threshold: Optional[float] = config.THRESHOLD_PROB_DEFAULT,
    a2_threshold: Optional[float] = 0.5,
    a3_threshold: Optional[float] = 0.5,
) -> List[Dict[str, Any]]:
    """Return structured breakdown items for Agents 1, 2, and 3."""
    a1 = results.get("agent1", {})
    a2 = results.get("agent2", {})
    a3 = results.get("agent3", {})

    # Agent 1
    a1_status = _normalize_status(a1)
    a1_prob: Optional[float] = float(a1["nb_score"]) if a1_status == "completed" else None
    a1_result: Optional[str] = None
    if a1_status == "completed":
        if a1.get("insufficient_evidence"):
            a1_result = "Unreliable (Insufficient Headers)"
        elif a1_threshold is None:
            a1_result = "Threshold Unavailable"
        elif a1_prob >= a1_threshold:
            a1_result = "Not Safe (Header Signal)"
        else:
            a1_result = "Safe (Header Signal)"

    # Agent 2
    a2_status = _normalize_status(a2)
    a2_prob: Optional[float] = float(a2["text_score"]) if a2_status == "completed" else None
    a2_result: Optional[str] = None
    if a2_status == "completed":
        if a2_threshold is None:
            a2_result = "Threshold Unavailable"
        else:
            a2_result = "Not Safe (Content Signal)" if a2_prob >= a2_threshold else "Safe (Content Signal)"

    # Agent 3
    a3_status = _normalize_status(a3)
    raw_a3_prob = a3.get("final_probability", a3.get("final_score"))
    a3_prob: Optional[float] = (
        float(raw_a3_prob)
        if a3_status == "completed"
        and not isinstance(raw_a3_prob, bool)
        and isinstance(raw_a3_prob, (int, float))
        else None
    )
    a3_result: Optional[str] = None
    if a3_status == "completed":
        if "final_class" in a3 and a3["final_class"] in (0, 1):
            a3_result = "Not Safe (1)" if a3["final_class"] == 1 else "Safe (0)"
        elif a3_prob is not None and a3_threshold is not None:
            a3_result = "Not Safe (1)" if a3_prob >= float(a3_threshold) else "Safe (0)"
        else:
            a3_result = "Threshold Unavailable"

    return [
        {
            "key":              "agent1",
            "title":            "Agent 1 – Header Analysis",
            "role":             "Evaluates sender, recipient, date, and subject headers using calibrated Naive Bayes.",
            "status":           a1_status,
            "result_label":     a1_result,
            "probability":      a1_prob,
            "probability_fmt":  format_probability(a1_prob),
            "probability_full": format_probability_full(a1_prob),
            "threshold":        a1_threshold,
            "error":            a1.get("error"),
            "color":            config.COLORS["agent1"],
            "extra":            (
                f"Header completeness: {a1.get('header_completeness', 0.0):.0%}"
                if a1_status == "completed"
                else None
            ),
        },
        {
            "key":              "agent2",
            "title":            "Agent 2 – Content Analysis",
            "role":             "Evaluates email subject and body text using fine-tuned DistilBERT (max 256 tokens).",
            "status":           a2_status,
            "result_label":     a2_result,
            "probability":      a2_prob,
            "probability_fmt":  format_probability(a2_prob),
            "probability_full": format_probability_full(a2_prob),
            "threshold":        a2_threshold,
            "error":            a2.get("error"),
            "color":            config.COLORS["agent2"],
            "extra":            None,
        },
        {
            "key":              "agent3",
            "title":            "Agent 3 – Decision Fusion",
            "role":             "Combines Agent 1 and Agent 2 probability scores via Logistic Regression for the final verdict.",
            "status":           a3_status,
            "result_label":     a3_result,
            "probability":      a3_prob,
            "probability_fmt":  format_probability(a3_prob),
            "probability_full": format_probability_full(a3_prob),
            "threshold":        a3_threshold,
            "error":            a3.get("error"),
            "color":            config.COLORS["agent3"],
            "extra":            None,
        },
    ]


# ---------------------------------------------------------------------------
# Session-based live activity & KPIs (in-memory only, never persisted)
# ---------------------------------------------------------------------------

def _sanitize_subject(subject: str, max_len: int = 56) -> str:
    clean = " ".join((subject or "").split())
    if not clean:
        return "(no subject)"
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1] + "…"


def append_session_history(
    history: List[Dict[str, Any]],
    email_dict: Dict[str, str],
    results: Dict[str, Any],
    max_entries: int = 50,
) -> List[Dict[str, Any]]:
    """Prepend one live analysis record to the in-memory session history."""
    a1 = results.get("agent1", {})
    a2 = results.get("agent2", {})
    a3 = results.get("agent3", {})
    a3_ok = bool(a3.get("available"))
    any_failed = any(
        _normalize_status(r) == "failed" for r in (a1, a2, a3)
    )

    if a3_ok and a3.get("final_class") in (0, 1):
        cls = int(a3["final_class"])
        verdict = "NOT SAFE" if cls == 1 else "SAFE"
        raw_prob = a3.get("final_probability", a3.get("final_score"))
        prob = float(raw_prob) if isinstance(raw_prob, (int, float)) and not isinstance(raw_prob, bool) else None
        status_label = "Completed"
    else:
        cls = None
        verdict = "Unavailable"
        prob = None
        status_label = "Failed" if any_failed else "Incomplete"

    total_ms = round(
        sum(
            float(r.get("duration_ms", 0.0))
            for r in (a1, a2, a3)
            if isinstance(r.get("duration_ms"), (int, float))
        ),
        1,
    )

    entry = {
        "timestamp":         datetime.now().strftime("%H:%M:%S"),
        "subject":           _sanitize_subject(email_dict.get("subject", "")),
        "final_class":       cls,
        "verdict":           verdict,
        "final_probability": prob,
        "status":            status_label,
        "duration_ms":       total_ms,
    }
    return [entry] + list(history)[: max_entries - 1]


def compute_session_kpis(history: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute live session-only counts from session history."""
    total = len(history)
    safe = sum(1 for h in history if h.get("final_class") == 0)
    not_safe = sum(1 for h in history if h.get("final_class") == 1)
    failures = sum(1 for h in history if h.get("final_class") not in (0, 1))
    return {
        "total":    total,
        "safe":     safe,
        "not_safe": not_safe,
        "failures": failures,
    }
