"""
PHISHGUARD — Result presentation helpers.

Builds structured, honest summaries for the Analyze Email results view:
  A. Final Verdict (ONLY when Agent 3 succeeded; never fabricated)
  B. Confidence / Probability (ONLY real continuous probability from Agent 3)
  C. Detection Indicators ("Why this decision?") grounded in real outputs
  D. Agent Breakdown (Agent 1, Agent 2, Agent 3 with completed/unavailable/failed)
"""
from typing import Any, Dict, List, Optional
import config


def _normalize_status(agent_result: Dict[str, Any]) -> str:
    """Return 'completed', 'failed', or 'unavailable'."""
    if agent_result.get("available"):
        return "completed"
    raw = str(agent_result.get("status", "")).lower()
    if raw in ("failed", "error"):
        return "failed"
    return "unavailable"


def build_verdict_summary(
    results: Dict[str, Any],
    a1_threshold: float = config.THRESHOLD_PROB_DEFAULT,
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

    # Agent 3 completed — check for real probability / class
    raw_score = a3.get("final_probability", a3.get("final_score"))
    final_score: Optional[float] = (
        float(raw_score)
        if not isinstance(raw_score, bool) and isinstance(raw_score, (int, float))
        else None
    )

    if "final_class" in a3 and a3["final_class"] in (0, 1):
        is_not_safe = a3["final_class"] == 1
    elif final_score is not None:
        thresh = float(a3.get("threshold", 0.5))
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
    a1_threshold: float = config.THRESHOLD_PROB_DEFAULT,
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
        elif nb_score >= a1_threshold:
            indicators.append({
                "level": "risk",
                "text": (
                    f"Agent 1 (Header Analysis) flagged elevated risk in the email headers "
                    f"(P(Not Safe) = {nb_score:.4f}, threshold ≈ {a1_threshold:.3f})."
                ),
            })
        else:
            indicators.append({
                "level": "safe",
                "text": (
                    f"Agent 1 (Header Analysis) found low header-based risk "
                    f"(P(Not Safe) = {nb_score:.4f}, threshold ≈ {a1_threshold:.3f})."
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
        if text_score >= 0.5:
            indicators.append({
                "level": "risk",
                "text": (
                    f"Agent 2 (Content Analysis) detected elevated risk in subject/body text "
                    f"(P(Not Safe) = {text_score:.4f})."
                ),
            })
        else:
            indicators.append({
                "level": "safe",
                "text": (
                    f"Agent 2 (Content Analysis) found low risk in subject/body text "
                    f"(P(Not Safe) = {text_score:.4f})."
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
    a1_threshold: float = config.THRESHOLD_PROB_DEFAULT,
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
        elif a1_prob >= a1_threshold:
            a1_result = "Not Safe (Header Signal)"
        else:
            a1_result = "Safe (Header Signal)"

    # Agent 2
    a2_status = _normalize_status(a2)
    a2_prob: Optional[float] = float(a2["text_score"]) if a2_status == "completed" else None
    a2_result: Optional[str] = None
    if a2_status == "completed":
        a2_result = "Not Safe (Content Signal)" if a2_prob >= 0.5 else "Safe (Content Signal)"

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
        elif a3_prob is not None:
            a3_result = "Not Safe (1)" if a3_prob >= float(a3.get("threshold", 0.5)) else "Safe (0)"

    return [
        {
            "key":          "agent1",
            "title":        "Agent 1 – Header Analysis",
            "role":         "Evaluates sender, recipient, date, and subject headers using calibrated Naive Bayes.",
            "status":       a1_status,
            "result_label": a1_result,
            "probability":  a1_prob,
            "error":        a1.get("error"),
            "color":        config.COLORS["agent1"],
            "extra":        (
                f"Header completeness: {a1.get('header_completeness', 0.0):.0%}"
                if a1_status == "completed"
                else None
            ),
        },
        {
            "key":          "agent2",
            "title":        "Agent 2 – Content Analysis",
            "role":         "Evaluates email subject and body text using fine-tuned DistilBERT (max 256 tokens).",
            "status":       a2_status,
            "result_label": a2_result,
            "probability":  a2_prob,
            "error":        a2.get("error"),
            "color":        config.COLORS["agent2"],
            "extra":        None,
        },
        {
            "key":          "agent3",
            "title":        "Agent 3 – Decision Fusion",
            "role":         "Combines Agent 1 and Agent 2 probability scores via Logistic Regression for the final verdict.",
            "status":       a3_status,
            "result_label": a3_result,
            "probability":  a3_prob,
            "error":        a3.get("error"),
            "color":        config.COLORS["agent3"],
            "extra":        None,
        },
    ]
