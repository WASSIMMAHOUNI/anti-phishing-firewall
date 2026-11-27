"""
Tests for Agent 1 adapter.

1. Status is "ready" when agent1_binary_outputs/ is present.
2. reference_examples.json — scores match within 1e-6 tolerance.
3. Empty headers produce insufficient_evidence=True.
4. Output keys are correct (receiver, not recipient).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_agent1_status_ready(agent1_adapter):
    s = agent1_adapter.status()
    assert s["status"] == "ready", f"Agent 1 not ready: {s['error']}"
    assert s["meta"] is not None


def test_agent1_meta_has_validation_metrics(agent1_adapter):
    meta = agent1_adapter.status()["meta"]
    vm = meta["validation_metrics"]
    assert vm["accuracy"] == pytest.approx(0.9512, abs=1e-3)
    assert vm["f1"]       == pytest.approx(0.9568, abs=1e-3)
    assert vm["roc_auc"]  == pytest.approx(0.9901, abs=1e-3)


def test_agent1_reference_examples(agent1_adapter, reference_examples):
    """
    Each reference example in reference_examples.json must reproduce
    its recorded nb_score to within 1e-6.
    """
    for ex in reference_examples:
        email_input = {
            "sender":   ex.get("sender",   ""),
            "receiver": ex.get("receiver", ""),
            "date":     ex.get("date",     ""),
            "subject":  ex.get("subject",  ""),
        }
        result = agent1_adapter.run(email_input)
        assert result["available"], f"Agent 1 failed on email_id={ex['email_id']}: {result.get('error')}"
        assert result["nb_score"] == pytest.approx(ex["nb_score"], abs=1e-6), (
            f"email_id={ex['email_id']}: got {result['nb_score']}, expected {ex['nb_score']}"
        )


def test_agent1_empty_headers_insufficient_evidence(agent1_adapter):
    """
    An email with no sender / receiver / date / subject should trigger
    insufficient_evidence=True and produce a high nb_score (≥ 0.90).
    """
    result = agent1_adapter.run(
        {"sender": "", "receiver": "", "date": "", "subject": ""}
    )
    assert result["available"]
    assert result["insufficient_evidence"] is True
    assert result["nb_score"] >= 0.90, (
        f"Expected high score for empty headers, got {result['nb_score']}"
    )


def test_agent1_output_keys(agent1_adapter, reference_examples):
    ex = reference_examples[0]
    result = agent1_adapter.run({
        "sender":   ex["sender"],
        "receiver": ex["receiver"],
        "date":     ex["date"],
        "subject":  ex["subject"],
    })
    assert "nb_score" in result
    assert "header_completeness" in result
    assert "insufficient_evidence" in result
    assert "recipient" not in result   # must be "receiver" in input, not "recipient"


def test_agent1_score_range(agent1_adapter, reference_examples):
    for ex in reference_examples:
        result = agent1_adapter.run({
            "sender":   ex["sender"],
            "receiver": ex["receiver"],
            "date":     ex["date"],
            "subject":  ex["subject"],
        })
        assert 0.0 <= result["nb_score"] <= 1.0, (
            f"Score out of range for email_id={ex['email_id']}: {result['nb_score']}"
        )
