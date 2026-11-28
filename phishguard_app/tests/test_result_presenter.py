"""
Tests for result_presenter.py — Final Verdict hierarchy, confidence handling,
Detection Indicators, and Agent Breakdown states.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from result_presenter import (
    append_session_history,
    build_agent_breakdown,
    build_detection_indicators,
    build_verdict_summary,
    compute_session_kpis,
)


def test_no_final_verdict_when_agent3_unavailable():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.99,
                   "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": False, "status": "unavailable", "error": "Missing weights"},
        "agent3": {"available": False, "status": "unavailable", "error": "Agent 3 requires Agent 1 and Agent 2"},
    }
    verdict = build_verdict_summary(results, a1_threshold=0.525)
    assert verdict["verdict_available"] is False
    assert verdict["verdict_class"] is None
    assert verdict["final_score"] is None
    assert "unavailable" in verdict["verdict_label"].lower()


def test_no_final_verdict_when_agent3_failed():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.1,
                   "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.1},
        "agent3": {"available": False, "status": "failed", "error": "Model file corrupted"},
    }
    verdict = build_verdict_summary(results, a1_threshold=0.525)
    assert verdict["verdict_available"] is False
    assert verdict["verdict_class"] is None
    assert verdict["final_score"] is None


def test_final_verdict_safe_when_agent3_returns_0():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.02,
                   "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.01},
        "agent3": {"available": True, "status": "completed", "final_score": 0.015, "final_class": 0},
    }
    verdict = build_verdict_summary(results, a1_threshold=0.525)
    assert verdict["verdict_available"] is True
    assert verdict["verdict_label"] == "SAFE"
    assert verdict["verdict_class"] == 0
    assert verdict["final_score"] == pytest.approx(0.015)


def test_final_verdict_not_safe_when_agent3_returns_1():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.94,
                   "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.99},
        "agent3": {"available": True, "status": "completed", "final_score": 0.985, "final_class": 1},
    }
    verdict = build_verdict_summary(results, a1_threshold=0.525)
    assert verdict["verdict_available"] is True
    assert verdict["verdict_label"] == "NOT SAFE"
    assert verdict["verdict_class"] == 1
    assert verdict["final_score"] == pytest.approx(0.985)


def test_no_invented_probability_from_hard_class():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.8,
                   "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.9},
        "agent3": {"available": True, "status": "completed", "final_class": 1},
    }
    verdict = build_verdict_summary(results)
    assert verdict["verdict_available"] is True
    assert verdict["verdict_label"] == "NOT SAFE"
    assert verdict["final_score"] is None


def test_detection_indicators_reflect_real_signals_and_limitations():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.98,
                   "header_completeness": 0.0, "insufficient_evidence": True},
        "agent2": {"available": False, "status": "unavailable", "error": "Missing weights"},
        "agent3": {"available": False, "status": "unavailable", "error": "Pending"},
    }
    data = build_detection_indicators(results, {"sender": "", "receiver": "", "date": "", "subject": ""})
    assert any("insufficient header evidence" in i["text"].lower() for i in data["indicators"])
    assert any("without per-feature" in lim.lower() for lim in data["limitations"])


def test_agent_breakdown_distinguishes_completed_unavailable_failed():
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.01,
                   "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": False, "status": "failed", "error": "RuntimeError: OOM"},
        "agent3": {"available": False, "status": "unavailable", "error": "Requires Agent 1 and Agent 2"},
    }
    items = build_agent_breakdown(results, a1_threshold=0.525)
    assert [i["status"] for i in items] == ["completed", "failed", "unavailable"]
    assert items[0]["probability"] == pytest.approx(0.01)
    assert items[1]["probability"] is None
    assert "OOM" in items[1]["error"]
    assert items[2]["probability"] is None


def test_session_history_and_kpis():
    assert compute_session_kpis([]) == {"total": 0, "safe": 0, "not_safe": 0, "failures": 0}

    hist = []
    hist = append_session_history(
        hist,
        {"subject": "Hello team " + "x" * 80},
        {
            "agent1": {"available": True, "status": "completed", "nb_score": 0.1, "duration_ms": 2.0},
            "agent2": {"available": True, "status": "completed", "text_score": 0.05, "duration_ms": 80.0},
            "agent3": {"available": True, "status": "completed", "final_probability": 0.02, "final_class": 0, "duration_ms": 1.0},
        },
    )
    hist = append_session_history(
        hist,
        {"subject": "URGENT login"},
        {
            "agent1": {"available": True, "status": "completed", "nb_score": 0.99, "duration_ms": 2.0},
            "agent2": {"available": True, "status": "completed", "text_score": 0.99, "duration_ms": 75.0},
            "agent3": {"available": True, "status": "completed", "final_probability": 0.999, "final_class": 1, "duration_ms": 1.0},
        },
    )
    hist = append_session_history(
        hist,
        {"subject": ""},
        {
            "agent1": {"available": True, "status": "completed", "nb_score": 0.5, "duration_ms": 1.5},
            "agent2": {"available": False, "status": "failed", "error": "boom", "duration_ms": 0.5},
            "agent3": {"available": False, "status": "unavailable", "error": "blocked"},
        },
    )

    kpis = compute_session_kpis(hist)
    assert kpis == {"total": 3, "safe": 1, "not_safe": 1, "failures": 1}
    assert hist[0]["subject"] == "(no subject)"
    assert hist[0]["status"] == "Failed"
    assert hist[2]["subject"].endswith("…")
    assert hist[2]["duration_ms"] == pytest.approx(83.0)
