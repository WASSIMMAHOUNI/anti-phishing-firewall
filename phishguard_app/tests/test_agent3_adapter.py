"""
Tests for Agent 3 adapter — live Logistic Regression meta-classifier and unavailable/invalid paths.
"""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_live_agent3_status_ready(live_agent3_adapter):
    s = live_agent3_adapter.status()
    assert s["status"] == "ready", f"Agent 3 not ready: {s['error']}"
    assert s["meta"] is not None
    assert s["config"] is not None


def test_live_agent3_reference_examples(live_agent3_adapter):
    ref_path = Path(__file__).parent.parent.parent / "agent3_outputs" / "reference_examples.json"
    with open(ref_path, encoding="utf-8") as fh:
        examples = json.load(fh)

    for ex in examples:
        res = live_agent3_adapter.run(ex["nb_score"], ex["text_score"])
        assert res["available"] is True
        assert res["status"] == "completed"
        assert res["final_probability"] == pytest.approx(ex["final_probability"], abs=1e-9)
        assert res["final_score"] == pytest.approx(ex["final_probability"], abs=1e-9)
        assert res["final_class"] == ex["final_class"]


@pytest.mark.parametrize("bad_nb,bad_text", [
    (float("nan"), 0.5),
    (0.5, float("inf")),
    (-0.1, 0.5),
    (0.5, 1.2),
    (True, 0.5),
    ("0.5", 0.5),
])
def test_live_agent3_rejects_invalid_probabilities(live_agent3_adapter, bad_nb, bad_text):
    res = live_agent3_adapter.run(bad_nb, bad_text)
    assert res["available"] is False
    assert res["status"] == "failed"
    assert "error" in res
