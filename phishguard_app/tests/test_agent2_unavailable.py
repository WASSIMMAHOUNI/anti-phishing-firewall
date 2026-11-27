"""
Tests for Agent 2 adapter — unavailable path.
Agent 2 weights are not in the repo, so the adapter must report unavailable
with a clear reason. No inference should succeed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_agent2_status_unavailable(agent2_adapter):
    s = agent2_adapter.status()
    assert s["status"] == "unavailable", (
        "Agent 2 should be unavailable (weights not in repo)"
    )
    assert s["error"] is not None and len(s["error"]) > 0


def test_agent2_error_message_is_informative(agent2_adapter):
    """The error message must explain WHY — not just 'error'."""
    error = agent2_adapter.status()["error"]
    # Should mention weights, model, or missing dependency
    keywords = ["weight", "model", "missing", "not found", "dependency", "import"]
    assert any(kw.lower() in error.lower() for kw in keywords), (
        f"Error message is not informative enough: {error!r}"
    )


def test_agent2_run_returns_unavailable(agent2_adapter):
    result = agent2_adapter.run("Test subject", "Test body")
    assert result["available"] is False
    assert "error" in result


def test_agent2_name_in_status(agent2_adapter):
    s = agent2_adapter.status()
    assert "Agent 2" in s["name"] or "DistilBERT" in s["name"]
