"""
Tests for Agent 3 adapter (stub) and orchestrator scenarios.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ---------------------------------------------------------------------------
# Agent 3 stub
# ---------------------------------------------------------------------------

def test_agent3_status_pending(agent3_adapter):
    s = agent3_adapter.status()
    assert s["status"] == "unavailable"
    assert "Agent 3" in s["name"] or "Meta" in s["name"]


def test_agent3_run_returns_unavailable(agent3_adapter):
    result = agent3_adapter.run(0.8, 0.9)
    assert result["available"] is False


# ---------------------------------------------------------------------------
# Orchestrator — full flow
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def orchestrator(tmp_path_factory):
    from pathlib import Path as P
    from agents.agent1_adapter import Agent1Adapter
    from agents.agent2_adapter import Agent2Adapter
    from agents.agent3_adapter import Agent3Adapter

    missing_dir = tmp_path_factory.mktemp("missing_a2") / "nonexistent_model"

    class _Orch:
        agent1 = Agent1Adapter(P(__file__).parent.parent.parent / "agent1_binary_outputs")
        agent2 = Agent2Adapter(
            model_dir=missing_dir,
            script_dir=P(__file__).parent.parent.parent,
            auto_download=False,
        )
        agent3 = Agent3Adapter(P(__file__).parent.parent.parent / "agent3_outputs")

        def get_all_status(self):
            return {
                "agent1": self.agent1.status(),
                "agent2": self.agent2.status(),
                "agent3": self.agent3.status(),
            }

        def analyze(self, email_dict):
            from orchestrator import Orchestrator
            # Use same logic as real orchestrator
            import config
            o = Orchestrator.__new__(Orchestrator)
            o.agent1 = self.agent1
            o.agent2 = self.agent2
            o.agent3 = self.agent3
            return o.analyze(email_dict)

    return _Orch()


def test_orchestrator_agent1_runs(orchestrator, reference_examples):
    ex = reference_examples[0]
    results = orchestrator.analyze({
        "sender":   ex["sender"],
        "receiver": ex["receiver"],
        "date":     ex["date"],
        "subject":  ex["subject"],
        "body":     "",
    })
    assert results["agent1"]["available"] is True
    assert results["agent1"]["nb_score"] == pytest.approx(ex["nb_score"], abs=1e-6)


def test_orchestrator_agent3_blocked_when_agent2_unavailable(orchestrator, reference_examples):
    """Agent 3 must NOT run if Agent 2 is unavailable."""
    ex = reference_examples[0]
    results = orchestrator.analyze({
        "sender":   ex["sender"],
        "receiver": ex["receiver"],
        "date":     ex["date"],
        "subject":  ex["subject"],
        "body":     "",
    })
    assert results["agent2"]["available"] is False
    assert results["agent3"]["available"] is False
    assert "Agent 2" in results["agent3"]["error"] or "agent" in results["agent3"]["error"].lower()


def test_orchestrator_all_status_keys(orchestrator):
    s = orchestrator.get_all_status()
    assert set(s.keys()) == {"agent1", "agent2", "agent3"}
    for key in ("agent1", "agent2", "agent3"):
        assert "status" in s[key]
        assert "name"   in s[key]


def test_orchestrator_handles_agent_exception_gracefully():
    from orchestrator import Orchestrator

    class _ExplodingAgent:
        def status(self):
            return {"name": "Exploding", "status": "error", "error": "boom"}
        def run(self, *args, **kwargs):
            raise RuntimeError("Simulated crash")

    o = Orchestrator.__new__(Orchestrator)
    o.agent1 = _ExplodingAgent()
    o.agent2 = _ExplodingAgent()
    o.agent3 = _ExplodingAgent()

    results = o.analyze({"sender": "a@b.com", "receiver": "c@d.com", "date": "", "subject": "Hi", "body": "Test"})
    assert results["agent1"]["available"] is False
    assert results["agent1"]["status"] == "failed"
    assert "Simulated crash" in results["agent1"]["error"]
    assert results["agent2"]["available"] is False
    assert results["agent2"]["status"] == "failed"
    assert results["agent3"]["available"] is False
    assert results["agent3"]["status"] == "unavailable"


def test_end_to_end_safe_and_suspicious_eml_files(orchestrator, reference_examples):
    from parser import parse_eml_bytes
    from result_presenter import build_verdict_summary

    safe_ex = next(ex for ex in reference_examples if ex["true_label"] == 0)
    susp_ex = next(ex for ex in reference_examples if ex["true_label"] == 1)

    for ex in (safe_ex, susp_ex):
        eml = (
            f"From: {ex['sender']}\n"
            f"To: {ex['receiver']}\n"
            f"Date: {ex['date']}\n"
            f"Subject: {ex['subject']}\n\n"
            "Email body text.\n"
        ).encode("utf-8")
        parsed = parse_eml_bytes(eml)
        results = orchestrator.analyze(parsed)
        assert results["agent1"]["available"] is True
        assert results["agent1"]["nb_score"] == pytest.approx(ex["nb_score"], abs=1e-6)
        assert results["agent2"]["available"] is False
        assert results["agent3"]["available"] is False
        verdict = build_verdict_summary(results)
        assert verdict["verdict_available"] is False


def test_live_full_pipeline_all_scenarios():
    """
    End-to-end tests of the real 3-agent pipeline (Agent 1 -> Agent 2 -> Agent 3):
      1. Normal safe email
      2. Suspicious email with urgency and suspicious link
      3. Spam-like email
      4. Missing optional headers
      5. Malformed header fields
      10. Repeated analyses do not leak/reuse previous predictions
    """
    from orchestrator import Orchestrator
    from result_presenter import build_verdict_summary

    orch = Orchestrator()
    assert orch.agent1.status()["status"] == "ready"
    assert orch.agent2.status()["status"] == "ready"
    assert orch.agent3.status()["status"] == "ready"

    scenarios = [
        ("safe", {
            "sender": "colleague@apache.org",
            "receiver": "team@apache.org",
            "date": "Tue, 05 Aug 2008 15:31:03 -0800",
            "subject": "Project sync tomorrow",
            "body": "Hi team, let us meet tomorrow at 10 AM in room B2 to review the quarterly roadmap.",
        }),
        ("suspicious_link", {
            "sender": "Security Alert <support@paypa1-verify.xyz>",
            "receiver": "user@example.com",
            "date": "Mon, 01 Jan 2024 10:00:00 +0000",
            "subject": "URGENT: Account Suspended - Action Required",
            "body": "URGENT: Your account will be permanently locked in 24 hours. Click http://paypa1-verify.xyz/login to verify your password immediately.",
        }),
        ("spam", {
            "sender": "Promo Deals <deals@cheap-meds-now.top>",
            "receiver": "undisclosed-recipients:;",
            "date": "Wed, 06 Aug 2008 05:02:41 +0530",
            "subject": "FREE $$$ Win a luxury watch today!!!",
            "body": "Congratulations! You have been selected for an exclusive cash prize and 90% discount. Claim now before midnight!",
        }),
        ("missing_optional_headers", {
            "sender": "alice@example.com",
            "receiver": "",
            "date": "",
            "subject": "Quick question",
            "body": "Hi, are you free for a quick call this afternoon?",
        }),
        ("malformed_headers", {
            "sender": "not-an-email-address",
            "receiver": "???",
            "date": "invalid-date-string-9999",
            "subject": "Test",
            "body": "Hello world.",
        }),
    ]

    recorded = {}
    for name, email_dict in scenarios:
        res = orch.analyze(email_dict)
        assert res["agent1"]["available"] is True
        assert res["agent2"]["available"] is True
        assert res["agent3"]["available"] is True

        nb = res["agent1"]["nb_score"]
        tx = res["agent2"]["text_score"]
        fp = res["agent3"]["final_probability"]
        fc = res["agent3"]["final_class"]

        assert 0.0 <= nb <= 1.0
        assert 0.0 <= tx <= 1.0
        assert 0.0 <= fp <= 1.0
        assert fc in (0, 1)

        # Verify Agent 3 received the exact upstream scores
        expected = orch.agent3.run(nb, tx)
        assert fp == pytest.approx(expected["final_probability"], abs=1e-12)
        assert fc == expected["final_class"]

        verdict = build_verdict_summary(res)
        assert verdict["verdict_available"] is True
        assert verdict["verdict_class"] == fc
        assert verdict["verdict_label"] == ("NOT SAFE" if fc == 1 else "SAFE")
        assert verdict["final_score"] == pytest.approx(fp, abs=1e-12)
        recorded[name] = (nb, tx, fp, fc)

    # Repeated analysis check: safe vs suspicious predictions must differ
    assert recorded["safe"][2] != recorded["suspicious_link"][2]
    assert recorded["safe"][3] == 0
    assert recorded["suspicious_link"][3] == 1


def test_orchestrator_blocks_agent3_on_invalid_upstream_prob():
    from orchestrator import Orchestrator
    from result_presenter import build_verdict_summary

    class _BadProbAgent:
        def status(self):
            return {"name": "Bad", "status": "ready", "error": None}
        def run(self, *args, **kwargs):
            return {"available": True, "status": "completed", "nb_score": float("nan"), "text_score": 1.5}

    class _CountingA3:
        called = False
        def status(self):
            return {"name": "A3", "status": "ready", "error": None}
        def run(self, nb, tx):
            self.called = True
            return {"available": True, "status": "completed", "final_probability": 0.5, "final_class": 1}

    o = Orchestrator.__new__(Orchestrator)
    o.agent1 = _BadProbAgent()
    o.agent2 = _BadProbAgent()
    o.agent3 = _CountingA3()

    res = o.analyze({"sender": "a@b.com", "receiver": "c@d.com", "date": "", "subject": "S", "body": "B"})
    assert o.agent3.called is False
    assert res["agent1"]["available"] is False
    assert res["agent1"]["status"] == "failed"
    assert res["agent2"]["available"] is False
    assert res["agent2"]["status"] == "failed"
    assert res["agent3"]["available"] is False
    assert build_verdict_summary(res)["verdict_available"] is False
