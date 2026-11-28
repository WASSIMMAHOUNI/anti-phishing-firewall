"""
PHISHGUARD Orchestrator.

Coordinates all three agent adapters, loads them once via Streamlit's
cache_resource, validates upstream probabilities, and exposes analyze().
"""
import math
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from agents.agent1_adapter import Agent1Adapter
from agents.agent2_adapter import Agent2Adapter
from agents.agent3_adapter import Agent3Adapter


def _is_valid_prob(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    val = float(value)
    return not math.isnan(val) and not math.isinf(val) and 0.0 <= val <= 1.0


class Orchestrator:
    """
    Load adapters once; reuse across Streamlit reruns.
    """

    def __init__(self):
        self.agent1 = Agent1Adapter(config.AGENT1_DIR)
        self.agent2 = Agent2Adapter(config.AGENT2_MODEL_DIR, config.AGENT2_SCRIPT_DIR)
        self.agent3 = Agent3Adapter(config.AGENT3_DIR)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_all_status(self) -> Dict[str, Dict]:
        return {
            "agent1": self.agent1.status(),
            "agent2": self.agent2.status(),
            "agent3": self.agent3.status(),
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, email_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Run all available agents on one email in sequence:
          Email -> Agent 1 (Headers) + Agent 2 (Text) -> Agent 3 (Fusion)
        """
        results: Dict[str, Any] = {}

        # --- Agent 1 (always attempted; header-only) -------------------
        a1_input = {
            k: email_dict.get(k, "")
            for k in ("sender", "receiver", "date", "subject")
        }
        t0 = time.perf_counter()
        try:
            a1_res = self.agent1.run(a1_input)
        except Exception as exc:
            a1_res = {
                "available": False,
                "status": "failed",
                "error": f"Agent 1 execution failed: {exc}",
            }
        a1_res["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if a1_res.get("available") and not _is_valid_prob(a1_res.get("nb_score")):
            a1_res = {
                "available": False,
                "status": "failed",
                "error": f"Agent 1 returned an invalid probability: {a1_res.get('nb_score')!r}",
                "duration_ms": a1_res["duration_ms"],
            }
        results["agent1"] = a1_res

        # --- Agent 2 (text; subject + body) ----------------------------
        subject = email_dict.get("subject", "")
        body    = email_dict.get("body", "")
        t0 = time.perf_counter()
        try:
            a2_res = self.agent2.run(subject, body)
        except Exception as exc:
            a2_res = {
                "available": False,
                "status": "failed",
                "error": f"Agent 2 execution failed: {exc}",
            }
        a2_res["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if a2_res.get("available") and not _is_valid_prob(a2_res.get("text_score")):
            a2_res = {
                "available": False,
                "status": "failed",
                "error": f"Agent 2 returned an invalid probability: {a2_res.get('text_score')!r}",
                "duration_ms": a2_res["duration_ms"],
            }
        results["agent2"] = a2_res

        # --- Agent 3 (meta; only if both A1 and A2 succeeded with valid probs)
        a1_ok = results["agent1"].get("available", False)
        a2_ok = results["agent2"].get("available", False)

        if a1_ok and a2_ok:
            t0 = time.perf_counter()
            try:
                a3_res = self.agent3.run(
                    float(results["agent1"]["nb_score"]),
                    float(results["agent2"]["text_score"]),
                )
            except Exception as exc:
                a3_res = {
                    "available": False,
                    "status": "failed",
                    "error": f"Agent 3 execution failed: {exc}",
                }
            a3_res["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            if a3_res.get("available"):
                prob = a3_res.get("final_probability", a3_res.get("final_score"))
                cls  = a3_res.get("final_class")
                if not _is_valid_prob(prob) or isinstance(cls, bool) or cls not in (0, 1):
                    a3_res = {
                        "available": False,
                        "status": "failed",
                        "error": f"Agent 3 returned invalid output (prob={prob!r}, class={cls!r}).",
                        "duration_ms": a3_res["duration_ms"],
                    }
            results["agent3"] = a3_res
        else:
            missing = []
            if not a1_ok:
                missing.append("Agent 1")
            if not a2_ok:
                missing.append("Agent 2")
            results["agent3"] = {
                "available": False,
                "status": "unavailable",
                "error": (
                    f"Agent 3 requires valid probabilities from both Agent 1 and Agent 2. "
                    f"Unavailable/failed upstream: {', '.join(missing)}."
                ),
            }

        return results
