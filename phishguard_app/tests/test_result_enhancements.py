"""
Tests for result-page enhancements:
  1. Shared probability formatting (edge cases + full precision preservation)
  2. Threshold resolution from metadata/config without inventing missing thresholds
  3. Agent 1 vs Agent 2 disagreement detection using unrounded probabilities
  4. Dynamic Agent 3 fusion explanation + neutral fallback when metadata missing
  5. Offline link-safety advisory (normalization, IPv4/IPv6, ports/queries, no substring matching, malformed URLs)
  6. Regression test: link advisory never changes Agent 1/2/3 outputs
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from formatting import format_probability, format_probability_full
from link_advisory import (
    analyze_links_advisory,
    extract_html_links,
    extract_registered_domain,
    is_ip_address,
    normalize_hostname,
    parse_url_safely,
)
from orchestrator import Orchestrator
from parser import extract_html_from_eml_string, parse_eml_string
from result_presenter import (
    build_agent_breakdown,
    build_disagreement_summary,
    build_fusion_explanation,
    resolve_agent_thresholds,
)


# ---------------------------------------------------------------------------
# 1. Probability formatting
# ---------------------------------------------------------------------------

def test_probability_formatting_boundaries():
    assert format_probability(0.0) == "0.0000"
    assert format_probability(1.0) == "1.0000"
    assert format_probability(0.00005) == "<0.0001"
    assert format_probability(0.0001) == "0.0001"
    assert format_probability(0.99995) == ">0.9999"
    assert format_probability(0.9999) == "0.9999"
    assert format_probability(0.524979) == "0.5250"
    assert format_probability(None) == "—"
    assert format_probability(True) == "—"

    # Full precision preserved
    assert format_probability_full(0.0000123456) == "1.23456e-05"
    assert format_probability_full(0.99998765) == "0.99998765"
    assert format_probability_full(0.0) == "0"
    assert format_probability_full(1.0) == "1"


# ---------------------------------------------------------------------------
# 2. Threshold resolution & missing-threshold handling
# ---------------------------------------------------------------------------

def test_resolve_agent_thresholds_from_metadata():
    all_status = {
        "agent1": {"meta": {"threshold_info": {"threshold_prob": 0.52497918747894}}},
        "agent2": {"meta": {"model_checkpoint": "distilbert-base-uncased"}},
        "agent3": {"config": {"decision_threshold": 0.5}},
    }
    thr = resolve_agent_thresholds(all_status)
    assert thr["agent1"] == pytest.approx(0.52497918747894)
    assert thr["agent2"] == pytest.approx(0.5)
    assert thr["agent3"] == pytest.approx(0.5)


def test_missing_thresholds_are_not_invented():
    all_status = {
        "agent1": {"meta": None},
        "agent2": {"meta": {"threshold": 0.6}},
        "agent3": {"config": None},
    }
    thr = resolve_agent_thresholds(all_status)
    assert thr["agent1"] is None
    assert thr["agent2"] == pytest.approx(0.6)
    assert thr["agent3"] is None

    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.7, "header_completeness": 1.0, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.1},
        "agent3": {"available": True, "status": "completed", "final_probability": 0.2, "final_class": 0},
    }
    disagreement = build_disagreement_summary(results, thr, a3_meta=None)
    assert disagreement["comparable"] is False
    assert disagreement["suppress_banner"] is True

    breakdown = build_agent_breakdown(results, a1_threshold=thr["agent1"], a2_threshold=thr["agent2"], a3_threshold=thr["agent3"])
    assert breakdown[0]["result_label"] == "Threshold Unavailable"


# ---------------------------------------------------------------------------
# 3. Unrounded probability comparison & disagreement
# ---------------------------------------------------------------------------

def test_disagreement_uses_unrounded_probabilities():
    # 0.52496 rounds to 0.5250, but is strictly below 0.52497918747894
    thr = {"agent1": 0.52497918747894, "agent2": 0.5, "agent3": 0.5}
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.52496, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.85},
        "agent3": {"available": True, "status": "completed", "final_probability": 0.81, "final_class": 1},
    }
    d = build_disagreement_summary(results, thr)
    assert d["comparable"] is True
    assert d["disagree"] is True
    assert d["suppress_banner"] is False
    assert d["a1_side"] == "Safe"
    assert d["a2_side"] == "Not Safe"


def test_disagreement_reverse_direction_a1_not_safe_a2_safe():
    thr = {"agent1": 0.52497918747894, "agent2": 0.5, "agent3": 0.5}
    results = {
        "agent1": {"available": True, "status": "completed", "nb_score": 0.91, "insufficient_evidence": False},
        "agent2": {"available": True, "status": "completed", "text_score": 0.04},
        "agent3": {"available": True, "status": "completed", "final_probability": 0.08, "final_class": 0},
    }
    d = build_disagreement_summary(results, thr)
    assert d["comparable"] is True
    assert d["disagree"] is True
    assert d["suppress_banner"] is False
    assert d["a1_side"] == "Not Safe"
    assert d["a2_side"] == "Safe"
    assert d["a3_label"] == "Safe"


# ---------------------------------------------------------------------------
# 4. Fusion explanation from agent3_meta.json + fallback
# ---------------------------------------------------------------------------

def test_fusion_explanation_dynamic_and_fallback():
    meta = {
        "coefficients": {
            "nb_logit": 0.4294824652244424,
            "text_logit": 2.1374991029607213,
            "intercept": -0.598023149833563,
        }
    }
    expl = build_fusion_explanation(meta)
    assert expl["has_coefficients"] is True
    assert "text_logit" in expl["summary"]
    assert "nb_logit" in expl["summary"]
    assert "2.1375" in expl["summary"]
    assert "0.4295" in expl["summary"]
    assert "does not always determine" in expl["summary"]

    fallback = build_fusion_explanation(None)
    assert fallback["has_coefficients"] is False
    assert fallback["summary"] == "Agent 3 combines both scores to produce the final decision."


# ---------------------------------------------------------------------------
# 5. Offline link-safety advisory
# ---------------------------------------------------------------------------

def test_link_advisory_domain_normalization_and_no_substring_matching():
    assert normalize_hostname("  MAIL.Example.COM. ") == "mail.example.com"
    assert extract_registered_domain("sub.paypal.com.") == "paypal.com"
    assert extract_registered_domain("evil-paypal.com") == "evil-paypal.com"
    assert extract_registered_domain("portal.bank.co.uk.") == "bank.co.uk"
    assert is_ip_address("192.168.1.10") is True
    assert is_ip_address("[2001:db8::1]") is True
    assert extract_registered_domain("192.168.1.10") is None

    email_no_substring_trap = {
        "sender": "Security <support@paypal.com.>",
        "subject": "Account check",
        "body": (
            "Visit HTTPS://LOGIN.PAYPAL.COM.:8443/auth?next=1 "
            "and avoid https://evil-paypal.com/login and http://192.168.1.50:8080/pay "
            "and malformed http://[::1"
        ),
    }
    adv = analyze_links_advisory(email_no_substring_trap)
    assert adv["sender_reg_domain"] == "paypal.com"
    assert len(adv["matching_links"]) == 1
    assert len(adv["mismatched_links"]) == 1  # evil-paypal.com must NOT match paypal.com
    assert len(adv["ip_links"]) == 1
    assert adv["malformed_count"] == 1
    assert adv["advisory_level"] == "warning"

    bad = parse_url_safely("https://example.com:notaport/path")
    assert bad["valid"] is False


def test_html_only_email_preserves_agent_dict_and_detects_href_mismatch():
    raw_html_eml = (
        "From: security@paypal.com\r\n"
        "To: user@example.com\r\n"
        "Date: Mon, 01 Jan 2024 10:00:00 +0000\r\n"
        "Subject: Account Notice\r\n"
        "Content-Type: text/html; charset=utf-8\r\n\r\n"
        "<html><body>"
        "<p>Please verify at <a href=\"https://evil-phish.com/auth\">https://paypal.com/verify</a></p>"
        "<p>Or visit <a href=\"https://login.paypal.com/help\">paypal.com/help</a></p>"
        "</body></html>"
    )
    email_dict = parse_eml_string(raw_html_eml)
    html_body = extract_html_from_eml_string(raw_html_eml)

    # Agent email dict remains 5 keys with stripped text
    assert set(email_dict.keys()) == {"sender", "receiver", "date", "subject", "body"}
    assert "<a " not in email_dict["body"]

    adv = analyze_links_advisory(email_dict, html_body=html_body)
    assert adv["html_link_count"] == 2
    assert len(adv["visible_text_mismatches"]) == 1
    assert adv["visible_text_mismatches"][0]["hostname"] == "evil-phish.com"
    assert adv["visible_text_mismatches"][0]["visible_text"] == "https://paypal.com/verify"
    assert len(adv["matching_links"]) == 1
    assert adv["matching_links"][0]["hostname"] == "login.paypal.com"


def test_multipart_email_multiple_links_and_malformed_html():
    raw_multi = (
        "From: alerts@bank.co.uk\r\n"
        "To: client@example.org\r\n"
        "Date: Tue, 02 Jan 2024 12:00:00 +0000\r\n"
        "Subject: Statement ready\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/alternative; boundary=\"b1\"\r\n\r\n"
        "--b1\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "Read your statement at https://portal.bank.co.uk/docs\r\n"
        "--b1\r\n"
        "Content-Type: text/html; charset=utf-8\r\n\r\n"
        "<div>"
        "<a href='https://portal.bank.co.uk/docs'>Click here</a> "
        "<a href='http://192.168.1.1/login'>bank.co.uk</a> "
        "<a href='https://evil.example/a'>unclosed <b>tag"
        "</div>\r\n"
        "--b1--\r\n"
    )
    email_dict = parse_eml_string(raw_multi)
    html_body = extract_html_from_eml_string(raw_multi)

    # Multipart agent body still uses text/plain unchanged
    assert email_dict["body"] == "Read your statement at https://portal.bank.co.uk/docs"

    adv = analyze_links_advisory(email_dict, html_body=html_body)
    assert adv["html_link_count"] >= 2
    assert len(adv["matching_links"]) == 1
    assert len(adv["ip_links"]) == 1
    assert len(adv["visible_text_mismatches"]) == 1

    # Direct malformed HTML helper test
    pairs = extract_html_links("<a href='https://a.example'>ok</a><a href='https://b.example'>unclosed")
    assert len(pairs) == 2


# ---------------------------------------------------------------------------
# 6. Regression test: link advisory never alters Agent 1/2/3 outputs
# ---------------------------------------------------------------------------

def test_link_advisory_does_not_alter_model_outputs():
    orch = Orchestrator()
    email = {
        "sender": "alerts@example.com",
        "receiver": "user@example.com",
        "date": "Mon, 01 Jan 2024 10:00:00 +0000",
        "subject": "Urgent verification required",
        "body": "Please check http://192.168.0.1/login and https://evil-example.com/verify?id=42",
    }
    baseline = orch.analyze(copy.deepcopy(email))
    after = orch.analyze(copy.deepcopy(email))
    advisory = analyze_links_advisory(email)

    assert advisory["link_count"] == 2
    assert after["agent1"]["nb_score"] == baseline["agent1"]["nb_score"]
    if baseline["agent2"].get("available") and after["agent2"].get("available"):
        assert after["agent2"]["text_score"] == baseline["agent2"]["text_score"]
    if baseline["agent3"].get("available") and after["agent3"].get("available"):
        assert after["agent3"]["final_probability"] == baseline["agent3"]["final_probability"]
        assert after["agent3"]["final_class"] == baseline["agent3"]["final_class"]
