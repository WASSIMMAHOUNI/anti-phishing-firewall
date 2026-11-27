"""
Tests for parser.py — .eml parsing and manual dict building.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import textwrap
from parser import parse_eml_bytes, parse_eml_string, build_manual


# ---------------------------------------------------------------------------
# Minimal valid .eml
# ---------------------------------------------------------------------------
SAMPLE_EML = textwrap.dedent("""\
    From: Alice <alice@example.com>
    To: Bob <bob@example.com>
    Date: Mon, 01 Jan 2024 10:00:00 +0000
    Subject: Hello from Alice
    MIME-Version: 1.0
    Content-Type: text/plain; charset=utf-8

    Hi Bob, just checking in. Best, Alice
""")


def test_parse_eml_string_headers():
    d = parse_eml_string(SAMPLE_EML)
    assert "alice@example.com" in d["sender"]
    assert "bob@example.com" in d["receiver"]
    assert "2024" in d["date"]
    assert "Hello from Alice" in d["subject"]


def test_parse_eml_string_body():
    d = parse_eml_string(SAMPLE_EML)
    assert "Hi Bob" in d["body"]


def test_parse_eml_bytes():
    d = parse_eml_bytes(SAMPLE_EML.encode("utf-8"))
    assert "alice@example.com" in d["sender"]


def test_parse_eml_missing_headers():
    minimal = "Subject: Only subject\n\nBody text"
    d = parse_eml_string(minimal)
    assert d["subject"] == "Only subject"
    assert d["sender"] == ""
    assert d["receiver"] == ""
    assert d["date"] == ""
    assert "Body text" in d["body"]


def test_build_manual_key_is_receiver():
    d = build_manual("a@x.com", "b@y.com", "Mon, 01 Jan 2024", "Test", "Body")
    assert "receiver" in d
    assert "recipient" not in d
    assert d["receiver"] == "b@y.com"


def test_build_manual_strips_whitespace():
    d = build_manual("  a@x.com  ", "  b@y.com  ", " date ", " subj ", " body ")
    assert d["sender"] == "a@x.com"
    assert d["receiver"] == "b@y.com"


def test_parse_eml_html_body():
    html_eml = (
        "From: a@x.com\nTo: b@y.com\nDate: Mon\nSubject: S\n"
        "Content-Type: text/html\n\n<html><body><p>Hello <b>World</b></p></body></html>"
    )
    d = parse_eml_string(html_eml)
    assert "Hello" in d["body"]
    assert "<html>" not in d["body"]


def test_parse_eml_multipart_with_attachments():
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "receiver@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 10:00:00 +0000"
    msg["Subject"] = "Multipart with attachments"
    msg.set_content("Actual email body.")
    msg.add_attachment(b"attachment text", maintype="text", subtype="plain", filename="note.txt")
    msg.add_attachment(b"\x89PNG\r\n\x1a\n", maintype="application", subtype="octet-stream", filename="doc.bin")

    d = parse_eml_bytes(msg.as_bytes())
    assert d["sender"] == "sender@example.com"
    assert d["receiver"] == "receiver@example.com"
    assert d["body"] == "Actual email body."
    assert "attachment text" not in d["body"]

