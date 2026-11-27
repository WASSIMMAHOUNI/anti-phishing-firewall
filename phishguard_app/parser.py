"""
PHISHGUARD — Email parser.
Converts a raw .eml file (bytes or string) into the dict expected by agents:
  {"sender": str, "receiver": str, "date": str, "subject": str, "body": str}

Note: the key is "receiver", not "recipient" — matches Agent 1's contract.
"""
import re
from email import policy
from email.parser import BytesParser, Parser
from typing import Dict


def parse_eml_bytes(data: bytes) -> Dict[str, str]:
    """Parse a raw .eml file given as bytes."""
    msg = BytesParser(policy=policy.default).parsebytes(data)
    return _extract(msg)


def parse_eml_string(text: str) -> Dict[str, str]:
    """Parse a raw .eml file given as a string."""
    msg = Parser(policy=policy.default).parsestr(text)
    return _extract(msg)


def build_manual(
    sender: str,
    receiver: str,
    date: str,
    subject: str,
    body: str = "",
) -> Dict[str, str]:
    """Build an email dict from manually-entered form fields."""
    return {
        "sender":   sender.strip(),
        "receiver": receiver.strip(),
        "date":     date.strip(),
        "subject":  subject.strip(),
        "body":     body.strip(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract(msg) -> Dict[str, str]:
    sender   = _safe_header(msg, "From")
    receiver = _safe_header(msg, "To")
    date     = _safe_header(msg, "Date")
    subject  = _safe_header(msg, "Subject")
    body     = _get_body(msg)
    return {
        "sender":   sender,
        "receiver": receiver,
        "date":     date,
        "subject":  subject,
        "body":     body,
    }


def _safe_header(msg, name: str) -> str:
    val = msg.get(name, "")
    return str(val).strip() if val else ""


def _get_body(msg) -> str:
    """Extract plain-text body; fall back to HTML stripped of tags."""
    # Multipart: prefer text/plain, then text/html (skip attachments)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                try:
                    return (part.get_content() or "").strip()
                except Exception:
                    pass
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/html":
                try:
                    return _strip_html(part.get_content() or "")
                except Exception:
                    pass
        return ""

    # Single-part
    try:
        content = msg.get_content() or ""
        if msg.get_content_type() == "text/html":
            content = _strip_html(content)
        return content.strip()
    except Exception:
        return ""


def _strip_html(html: str) -> str:
    """Minimal HTML tag stripper (no external deps)."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
