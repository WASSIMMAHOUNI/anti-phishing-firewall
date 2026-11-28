"""
PHISHGUARD — Pure offline link-safety advisory (post-inference, display-only).

Strictly separate from Agent 1, Agent 2, and Agent 3:
  - Never modifies any agent score, final_probability, or final_class.
  - Never fetches URLs, resolves DNS, or contacts external services.
  - Never logs complete email bodies or raw URLs.
  - Never persists extracted links beyond the current analysis.
  - Uses a conservative registered-domain heuristic (not a complete PSL).
"""
from email.utils import parseaddr
from html.parser import HTMLParser
import ipaddress
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit


_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'\]\)]+",
    re.IGNORECASE,
)

# Conservative set of common two-label public suffixes for heuristic comparison.
# This is intentionally NOT a full Public Suffix List (PSL) implementation.
_COMMON_TWO_LEVEL_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "net.uk",
    "com.au", "net.au", "org.au", "edu.au",
    "co.jp", "ne.jp", "or.jp", "ac.jp",
    "co.nz", "org.nz", "net.nz",
    "co.in", "org.in", "net.in",
    "com.br", "org.br", "net.br",
    "com.cn", "org.cn", "net.cn",
    "co.za", "org.za",
    "com.mx", "com.tr", "com.sg", "co.kr", "com.hk", "com.tw",
})

_TRAILING_PUNCT = ".,;:!?)]}>\"'"

_DOMAIN_CLAIM_PATTERN = re.compile(
    r"^(?:https?://)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d+)?(?:[/?#].*)?$",
    re.IGNORECASE,
)

_HEURISTIC_NOTE = (
    "Advisory only — does not affect model scores or the final verdict. "
    "No URLs were fetched, resolved via DNS, or sent to external services. "
    "Registered-domain comparison uses a conservative heuristic (not a complete "
    "Public Suffix List), so uncommon multi-level ccTLDs may be approximated."
)


def normalize_hostname(host: Optional[str]) -> Optional[str]:
    """Lowercase a hostname and strip whitespace and trailing dots."""
    if not host or not isinstance(host, str):
        return None
    cleaned = host.strip().lower().rstrip(".")
    if not cleaned:
        return None
    # Strip IPv6 brackets if passed directly
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def is_ip_address(host: Optional[str]) -> bool:
    """Return True if host is a valid IPv4 or IPv6 literal."""
    norm = normalize_hostname(host)
    if not norm:
        return False
    try:
        ipaddress.ip_address(norm)
        return True
    except ValueError:
        return False


def extract_registered_domain(host: Optional[str]) -> Optional[str]:
    """
    Conservative registered-domain heuristic (NOT a full PSL).
    Returns None for IP addresses, single-label hosts, or invalid hostnames.
    Never uses substring matching.
    """
    norm = normalize_hostname(host)
    if not norm or is_ip_address(norm):
        return None
    if any(ch.isspace() for ch in norm):
        return None
    labels = [lbl for lbl in norm.split(".") if lbl]
    if len(labels) < 2:
        return None
    last_two = f"{labels[-2]}.{labels[-1]}"
    if len(labels) >= 3 and last_two in _COMMON_TWO_LEVEL_SUFFIXES:
        return f"{labels[-3]}.{last_two}"
    return last_two


def extract_sender_domain(sender: str) -> Optional[str]:
    """Extract and normalize the domain part from a From header."""
    if not sender or not isinstance(sender, str):
        return None
    _, addr = parseaddr(sender)
    candidate = addr or sender
    if "@" not in candidate:
        return None
    domain_part = candidate.rsplit("@", 1)[-1].strip()
    # Strip optional '>' or whitespace
    domain_part = domain_part.rstrip(">").strip()
    return normalize_hostname(domain_part)


class _AnchorExtractor(HTMLParser):
    """Stdlib HTMLParser extracting <a href=...> and its visible text safely."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors: List[Dict[str, str]] = []
        self._in_a = False
        self._current_href: Optional[str] = None
        self._text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() == "a":
            if self._in_a and self._current_href is not None:
                vis = " ".join("".join(self._text_parts).split())
                self.anchors.append({"href": self._current_href, "visible_text": vis})
                self._in_a = False
                self._current_href = None
                self._text_parts = []
            href_val = None
            for k, v in attrs:
                if k and k.lower() == "href" and v is not None:
                    href_val = str(v).strip()
                    break
            if href_val is not None:
                self._in_a = True
                self._current_href = href_val
                self._text_parts = []

    def handle_data(self, data: str):
        if self._in_a and data:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str):
        if tag.lower() == "a" and self._in_a:
            vis = " ".join("".join(self._text_parts).split())
            self.anchors.append({
                "href": self._current_href or "",
                "visible_text": vis,
            })
            self._in_a = False
            self._current_href = None
            self._text_parts = []


def extract_html_links(html_text: str) -> List[Dict[str, str]]:
    """
    Extract <a href="...">visible text</a> pairs from HTML without crashing
    on malformed markup. Preserves href and visible_text separately.
    """
    if not html_text or not isinstance(html_text, str):
        return []
    parser = _AnchorExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        pass
    if parser._in_a and parser._current_href is not None:
        vis = " ".join("".join(parser._text_parts).split())
        parser.anchors.append({"href": parser._current_href, "visible_text": vis})
        parser._in_a = False
    return [
        a for a in parser.anchors
        if a["href"].lower().startswith(("http://", "https://", "www."))
    ]


def _extract_visible_claim(visible_text: str) -> Optional[Dict[str, Any]]:
    """
    If visible link text looks like a URL or domain/IP claim, parse its
    hostname and registered domain (or IP). Returns None for normal prose
    like 'Click here' or 'Verify account'.
    """
    if not visible_text or not isinstance(visible_text, str):
        return None
    cleaned = visible_text.strip().rstrip(_TRAILING_PUNCT)
    if not cleaned or any(ch.isspace() for ch in cleaned):
        return None

    # Explicit URL or www.
    if cleaned.lower().startswith(("http://", "https://", "www.")):
        parsed = parse_url_safely(cleaned)
        if parsed.get("valid"):
            return {
                "hostname": parsed["hostname"],
                "is_ip": parsed["is_ip"],
                "registered_domain": parsed["registered_domain"],
            }
        return None

    # Bare domain/host claim like "paypal.com" or "login.paypal.com/auth"
    m = _DOMAIN_CLAIM_PATTERN.match(cleaned)
    if not m:
        return None
    parsed = parse_url_safely(f"http://{cleaned}")
    if parsed.get("valid"):
        return {
            "hostname": parsed["hostname"],
            "is_ip": parsed["is_ip"],
            "registered_domain": parsed["registered_domain"],
        }
    return None


def extract_urls_from_text(text: str) -> List[str]:
    """Extract candidate http(s) and www URLs from plain text."""
    if not text or not isinstance(text, str):
        return []
    found: List[str] = []
    seen = set()
    for match in _URL_PATTERN.findall(text):
        cleaned = match.rstrip(_TRAILING_PUNCT)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            found.append(cleaned)
    return found


def parse_url_safely(raw_url: str) -> Dict[str, Any]:
    """
    Parse one URL offline without crashing on malformed inputs.
    Normalizes hostnames, separates IP addresses, and handles ports/paths/queries.
    """
    if not raw_url or not isinstance(raw_url, str):
        return {"raw": "", "valid": False, "reason": "empty"}

    trimmed = raw_url.strip()
    has_scheme = trimmed.lower().startswith(("http://", "https://"))
    parse_target = trimmed if has_scheme else f"http://{trimmed}"

    try:
        parts = urlsplit(parse_target)
        scheme = parts.scheme.lower() if has_scheme else "http (assumed)"
        raw_host = parts.hostname  # urlsplit strips port and IPv6 brackets
        # Validate port if present (urlsplit.port raises ValueError on invalid port)
        port = parts.port
    except ValueError:
        return {
            "raw": trimmed,
            "valid": False,
            "reason": "malformed URL or invalid port",
        }

    host = normalize_hostname(raw_host)
    if not host or any(ch.isspace() for ch in host):
        return {
            "raw": trimmed,
            "valid": False,
            "reason": "missing or invalid hostname",
        }

    ip_literal = is_ip_address(host)
    reg_domain = None if ip_literal else extract_registered_domain(host)
    if not ip_literal and reg_domain is None:
        return {
            "raw": trimmed,
            "valid": False,
            "reason": "unrecognized or single-label hostname",
        }

    is_punycode = any(lbl.startswith("xn--") for lbl in host.split("."))
    has_userinfo = parts.username is not None or parts.password is not None

    return {
        "raw": trimmed,
        "valid": True,
        "scheme": scheme,
        "is_https": scheme == "https",
        "hostname": host,
        "port": port,
        "is_ip": ip_literal,
        "registered_domain": reg_domain,
        "is_punycode": is_punycode,
        "has_userinfo": has_userinfo,
    }


def analyze_links_advisory(
    email_dict: Dict[str, str],
    html_body: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run offline link-safety checks on an email's subject, body, and optional HTML.
    Completely independent of Agent 1, Agent 2, and Agent 3.
    """
    sender = email_dict.get("sender", "") if email_dict else ""
    subject = email_dict.get("subject", "") if email_dict else ""
    body = email_dict.get("body", "") if email_dict else ""

    sender_domain = extract_sender_domain(sender)
    sender_is_ip = is_ip_address(sender_domain) if sender_domain else False
    sender_reg = None if sender_is_ip else extract_registered_domain(sender_domain)

    # 1. Extract HTML <a href="...">visible text</a> pairs (if HTML present)
    html_sources = []
    if html_body and isinstance(html_body, str):
        html_sources.append(html_body)
    if "<a " in body.lower() or "<a\t" in body.lower():
        html_sources.append(body)

    html_anchors: List[Dict[str, str]] = []
    seen_anchor_pairs = set()
    for src in html_sources:
        for anchor in extract_html_links(src):
            key = (anchor["href"], anchor["visible_text"])
            if key not in seen_anchor_pairs:
                seen_anchor_pairs.add(key)
                html_anchors.append(anchor)

    # 2. Extract plain-text URLs (avoid duplicating href or visible_text already captured as HTML anchor)
    anchor_seen_urls = set()
    for a in html_anchors:
        anchor_seen_urls.add(a["href"].rstrip(_TRAILING_PUNCT))
        if a["visible_text"]:
            anchor_seen_urls.add(a["visible_text"].strip().rstrip(_TRAILING_PUNCT))
    plain_urls = [
        u for u in extract_urls_from_text(f"{subject}\n{body}")
        if u not in anchor_seen_urls
    ]

    candidate_items: List[Dict[str, Optional[str]]] = [
        {"url": a["href"], "visible_text": a["visible_text"], "source": "html_anchor"}
        for a in html_anchors
    ] + [
        {"url": u, "visible_text": None, "source": "plain_text"}
        for u in plain_urls
    ]

    parsed_links: List[Dict[str, Any]] = []
    malformed_count = 0
    ip_links: List[Dict[str, Any]] = []
    http_links: List[Dict[str, Any]] = []
    mismatched_links: List[Dict[str, Any]] = []
    matching_links: List[Dict[str, Any]] = []
    punycode_links: List[Dict[str, Any]] = []
    userinfo_links: List[Dict[str, Any]] = []
    visible_text_mismatches: List[Dict[str, Any]] = []

    for item in candidate_items:
        info = parse_url_safely(item["url"] or "")
        info["source"] = item["source"]
        info["visible_text"] = item["visible_text"]
        info["visible_text_mismatch"] = False
        info["visible_claimed_domain"] = None

        if not info["valid"]:
            malformed_count += 1
            parsed_links.append(info)
            continue

        # Visible-text vs actual-href domain comparison (exact registered-domain equality)
        if item["visible_text"]:
            claim = _extract_visible_claim(item["visible_text"])
            if claim is not None:
                claimed_target = (
                    claim["hostname"] if claim["is_ip"] else claim["registered_domain"]
                )
                actual_target = (
                    info["hostname"] if info["is_ip"] else info["registered_domain"]
                )
                info["visible_claimed_domain"] = claimed_target
                if claimed_target and actual_target and (
                    claim["is_ip"] != info["is_ip"] or claimed_target != actual_target
                ):
                    info["visible_text_mismatch"] = True
                    visible_text_mismatches.append(info)

        # Sender-domain comparison (exact registered-domain equality; never substring)
        if info["is_ip"]:
            info["domain_match"] = "ip_literal"
            ip_links.append(info)
        elif sender_reg is None:
            info["domain_match"] = "sender_unavailable"
        elif info["registered_domain"] == sender_reg:
            info["domain_match"] = "match"
            matching_links.append(info)
        else:
            info["domain_match"] = "mismatch"
            mismatched_links.append(info)

        if not info["is_https"]:
            http_links.append(info)
        if info["is_punycode"]:
            punycode_links.append(info)
        if info["has_userinfo"]:
            userinfo_links.append(info)

        parsed_links.append(info)

    findings: List[Dict[str, str]] = []
    if not candidate_items:
        findings.append({
            "level": "info",
            "text": "No HTTP/HTTPS or www links were detected in the subject, body, or HTML anchors.",
        })
    else:
        if visible_text_mismatches:
            findings.append({
                "level": "warning",
                "text": (
                    f"{len(visible_text_mismatches)} HTML link(s) display a visible domain/URL "
                    "that differs from the actual href destination."
                ),
            })
        if ip_links:
            findings.append({
                "level": "warning",
                "text": (
                    f"{len(ip_links)} link(s) use a raw IP address instead of a domain name."
                ),
            })
        if mismatched_links and sender_reg:
            hosts_preview = ", ".join(
                sorted({lnk["registered_domain"] for lnk in mismatched_links if lnk.get("registered_domain")})[:4]
            )
            findings.append({
                "level": "warning",
                "text": (
                    f"{len(mismatched_links)} link(s) point to a different registered domain than "
                    f"the sender ({sender_reg}): {hosts_preview}."
                ),
            })
        if http_links:
            findings.append({
                "level": "warning",
                "text": f"{len(http_links)} link(s) use unencrypted HTTP rather than HTTPS.",
            })
        if punycode_links:
            findings.append({
                "level": "warning",
                "text": f"{len(punycode_links)} link(s) contain Punycode/IDN (xn--) domain labels.",
            })
        if userinfo_links:
            findings.append({
                "level": "warning",
                "text": f"{len(userinfo_links)} link(s) include embedded userinfo (@) before the host.",
            })
        if malformed_count:
            findings.append({
                "level": "info",
                "text": f"{malformed_count} malformed URL string(s) could not be parsed cleanly.",
            })
        if (
            matching_links
            and not visible_text_mismatches
            and not ip_links
            and not mismatched_links
            and not http_links
            and not punycode_links
            and not userinfo_links
        ):
            findings.append({
                "level": "safe",
                "text": (
                    f"All {len(matching_links)} extracted HTTPS link(s) match the sender's "
                    f"registered domain ({sender_reg})."
                ),
            })
        elif not findings:
            findings.append({
                "level": "info",
                "text": f"{len(candidate_items)} link(s) extracted; sender domain unavailable for comparison.",
            })

    has_warnings = any(f["level"] == "warning" for f in findings)
    advisory_level = "warning" if has_warnings else ("info" if candidate_items else "none")

    return {
        "link_count": len(candidate_items),
        "html_link_count": len(html_anchors),
        "plain_link_count": len(plain_urls),
        "valid_link_count": sum(1 for lnk in parsed_links if lnk.get("valid")),
        "malformed_count": malformed_count,
        "sender_domain": sender_domain,
        "sender_reg_domain": sender_reg,
        "links": parsed_links,
        "ip_links": ip_links,
        "http_links": http_links,
        "mismatched_links": mismatched_links,
        "matching_links": matching_links,
        "visible_text_mismatches": visible_text_mismatches,
        "findings": findings,
        "advisory_level": advisory_level,
        "heuristic_note": _HEURISTIC_NOTE,
    }
