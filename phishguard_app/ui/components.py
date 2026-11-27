"""
PHISHGUARD — Shared UI components and CSS helpers.
"""
import streamlit as st
from typing import Optional, Dict, Any
import config


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

def inject_css():
    c = config.COLORS
    st.markdown(
        f"""
        <style>
        /* ---- Base ---- */
        .stApp {{ background-color: {c['bg']}; }}

        /* ---- Agent cards ---- */
        .pg-card {{
            background: {c['card_bg']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.6rem;
        }}
        .pg-card-title {{
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {c['text_muted']};
            margin-bottom: 0.3rem;
        }}
        .pg-card-value {{
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        .pg-card-sub {{
            font-size: 0.8rem;
            color: {c['text_muted']};
            margin-top: 0.2rem;
        }}

        /* ---- Status badges ---- */
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .badge-ready     {{ background:#1a3a26; color:{c['safe']}; border:1px solid {c['safe']}; }}
        .badge-error     {{ background:#3a1a1a; color:{c['not_safe']}; border:1px solid {c['not_safe']}; }}
        .badge-unavail   {{ background:#2a2520; color:{c['warning']}; border:1px solid {c['warning']}; }}
        .badge-pending   {{ background:#222228; color:{c['pending']}; border:1px solid {c['pending']}; }}

        /* ---- Score gauge label ---- */
        .score-label {{
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {c['text_muted']};
        }}
        .score-value {{
            font-size: 2.4rem;
            font-weight: 800;
        }}
        .score-safe    {{ color: {c['safe']}; }}
        .score-notsafe {{ color: {c['not_safe']}; }}
        .score-neutral {{ color: {c['text']}; }}

        /* ---- Warning banner ---- */
        .pg-warning {{
            background: #2a2510;
            border: 1px solid {c['warning']};
            border-left: 4px solid {c['warning']};
            border-radius: 6px;
            padding: 0.65rem 1rem;
            font-size: 0.85rem;
            color: {c['warning']};
            margin: 0.5rem 0;
        }}
        .pg-info {{
            background: #0e1a2a;
            border: 1px solid {c['agent1']};
            border-left: 4px solid {c['agent1']};
            border-radius: 6px;
            padding: 0.65rem 1rem;
            font-size: 0.85rem;
            color: {c['agent1']};
            margin: 0.5rem 0;
        }}

        /* ---- Section divider ---- */
        .pg-divider {{
            border: none;
            border-top: 1px solid {c['border']};
            margin: 1.2rem 0;
        }}

        /* ---- Metric row ---- */
        .metric-row {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 0.8rem;
        }}
        .metric-item {{
            background: {c['card_bg']};
            border: 1px solid {c['border']};
            border-radius: 8px;
            padding: 0.5rem 0.9rem;
            min-width: 120px;
        }}
        .metric-label {{
            font-size: 0.7rem;
            color: {c['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .metric-val {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {c['text']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Badge helpers
# ---------------------------------------------------------------------------

def status_badge(status: str) -> str:
    mapping = {
        "ready":       ("badge-ready",   "✓ Ready"),
        "completed":   ("badge-ready",   "✓ Completed"),
        "error":       ("badge-error",   "✗ Failed"),
        "failed":      ("badge-error",   "✗ Failed"),
        "unavailable": ("badge-unavail", "⚠ Unavailable"),
        "pending":     ("badge-pending", "◌ Unavailable (Pending)"),
    }
    cls, label = mapping.get(status, ("badge-pending", status.capitalize()))
    return f'<span class="badge {cls}">{label}</span>'


# ---------------------------------------------------------------------------
# Agent status card (overview page)
# ---------------------------------------------------------------------------

def agent_status_card(
    label: str,
    agent_color: str,
    status_dict: Dict[str, Any],
):
    status  = status_dict.get("status", "pending")
    error   = status_dict.get("error", "")
    meta    = status_dict.get("meta")

    badge_html = status_badge(status)
    border_color = {
        "ready":       config.COLORS["safe"],
        "error":       config.COLORS["not_safe"],
        "unavailable": config.COLORS["warning"],
        "pending":     config.COLORS["pending"],
    }.get(status, config.COLORS["border"])

    extra = ""
    if status == "ready" and meta:
        val_m = meta.get("validation_metrics") or meta.get("test_metrics", {}).get("agent3_fusion", {})
        split_lbl = "Test" if "test_metrics" in meta else "Val"
        acc   = val_m.get("accuracy", None)
        f1    = val_m.get("f1", None)
        auc   = val_m.get("roc_auc", None)
        if acc is not None:
            extra = (
                f'<div class="metric-row">'
                f'<div class="metric-item"><div class="metric-label">{split_lbl} Acc</div>'
                f'<div class="metric-val">{acc:.4f}</div></div>'
                f'<div class="metric-item"><div class="metric-label">F1</div>'
                f'<div class="metric-val">{f1:.4f}</div></div>'
                f'<div class="metric-item"><div class="metric-label">ROC-AUC</div>'
                f'<div class="metric-val">{auc:.4f}</div></div>'
                f'</div>'
            )
    elif error:
        short = error[:160] + ("…" if len(error) > 160 else "")
        extra = f'<div class="pg-card-sub" style="color:#8b949e">{short}</div>'

    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {border_color}">
          <div class="pg-card-title">{label}</div>
          <div style="margin-bottom:0.4rem">{badge_html}</div>
          {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Score display
# ---------------------------------------------------------------------------

THRESHOLD_PROB = 0.525   # will be overridden at runtime from meta


def score_gauge(
    label: str,
    score: float,
    agent_color: str,
    threshold: float = THRESHOLD_PROB,
    insufficient: bool = False,
):
    """Render a score with a progress bar and verdict label."""
    pct = int(score * 100)
    verdict = "NOT SAFE" if score >= threshold else "SAFE"
    css_cls  = "score-notsafe" if score >= threshold else "score-safe"
    if insufficient:
        css_cls = "score-neutral"
        verdict = "UNRELIABLE"

    bar_color = (
        config.COLORS["not_safe"] if score >= threshold
        else config.COLORS["safe"]
    )
    if insufficient:
        bar_color = config.COLORS["warning"]

    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {agent_color}">
          <div class="pg-card-title">{label}</div>
          <div class="score-value {css_cls}">{score:.4f}</div>
          <div style="margin:0.4rem 0 0.2rem 0">
            <div style="background:#30363d;border-radius:4px;height:8px;overflow:hidden">
              <div style="width:{pct}%;background:{bar_color};height:100%;
                          border-radius:4px;transition:width .3s"></div>
            </div>
          </div>
          <div class="pg-card-sub">
            P(Not Safe) &nbsp;|&nbsp;
            Verdict: <strong class="{css_cls}">{verdict}</strong>
            &nbsp;|&nbsp; threshold ≈ {threshold:.3f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def unavailable_card(label: str, agent_color: str, reason: str):
    """Render a greyed-out card when an agent can't run."""
    short = reason[:200] + ("…" if len(reason) > 200 else "")
    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {config.COLORS['pending']};
                                     opacity:0.75">
          <div class="pg-card-title">{label}</div>
          {status_badge("unavailable" if "pending" not in reason.lower() else "pending")}
          <div class="pg-card-sub" style="margin-top:0.4rem">{short}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_banner(message: str):
    st.markdown(
        f'<div class="pg-warning">⚠ {message}</div>',
        unsafe_allow_html=True,
    )


def info_banner(message: str):
    st.markdown(
        f'<div class="pg-info">ℹ {message}</div>',
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<hr class="pg-divider">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<h2 style='margin-bottom:0.1rem'>{title}</h2>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<p style='color:{config.COLORS['text_muted']};margin-top:0'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    divider()


# ---------------------------------------------------------------------------
# Redesigned Analyze Email result components
# ---------------------------------------------------------------------------

def render_verdict_banner(verdict: Dict[str, Any]):
    """
    Section A & B: Prominent Final Verdict + real probability (only when Agent 3 succeeded).
    """
    c = config.COLORS
    if verdict["verdict_available"]:
        is_not_safe = verdict["verdict_class"] == 1
        accent = c["not_safe"] if is_not_safe else c["safe"]
        bg = "#251417" if is_not_safe else "#102419"
        icon = "🛡️⚠️" if is_not_safe else "🛡️✓"
        label = verdict["verdict_label"]
        prob = verdict.get("final_score")
        prob_html = (
            f"<div style='margin-top:0.75rem;font-size:0.95rem;color:{c['text']}'>"
            f"<strong>Final Probability P(Not Safe):</strong> "
            f"<span style='color:{accent};font-weight:700;font-size:1.1rem'>{prob:.4f}</span>"
            f"</div>"
            if prob is not None
            else ""
        )
    else:
        accent = c["warning"]
        bg = "#1b1d22"
        icon = "⏳"
        label = verdict["verdict_label"]
        prob_html = (
            f"<div style='margin-top:0.75rem;font-size:0.88rem;color:{c['text_muted']}'>"
            f"<strong>Final Probability:</strong> Unavailable (requires Agent 3)"
            f"</div>"
        )

    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {accent};border-left:6px solid {accent};
                    border-radius:8px;padding:1.4rem 1.6rem;margin-bottom:1.2rem">
          <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;
                      text-transform:uppercase;color:{c['text_muted']};margin-bottom:0.35rem">
            Final Security Verdict
          </div>
          <div style="font-size:2rem;font-weight:800;color:{accent};line-height:1.15">
            {icon} &nbsp;{label}
          </div>
          <p style="margin:0.65rem 0 0 0;font-size:0.98rem;color:{c['text']}">
            {verdict['explanation']}
          </p>
          {prob_html}
          <div style="margin-top:0.6rem;font-size:0.78rem;color:{c['text_muted']}">
            Experimental detection system — may produce false positives or false negatives.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_indicators(data: Dict[str, Any]):
    """
    Section C: 'Why This Decision? (Detection Indicators)' grounded in real signals.
    """
    c = config.COLORS
    st.subheader("Why This Decision? — Detection Indicators")

    for item in data.get("indicators", []):
        lvl = item["level"]
        if lvl == "risk":
            icon, color = "🔴", c["not_safe"]
        elif lvl == "safe":
            icon, color = "🟢", c["safe"]
        elif lvl == "warning":
            icon, color = "🟡", c["warning"]
        else:
            icon, color = "⚪", c["text_muted"]
        st.markdown(
            f"<div style='padding:0.35rem 0;color:{c['text']}'>"
            f"{icon} &nbsp;<span style='color:{color}'>{item['text']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    limitations = data.get("limitations", [])
    if limitations:
        st.markdown(
            f"<div style='margin-top:0.65rem;padding:0.75rem 1rem;background:{c['card_bg']};"
            f"border:1px solid {c['border']};border-radius:6px;font-size:0.85rem;color:{c['text_muted']}'>"
            + "<br>".join(f"• {lim}" for lim in limitations)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_agent_breakdown_item(item: Dict[str, Any]):
    """
    Section D: Clean row/block for one agent in the Agent Breakdown.
    """
    c = config.COLORS
    status = item["status"]
    badge = status_badge(status)

    if status == "completed":
        prob = item["probability"]
        prob_str = f"{prob:.4f}" if prob is not None else "—"
        detail_html = (
            f"<div style='display:flex;gap:2rem;flex-wrap:wrap;margin-top:0.55rem'>"
            f"<div><span style='color:{c['text_muted']};font-size:0.78rem'>RESULT</span><br>"
            f"<strong>{item['result_label']}</strong></div>"
            f"<div><span style='color:{c['text_muted']};font-size:0.78rem'>P(NOT SAFE)</span><br>"
            f"<strong>{prob_str}</strong></div>"
            + (
                f"<div><span style='color:{c['text_muted']};font-size:0.78rem'>SIGNAL</span><br>"
                f"<span>{item['extra']}</span></div>"
                if item.get("extra") else ""
            )
            + "</div>"
        )
    else:
        err = item.get("error") or "Unavailable"
        detail_html = (
            f"<div style='margin-top:0.45rem;font-size:0.86rem;color:{c['text_muted']}'>"
            f"{err}</div>"
        )

    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {item['color']}">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
            <strong style="font-size:1rem;color:{c['text']}">{item['title']}</strong>
            {badge}
          </div>
          <div style="font-size:0.84rem;color:{c['text_muted']};margin-top:0.25rem">
            {item['role']}
          </div>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_email_info(email_dict: Dict[str, str]):
    """
    Section E: Secondary block displaying Sender, Recipient, Subject, Date.
    """
    import html
    c = config.COLORS
    fields = [
        ("Sender",    email_dict.get("sender") or "—"),
        ("Recipient", email_dict.get("receiver") or "—"),
        ("Subject",   email_dict.get("subject") or "—"),
        ("Date",      email_dict.get("date") or "—"),
    ]
    rows = "".join(
        f"<tr><td style='color:{c['text_muted']};padding:0.25rem 1rem 0.25rem 0;width:110px'>"
        f"<strong>{k}</strong></td>"
        f"<td style='color:{c['text']};padding:0.25rem 0'>{html.escape(v)}</td></tr>"
        for k, v in fields
    )
    st.markdown(
        f"""
        <div class="pg-card">
          <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
            {rows}
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
