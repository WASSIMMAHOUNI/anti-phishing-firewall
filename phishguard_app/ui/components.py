"""
PHISHGUARD — Shared SaaS UI components, CSS design system, and Altair charts.
"""
import html
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import config
from formatting import format_probability, format_probability_full


# ---------------------------------------------------------------------------
# Global CSS — Dark Violet / Deep Navy Cybersecurity SaaS Design System
# ---------------------------------------------------------------------------

def inject_css():
    c = config.COLORS
    st.markdown(
        f"""
        <style>
        /* ---- Base canvas & sidebar ---- */
        .stApp {{
            background-color: {c['bg']};
            color: {c['text']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {c['bg_secondary']};
            border-right: 1px solid {c['border']};
        }}

        /* ---- Primary buttons & tabs ---- */
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button {{
            background-color: {c['primary']};
            color: #ffffff;
            border: 1px solid {c['primary']};
            border-radius: 8px;
            font-weight: 600;
        }}
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button:hover {{
            background-color: #7c3aed;
            border-color: #7c3aed;
        }}
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {c['border']};
            background-color: {c['card_bg']};
            color: {c['text']};
        }}
        .stButton > button:hover {{
            border-color: {c['primary']};
            color: #ffffff;
        }}

        /* ---- Cards ---- */
        .pg-card {{
            background: {c['card_bg']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        }}
        .pg-card-title {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {c['text_muted']};
            margin-bottom: 0.35rem;
        }}
        .pg-card-sub {{
            font-size: 0.84rem;
            color: {c['text_muted']};
            line-height: 1.45;
        }}

        /* ---- Hero banner ---- */
        .pg-hero {{
            background: {c['card_bg']};
            border: 1px solid {c['border']};
            border-top: 3px solid {c['primary']};
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }}

        /* ---- KPI tiles ---- */
        .pg-kpi {{
            background: {c['card_bg']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.6rem;
        }}
        .pg-kpi-label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {c['text_muted']};
        }}
        .pg-kpi-val {{
            font-size: 1.75rem;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
            margin: 0.2rem 0;
        }}
        .pg-kpi-sub {{
            font-size: 0.78rem;
            color: {c['text_muted']};
        }}

        /* ---- Status badges & dots ---- */
        .badge {{
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
        }}
        .badge-ready   {{ background:{c['safe_bg']}; color:{c['safe']}; border:1px solid {c['safe']}; }}
        .badge-error   {{ background:{c['not_safe_bg']}; color:{c['not_safe']}; border:1px solid {c['not_safe']}; }}
        .badge-unavail {{ background:{c['warning_bg']}; color:{c['warning']}; border:1px solid {c['warning']}; }}
        .badge-pending {{ background:#181b28; color:{c['pending']}; border:1px solid {c['pending']}; }}

        .status-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        /* ---- Metric row inside cards ---- */
        .metric-row {{
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin: 0.65rem 0 0.25rem 0;
        }}
        .metric-item {{
            background: {c['card_elevated']};
            border: 1px solid {c['border']};
            border-radius: 8px;
            padding: 0.45rem 0.75rem;
            min-width: 92px;
            flex: 1 1 92px;
        }}
        .metric-label {{
            font-size: 0.68rem;
            color: {c['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-val {{
            font-size: 1rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: {c['text']};
        }}

        /* ---- Pipeline flow ---- */
        .pg-flow {{
            display: flex;
            align-items: stretch;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 1rem 0;
        }}
        .pg-flow-node {{
            background: {c['card_bg']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            padding: 0.85rem 1rem;
            flex: 1 1 180px;
        }}
        .pg-flow-arrow {{
            display: flex;
            align-items: center;
            color: {c['text_muted']};
            font-weight: 700;
            font-size: 1.1rem;
        }}

        /* ---- Warning & info banners ---- */
        .pg-warning {{
            background: {c['warning_bg']};
            border: 1px solid {c['warning']};
            border-left: 4px solid {c['warning']};
            border-radius: 8px;
            padding: 0.7rem 1rem;
            font-size: 0.86rem;
            color: {c['warning']};
            margin: 0.5rem 0;
        }}
        .pg-info {{
            background: #10162b;
            border: 1px solid {c['secondary']};
            border-left: 4px solid {c['secondary']};
            border-radius: 8px;
            padding: 0.7rem 1rem;
            font-size: 0.86rem;
            color: {c['text']};
            margin: 0.5rem 0;
        }}

        /* ---- Divider ---- */
        .pg-divider {{
            border: none;
            border-top: 1px solid {c['border']};
            margin: 1.25rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar (shared across all 4 pages)
# ---------------------------------------------------------------------------

def render_sidebar(all_status: Dict[str, Dict[str, Any]]):
    c = config.COLORS
    ready_count = sum(
        1 for k in ("agent1", "agent2", "agent3")
        if all_status.get(k, {}).get("status") == "ready"
    )
    pill_cls = "badge-ready" if ready_count == 3 else ("badge-unavail" if ready_count > 0 else "badge-error")

    with st.sidebar:
        st.markdown(
            f"<div style='padding-bottom:0.4rem'>"
            f"<div style='font-size:1.15rem;font-weight:800;letter-spacing:0.04em;color:{c['text']}'>"
            f"🛡️ {config.APP_TITLE}</div>"
            f"<div style='font-size:0.78rem;color:{c['text_muted']};margin-top:0.15rem'>"
            f"{config.APP_SUBTITLE}</div>"
            f"<div style='margin-top:0.55rem'><span class='badge {pill_cls}'>"
            f"{ready_count}/3 Agents Operational</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f"<div class='pg-card-title' style='margin-bottom:0.5rem'>Agent Readiness</div>",
            unsafe_allow_html=True,
        )

        labels = {
            "agent1": "Agent 1 · Header NB",
            "agent2": "Agent 2 · DistilBERT",
            "agent3": "Agent 3 · Fusion LR",
        }
        for key in ("agent1", "agent2", "agent3"):
            s = all_status.get(key, {}).get("status", "unavailable")
            dot_color = {
                "ready":       c["safe"],
                "unavailable": c["warning"],
                "pending":     c["pending"],
                "error":       c["not_safe"],
            }.get(s, c["pending"])
            st.markdown(
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"padding:0.28rem 0;font-size:0.84rem;color:{c['text']}'>"
                f"<span><span class='status-dot' style='background:{dot_color}'></span>{labels[key]}</span>"
                f"<span style='font-size:0.72rem;color:{c['text_muted']};text-transform:uppercase'>{s}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(f"v{config.APP_VERSION} · [GitHub]({config.PROJECT_REPO})")


# ---------------------------------------------------------------------------
# Basic layout helpers
# ---------------------------------------------------------------------------

def status_badge(status: str) -> str:
    mapping = {
        "ready":       ("badge-ready",   "✓ Ready"),
        "completed":   ("badge-ready",   "✓ Completed"),
        "error":       ("badge-error",   "✗ Failed"),
        "failed":      ("badge-error",   "✗ Failed"),
        "unavailable": ("badge-unavail", "⚠ Unavailable"),
        "pending":     ("badge-pending", "◌ Unavailable"),
    }
    cls, label = mapping.get(status, ("badge-pending", status.capitalize()))
    return f'<span class="badge {cls}">{label}</span>'


def warning_banner(message: str):
    st.markdown(
        f'<div class="pg-warning">⚠ {html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def info_banner(message: str):
    st.markdown(
        f'<div class="pg-info">ℹ {html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<hr class="pg-divider">', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<h2 style='margin-bottom:0.15rem;font-weight:800;letter-spacing:-0.01em'>{title}</h2>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<p style='color:{config.COLORS['text_muted']};margin-top:0;font-size:0.95rem'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    divider()


def kpi_tile(label: str, value: str, subtitle: str = "", accent: Optional[str] = None):
    c = config.COLORS
    val_color = accent or c["text"]
    st.markdown(
        f"""
        <div class="pg-kpi" style="border-top:2px solid {accent or c['border']}">
          <div class="pg-kpi-label">{html.escape(label)}</div>
          <div class="pg-kpi-val" style="color:{val_color}">{html.escape(value)}</div>
          <div class="pg-kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overview Agent Status Cards (consistent layout, agent-specific metadata)
# ---------------------------------------------------------------------------

def render_overview_agent_card(
    title: str,
    subtitle: str,
    accent: str,
    status_dict: Dict[str, Any],
    specs: List[tuple[str, str]],
):
    c = config.COLORS
    status = status_dict.get("status", "unavailable")
    error  = status_dict.get("error")
    badge  = status_badge(status)

    spec_html = "".join(
        f"<div class='metric-item'>"
        f"<div class='metric-label'>{html.escape(k)}</div>"
        f"<div class='metric-val'>{html.escape(v)}</div>"
        f"</div>"
        for k, v in specs
    )

    err_html = ""
    if error and status != "ready":
        err_html = (
            f"<div style='margin-top:0.55rem;font-size:0.8rem;color:{c['warning']}'>"
            f"{html.escape(error[:180])}</div>"
        )

    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {accent}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.5rem">
            <div>
              <div style="font-size:1rem;font-weight:700;color:{c['text']}">{html.escape(title)}</div>
              <div style="font-size:0.8rem;color:{c['text_muted']};margin-top:0.1rem">{html.escape(subtitle)}</div>
            </div>
            {badge}
          </div>
          <div class="metric-row">{spec_html}</div>
          {err_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline flow diagram (replaces ASCII terminal box)
# ---------------------------------------------------------------------------

def render_pipeline_flow(all_status: Dict[str, Dict[str, Any]]):
    c = config.COLORS
    nodes = [
        ("1. Email Input", "Headers + Subject + Body", c["secondary"], "ready"),
        ("2. Agent 1 · Headers", "Naive Bayes (T=1.572)", c["agent1"], all_status["agent1"]["status"]),
        ("2. Agent 2 · Content", "DistilBERT (256 tok)", c["agent2"], all_status["agent2"]["status"]),
        ("3. Agent 3 · Fusion", "LogReg (thr = 0.50)", c["agent3"], all_status["agent3"]["status"]),
    ]
    parts = []
    for i, (title, sub, color, st_val) in enumerate(nodes):
        badge = status_badge(st_val) if i > 0 else "<span class='badge badge-ready'>Input</span>"
        parts.append(
            f"<div class='pg-flow-node' style='border-top:3px solid {color}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:0.4rem'>"
            f"<strong style='font-size:0.88rem'>{html.escape(title)}</strong>{badge}</div>"
            f"<div style='font-size:0.78rem;color:{c['text_muted']};margin-top:0.35rem'>{html.escape(sub)}</div>"
            f"</div>"
        )
        if i < len(nodes) - 1:
            parts.append("<div class='pg-flow-arrow'>→</div>")

    st.markdown(f"<div class='pg-flow'>{''.join(parts)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Analyze Email result components
# ---------------------------------------------------------------------------

def render_verdict_banner(verdict: Dict[str, Any]):
    c = config.COLORS
    if verdict["verdict_available"]:
        is_not_safe = verdict["verdict_class"] == 1
        accent = c["not_safe"] if is_not_safe else c["safe"]
        bg = c["not_safe_bg"] if is_not_safe else c["safe_bg"]
        icon = "🛡️⚠" if is_not_safe else "🛡️✓"
        label = verdict["verdict_label"]
        prob = verdict.get("final_score")
        prob_fmt = format_probability(prob)
        prob_full = format_probability_full(prob)
        prob_html = (
            f"<div style='margin-top:0.75rem;font-size:0.95rem;color:{c['text']}'>"
            f"<strong>Final Probability P(Not Safe):</strong> "
            f"<span title='Exact value: {html.escape(prob_full)}' "
            f"style='color:{accent};font-weight:800;font-size:1.15rem;font-variant-numeric:tabular-nums'>"
            f"{html.escape(prob_fmt)}</span>"
            f" <span style='font-size:0.78rem;color:{c['text_muted']}'>(exact: {html.escape(prob_full)})</span>"
            f"</div>"
            if prob is not None
            else ""
        )
    else:
        accent = c["warning"]
        bg = c["warning_bg"]
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
                    border-radius:10px;padding:1.35rem 1.55rem;margin-bottom:1.15rem">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;
                      text-transform:uppercase;color:{c['text_muted']};margin-bottom:0.3rem">
            Final Security Verdict
          </div>
          <div style="font-size:2rem;font-weight:800;color:{accent};line-height:1.15">
            {icon} &nbsp;{html.escape(label)}
          </div>
          <p style="margin:0.6rem 0 0 0;font-size:0.96rem;color:{c['text']}">
            {html.escape(verdict['explanation'])}
          </p>
          {prob_html}
          <div style="margin-top:0.6rem;font-size:0.78rem;color:{c['text_muted']}">
            Experimental academic detection system — may produce false positives or false negatives.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_indicators(data: Dict[str, Any]):
    c = config.COLORS
    st.markdown(
        f"<div style='font-size:1.05rem;font-weight:700;margin-bottom:0.45rem'>"
        f"Why This Decision? — Detection Indicators</div>",
        unsafe_allow_html=True,
    )

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
            f"<div style='padding:0.3rem 0;font-size:0.9rem;color:{c['text']}'>"
            f"{icon} &nbsp;<span style='color:{color}'>{html.escape(item['text'])}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    limitations = data.get("limitations", [])
    if limitations:
        st.markdown(
            f"<div style='margin-top:0.65rem;padding:0.75rem 1rem;background:{c['card_bg']};"
            f"border:1px solid {c['border']};border-radius:8px;font-size:0.82rem;color:{c['text_muted']}'>"
            + "<br>".join(f"• {html.escape(lim)}" for lim in limitations)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_agent_breakdown_item(item: Dict[str, Any]):
    c = config.COLORS
    status = item["status"]
    badge = status_badge(status)

    if status == "completed":
        prob = item["probability"]
        prob_str = item.get("probability_fmt") or format_probability(prob)
        prob_full = item.get("probability_full") or format_probability_full(prob)
        thr_val = item.get("threshold")
        thr_str = f"{float(thr_val):.4f}" if isinstance(thr_val, (int, float)) else "Unavailable"
        detail_html = (
            f"<div class='metric-row'>"
            f"<div class='metric-item'><div class='metric-label'>Result</div>"
            f"<div class='metric-val' style='font-size:0.92rem'>{html.escape(str(item['result_label']))}</div></div>"
            f"<div class='metric-item' title='Exact: {html.escape(prob_full)} | Threshold: {html.escape(thr_str)}'>"
            f"<div class='metric-label'>P(Not Safe)</div>"
            f"<div class='metric-val'>{html.escape(prob_str)}</div></div>"
            + (
                f"<div class='metric-item'><div class='metric-label'>Signal</div>"
                f"<div class='metric-val' style='font-size:0.9rem'>{html.escape(str(item['extra']))}</div></div>"
                if item.get("extra") else ""
            )
            + "</div>"
        )
    else:
        err = item.get("error") or "Unavailable"
        detail_html = (
            f"<div style='margin-top:0.45rem;font-size:0.85rem;color:{c['text_muted']}'>"
            f"{html.escape(err)}</div>"
        )

    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {item['color']}">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
            <strong style="font-size:0.98rem;color:{c['text']}">{html.escape(item['title'])}</strong>
            {badge}
          </div>
          <div style="font-size:0.82rem;color:{c['text_muted']};margin-top:0.25rem">
            {html.escape(item['role'])}
          </div>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_email_info(email_dict: Dict[str, str]):
    c = config.COLORS
    fields = [
        ("Sender",    email_dict.get("sender") or "—"),
        ("Recipient", email_dict.get("receiver") or "—"),
        ("Subject",   email_dict.get("subject") or "—"),
        ("Date",      email_dict.get("date") or "—"),
    ]
    rows = "".join(
        f"<tr><td style='color:{c['text_muted']};padding:0.28rem 1rem 0.28rem 0;width:110px'>"
        f"<strong>{k}</strong></td>"
        f"<td style='color:{c['text']};padding:0.28rem 0'>{html.escape(v)}</td></tr>"
        for k, v in fields
    )
    st.markdown(
        f"""
        <div class="pg-card">
          <table style="width:100%;border-collapse:collapse;font-size:0.88rem">
            {rows}
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Altair Visualizations
# ---------------------------------------------------------------------------

def _altair_theme(chart: alt.Chart) -> alt.Chart:
    c = config.COLORS
    return (
        chart.configure(background="transparent")
        .configure_axis(
            labelColor=c["text_muted"],
            titleColor=c["text"],
            gridColor=c["border"],
            domainColor=c["border"],
        )
        .configure_legend(labelColor=c["text"], titleColor=c["text"])
        .configure_title(color=c["text"], fontSize=14)
        .configure_view(strokeWidth=0)
    )


def render_score_comparison_chart(results: Dict[str, Any]):
    """Horizontal bar chart of real agent probabilities for the analyzed email."""
    c = config.COLORS
    rows = []
    if results.get("agent1", {}).get("available"):
        s1 = float(results["agent1"]["nb_score"])
        rows.append({
            "Agent": "Agent 1 (Headers)",
            "Score": s1,
            "ScoreLabel": format_probability(s1),
            "FullPrecision": format_probability_full(s1),
            "Color": c["agent1"],
        })
    if results.get("agent2", {}).get("available"):
        s2 = float(results["agent2"]["text_score"])
        rows.append({
            "Agent": "Agent 2 (Content)",
            "Score": s2,
            "ScoreLabel": format_probability(s2),
            "FullPrecision": format_probability_full(s2),
            "Color": c["agent2"],
        })
    if results.get("agent3", {}).get("available"):
        s3 = float(results["agent3"]["final_probability"])
        rows.append({
            "Agent": "Agent 3 (Fusion)",
            "Score": s3,
            "ScoreLabel": format_probability(s3),
            "FullPrecision": format_probability_full(s3),
            "Color": c["agent3"],
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=22)
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="P(Not Safe)"),
            y=alt.Y("Agent:N", sort=None, title=None),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Agent:N"),
                alt.Tooltip("ScoreLabel:N", title="Formatted"),
                alt.Tooltip("FullPrecision:N", title="Exact"),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", dx=6, color=c["text"], fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("Score:Q"),
            y=alt.Y("Agent:N", sort=None),
            text="ScoreLabel:N",
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [0.5]}))
        .mark_rule(color=c["warning"], strokeDash=[4, 4], strokeWidth=1.5)
        .encode(x="x:Q")
    )
    chart = _altair_theme((bars + labels + rule).properties(height=165, title="P(Not Safe) Across Agents (0.50 fusion line)"))
    st.altair_chart(chart, use_container_width=True)


def render_disagreement_banner(disagreement: Dict[str, Any]):
    """
    Render the Agent 1 vs Agent 2 disagreement banner ONLY when both agents
    and their thresholds are available and their unrounded scores disagree.
    """
    if not disagreement.get("comparable") or disagreement.get("suppress_banner") or not disagreement.get("disagree"):
        return

    c = config.COLORS
    a1_side = disagreement["a1_side"]
    a2_side = disagreement["a2_side"]
    a1_fmt = disagreement["a1_score_fmt"]
    a2_fmt = disagreement["a2_score_fmt"]
    a1_thr = disagreement["a1_threshold"]
    a2_thr = disagreement["a2_threshold"]
    a3_label = disagreement.get("a3_label")
    a3_fmt = disagreement.get("a3_prob_fmt")
    fusion_text = (disagreement.get("fusion") or {}).get("summary", "")

    a3_line = (
        f"Agent 3 (Decision Fusion) resolved this disagreement as <strong>{html.escape(a3_label)}</strong> "
        f"(P(Not Safe) = <strong>{html.escape(str(a3_fmt))}</strong>)."
        if a3_label and a3_fmt
        else "Agent 3 final decision is unavailable for this run."
    )

    ev_note = ""
    if disagreement.get("a1_insufficient_evidence"):
        ev_note = (
            f"<div style='margin-top:0.35rem;font-size:0.82rem;color:{c['warning']}'>"
            "Note: Agent 1 also flagged insufficient header evidence for this email."
            "</div>"
        )

    st.markdown(
        f"""
        <div class="pg-card" style="border-left:4px solid {c['warning']};background:{c['warning_bg']}">
          <div style="font-size:0.74rem;font-weight:700;letter-spacing:0.08em;
                      text-transform:uppercase;color:{c['warning']};margin-bottom:0.25rem">
            Signal Disagreement Detected (Agent 1 vs Agent 2)
          </div>
          <div style="font-size:0.92rem;color:{c['text']};line-height:1.5">
            <strong>Agent 1 (Headers):</strong> {html.escape(a1_side)}
            (P = {html.escape(a1_fmt)}, thr = {a1_thr:.4f}) &nbsp;|&nbsp;
            <strong>Agent 2 (Content):</strong> {html.escape(a2_side)}
            (P = {html.escape(a2_fmt)}, thr = {a2_thr:.4f})
          </div>
          <div style="margin-top:0.45rem;font-size:0.88rem;color:{c['text']}">
            {a3_line}
          </div>
          <div style="margin-top:0.4rem;font-size:0.83rem;color:{c['text_muted']}">
            {html.escape(fusion_text)}
          </div>
          {ev_note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_link_advisory(advisory: Dict[str, Any]):
    """
    Render the offline Link-Safety Advisory card (strictly display-only;
    does not alter Agent 1, Agent 2, or Agent 3 outputs).
    """
    c = config.COLORS
    lvl = advisory.get("advisory_level", "none")
    accent = c["warning"] if lvl == "warning" else (c["secondary"] if lvl == "info" else c["border"])

    st.markdown(
        f"<div class='pg-card' style='border-left:4px solid {accent}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem'>"
        f"<strong style='font-size:0.98rem;color:{c['text']}'>🔗 Offline Link-Safety Advisory (Non-Model Check)</strong>"
        f"<span class='badge badge-pending'>Advisory Only</span>"
        f"</div>"
        f"<div style='font-size:0.82rem;color:{c['text_muted']};margin-top:0.25rem'>"
        f"Links detected: <strong>{advisory.get('link_count', 0)}</strong>"
        + (
            f" · Sender registered domain (heuristic): <strong>{html.escape(advisory['sender_reg_domain'])}</strong>"
            if advisory.get("sender_reg_domain")
            else " · Sender domain: unavailable"
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )

    for item in advisory.get("findings", []):
        item_lvl = item.get("level", "info")
        icon, color = {
            "warning": ("🟡", c["warning"]),
            "safe":    ("🟢", c["safe"]),
            "info":    ("🔵", c["secondary"]),
        }.get(item_lvl, ("⚪", c["text_muted"]))
        st.markdown(
            f"<div style='padding:0.2rem 0;font-size:0.88rem;color:{c['text']}'>"
            f"{icon} &nbsp;<span style='color:{color}'>{html.escape(item['text'])}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    links = advisory.get("links", [])
    if links:
        with st.expander(f"Inspect extracted link hostnames ({len(links)})"):
            rows = []
            for lnk in links:
                if lnk.get("valid"):
                    vis = lnk.get("visible_text") or "—"
                    if len(vis) > 48:
                        vis = vis[:45] + "…"
                    rows.append({
                        "Source": "HTML <a>" if lnk.get("source") == "html_anchor" else "Plain Text",
                        "Scheme": lnk.get("scheme", ""),
                        "Hostname": lnk.get("hostname", ""),
                        "Port": str(lnk["port"]) if lnk.get("port") else "—",
                        "Registered Domain (Heuristic)": lnk.get("registered_domain") or ("IP Address" if lnk.get("is_ip") else "—"),
                        "Visible Text": vis,
                        "Text vs Href": "Mismatch ⚠" if lnk.get("visible_text_mismatch") else ("OK" if lnk.get("visible_claimed_domain") else "—"),
                        "Sender Match": lnk.get("domain_match", "—"),
                    })
                else:
                    rows.append({
                        "Source": "HTML <a>" if lnk.get("source") == "html_anchor" else "Plain Text",
                        "Scheme": "invalid",
                        "Hostname": "—",
                        "Port": "—",
                        "Registered Domain (Heuristic)": lnk.get("reason", "malformed"),
                        "Visible Text": lnk.get("visible_text") or "—",
                        "Text vs Href": "—",
                        "Sender Match": "unavailable",
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(advisory.get("heuristic_note", ""))


def render_shared_test_comparison_chart(test_metrics: Dict[str, Dict[str, Any]]):
    """
    Grouped bar chart on the shared 7,343 test emails (from agent3_meta.json).
    Never mixes validation and test splits.
    """
    c = config.COLORS
    mapping = [
        ("agent1_alone", "Agent 1 (Headers)"),
        ("agent2_alone", "Agent 2 (DistilBERT)"),
        ("agent3_fusion", "Agent 3 (Fusion)"),
    ]
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    rows = []
    for key, agent_label in mapping:
        m = test_metrics.get(key)
        if not m:
            continue
        for mk, ml in metric_keys:
            if mk in m and m[mk] is not None:
                rows.append({"Agent": agent_label, "Metric": ml, "Value": float(m[mk])})

    if not rows:
        return

    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Metric:N", sort=[ml for _, ml in metric_keys], title=None),
            xOffset="Agent:N",
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0.90, 1.0]), title="Score"),
            color=alt.Color(
                "Agent:N",
                scale=alt.Scale(
                    domain=["Agent 1 (Headers)", "Agent 2 (DistilBERT)", "Agent 3 (Fusion)"],
                    range=[c["agent1"], c["agent2"], c["safe"]],
                ),
            ),
            tooltip=["Agent:N", "Metric:N", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=270, title="Offline Test Evaluation — Same 7,343 Held-Out Emails")
    )
    st.altair_chart(_altair_theme(chart), use_container_width=True)


def render_confusion_matrix_chart(cm: List[List[int]], title: str):
    """Render a 2x2 confusion matrix heatmap from real TN/FP/FN/TP counts."""
    c = config.COLORS
    rows = [
        {"Actual": "Safe (0)", "Predicted": "Safe (0)", "Count": int(cm[0][0])},
        {"Actual": "Safe (0)", "Predicted": "Not Safe (1)", "Count": int(cm[0][1])},
        {"Actual": "Not Safe (1)", "Predicted": "Safe (0)", "Count": int(cm[1][0])},
        {"Actual": "Not Safe (1)", "Predicted": "Not Safe (1)", "Count": int(cm[1][1])},
    ]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=["Safe (0)", "Not Safe (1)"]),
        y=alt.Y("Actual:N", sort=["Safe (0)", "Not Safe (1)"]),
    )
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(range=[c["card_elevated"], c["primary"]]), legend=None),
        tooltip=["Actual:N", "Predicted:N", "Count:Q"],
    )
    text = base.mark_text(fontSize=14, fontWeight="bold", color=c["text"]).encode(
        text=alt.Text("Count:Q", format=",d"),
    )
    chart = _altair_theme((heat + text).properties(height=210, title=title))
    st.altair_chart(chart, use_container_width=True)


def render_fusion_weights_chart(coeffs: Dict[str, float]):
    c = config.COLORS
    rows = [
        {"Parameter": "text_logit (Agent 2)", "Weight": float(coeffs["text_logit"]), "Color": c["agent2"]},
        {"Parameter": "nb_logit (Agent 1)", "Weight": float(coeffs["nb_logit"]), "Color": c["agent1"]},
        {"Parameter": "intercept (bias)", "Weight": float(coeffs["intercept"]), "Color": c["pending"]},
    ]
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=24)
        .encode(
            x=alt.X("Weight:Q", title="Logistic Regression Coefficient"),
            y=alt.Y("Parameter:N", sort=None, title=None),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Parameter:N", alt.Tooltip("Weight:Q", format=".4f")],
        )
        .properties(height=180, title="Agent 3 Fusion Coefficients (C = 0.1)")
    )
    st.altair_chart(_altair_theme(chart), use_container_width=True)


def render_dataset_split_chart():
    c = config.COLORS
    df = pd.DataFrame([
        {"Split": "Train (70%)", "Rows": 34256, "Color": c["secondary"]},
        {"Split": "Validation (15%)", "Rows": 7340, "Color": c["primary"]},
        {"Split": "Test (15%)", "Rows": 7343, "Color": c["safe"]},
    ])
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=24)
        .encode(
            x=alt.X("Rows:Q", title="Emails (48,939 total)"),
            y=alt.Y("Split:N", sort=None, title=None),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Split:N", alt.Tooltip("Rows:Q", format=",d")],
        )
        .properties(height=170, title="Dataset Split Distribution (emails_clean.csv)")
    )
    st.altair_chart(_altair_theme(chart), use_container_width=True)
