"""
theme.py
========
Single source of truth for Review Shield's visual palette and Plotly template.

All Streamlit pages and detector modules must import colours from here —
never hard-code hex strings elsewhere.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = {
    # Primary brand
    "brand_primary": "#7C3AED",      # Violet-700
    "brand_secondary": "#A78BFA",    # Violet-400
    "brand_accent": "#06B6D4",       # Cyan-500

    # Semantic
    "danger": "#EF4444",             # Red-500
    "warning": "#F59E0B",            # Amber-500
    "success": "#10B981",            # Emerald-500
    "info": "#3B82F6",               # Blue-500

    # Neutrals (dark mode)
    "bg_dark": "#0F0F1A",
    "bg_card": "#1A1A2E",
    "bg_surface": "#16213E",
    "border": "#2D2D52",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
}

# Shorthand aliases
P = PALETTE

# ---------------------------------------------------------------------------
# Risk-level colour map  (score 0-1)
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "low":      P["success"],
    "medium":   P["warning"],
    "high":     P["danger"],
    "critical": "#B91C1C",  # Red-700
}


def score_to_risk(score: float) -> str:
    """Return a risk label given a 0-1 ensemble score."""
    if score < 0.35:
        return "low"
    if score < 0.60:
        return "medium"
    if score < 0.80:
        return "high"
    return "critical"


def score_to_color(score: float) -> str:
    """Return a hex colour string for a given 0-1 risk score."""
    return RISK_COLORS[score_to_risk(score)]


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
_LAYOUT = dict(
    paper_bgcolor=P["bg_dark"],
    plot_bgcolor=P["bg_card"],
    font=dict(family="Inter, system-ui, sans-serif", color=P["text_primary"], size=13),
    title_font=dict(family="Inter, system-ui, sans-serif", color=P["text_primary"], size=16),
    xaxis=dict(
        gridcolor=P["border"],
        zerolinecolor=P["border"],
        tickfont=dict(color=P["text_secondary"]),
    ),
    yaxis=dict(
        gridcolor=P["border"],
        zerolinecolor=P["border"],
        tickfont=dict(color=P["text_secondary"]),
    ),
    legend=dict(
        bgcolor=P["bg_surface"],
        bordercolor=P["border"],
        borderwidth=1,
        font=dict(color=P["text_secondary"]),
    ),
    colorway=[
        P["brand_primary"],
        P["brand_accent"],
        P["warning"],
        P["danger"],
        P["success"],
        P["brand_secondary"],
    ],
    hoverlabel=dict(
        bgcolor=P["bg_surface"],
        bordercolor=P["border"],
        font=dict(color=P["text_primary"]),
    ),
    margin=dict(l=50, r=30, t=50, b=40),
)

pio.templates["review_shield"] = go.layout.Template(layout=_LAYOUT)
pio.templates.default = "review_shield"

# ---------------------------------------------------------------------------
# CSS injected into Streamlit
# ---------------------------------------------------------------------------
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --brand:     {P['brand_primary']};
    --brand-lt:  {P['brand_secondary']};
    --accent:    {P['brand_accent']};
    --danger:    {P['danger']};
    --warning:   {P['warning']};
    --success:   {P['success']};
    --bg-dark:   {P['bg_dark']};
    --bg-card:   {P['bg_card']};
    --bg-surf:   {P['bg_surface']};
    --border:    {P['border']};
    --txt:       {P['text_primary']};
    --txt-sec:   {P['text_secondary']};
    --txt-muted: {P['text_muted']};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--txt) !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}}

/* Cards */
.rs-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .2s;
}}
.rs-card:hover {{ border-color: var(--brand-lt); }}

/* Risk badge */
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .03em;
    text-transform: uppercase;
}}
.badge-low      {{ background: #064E3B22; color: {P['success']}; border: 1px solid {P['success']}44; }}
.badge-medium   {{ background: #78350F22; color: {P['warning']}; border: 1px solid {P['warning']}44; }}
.badge-high     {{ background: #7F1D1D22; color: {P['danger']};  border: 1px solid {P['danger']}44; }}
.badge-critical {{ background: #450A0A33; color: #FCA5A5;        border: 1px solid #F8717144; }}

/* Metric tile */
.metric-tile {{
    background: linear-gradient(135deg, var(--bg-card), var(--bg-surf));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}}
.metric-value {{
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
}}
.metric-label {{
    font-size: .8rem;
    color: var(--txt-sec);
    margin-top: .25rem;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

/* Reply workbench */
.reply-box {{
    background: var(--bg-surf);
    border: 1px solid var(--border);
    border-left: 3px solid var(--brand);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    white-space: pre-wrap;
    font-size: .9rem;
    line-height: 1.6;
}}

/* Evidence pill */
.evidence-item {{
    display: inline-block;
    background: #7C3AED18;
    border: 1px solid #7C3AED44;
    border-radius: 6px;
    padding: 3px 9px;
    font-size: .78rem;
    color: var(--brand-lt);
    margin: 2px 3px;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-dark); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

/* Mobile */
@media (max-width: 640px) {{
    .metric-value {{ font-size: 1.8rem; }}
    .rs-card {{ padding: .9rem 1rem; }}
}}
</style>
"""
