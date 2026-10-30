"""
app.py
======
Review Shield – Streamlit entry-point.

Pages:
  1. 🏠 Dashboard      – Brand-wide risk overview, KPI tiles, charts
  2. 🔍 Review Explorer – Filter & drill into individual reviews
  3. ✍️  Reply Workbench – Draft / Approve / Edit / Reject replies
  4. ⚙️  Settings        – Thresholds, API key status, data upload
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

import theme  # noqa: E402 – must import after sys.path fix
from src.data_loader import LOCATIONS, PLATFORMS, load_reviews
from src.detectors.ensemble import WEIGHTS, run_ensemble
from src.reply_engine import draft_reply
from src.utils import seed_everything

# ---------------------------------------------------------------------------
# App-level config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Review Shield – Revio AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
seed_everything()

# Inject global CSS
st.markdown(theme.GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "ensemble_cache" not in st.session_state:
    st.session_state.ensemble_cache: dict[str, dict] = {}
if "replies" not in st.session_state:
    st.session_state.replies: dict[str, dict] = {}  # review_id -> {text, status}


# ---------------------------------------------------------------------------
# Helper: load / cache data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating synthetic dataset…")
def get_data(uploaded=None) -> pd.DataFrame:
    return load_reviews(uploaded)


@st.cache_data(show_spinner="Running detectors…")
def get_ensemble(loc: str, plat: str, _df_hash: int) -> dict:
    """Cached ensemble result; _df_hash forces re-run when data changes."""
    from src.detectors.ensemble import run_ensemble as _re
    df = st.session_state.df
    return _re(df, location=loc or None, platform=plat or None)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;padding:1rem 0 1.5rem">
            <div style="font-size:2.2rem;margin-bottom:.3rem">🛡️</div>
            <div style="font-size:1.2rem;font-weight:700;color:{theme.P['brand_secondary']}">
                Review Shield
            </div>
            <div style="font-size:.75rem;color:{theme.P['text_muted']};margin-top:.2rem">
                Powered by Revio AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Review Explorer", "✍️ Reply Workbench", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.divider()

    # Data source
    st.markdown("**Data Source**")
    uploaded = st.file_uploader(
        "Upload reviews CSV",
        type="csv",
        help="CSV must have: reviewer_id, location, platform, rating, text, timestamp, account_age_days",
    )

    if uploaded:
        st.session_state.df = load_reviews(uploaded)
        st.session_state.ensemble_cache = {}
        st.success(f"Loaded {len(st.session_state.df):,} reviews")
    elif st.session_state.df is None:
        st.session_state.df = get_data()
        st.caption("Using synthetic demo data")

    df: pd.DataFrame = st.session_state.df

    st.divider()

    # API key status
    api_key = os.environ.get("ANTHROPIC_API_KEY") or (
        st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    )
    if api_key:
        st.markdown(
            f'<span style="color:{theme.P["success"]}">● Claude API connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{theme.P["text_muted"]}">● Rule-based mode (no API key)</span>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------
def risk_badge(risk: str) -> str:
    return f'<span class="badge badge-{risk}">{risk}</span>'


def metric_tile(label: str, value: str, color: str = theme.P["brand_secondary"]) -> str:
    return f"""
    <div class="metric-tile">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def evidence_pills(evidence: dict) -> str:
    pills = []
    skip = {"detector_evidence", "cluster_sample_texts", "cluster_sample_review_ids"}
    for k, v in evidence.items():
        if k in skip:
            continue
        pills.append(f'<span class="evidence-item">{k}: {v}</span>')
    return " ".join(pills)


# ---------------------------------------------------------------------------
# PAGE 1 – Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    st.markdown(
        f"""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.25rem">
        🏠 Brand Risk Dashboard</h1>
        <p style="color:{theme.P['text_secondary']};margin-bottom:1.5rem">
        Real-time campaign-detection overview across all locations & platforms.</p>""",
        unsafe_allow_html=True,
    )

    # ---- Compute ensemble for each location ----------------------------
    location_scores: list[dict] = []
    progress = st.progress(0, text="Analysing locations…")
    for i, loc in enumerate(LOCATIONS):
        result = run_ensemble(df, location=loc)
        score = result["score"]
        risk = theme.score_to_risk(score)
        location_scores.append(
            {
                "location": loc,
                "score": score,
                "risk": risk,
                "color": theme.score_to_color(score),
                "verdict": result["evidence"]["verdict"],
                "triggered": result["evidence"]["triggered_detectors"],
                "detector_scores": result["evidence"]["detector_scores"],
            }
        )
        progress.progress((i + 1) / len(LOCATIONS), text=f"Analysed: {loc}")
    progress.empty()

    loc_df = pd.DataFrame(location_scores)
    high_risk = loc_df[loc_df["risk"].isin(["high", "critical"])]
    total_reviews = len(df)
    attack_reviews = int(df["is_attack"].sum()) if "is_attack" in df.columns else 0

    # ---- KPI Tiles -------------------------------------------------------
    cols = st.columns(4)
    kpis = [
        ("Total Reviews", f"{total_reviews:,}", theme.P["brand_secondary"]),
        ("Flagged Reviews", f"{attack_reviews:,}", theme.P["danger"]),
        (
            "High-Risk Locations",
            str(len(high_risk)),
            theme.P["warning"] if len(high_risk) > 0 else theme.P["success"],
        ),
        (
            "Overall Brand Risk",
            theme.pct(float(loc_df["score"].mean())),
            theme.score_to_color(float(loc_df["score"].mean())),
        ),
    ]
    for col, (label, val, color) in zip(cols, kpis):
        col.markdown(metric_tile(label, val, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Risk Bar Chart --------------------------------------------------
    col_chart, col_table = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown("#### Risk Score by Location")
        fig = go.Figure(
            go.Bar(
                x=loc_df["score"],
                y=loc_df["location"],
                orientation="h",
                marker=dict(
                    color=loc_df["color"],
                    line=dict(color=theme.P["border"], width=0.5),
                ),
                text=[f"{s:.0%}" for s in loc_df["score"]],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Risk Score: %{x:.1%}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            xaxis=dict(tickformat=".0%", range=[0, 1.05]),
            yaxis=dict(autorange="reversed"),
            height=320,
            margin=dict(l=160, r=60, t=20, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Location Alert Summary")
        for _, row in loc_df.sort_values("score", ascending=False).iterrows():
            badge = risk_badge(row["risk"])
            triggered_str = (
                ", ".join(row["triggered"]) if row["triggered"] else "none"
            )
            st.markdown(
                f"""
                <div class="rs-card" style="margin-bottom:.6rem;padding:.8rem 1rem">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:600;font-size:.9rem">{row['location']}</span>
                        {badge}
                    </div>
                    <div style="font-size:.78rem;color:{theme.P['text_secondary']};margin-top:.4rem">
                        {row['verdict']}
                    </div>
                    <div style="font-size:.72rem;color:{theme.P['text_muted']};margin-top:.3rem">
                        Triggered: {triggered_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Detector Radar Chart -------------------------------------------
    st.markdown("#### Detector Signal Heatmap (all locations)")
    detector_names = list(WEIGHTS.keys())
    heat_data = pd.DataFrame(
        [
            {
                "location": row["location"],
                **row["detector_scores"],
            }
            for _, row in loc_df.iterrows()
        ]
    ).set_index("location")

    fig2 = px.imshow(
        heat_data,
        color_continuous_scale=[
            [0, theme.P["success"]],
            [0.35, theme.P["warning"]],
            [0.6, theme.P["danger"]],
            [1, "#7F1D1D"],
        ],
        zmin=0,
        zmax=1,
        labels=dict(color="Risk"),
        aspect="auto",
        text_auto=".2f",
    )
    fig2.update_layout(height=260, margin=dict(l=160, r=20, t=20, b=60))
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Review Volume Timeline -----------------------------------------
    st.markdown("#### Review Volume Over Time")
    timeline = (
        df.copy()
        .assign(day=df["timestamp"].dt.tz_convert("UTC").dt.normalize())
        .groupby(["day", "is_attack" if "is_attack" in df.columns else "day"])
        .size()
        .reset_index(name="count")
    )

    daily = (
        df.assign(day=df["timestamp"].dt.tz_convert("UTC").dt.normalize())
        .groupby("day")["rating"]
        .agg(count="count", avg_rating="mean")
        .reset_index()
    )

    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=daily["day"],
            y=daily["count"],
            name="Review Count",
            marker_color=theme.P["brand_primary"] + "99",
            yaxis="y",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=daily["day"],
            y=daily["avg_rating"],
            name="Avg Rating",
            mode="lines+markers",
            line=dict(color=theme.P["brand_accent"], width=2),
            marker=dict(size=4),
            yaxis="y2",
        )
    )
    fig3.update_layout(
        height=280,
        yaxis=dict(title="# Reviews"),
        yaxis2=dict(
            title="Avg Rating",
            overlaying="y",
            side="right",
            range=[1, 5.2],
            tickformat=".1f",
        ),
        legend=dict(orientation="h", y=1.1),
        barmode="overlay",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 2 – Review Explorer
# ---------------------------------------------------------------------------
def page_explorer() -> None:
    st.markdown(
        f"""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.25rem">
        🔍 Review Explorer</h1>
        <p style="color:{theme.P['text_secondary']};margin-bottom:1.5rem">
        Drill into individual reviews, filter by risk level, and see why each was flagged.</p>""",
        unsafe_allow_html=True,
    )

    # ---- Filters -------------------------------------------------------
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sel_loc = st.selectbox("Location", ["All"] + LOCATIONS, key="exp_loc")
    with col_f2:
        sel_plat = st.selectbox("Platform", ["All"] + PLATFORMS, key="exp_plat")
    with col_f3:
        sel_risk = st.selectbox(
            "Risk Level", ["All", "critical", "high", "medium", "low"], key="exp_risk"
        )
    with col_f4:
        sel_type = st.selectbox(
            "Attack Type",
            ["All", "none", "review_bomb", "fake_positive"],
            key="exp_type",
        )

    filtered = df.copy()
    if sel_loc != "All":
        filtered = filtered[filtered["location"] == sel_loc]
    if sel_plat != "All":
        filtered = filtered[filtered["platform"] == sel_plat]
    if sel_type != "All" and "attack_type" in filtered.columns:
        filtered = filtered[filtered["attack_type"] == sel_type]

    # ---- Run ensemble on filtered slice --------------------------------
    loc_arg = sel_loc if sel_loc != "All" else None
    plat_arg = sel_plat if sel_plat != "All" else None

    with st.spinner("Running detectors on filtered slice…"):
        ensemble = run_ensemble(df, location=loc_arg, platform=plat_arg)

    score = ensemble["score"]
    risk = theme.score_to_risk(score)

    # Summary banner
    st.markdown(
        f"""
        <div class="rs-card" style="border-left:4px solid {theme.score_to_color(score)}">
            <div style="display:flex;align-items:center;gap:1rem">
                <div>
                    <div style="font-size:2rem;font-weight:800;color:{theme.score_to_color(score)}">
                        {score:.0%}
                    </div>
                    <div style="font-size:.75rem;color:{theme.P['text_secondary']}">
                        Ensemble Risk Score
                    </div>
                </div>
                <div style="flex:1">
                    <div style="font-weight:600">{ensemble['evidence']['verdict']}</div>
                    <div style="font-size:.8rem;color:{theme.P['text_secondary']};margin-top:.3rem">
                        {evidence_pills(ensemble['evidence'])}
                    </div>
                </div>
                {risk_badge(risk)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Per-detector breakdown ----------------------------------------
    st.markdown("#### Detector Breakdown")
    det_cols = st.columns(4)
    detector_icons = {
        "velocity": "⚡",
        "similarity": "🔗",
        "account": "👤",
        "rating_drop": "📉",
    }
    for col, (det_name, det_score) in zip(
        det_cols, ensemble["evidence"]["detector_scores"].items()
    ):
        det_risk = theme.score_to_risk(det_score)
        det_evidence = ensemble["evidence"]["detector_evidence"][det_name]
        with col:
            st.markdown(
                f"""
                <div class="rs-card" style="text-align:center">
                    <div style="font-size:1.6rem">{detector_icons.get(det_name,'🔎')}</div>
                    <div style="font-weight:600;font-size:.85rem;margin:.3rem 0">
                        {det_name.replace('_',' ').title()}
                    </div>
                    <div style="font-size:1.5rem;font-weight:700;
                        color:{theme.score_to_color(det_score)}">
                        {det_score:.0%}
                    </div>
                    {risk_badge(det_risk)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Evidence", expanded=False):
                for k, v in det_evidence.items():
                    if k not in {"cluster_sample_texts", "cluster_sample_review_ids"}:
                        st.markdown(f"**{k}:** `{v}`")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Review Table --------------------------------------------------
    st.markdown(f"#### Reviews ({len(filtered):,} shown)")

    display_cols = ["review_id", "location", "platform", "rating", "text", "timestamp"]
    if "is_attack" in filtered.columns:
        display_cols += ["is_attack", "attack_type"]

    show_df = filtered[display_cols].sort_values("timestamp", ascending=False).head(200)

    # Apply risk filter (approximate per-review risk by attack label for demo)
    if sel_risk != "All" and "is_attack" in show_df.columns:
        if sel_risk in ("high", "critical"):
            show_df = show_df[show_df["is_attack"] == True]  # noqa: E712
        elif sel_risk == "low":
            show_df = show_df[show_df["is_attack"] == False]  # noqa: E712

    st.dataframe(
        show_df.reset_index(drop=True),
        use_container_width=True,
        column_config={
            "rating": st.column_config.NumberColumn("⭐ Rating", format="%d ⭐", min_value=1, max_value=5),
            "is_attack": st.column_config.CheckboxColumn("Flagged"),
            "text": st.column_config.TextColumn("Review Text", width="large"),
            "timestamp": st.column_config.DatetimeColumn("Date", format="MMM D, YYYY HH:mm"),
        },
        hide_index=True,
        height=420,
    )


# ---------------------------------------------------------------------------
# PAGE 3 – Reply Workbench
# ---------------------------------------------------------------------------
def page_reply_workbench() -> None:
    st.markdown(
        f"""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.25rem">
        ✍️ Reply Workbench</h1>
        <p style="color:{theme.P['text_secondary']};margin-bottom:1.5rem">
        Draft, review, and action brand-safe replies. No reply is ever posted automatically.</p>""",
        unsafe_allow_html=True,
    )

    # Pick reviews to reply to
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_loc = st.selectbox("Filter Location", ["All"] + LOCATIONS, key="wb_loc")
    with col_sel2:
        show_flagged_only = st.toggle("Show flagged reviews only", value=True, key="wb_flag")

    filtered = df.copy()
    if sel_loc != "All":
        filtered = filtered[filtered["location"] == sel_loc]
    if show_flagged_only and "is_attack" in filtered.columns:
        filtered = filtered[filtered["is_attack"] == True]  # noqa: E712

    if filtered.empty:
        st.info("No reviews match the current filters.")
        return

    # Select a review
    filtered = filtered.sort_values("timestamp", ascending=False).reset_index(drop=True)
    review_options = {
        f"[{row.rating}⭐] {row.text[:80]}…  — {row.location}": idx
        for idx, row in filtered.head(30).iterrows()
    }
    selected_label = st.selectbox(
        "Select review to draft a reply for",
        options=list(review_options.keys()),
        key="wb_review_sel",
    )
    selected_idx = review_options[selected_label]
    review = filtered.iloc[selected_idx]

    # Show original review
    st.markdown("#### Selected Review")
    is_attack = bool(review.get("is_attack", False))
    attack_type = review.get("attack_type", "unknown")
    ensemble_score_for_review = 0.5 if is_attack else 0.1

    attack_badge = (
        f'<span class="badge badge-high">⚠ {attack_type.replace("_"," ").title()}</span>'
        if is_attack
        else f'<span class="badge badge-low">✓ Organic</span>'
    )

    st.markdown(
        f"""
        <div class="rs-card">
            <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:.7rem">
                <div>
                    <span style="font-size:1.1rem">{"⭐"*int(review.rating)}</span>
                    <span style="color:{theme.P['text_muted']};font-size:.8rem;margin-left:.5rem">
                        {review.location} · {review.platform} ·
                        {pd.Timestamp(review.timestamp).strftime('%b %d, %Y')}
                    </span>
                </div>
                {attack_badge}
            </div>
            <div style="font-size:.95rem;line-height:1.6">{review.text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Draft button
    review_id = str(review.get("review_id", selected_idx))
    cached = st.session_state.replies.get(review_id, {})

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        draft_btn = st.button("✨ Draft Reply", key=f"draft_{review_id}", type="primary")

    if draft_btn or cached.get("text"):
        if draft_btn or not cached.get("text"):
            with st.spinner("Drafting reply…"):
                result = draft_reply(
                    review_text=str(review.text),
                    location=str(review.location),
                    rating=int(review.rating),
                    is_flagged=is_attack,
                    ensemble_score=ensemble_score_for_review,
                )
            st.session_state.replies[review_id] = {
                "text": result["text"],
                "backend": result["backend"],
                "sentiment": result["sentiment"],
                "status": "draft",
            }
            cached = st.session_state.replies[review_id]

        st.markdown("#### Draft Reply")
        st.markdown(
            f'<div style="font-size:.78rem;color:{theme.P["text_muted"]};margin-bottom:.5rem">'
            f'Backend: {cached["backend"]}  ·  Detected sentiment: {cached["sentiment"]}'
            "</div>",
            unsafe_allow_html=True,
        )

        # Editable text area
        edited_text = st.text_area(
            "Edit reply (optional)",
            value=cached["text"],
            height=180,
            key=f"edit_{review_id}",
            label_visibility="collapsed",
        )

        # Action buttons
        col_a, col_b, col_c, _ = st.columns([1, 1, 1, 3])
        with col_a:
            if st.button("✅ Approve", key=f"approve_{review_id}", type="primary"):
                st.session_state.replies[review_id]["text"] = edited_text
                st.session_state.replies[review_id]["status"] = "approved"
                st.success("Reply approved (not yet posted — requires manual submission).")
        with col_b:
            if st.button("📝 Save Edit", key=f"save_{review_id}"):
                st.session_state.replies[review_id]["text"] = edited_text
                st.session_state.replies[review_id]["status"] = "edited"
                st.info("Edits saved.")
        with col_c:
            if st.button("❌ Reject", key=f"reject_{review_id}"):
                st.session_state.replies[review_id]["status"] = "rejected"
                st.warning("Reply rejected.")

        status = st.session_state.replies[review_id].get("status", "draft")
        status_colors = {
            "draft": theme.P["text_muted"],
            "edited": theme.P["info"],
            "approved": theme.P["success"],
            "rejected": theme.P["danger"],
        }
        st.markdown(
            f'<div style="margin-top:.5rem;font-size:.8rem;color:{status_colors.get(status, theme.P[\"text_muted\"])}">'
            f"Status: <strong>{status.upper()}</strong></div>",
            unsafe_allow_html=True,
        )

    # ---- Approved replies log ------------------------------------------
    approved = {
        rid: r
        for rid, r in st.session_state.replies.items()
        if r.get("status") == "approved"
    }
    if approved:
        st.markdown("---")
        st.markdown(f"#### Approved Reply Queue ({len(approved)} ready)")
        for rid, rep in approved.items():
            st.markdown(
                f"""
                <div class="rs-card">
                    <div style="font-size:.75rem;color:{theme.P['text_muted']};margin-bottom:.5rem">
                        Review ID: {rid}
                    </div>
                    <div class="reply-box">{rep['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE 4 – Settings
# ---------------------------------------------------------------------------
def page_settings() -> None:
    st.markdown(
        f"""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.25rem">
        ⚙️ Settings</h1>
        <p style="color:{theme.P['text_secondary']};margin-bottom:1.5rem">
        Configure detector thresholds, API keys, and data options.</p>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🔬 Detector Thresholds")
        st.markdown(
            f'<p style="color:{theme.P["text_muted"]};font-size:.85rem">'
            "Adjust sensitivity. Higher values = fewer (but more precise) alerts.</p>",
            unsafe_allow_html=True,
        )
        vel_mult = st.slider(
            "Velocity burst multiplier threshold",
            min_value=2.0, max_value=10.0, value=4.0, step=0.5,
            help="How many times the baseline rate before flagging a velocity burst.",
        )
        sim_thresh = st.slider(
            "Similarity cluster threshold",
            min_value=0.60, max_value=0.99, value=0.82, step=0.01,
            help="Cosine similarity above which two reviews are near-duplicates.",
        )
        acc_age = st.slider(
            "Account age threshold (days)",
            min_value=3, max_value=60, value=14, step=1,
            help="Accounts younger than this are considered suspicious.",
        )
        drop_thresh = st.slider(
            "Rating drop threshold (stars)",
            min_value=0.3, max_value=2.0, value=0.8, step=0.1,
            help="Minimum star-rating drop to trigger rating-drop detector.",
        )
        st.caption("Threshold changes take effect on next detector run (clear cache below).")

        if st.button("🗑️ Clear Detector Cache"):
            st.cache_data.clear()
            st.session_state.ensemble_cache = {}
            st.success("Cache cleared. Detectors will re-run on next page visit.")

    with col2:
        st.markdown("#### 🔑 API & Integrations")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            st.success("✅ ANTHROPIC_API_KEY detected via environment variable.")
            st.markdown(
                "Claude will be used for reply drafting. "
                "Key is read from the environment — it is never displayed or logged."
            )
        else:
            st.warning(
                "⚠️ No ANTHROPIC_API_KEY found. "
                "The reply engine will use the rule-based + VADER fallback. "
                "Set the key via environment variable or `.streamlit/secrets.toml`."
            )
            st.code(
                "# .streamlit/secrets.toml\nANTHROPIC_API_KEY = \"sk-ant-...\"\n\n"
                "# or shell\nexport ANTHROPIC_API_KEY=sk-ant-...",
                language="toml",
            )

        st.markdown("#### 📊 Dataset Info")
        st.markdown(
            f"""
            <div class="rs-card">
                <div><strong>Total reviews:</strong> {len(df):,}</div>
                <div><strong>Date range:</strong>
                    {df['timestamp'].min().strftime('%Y-%m-%d')} →
                    {df['timestamp'].max().strftime('%Y-%m-%d')}
                </div>
                <div><strong>Locations:</strong> {df['location'].nunique()}</div>
                <div><strong>Platforms:</strong> {df['platform'].nunique()}</div>
                {"<div><strong>Ground-truth attacks:</strong> "
                 + str(int(df['is_attack'].sum()))
                 + f" ({df['is_attack'].mean():.1%})</div>"
                 if 'is_attack' in df.columns else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📥 Export")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download dataset as CSV",
            data=csv,
            file_name="review_shield_data.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🔍 Review Explorer":
    page_explorer()
elif page == "✍️ Reply Workbench":
    page_reply_workbench()
elif page == "⚙️ Settings":
    page_settings()
