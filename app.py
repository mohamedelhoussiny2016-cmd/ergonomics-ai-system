"""
app.py — Industrial Ergonomics Platform (UPGRADED)
TuMeke-style Streamlit Dashboard

FIXES APPLIED:
  1. Results caching — "View Results" NEVER re-runs analysis
  2. Video display constrained to responsive column (not full-width)
  3. Cross-platform PDF paths via config.get_output_path()
  4. body_risk crash fixed in calc_body_part_distribution
  5. REBA body_risk field now populated
  6. PDF passes body_stats + dynamic recommendations
  7. Improved industrial dark UI
  8. Alert toasts (not spam)
  9. Session summary card in results
  10. Angle box-plot from cached results
"""

import streamlit as st
st.set_page_config(layout="wide")
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile, os, io
from pathlib import Path
from datetime import datetime
from collections import Counter

from config import (
    SystemConfig,
    RULA_RISK,
    REBA_RISK,
    FINAL_RISK_COLORS,
    get_output_path,
)
from processor import VideoProcessor
from smoother import SessionStats
from report import ReportGenerator

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial Ergonomics Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #0b1622; }
[data-testid="stSidebar"]          { background: #0f1e2e; }

.kpi-card {
    background: linear-gradient(135deg,#1a2a3a,#0f1922);
    border: 1px solid #2980b9; border-radius:10px;
    padding:16px 10px; text-align:center; margin-bottom:8px;
}
.kpi-value  { font-size:2.2rem; font-weight:700; line-height:1.1; }
.kpi-label  { font-size:0.75rem; color:#8aa; margin-top:4px; }
.risk-low   { color:#27ae60; }
.risk-med   { color:#f39c12; }
.risk-high  { color:#e74c3c; }
.risk-vhigh { color:#c0392b; }

.step-card {
    background:#111e2d; border:2px solid #2980b9; border-radius:14px;
    padding:26px 20px; text-align:center; margin:6px;
}
.step-card.selected { border-color:#27ae60; background:#0d2010; }
.step-number { font-size:0.7rem; color:#7fb3d3; font-weight:600;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:6px; }
.step-title { font-size:1.15rem; font-weight:700; color:#e0e8f0; margin-bottom:6px; }
.step-desc  { font-size:0.8rem; color:#7a9bb5; line-height:1.5; }

.alert-high {
    background:linear-gradient(90deg,#c0392b,#e74c3c);
    color:#fff; padding:12px 20px; border-radius:8px;
    font-weight:700; font-size:1.05rem; text-align:center; margin-bottom:12px;
}
.alert-warn {
    background:linear-gradient(90deg,#e67e22,#f39c12);
    color:#fff; padding:10px 20px; border-radius:8px;
    font-weight:600; font-size:0.95rem; text-align:center; margin-bottom:10px;
}
.sec-hdr {
    border-left:4px solid #2980b9; padding-left:12px;
    margin:14px 0 6px; color:#c8dced; font-size:1.0rem; font-weight:700;
}
.upload-zone {
    border:2px dashed #2980b9; border-radius:12px; padding:28px;
    text-align:center; background:#0d1e2e; color:#7fb3d3; font-size:1rem;
    margin-bottom:12px;
}
.pill {
    background:#1a3a5c; border-radius:20px; padding:3px 12px;
    font-size:0.75rem; color:#7fb3d3; display:inline-block; margin:2px;
}
.summary-card {
    background:#111e2d; border:1px solid #2980b9; border-radius:10px;
    padding:16px; margin-bottom:12px;
}

/* Fix 3: hard cap video width + center + industrial border */
div[data-testid="stImage"] img {
    max-width: 90% !important;
    height: auto !important;
    display: block;
    margin-left: auto;
    margin-right: auto;
    border: 2px solid #2980b9;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session state ────────────────────────────────────────────────────────────


def _init():
    D = {
        "step": 1,
        "assessment": None,
        "tmp_video": None,
        "uploaded_name": None,
        "uploaded_size": 0,
        # ── Analysis results (CACHED — never re-run on tab switch) ──
        "stats": None,
        "all_results": [],  # list of RULAResult | REBAResult
        "final_risk": None,
        "alert_events": [],
        "hr_frames": [],
        "body_stats": {},  # calc_body_part_distribution output
        "video_info_str": "",
        "output_video": None,
        "report_bytes": None,
        "csv_data": None,
        "cfg": None,
        "_source_mode": "file",
        "_cam_idx": 0,
        "_analysis_done": False,  # gate: if True, step 3 skips re-run
    }
    for k, v in D.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()
SS = st.session_state

# ─── Helpers ──────────────────────────────────────────────────────────────────


def risk_css(r):
    return {
        "Low": "risk-low",
        "Low Risk": "risk-low",
        "Negligible": "risk-low",
        "Medium": "risk-med",
        "Medium Risk": "risk-med",
        "Investigate": "risk-med",
        "High": "risk-high",
        "High Risk": "risk-high",
        "Very High": "risk-vhigh",
        "Very High Risk": "risk-vhigh",
    }.get(r, "risk-med")


def score_to_risk_css(score, max_s):
    r = score / max_s
    if r < 0.43:
        return "risk-low"
    elif r < 0.72:
        return "risk-med"
    elif r < 0.86:
        return "risk-high"
    else:
        return "risk-vhigh"


def kpi(col, value, label, css):
    col.markdown(
        f"""
    <div class="kpi-card">
      <div class="kpi-value {css}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>""",
        unsafe_allow_html=True,
    )


# ─── Chart builders ───────────────────────────────────────────────────────────


def make_timeline(smoothed, max_score, assessment):
    N = min(len(smoothed), 120)
    step = max(1, len(smoothed) // N)
    ys = smoothed[::step]
    xs = [i * step for i in range(len(ys))]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="Smoothed Score",
            line=dict(color="#3498db", width=2),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.12)",
        )
    )
    fig.add_hrect(
        y0=0,
        y1=max_score * 0.43,
        fillcolor="rgba(39,174,96,0.07)",
        line_width=0,
        annotation_text="Low",
        annotation_position="right",
    )
    fig.add_hrect(
        y0=max_score * 0.43,
        y1=max_score * 0.72,
        fillcolor="rgba(243,156,18,0.07)",
        line_width=0,
        annotation_text="Med",
        annotation_position="right",
    )
    fig.add_hrect(
        y0=max_score * 0.72,
        y1=max_score * 0.86,
        fillcolor="rgba(231,76,60,0.07)",
        line_width=0,
        annotation_text="High",
        annotation_position="right",
    )
    fig.add_hrect(
        y0=max_score * 0.86,
        y1=max_score,
        fillcolor="rgba(192,57,43,0.10)",
        line_width=0,
        annotation_text="V.High",
        annotation_position="right",
    )
    fig.add_hline(
        y=6 if assessment == "RULA" else 10,
        line_dash="dash",
        line_color="#c0392b",
        annotation_text="Alert threshold",
        annotation_position="left",
    )
    fig.update_layout(
        paper_bgcolor="#0b1622",
        plot_bgcolor="#0b1622",
        font=dict(color="#c8d8e8"),
        xaxis=dict(title="Frame index", gridcolor="#162233"),
        yaxis=dict(
            title=f"{assessment} Score", range=[0, max_score + 0.5], gridcolor="#162233"
        ),
        margin=dict(l=40, r=80, t=20, b=40),
        height=300,
        legend=dict(bgcolor="#0b1622", bordercolor="#2980b9", borderwidth=1),
    )
    return fig


def make_dist_chart(stats, max_score, assessment):
    dist = stats.score_distribution(max_score)
    sx = list(range(1, max_score + 1))
    sy = [dist.get(s, 0) for s in sx]

    def sc(s):
        r = s / max_score
        if r < 0.43:
            return "#27ae60"
        elif r < 0.72:
            return "#f39c12"
        elif r < 0.86:
            return "#e74c3c"
        else:
            return "#c0392b"

    fig = go.Figure(
        go.Bar(
            x=[str(s) for s in sx],
            y=sy,
            marker_color=[sc(s) for s in sx],
            text=sy,
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor="#0b1622",
        plot_bgcolor="#0b1622",
        font=dict(color="#c8d8e8"),
        xaxis=dict(title=f"{assessment} Score", gridcolor="#162233"),
        yaxis=dict(title="Frames", gridcolor="#162233"),
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
        height=240,
    )
    return fig


def make_angle_box(results_cache):
    """Box plot of joint angles from cached results."""
    fig = go.Figure()
    for k, label, c in [
        ("neck", "Neck", "#3498db"),
        ("trunk", "Trunk", "#2ecc71"),
        ("upper_arm", "Upper Arm", "#e67e22"),
        ("lower_arm", "Lower Arm", "#9b59b6"),
        ("wrist", "Wrist", "#1abc9c"),
    ]:
        vals = [
            r.angles.get(k, 0)
            for r in results_cache
            if r and r.angles and k in r.angles
        ]
        if vals:
            fig.add_trace(go.Box(y=vals, name=label, marker_color=c, boxmean=True))
    fig.update_layout(
        paper_bgcolor="#0b1622",
        plot_bgcolor="#0b1622",
        font=dict(color="#c8d8e8"),
        yaxis=dict(title="Degrees", gridcolor="#162233"),
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
        height=240,
    )
    return fig


def make_body_part_chart(body_stats):
    """Grouped bar chart of body part risk distribution."""
    if not body_stats:
        return go.Figure()
    parts = list(body_stats.keys())
    labels = {
        "neck": "Neck",
        "trunk": "Trunk",
        "upper_arm": "Upper Arm",
        "lower_arm": "Lower Arm",
        "wrist": "Wrist",
        "legs": "Legs",
    }
    fig = go.Figure()
    for level, color in [
        ("Low", "#27ae60"),
        ("Medium", "#f39c12"),
        ("High", "#e74c3c"),
    ]:
        fig.add_trace(
            go.Bar(
                name=level,
                x=[labels.get(p, p) for p in parts],
                y=[body_stats[p].get(level, 0) for p in parts],
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0b1622",
        plot_bgcolor="#0b1622",
        font=dict(color="#c8d8e8"),
        xaxis=dict(gridcolor="#162233"),
        yaxis=dict(title="% of time", range=[0, 100], gridcolor="#162233"),
        legend=dict(bgcolor="#0b1622", bordercolor="#2980b9", borderwidth=1),
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
    )
    return fig


# ─── Data helpers ─────────────────────────────────────────────────────────────


def results_csv(stats):
    rows = []
    max_sc = 15 if stats.assessment_type == "REBA" else 7
    rmap = REBA_RISK if max_sc == 15 else RULA_RISK
    for i, s in enumerate(stats.smoothed_scores):
        label = rmap.get(s, ("",))[0]
        rows.append({"frame_index": i + 1, "smoothed_score": s, "risk_level": label})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def calc_body_part_distribution(results):
    """
    FIXED: Safely handles missing body_risk with hasattr check.
    Supports both RULA (neck/trunk/upper_arm/lower_arm/wrist)
    and REBA (adds legs).
    """
    all_parts = ["neck", "trunk", "upper_arm", "lower_arm", "wrist", "legs"]
    stats = {}

    for p in all_parts:
        # FIXED: guard for missing body_risk
        values = [
            r.body_risk[p]
            for r in results
            if hasattr(r, "body_risk") and r.body_risk and p in r.body_risk
        ]
        if not values:
            continue
        c = Counter(values)
        total = len(values)
        stats[p] = {
            "Low": round(c.get("Low", 0) / total * 100, 1),
            "Medium": round(c.get("Medium", 0) / total * 100, 1),
            "High": round(c.get("High", 0) / total * 100, 1),
        }
    return stats


# ─── PDF path helper ──────────────────────────────────────────────────────────


def new_pdf_path(assessment):
    """Cross-platform PDF output path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_output_path(f"{assessment}_report_{ts}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<div style="text-align:center;padding:6px 0 4px">
  <span style="color:#2980b9;font-size:1.9rem;font-weight:800">🏭 Industrial Ergonomics Platform</span><br>
  <span style="color:#7fb3d3;font-size:0.85rem">AI-powered RULA &amp; REBA · TuMeke-style stability · Factory-grade</span>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─── Step indicator ───────────────────────────────────────────────────────────


def step_indicator():
    steps = ["Select Method", "Upload Video", "Analysis", "Results"]
    cur = SS["step"]
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, steps), 1):
        done = i < cur
        active = i == cur
        icon = "✅" if done else ("🔵" if active else "⚪")
        color = "#27ae60" if done else ("#2980b9" if active else "#455a6a")
        col.markdown(
            f"""
        <div style="text-align:center">
          <div style="font-size:1.2rem">{icon}</div>
          <div style="font-size:0.75rem;color:{color};font-weight:{'700' if active else '400'}">{label}</div>
        </div>""",
            unsafe_allow_html=True,
        )


step_indicator()
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SELECT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

if SS["step"] == 1:
    st.markdown(
        '<div class="sec-hdr">Step 1 — Select Assessment Method</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Choose the ergonomic standard for your task:")
    st.markdown("")

    c1, c2, _ = st.columns([1, 1, 0.2])

    with c1:
        sel = SS["assessment"] == "RULA"
        st.markdown(
            f"""
        <div class="step-card {'selected' if sel else ''}">
          <div class="step-number">Assessment A</div>
          <div class="step-title">📐 RULA</div>
          <div style="font-size:1.4rem;margin:4px 0">Score: 1–7</div>
          <div class="step-desc">
            <b>Rapid Upper Limb Assessment</b><br><br>
            Best for: assembly lines, electronics,<br>
            repetitive upper-body tasks.<br><br>
            Analyzes: Neck · Trunk · Upper Arm · Lower Arm · Wrist
          </div>
        </div>""",
            unsafe_allow_html=True,
        )
        if st.button("✅ Select RULA", use_container_width=True, key="sel_rula"):
            SS["assessment"] = "RULA"
            SS["_analysis_done"] = False

    with c2:
        sel2 = SS["assessment"] == "REBA"
        st.markdown(
            f"""
        <div class="step-card {'selected' if sel2 else ''}">
          <div class="step-number">Assessment B</div>
          <div class="step-title">🧍 REBA</div>
          <div style="font-size:1.4rem;margin:4px 0">Score: 1–15</div>
          <div class="step-desc">
            <b>Rapid Entire Body Assessment</b><br><br>
            Best for: lifting, bending, logistics,<br>
            healthcare, heavy industry.<br><br>
            Analyzes: Neck · Trunk · Legs · Upper Arm · Lower Arm · Wrist
          </div>
        </div>""",
            unsafe_allow_html=True,
        )
        if st.button("✅ Select REBA", use_container_width=True, key="sel_reba"):
            SS["assessment"] = "REBA"
            SS["_analysis_done"] = False

    st.markdown("")
    if SS["assessment"]:
        st.success(f"✅ **{SS['assessment']}** selected.")
        if st.button("Next →", type="primary"):
            SS["step"] = 2
            st.rerun()
    else:
        st.info("Select an assessment method above.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — UPLOAD VIDEO
# ══════════════════════════════════════════════════════════════════════════════

elif SS["step"] == 2:
    st.markdown(
        f'<div class="sec-hdr">Step 2 — Upload Video  <span class="pill">{SS["assessment"]}</span></div>',
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Input source",
        ["📁 Video File", "🎥 Webcam"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if "📁" in source_mode:
        uploaded = st.file_uploader(
            "Upload factory floor video (MP4 · AVI · MOV · MKV, up to 500 MB)",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded:
            new = (
                uploaded.name != SS["uploaded_name"]
                or uploaded.size != SS["uploaded_size"]
            )
            if new:
                old = SS.get("tmp_video")
                if old and os.path.exists(old):
                    try:
                        os.unlink(old)
                    except:
                        pass
                suffix = Path(uploaded.name).suffix or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.close()
                SS["tmp_video"] = tmp.name
                SS["uploaded_name"] = uploaded.name
                SS["uploaded_size"] = uploaded.size
                SS["_analysis_done"] = False  # force re-analysis on new file

            ca, cb, cc = st.columns(3)
            ca.success(f"✅ {SS['uploaded_name']}")
            cb.info(f"📦 {SS['uploaded_size']/1024/1024:.1f} MB")
            cc.info("💾 Saved — ready for analysis")

        elif SS.get("tmp_video") and os.path.exists(SS["tmp_video"]):
            st.info(f"📹 **{SS['uploaded_name']}** loaded and ready.")
        else:
            st.markdown(
                '<div class="upload-zone">📹 Upload a factory floor video to begin</div>',
                unsafe_allow_html=True,
            )

        ready = bool(SS.get("tmp_video") and os.path.exists(SS["tmp_video"]))
        SS["_source_mode"] = "file"

    else:
        cam_idx = st.number_input("Camera index", 0, 4, 0)
        SS["_cam_idx"] = cam_idx
        SS["_source_mode"] = "webcam"
        SS["_analysis_done"] = False
        ready = True
        st.info(f"Webcam index {cam_idx} selected.")

    st.markdown("")
    cb, cn = st.columns([1, 5])
    if cb.button("← Back"):
        SS["step"] = 1
        st.rerun()
    if cn.button("▶ Start Analysis", type="primary", disabled=not ready):
        SS["step"] = 3
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ANALYSIS
# FIX #1: If _analysis_done is True, skip processing and go straight to results
# ══════════════════════════════════════════════════════════════════════════════

elif SS["step"] == 3:
    assessment = SS["assessment"]

    # ── CACHE GATE: if analysis already done, jump to results ─────────────────
    if SS.get("_analysis_done") and SS.get("stats") and SS["stats"].smoothed_scores:
        st.success(f"✅ Analysis already complete. Redirecting to Results…")
        SS["step"] = 4
        st.rerun()

    st.markdown(
        f'<div class="sec-hdr">Step 3 — Analysing with {assessment} '
        f'<span class="pill">Auto-calibrated</span></div>',
        unsafe_allow_html=True,
    )

    cfg = SystemConfig()
    processor = VideoProcessor(cfg, assessment=assessment)
    SS["cfg"] = cfg

    video_source = (
        SS.get("_cam_idx", 0) if SS.get("_source_mode") == "webcam" else SS["tmp_video"]
    )
    vid_str = (
        f"Webcam {video_source}"
        if SS.get("_source_mode") == "webcam"
        else SS["uploaded_name"]
    )

    # ── Layout: constrained video on right ────────────────────────────────────
    col_cfg, col_vid = st.columns([1, 1.8])

    with col_cfg:
        st.markdown("**Auto-calibrated config**")
        cfg_ph = st.empty()
        score_ph = st.empty()
        avg_ph = st.empty()
        pct_ph = st.empty()
        alert_ph = st.empty()
        # Task 4: alert box lives in left panel — never covers video
        alert_box = st.empty()

    with col_vid:
        # Fix 2: inner sub-columns to center video
        c_left, c_center, c_right = st.columns([1, 3, 1])
        with c_center:
            st.markdown("### 🎥 Live Analysis")
            frame_container = st.container()
            frame_ph = st.empty()
        prog_bar = st.progress(0)
        status_ph = st.empty()
        live_chart_ph = st.empty()

    stop_btn = st.button("⏹ Stop", type="secondary")

    # Task 5: alert spam prevention
    import time

    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0

    all_results = []
    frame_count = 0
    cfg_shown = False

    try:
        for bundle in processor.stream(
            video_source, save_output=True, high_risk_frame_limit=10
        ):
            frame_count += 1

            if not cfg_shown and frame_count == 1:
                c = processor.cfg
                cfg_ph.markdown(
                    f"""
| Setting | Value |
|---------|-------|
| Source FPS | {c.source_fps:.0f} |
| Frame skip | {c.frame_skip} (auto) |
| Eff. FPS | {c.effective_fps:.1f} |
| Resolution | {c.process_width}px |
| Warn window | {c.warning_seconds}s |
| Alert window | {c.high_risk_seconds}s |
"""
                )
                cfg_shown = True

            if bundle.result:
                all_results.append(bundle.result)

            # Fix 1: fixed VIDEO_WIDTH resize — always fits, no cropping
            rgb = cv2.cvtColor(bundle.annotated_frame, cv2.COLOR_BGR2RGB)
            if frame_count % 2 == 0:
                VIDEO_WIDTH = min(
                    700, int(st.session_state.get("screen_width", 1200) * 0.6)
                )
                h, w = rgb.shape[:2]
                scale = VIDEO_WIDTH / w
                new_h = int(h * scale)
                rgb_resized = cv2.resize(rgb, (VIDEO_WIDTH, new_h))
                with frame_container:
                    frame_ph.image(rgb_resized, channels="RGB")

            s = bundle.smoothed_score
            ms = 15 if assessment == "REBA" else 7
            css = score_to_risk_css(s, ms)
            score_ph.markdown(
                f"""
<div class="kpi-card">
  <div class="kpi-value {css}">{s}/{ms}</div>
  <div class="kpi-label">Current {assessment} Score</div>
</div>""",
                unsafe_allow_html=True,
            )
            avg_ph.metric("Avg", f"{bundle.stats.avg_score:.1f}")
            pct_ph.metric("High Risk %", f"{bundle.stats.pct_high_risk:.0f}%")

            al = bundle.alert_state
            al_color = {"OK": "#27ae60", "WARNING": "#e67e22", "HIGH_RISK": "#e74c3c"}[
                al
            ]
            al_label = {
                "OK": "✅ OK",
                "WARNING": "⚠️ Warning",
                "HIGH_RISK": "🚨 High Risk!",
            }[al]
            alert_ph.markdown(
                f"""
<div style="background:{al_color};color:#fff;padding:8px;border-radius:8px;
     text-align:center;font-weight:700;font-size:0.95rem">{al_label}</div>""",
                unsafe_allow_html=True,
            )

            prog_bar.progress(min(bundle.progress, 1.0))
            status_ph.text(
                f"Frame {bundle.frame_idx} | Score {bundle.smoothed_score} | {al}"
            )

            if len(all_results) % 12 == 0 and len(all_results) > 2:
                ss = bundle.stats.smoothed_scores
                live_chart_ph.plotly_chart(
                    make_timeline(ss, ms, assessment),
                    use_container_width=True,
                    key=f"lc_{frame_count}",
                )

            # Task 4+5: alerts in left panel only, with spam prevention (no toast)
            if len(processor.alert_events) > len(SS.get("alert_events", [])):
                latest = processor.alert_events[-1]
                if time.time() - st.session_state.last_alert_time > 2:
                    alert_box.warning(f"⚠ {latest.message}")
                    st.session_state.last_alert_time = time.time()
                    # Fix 4: spacing so alert doesn't merge with elements below

            if stop_btn:
                processor.stop()
                break

    except Exception as e:
        st.error(f"Analysis error: {e}")
        st.exception(e)

    # ── Cache all results (FIX #1) ────────────────────────────────────────────
    final_stats = processor.session_stats
    final_risk = final_stats.final_classification(cfg)
    alert_events = processor.alert_events
    body_stats = calc_body_part_distribution(all_results)  # FIX #4 applied inside

    SS["stats"] = final_stats
    SS["all_results"] = all_results
    SS["final_risk"] = final_risk
    SS["alert_events"] = alert_events
    SS["hr_frames"] = processor.high_risk_frames
    SS["body_stats"] = body_stats
    SS["video_info_str"] = vid_str
    SS["csv_data"] = results_csv(final_stats)
    SS["_analysis_done"] = True  # ← gate flag

    vid_out = cfg.output_video_path
    if os.path.exists(vid_out):
        SS["output_video"] = vid_out

    prog_bar.progress(1.0)
    status_ph.text(f"✅ Complete — {len(all_results)} frames | Final: {final_risk}")

    if final_stats.smoothed_scores:
        ms = 15 if assessment == "REBA" else 7
        live_chart_ph.plotly_chart(
            make_timeline(final_stats.smoothed_scores, ms, assessment),
            use_container_width=True,
        )

    st.success(f"✅ Analysis done! **Final Risk: {final_risk}** — view results below.")
    if st.button("📊 View Results →", type="primary"):
        SS["step"] = 4
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RESULTS DASHBOARD
# FIX #1: Reads purely from session_state — no re-analysis
# ══════════════════════════════════════════════════════════════════════════════

elif SS["step"] == 4:
    assessment = SS["assessment"]
    stats: SessionStats = SS.get("stats")
    final_risk = SS.get("final_risk", "Unknown")
    alert_events = SS.get("alert_events", [])
    body_stats = SS.get("body_stats", {})
    results_cache = SS.get("all_results", [])
    ms = 15 if assessment == "REBA" else 7

    if not stats or not stats.smoothed_scores:
        st.warning("No analysis data. Please run an analysis first.")
        if st.button("← Start Over"):
            SS["step"] = 1
            st.rerun()
        st.stop()

    st.markdown(
        f'<div class="sec-hdr">📊 Results Dashboard '
        f'<span class="pill">{assessment}</span> '
        f'<span class="pill">{SS["video_info_str"]}</span></div>',
        unsafe_allow_html=True,
    )

    risk_hex = FINAL_RISK_COLORS.get(final_risk, "#7f8c8d")
    if final_risk == "High Risk":
        st.markdown(
            '<div class="alert-high">🚨 HIGH RISK — Immediate ergonomic intervention required</div>',
            unsafe_allow_html=True,
        )
    elif final_risk == "Medium Risk":
        st.markdown(
            '<div class="alert-warn">⚠️ MEDIUM RISK — Workstation changes recommended</div>',
            unsafe_allow_html=True,
        )

    # Tabs
    tab_dash, tab_body, tab_export, tab_info = st.tabs(
        ["📊 Dashboard", "🧍 Body Parts", "📥 Export", "ℹ️ Session Info"]
    )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    with tab_dash:
        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi(
            k1,
            f"{stats.avg_score:.1f}",
            f"Avg (/{ms})",
            score_to_risk_css(int(round(stats.avg_score)), ms),
        )
        kpi(
            k2,
            str(stats.max_score),
            f"Peak (/{ms})",
            score_to_risk_css(stats.max_score, ms),
        )
        kpi(
            k3,
            f"{stats.pct_high_risk:.0f}%",
            "High Risk Time",
            "risk-high" if stats.pct_high_risk > 20 else "risk-med",
        )
        kpi(k4, f"{stats.pct_warning:.0f}%", "Warning Time", "risk-med")
        kpi(
            k5,
            str(len(alert_events)),
            "Alert Events",
            "risk-high" if alert_events else "risk-low",
        )
        st.markdown("")

        # Final classification card
        fr_css = risk_css(final_risk)
        st.markdown(
            f"""
<div class="kpi-card" style="border-color:{risk_hex}">
  <div class="kpi-value {fr_css}">{final_risk}</div>
  <div class="kpi-label">Final Session Classification (time-based TuMeke logic)</div>
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Session summary card
        st.markdown(
            f"""
<div class="summary-card">
  <b style="color:#7fb3d3">Session Summary</b><br>
  Frames analysed: <b>{len(stats.smoothed_scores)}</b> &nbsp;|&nbsp;
  Duration: <b>{stats.duration_sec:.1f}s</b> &nbsp;|&nbsp;
  Pose detected: <b>{stats.detected_frames}</b> frames &nbsp;|&nbsp;
  Avg score: <b>{stats.avg_score:.1f}/{ms}</b> &nbsp;|&nbsp;
  Peak: <b>{stats.max_score}/{ms}</b>
</div>""",
            unsafe_allow_html=True,
        )

        # Timeline + distribution
        st.markdown('<div class="sec-hdr">Score Timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_timeline(stats.smoothed_scores, ms, assessment),
            use_container_width=True,
        )

        cd, ca = st.columns(2)
        with cd:
            st.markdown(
                '<div class="sec-hdr">Score Distribution</div>', unsafe_allow_html=True
            )
            st.plotly_chart(
                make_dist_chart(stats, ms, assessment), use_container_width=True
            )
        with ca:
            st.markdown(
                '<div class="sec-hdr">Joint Angle Ranges</div>', unsafe_allow_html=True
            )
            if results_cache:
                st.plotly_chart(make_angle_box(results_cache), use_container_width=True)
            else:
                st.info("Angle data will appear after a full analysis run.")

        # Alert events
        if alert_events:
            st.markdown(
                '<div class="sec-hdr">🚨 Alert Events</div>', unsafe_allow_html=True
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time (s)": f"{e.timestamp_sec:.1f}",
                            "Frame": e.frame_idx,
                            "Score": e.score,
                            "State": e.state,
                            "Message": e.message,
                        }
                        for e in alert_events
                    ]
                ),
                use_container_width=True,
            )

    # ── Body Parts ────────────────────────────────────────────────────────────
    with tab_body:
        if not body_stats:
            st.info("Body part data not available. Run an analysis first.")
        else:
            st.markdown(
                '<div class="sec-hdr">🧍 Body Part Risk Breakdown</div>',
                unsafe_allow_html=True,
            )

            # Stacked bar chart
            st.plotly_chart(make_body_part_chart(body_stats), use_container_width=True)

            # Summary table
            PART_LABELS = {
                "neck": "Neck",
                "trunk": "Trunk",
                "upper_arm": "Upper Arm",
                "lower_arm": "Lower Arm",
                "wrist": "Wrist",
                "legs": "Legs",
            }
            rows = []
            for part, label in PART_LABELS.items():
                if part not in body_stats:
                    continue
                d = body_stats[part]
                dominant = max(
                    [("Low", d["Low"]), ("Medium", d["Medium"]), ("High", d["High"])],
                    key=lambda x: x[1],
                )[0]
                rows.append(
                    {
                        "Body Part": label,
                        "Low %": f"{d['Low']:.0f}%",
                        "Medium %": f"{d['Medium']:.0f}%",
                        "High %": f"{d['High']:.0f}%",
                        "Dominant": dominant,
                    }
                )
            if rows:
                df_body = pd.DataFrame(rows)
                st.dataframe(df_body, use_container_width=True, hide_index=True)

            # Per-part mini charts (3 columns)
            st.markdown(
                '<div class="sec-hdr">Per-Part Distribution</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for i, (part, label) in enumerate(PART_LABELS.items()):
                if part not in body_stats:
                    continue
                d = body_stats[part]
                fig = go.Figure(
                    go.Bar(
                        x=["Low", "Medium", "High"],
                        y=[d["Low"], d["Medium"], d["High"]],
                        marker_color=["#27ae60", "#f39c12", "#e74c3c"],
                        text=[
                            f"{d['Low']:.0f}%",
                            f"{d['Medium']:.0f}%",
                            f"{d['High']:.0f}%",
                        ],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title=label,
                    paper_bgcolor="#0b1622",
                    plot_bgcolor="#0b1622",
                    font=dict(color="#c8d8e8"),
                    height=220,
                    margin=dict(l=10, r=10, t=35, b=10),
                    yaxis=dict(range=[0, 100]),
                )
                cols[i % 3].plotly_chart(fig, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("### 📦 Export Options")
        e1, e2, e3 = st.columns(3)

        with e1:
            st.markdown("#### 📊 CSV")
            csv = SS.get("csv_data") or results_csv(stats)
            st.download_button(
                "⬇ Download CSV",
                data=csv,
                file_name=f"{assessment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with e2:
            st.markdown("#### 📄 PDF Report")
            lang = st.selectbox("Language", ["English", "Arabic"], key="pdf_lang")
            if st.button("📝 Generate PDF", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    try:
                        # FIX #3: cross-platform path
                        pdf_path = new_pdf_path(assessment)
                        gen = ReportGenerator(
                            language="en" if lang == "English" else "ar"
                        )
                        gen.generate(
                            stats=stats,
                            high_risk_frames=SS.get("hr_frames", []),
                            alert_events=SS.get("alert_events", []),
                            final_risk=final_risk,
                            assessment_type=assessment,
                            video_info_str=SS["video_info_str"],
                            output_path=pdf_path,
                            max_score=ms,
                            body_stats=body_stats,  # FIX #6
                            results_cache=results_cache,
                        )
                        with open(pdf_path, "rb") as f:
                            SS["report_bytes"] = f.read()
                        st.success(f"✅ PDF saved to: {pdf_path}")
                    except Exception as ex:
                        st.error(f"PDF error: {ex}")
                        st.exception(ex)

            if SS.get("report_bytes"):
                st.download_button(
                    "⬇ Download PDF",
                    data=SS["report_bytes"],
                    file_name=f"{assessment}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        with e3:
            st.markdown("#### 🎥 Annotated Video")
            vid = SS.get("output_video")
            if vid and os.path.exists(vid):
                with open(vid, "rb") as f:
                    st.download_button(
                        "⬇ Download Video",
                        data=f.read(),
                        file_name=f"{assessment}_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            else:
                st.info("Video appears here after analysis.")

    # ── Session Info ──────────────────────────────────────────────────────────
    with tab_info:
        cfg = SS.get("cfg", SystemConfig())
        st.markdown("### ⚙️ Auto-Calibration")
        st.json(
            {
                "assessment": assessment,
                "source_fps": cfg.source_fps,
                "frame_skip": cfg.frame_skip,
                "effective_fps": round(cfg.effective_fps, 1),
                "process_width": cfg.process_width,
                "smoothing_window": cfg.smoothing_window,
                "spike_threshold": cfg.spike_threshold,
                "warning_seconds": cfg.warning_seconds,
                "high_risk_seconds": cfg.high_risk_seconds,
            }
        )
        st.markdown("### 📊 Session Stats")
        st.json(
            {
                "total_frames": stats.total_frames,
                "detected_frames": stats.detected_frames,
                "analyzed_frames": len(stats.smoothed_scores),
                "avg_score": round(stats.avg_score, 2),
                "max_score": stats.max_score,
                "pct_high_risk": round(stats.pct_high_risk, 1),
                "pct_warning": round(stats.pct_warning, 1),
                "duration_sec": round(stats.duration_sec, 1),
                "final_risk": final_risk,
                "alert_events": len(alert_events),
            }
        )

    st.markdown("---")
    if st.button("🔄 New Analysis", type="primary"):
        for k in [
            "stats",
            "all_results",
            "final_risk",
            "alert_events",
            "hr_frames",
            "body_stats",
            "output_video",
            "report_bytes",
            "csv_data",
            "_analysis_done",
        ]:
            SS[k] = (
                []
                if k in ("all_results", "alert_events", "hr_frames")
                else ({} if k == "body_stats" else None)
            )
        SS["step"] = 1
        st.rerun()
