"""
report.py — PDF report generator (TuMeke-style, redesigned)
Industrial Ergonomics Platform
FIXED:
  - Cross-platform paths (no /tmp)
  - Body part table section
  - Dynamic recommendations from worst body parts
  - Risk distribution section
  - Improved visual design
"""

import io
import os
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np
import cv2

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.lineplots import LinePlot

from config import RECOMMENDATIONS, FINAL_RISK_COLORS, BODY_PART_RECOMMENDATIONS, get_output_path
from smoother import SessionStats, AlertEvent

# ─── Colors ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0f1923")
BLUE   = colors.HexColor("#2980b9")
BLDARK = colors.HexColor("#1a3a5c")
GREEN  = colors.HexColor("#27ae60")
ORANGE = colors.HexColor("#e67e22")
RED    = colors.HexColor("#e74c3c")
DKRED  = colors.HexColor("#c0392b")
LIGHT  = colors.HexColor("#f0f4f8")
MGRAY  = colors.HexColor("#bdc3c7")
DARK   = colors.HexColor("#2c3e50")
WHITE  = colors.white
LGREEN = colors.Color(0.15, 0.68, 0.38, alpha=0.18)
LORANGE= colors.Color(0.95, 0.61, 0.07, alpha=0.18)
LRED   = colors.Color(0.91, 0.30, 0.24, alpha=0.18)

# ─── Styles ───────────────────────────────────────────────────────────────────

def _style(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE   = _style("T",  fontSize=22, textColor=BLUE,   alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
H1      = _style("H1", fontSize=13, textColor=BLDARK, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5)
H2      = _style("H2", fontSize=10, textColor=BLDARK, fontName="Helvetica-Bold", spaceBefore=6,  spaceAfter=3)
BODY    = _style("BD", fontSize=9.5, textColor=DARK, leading=15, spaceAfter=4)
BODY_AR = _style("AR", fontSize=10,  textColor=DARK, leading=15, spaceAfter=4, alignment=TA_RIGHT)
CAPTION = _style("CA", fontSize=8, textColor=colors.gray, alignment=TA_CENTER)

def _risk_color_rl(risk: str) -> colors.Color:
    return {"Low": GREEN, "Medium": ORANGE, "High": RED}.get(risk, ORANGE)

def _score_color(score: int, max_score: int) -> colors.Color:
    r = score / max_score
    if r < 0.43:   return GREEN
    elif r < 0.72: return ORANGE
    elif r < 0.86: return RED
    else:          return DKRED

def _tint(risk: str):
    return {"Low": LGREEN, "Medium": LORANGE, "High": LRED}.get(risk, LORANGE)

# ─── Frame → Image ────────────────────────────────────────────────────────────

def _frame_to_image(frame: np.ndarray, max_w: float = 8.5*cm) -> Optional[Image]:
    h, w   = frame.shape[:2]
    scale  = min(max_w / w, 1.0)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return Image(io.BytesIO(buf.tobytes()), width=w*scale, height=h*scale)

# ─── Timeline drawing ─────────────────────────────────────────────────────────

def _timeline_drawing(smoothed_scores: List[int], max_score: int) -> Drawing:
    if len(smoothed_scores) < 2:
        return Drawing(400, 120)
    N    = min(len(smoothed_scores), 100)
    step = max(1, len(smoothed_scores) // N)
    ys   = smoothed_scores[::step]
    xs   = list(range(len(ys)))

    wc, hc = 16*cm, 5.5*cm
    d  = Drawing(wc, hc)
    lp = LinePlot()
    lp.x      = 1.0*cm; lp.y      = 0.8*cm
    lp.width  = wc - 1.5*cm; lp.height = hc - 1.0*cm
    lp.data   = [list(zip(xs, ys))]
    lp.lines[0].strokeColor = colors.HexColor("#3498db")
    lp.lines[0].strokeWidth = 1.5
    lp.xValueAxis.valueMin  = 0
    lp.xValueAxis.valueMax  = max(len(ys)-1, 1)
    lp.yValueAxis.valueMin  = 0
    lp.yValueAxis.valueMax  = max_score
    lp.yValueAxis.valueStep = 1 if max_score <= 7 else 3
    d.add(lp)
    thr_y = lp.y + lp.height * (6 / max_score)
    d.add(Line(lp.x, thr_y, lp.x+lp.width, thr_y,
               strokeColor=DKRED, strokeDashArray=[4,3], strokeWidth=1))
    return d

# ─── Dynamic recommendations ─────────────────────────────────────────────────

def _build_dynamic_recommendations(body_stats: Dict, final_risk: str,
                                   assessment: str) -> List[str]:
    """Generate prioritized recommendations based on worst body parts."""
    recs = []

    # Priority order
    priority = ["trunk", "neck", "upper_arm", "wrist", "lower_arm", "legs"]

    # Find high-risk parts first, then medium
    for level in ("High", "Medium"):
        for part in priority:
            if part not in body_stats:
                continue
            pct = body_stats[part].get(level, 0)
            if pct > 20:  # at least 20% of time in this level
                rec = BODY_PART_RECOMMENDATIONS.get(part, {}).get(level)
                if rec and rec not in recs:
                    recs.append(rec)

    # Add general fallback from RECOMMENDATIONS
    general_rec = RECOMMENDATIONS.get(assessment, {}).get(final_risk, "")
    if general_rec and general_rec not in recs:
        recs.insert(0, general_rec)

    return recs or ["Consult an ergonomics specialist for a detailed workstation assessment."]

# ─── Main generator ───────────────────────────────────────────────────────────

class ReportGenerator:

    def __init__(self, language: str = "en"):
        self.lang = language if language in ("en", "ar") else "en"

    def generate(
        self,
        stats:            SessionStats,
        high_risk_frames: List[np.ndarray],
        alert_events:     List[AlertEvent],
        final_risk:       str,
        assessment_type:  str,
        video_info_str:   str,
        output_path:      str,
        max_score:        int = 7,
        body_stats:       Optional[Dict] = None,
        results_cache:    Optional[list] = None,
    ) -> str:
        # FIXED: ensure parent directory exists (cross-platform)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=18*mm,  bottomMargin=18*mm,
        )
        story = self._build(stats, high_risk_frames, alert_events,
                            final_risk, assessment_type, video_info_str,
                            max_score, body_stats or {}, results_cache or [])
        doc.build(story)
        return output_path

    # ── Build story ───────────────────────────────────────────────────────────

    def _build(self, stats, hrframes, alert_events, final_risk,
               assessment, video_info_str, max_score, body_stats, results_cache):
        s = []
        s += self._header(stats, assessment, video_info_str, final_risk, max_score)
        s += self._executive_summary(stats, final_risk, max_score)
        s += self._body_part_table(body_stats, assessment)
        s += self._risk_distribution_section(body_stats)
        s += self._timeline_section(stats, max_score)
        s += self._distribution(stats, max_score)
        if alert_events:
            s += self._alerts_section(alert_events)
        if hrframes:
            s += self._frame_gallery(hrframes)
        s += self._recommendations_section(body_stats, final_risk, assessment)
        s += self._footer()
        return s

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self, stats, assessment, video_info_str, final_risk, max_score):
        items = []

        # Color bar across top
        risk_hex = FINAL_RISK_COLORS.get(final_risk, "#7f8c8d")
        bar_color = colors.HexColor(risk_hex)
        bar = Table([["  "]], colWidths=[17.4*cm], rowHeights=[0.5*cm])
        bar.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1), bar_color)]))
        items.append(bar)
        items.append(Spacer(1, 3*mm))

        items.append(Paragraph("🏭  INDUSTRIAL ERGONOMICS PLATFORM", _style(
            "fac", fontSize=9, textColor=BLUE, alignment=TA_CENTER)))
        items.append(Spacer(1, 2*mm))
        items.append(Paragraph(f"{assessment} VIDEO ANALYSIS REPORT", TITLE))
        items.append(Paragraph(
            "Rapid Upper Limb Assessment" if assessment=="RULA" else "Rapid Entire Body Assessment",
            _style("st", fontSize=10, textColor=DARK, alignment=TA_CENTER, spaceAfter=8)))

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        meta = [
            ["Report Date:", ts,          "Video Source:",    video_info_str],
            ["Assessment:", assessment,   "Duration:",        f"{stats.duration_sec:.1f}s"],
            ["Frames Analysed:", str(len(stats.smoothed_scores)),
             "Pose Detected:",   f"{stats.detected_frames} frames"],
        ]
        mt = Table(meta, colWidths=[3*cm, 6.2*cm, 3*cm, 6.2*cm])
        mt.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0),(2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8.5),
            ("TEXTCOLOR",     (0,0),(-1,-1), DARK),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.5, MGRAY),
            ("PADDING",       (0,0),(-1,-1), 4),
        ]))
        items.append(mt)
        items.append(Spacer(1, 4*mm))
        items.append(HRFlowable(width="100%", thickness=2, color=BLUE))
        items.append(Spacer(1, 3*mm))
        return items

    # ── Executive summary ─────────────────────────────────────────────────────

    def _executive_summary(self, stats, final_risk, max_score):
        items = [Paragraph("📊 Executive Summary", H1)]
        avg = stats.avg_score; mx = stats.max_score
        rhex = FINAL_RISK_COLORS.get(final_risk, "#7f8c8d")

        kpi_data = [[
            Paragraph(f"{avg:.1f}", _style("av", fontSize=28, alignment=TA_CENTER,
                fontName="Helvetica-Bold", textColor=_score_color(int(round(avg)), max_score))),
            Paragraph(f"{mx}", _style("mx", fontSize=28, alignment=TA_CENTER,
                fontName="Helvetica-Bold", textColor=_score_color(mx, max_score))),
            Paragraph(f"{stats.pct_high_risk:.0f}%", _style("hr", fontSize=28,
                alignment=TA_CENTER, fontName="Helvetica-Bold",
                textColor=RED if stats.pct_high_risk > 20 else ORANGE)),
            Paragraph(final_risk, _style("fr", fontSize=14, alignment=TA_CENTER,
                fontName="Helvetica-Bold", textColor=colors.HexColor(rhex))),
        ],[
            Paragraph(f"Avg Score (/{max_score})", CAPTION),
            Paragraph(f"Peak Score (/{max_score})", CAPTION),
            Paragraph("High Risk %", CAPTION),
            Paragraph("Final Classification", CAPTION),
        ]]
        kpi_t = Table(kpi_data, colWidths=[4.5*cm]*4)
        kpi_t.setStyle(TableStyle([
            ("ALIGN",  (0,0),(-1,-1), "CENTER"),
            ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
            ("PADDING",(0,0),(-1,-1), 8),
            ("BOX",    (0,0),(2,-1),  1, BLUE),
            ("BOX",    (3,0),(3,-1),  1, colors.HexColor(rhex)),
            ("BACKGROUND",(0,0),(2,0), LIGHT),
        ]))
        items.append(kpi_t)
        items.append(Spacer(1, 3*mm))

        body = (
            f"Analysis of <b>{len(stats.smoothed_scores)}</b> processed frames "
            f"over <b>{stats.duration_sec:.1f}s</b>. "
            f"Average score: <b>{avg:.1f}/{max_score}</b>. "
            f"Peak: <b>{mx}/{max_score}</b>. "
            f"<b>{stats.pct_high_risk:.1f}%</b> of recorded time in high-risk postures."
        )
        items.append(Paragraph(body, BODY))
        if self.lang == "ar":
            items.append(Paragraph(
                f"تحليل <b>{len(stats.smoothed_scores)}</b> إطاراً على مدار "
                f"<b>{stats.duration_sec:.1f} ثانية</b>. متوسط الدرجة: <b>{avg:.1f}/{max_score}</b>.",
                BODY_AR))
        items.append(Spacer(1, 4*mm))
        return items

    # ── A. Body Part Table ────────────────────────────────────────────────────

    def _body_part_table(self, body_stats: Dict, assessment: str) -> list:
        if not body_stats:
            return []

        items = [Paragraph("🧍 Body Part Risk Assessment", H1)]

        PART_LABELS = {
            "neck":      "Neck",
            "trunk":     "Trunk",
            "upper_arm": "Upper Arm",
            "lower_arm": "Lower Arm",
            "wrist":     "Wrist",
            "legs":      "Legs",
        }

        rows = [["Body Part", "Low %", "Medium %", "High %", "Dominant Risk"]]

        for part, label in PART_LABELS.items():
            if part not in body_stats:
                continue
            d    = body_stats[part]
            lo   = d.get("Low",   0)
            med  = d.get("Medium",0)
            hi   = d.get("High",  0)
            # Dominant = highest percentage category
            dominant = max([("Low", lo), ("Medium", med), ("High", hi)], key=lambda x: x[1])[0]
            rows.append([label, f"{lo:.0f}%", f"{med:.0f}%", f"{hi:.0f}%", dominant])

        t = Table(rows, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 6.4*cm])
        ts = [
            ("BACKGROUND",    (0,0),(-1,0), BLDARK),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("GRID",          (0,0),(-1,-1), 0.4, MGRAY),
            ("PADDING",       (0,0),(-1,-1), 6),
            ("ALIGN",         (1,0),(-1,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
        ]
        # Color the Dominant Risk column cells
        for i, row in enumerate(rows[1:], start=1):
            dominant = row[-1]
            ts.append(("TEXTCOLOR", (4,i),(4,i), _risk_color_rl(dominant)))
            ts.append(("FONTNAME",  (4,i),(4,i), "Helvetica-Bold"))
        t.setStyle(TableStyle(ts))
        items.append(t)
        items.append(Spacer(1, 4*mm))
        return items

    # ── B. Risk Distribution ──────────────────────────────────────────────────

    def _risk_distribution_section(self, body_stats: Dict) -> list:
        if not body_stats:
            return []

        items = [Paragraph("📊 Risk Distribution by Body Part", H1)]

        # Overall distribution
        all_lows = [d["Low"]    for d in body_stats.values()]
        all_meds = [d["Medium"] for d in body_stats.values()]
        all_his  = [d["High"]   for d in body_stats.values()]

        avg_low = sum(all_lows) / len(all_lows) if all_lows else 0
        avg_med = sum(all_meds) / len(all_meds) if all_meds else 0
        avg_hi  = sum(all_his)  / len(all_his)  if all_his  else 0

        dist_data = [[
            Paragraph(f"{avg_low:.0f}%", _style("dl", fontSize=22, alignment=TA_CENTER,
                fontName="Helvetica-Bold", textColor=GREEN)),
            Paragraph(f"{avg_med:.0f}%", _style("dm", fontSize=22, alignment=TA_CENTER,
                fontName="Helvetica-Bold", textColor=ORANGE)),
            Paragraph(f"{avg_hi:.0f}%",  _style("dh", fontSize=22, alignment=TA_CENTER,
                fontName="Helvetica-Bold", textColor=RED)),
        ],[
            Paragraph("Low Risk (avg)", CAPTION),
            Paragraph("Medium Risk (avg)", CAPTION),
            Paragraph("High Risk (avg)", CAPTION),
        ]]
        dt = Table(dist_data, colWidths=[5.8*cm]*3)
        dt.setStyle(TableStyle([
            ("ALIGN",  (0,0),(-1,-1), "CENTER"),
            ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
            ("PADDING",(0,0),(-1,-1), 10),
            ("BOX",    (0,0),(0,-1), 1, GREEN),
            ("BOX",    (1,0),(1,-1), 1, ORANGE),
            ("BOX",    (2,0),(2,-1), 1, RED),
            ("BACKGROUND",(0,0),(0,0), colors.Color(0.15,0.68,0.38,alpha=0.08)),
            ("BACKGROUND",(1,0),(1,0), colors.Color(0.95,0.61,0.07,alpha=0.08)),
            ("BACKGROUND",(2,0),(2,0), colors.Color(0.91,0.30,0.24,alpha=0.08)),
        ]))
        items.append(dt)
        items.append(Spacer(1, 4*mm))
        return items

    # ── Timeline ──────────────────────────────────────────────────────────────

    def _timeline_section(self, stats, max_score):
        items = [Paragraph("📈 Score Timeline", H1)]
        if len(stats.smoothed_scores) > 1:
            items.append(_timeline_drawing(stats.smoothed_scores, max_score))
            items.append(Paragraph(
                "Blue line: smoothed score per frame  |  Dashed red: alert threshold", CAPTION))
        else:
            items.append(Paragraph("Insufficient frames for timeline chart.", BODY))
        items.append(Spacer(1, 4*mm))
        return items

    # ── Score distribution ────────────────────────────────────────────────────

    def _distribution(self, stats, max_score):
        items = [Paragraph("📊 Score Distribution", H1)]
        dist  = stats.score_distribution(max_score)
        total = max(len(stats.smoothed_scores), 1)
        rows  = [["Score", "Frames", "Percentage", "Risk Level"]]

        from config import RULA_RISK, REBA_RISK
        rmap = REBA_RISK if max_score == 15 else RULA_RISK
        tints = [LGREEN, LGREEN, LORANGE, LORANGE,
                 colors.Color(0.90,0.49,0.13,alpha=0.15), LRED, LRED]

        for score in range(1, max_score+1):
            cnt   = dist.get(score, 0)
            pct   = 100 * cnt / total
            label = rmap.get(score, ("",))[0]
            rows.append([str(score), str(cnt), f"{pct:.1f}%", label])

        t = Table(rows, colWidths=[2*cm, 3*cm, 4*cm, 8.4*cm])
        ts = [
            ("BACKGROUND",    (0,0),(-1,0), BLDARK),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8.5),
            ("GRID",          (0,0),(-1,-1), 0.4, MGRAY),
            ("PADDING",       (0,0),(-1,-1), 5),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ]
        for i in range(1, min(len(rows), len(tints)+1)):
            ts.append(("BACKGROUND", (0,i),(-1,i), tints[i-1]))
        t.setStyle(TableStyle(ts))
        items.append(t)
        items.append(Spacer(1, 4*mm))
        return items

    # ── Alert events ──────────────────────────────────────────────────────────

    def _alerts_section(self, events):
        items = [Paragraph(f"🚨 Alert Events ({len(events)})", H1)]
        rows  = [["Time (s)", "Frame", "Score", "State", "Message"]]
        for ev in events[:20]:
            rows.append([f"{ev.timestamp_sec:.1f}", str(ev.frame_idx),
                         str(ev.score), ev.state, ev.message[:50]])
        t = Table(rows, colWidths=[2*cm, 2*cm, 1.8*cm, 3*cm, 8.6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), RED),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.4, MGRAY),
            ("PADDING",       (0,0),(-1,-1), 4),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ]))
        items.append(t)
        items.append(Spacer(1, 4*mm))
        return items

    # ── Frame gallery ─────────────────────────────────────────────────────────

    def _frame_gallery(self, frames):
        items = [
            Paragraph("🖼 High-Risk Frame Samples", H1),
            Paragraph("Frames captured during sustained high-risk posture events.", BODY),
            Spacer(1, 3*mm),
        ]
        for i in range(0, min(len(frames), 6), 2):
            pair = frames[i:i+2]
            imgs = [_frame_to_image(f, max_w=8.5*cm) for f in pair]
            caps = [Paragraph(f"Sample {i+j+1}", CAPTION) for j in range(len(pair))]
            if len(imgs) == 1:
                imgs.append(""); caps.append("")
            row_t = Table([[imgs[0], imgs[1]], [caps[0], caps[1]]],
                          colWidths=[9.2*cm, 9.2*cm])
            row_t.setStyle(TableStyle([
                ("ALIGN", (0,0),(-1,-1),"CENTER"),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("PADDING",(0,0),(-1,-1),4),
            ]))
            items.append(row_t)
            items.append(Spacer(1, 3*mm))
        return items

    # ── E. Recommendations (dynamic) ─────────────────────────────────────────

    def _recommendations_section(self, body_stats: Dict, final_risk: str, assessment: str) -> list:
        items = [Paragraph("💡 Recommendations", H1)]

        recs = _build_dynamic_recommendations(body_stats, final_risk, assessment)

        items.append(Paragraph(
            f"<b>Priority Actions for {final_risk} ({assessment}):</b>", H2))
        for i, rec in enumerate(recs[:8], 1):
            items.append(Paragraph(f"{i}. {rec}", BODY))

        # Arabic translations if requested
        if self.lang == "ar":
            ar_recs = [
                "ضبط ارتفاع محطة العمل بشكل مناسب لكل عامل.",
                "تطبيق التناوب الوظيفي كل ساعتين للمهام عالية الخطورة.",
                "جدولة فترات راحة قصيرة كل 30 دقيقة.",
                "توفير حصائر مضادة للإجهاد للعمال الواقفين.",
                "إعادة تصميم الأدوات لتقليل الوضعيات غير المريحة.",
                "تدريب المشرفين على منهجية تقييم RULA/REBA.",
            ]
            items.append(Spacer(1, 3*mm))
            items.append(Paragraph("<b>توصيات عامة:</b>",
                _style("arh", fontSize=10, fontName="Helvetica-Bold",
                       textColor=BLDARK, alignment=TA_RIGHT)))
            for r in ar_recs:
                items.append(Paragraph(f"• {r}", BODY_AR))

        items.append(Spacer(1, 4*mm))
        return items

    # ── Footer ────────────────────────────────────────────────────────────────

    def _footer(self):
        return [
            HRFlowable(width="100%", thickness=1, color=MGRAY),
            Spacer(1, 3*mm),
            Paragraph(
                f"Industrial Ergonomics Platform  |  "
                f"RULA: McAtamney & Corlett (1993)  |  "
                f"REBA: Hignett & McAtamney (2000)  |  "
                f"Generated {datetime.now().strftime('%Y-%m-%d')}",
                _style("ft", fontSize=7.5, textColor=colors.gray, alignment=TA_CENTER)),
        ]
