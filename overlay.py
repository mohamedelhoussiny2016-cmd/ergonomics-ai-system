"""
overlay.py — Industrial HUD renderer for annotated video frames
Industrial Ergonomics Platform
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional
from smoother import AlertState


# ─── Color palette ────────────────────────────────────────────────────────────
C_SAFE     = (39,  174, 96)
C_WARN     = (230, 126, 34)
C_HIGH     = (231, 76,  60)
C_DARK_RED = (192, 57,  43)
C_WHITE    = (255, 255, 255)
C_BLACK    = (0,   0,   0)
C_DARK_BG  = (15,  25,  35)
C_BLUE     = (52,  152, 219)
C_YELLOW   = (50,  205, 255)


def score_color(score: int, max_score: int = 7) -> tuple:
    ratio = score / max_score
    if ratio < 0.43:   return C_SAFE
    elif ratio < 0.72: return C_WARN
    elif ratio < 0.86: return C_HIGH
    else:              return C_DARK_RED


def alert_color(state: AlertState) -> tuple:
    return {"OK": C_SAFE, "WARNING": C_WARN, "HIGH_RISK": C_HIGH}[state]


# ─── Main HUD renderer ────────────────────────────────────────────────────────

def draw_hud(
    frame:          np.ndarray,
    score:          int,
    max_score:      int,
    risk_level:     str,
    assessment:     str,
    alert_state:    AlertState,
    timestamp_sec:  float,
    frame_idx:      int,
    angles:         Dict[str, float],
    avg_score:      float = 0.0,
    pct_high:       float = 0.0,
) -> np.ndarray:
    """Draw full industrial HUD on a copy of frame. Returns new frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    col  = score_color(score, max_score)

    # ── Left info panel ────────────────────────────────────────────────────
    panel_w, panel_h = 300, 210
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), C_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    # Brand header
    cv2.putText(out, f"{assessment} ERGONOMICS ANALYZER",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BLUE, 1)

    # Score display
    cv2.putText(out, f"Score: {score}/{max_score}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2)

    # Score bar
    bar_x, bar_y, bar_h2 = 8, 56, 11
    bar_w = panel_w - 16
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h2), (45, 55, 65), -1)
    filled = int(bar_w * (score / max_score))
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h2), col, -1)

    # Risk level
    cv2.putText(out, f"Risk: {risk_level}",
                (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)

    # Session averages
    cv2.putText(out, f"Avg:{avg_score:.1f}  Hi%:{pct_high:.0f}%",
                (8, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1)

    # Time
    cv2.putText(out, f"t={timestamp_sec:.1f}s  frame={frame_idx}",
                (8, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 150, 170), 1)

    # Angles mini-readout
    neck  = angles.get("neck",      0)
    trunk = angles.get("trunk",     0)
    ua    = angles.get("upper_arm", 0)
    la    = angles.get("lower_arm", 0)
    cv2.putText(out,
        f"N:{neck:.0f} T:{trunk:.0f} UA:{ua:.0f} LA:{la:.0f}",
        (8, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140, 200, 255), 1)

    # Alert state pill
    al_col  = alert_color(alert_state)
    al_text = {"OK": "OK", "WARNING": "! WARNING", "HIGH_RISK": "!! HIGH RISK"}[alert_state]
    cv2.rectangle(out, (8, 162), (panel_w - 8, 184), al_col, -1)
    cv2.putText(out, al_text, (14, 178),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 2)

    # Top-right clock
    ts = time.strftime("%H:%M:%S")
    cv2.putText(out, ts, (w - 95, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 180, 200), 1)

    # ── HIGH RISK overlay ──────────────────────────────────────────────────
    if alert_state == "HIGH_RISK":
        # Pulsing tint
        pulse_alpha = 0.18 + 0.10 * abs(np.sin(time.time() * 2.5))
        tint = out.copy()
        cv2.rectangle(tint, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(tint, pulse_alpha, out, 1 - pulse_alpha, 0, out)

        # Red border
        thick = 7
        cv2.rectangle(out, (thick, thick), (w - thick, h - thick), C_HIGH, thick)

        # Bottom warning banner
        banner_h = 52
        bov = out.copy()
        cv2.rectangle(bov, (0, h - banner_h), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(bov, 0.88, out, 0.12, 0, out)
        cv2.putText(out, "HIGH RISK POSTURE DETECTED",
                    (w // 2 - 220, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_YELLOW, 3)
        cv2.putText(out, "ERGONOMIC INTERVENTION REQUIRED",
                    (w // 2 - 215, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, C_WHITE, 1)

    elif alert_state == "WARNING":
        # Subtle orange border
        cv2.rectangle(out, (4, 4), (w - 4, h - 4), C_WARN, 4)
        # Small warning tag top-right
        cv2.rectangle(out, (w - 170, 0), (w, 30), C_WARN, -1)
        cv2.putText(out, "⚠ WARNING", (w - 158, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 2)

    return out


def draw_no_detection(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """Frame shown when no pose is detected."""
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (260, 42), C_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.putText(out, "No pose detected", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 120, 200), 2)
    cv2.putText(out, f"frame {frame_idx}", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 120, 140), 1)
    return out
