"""
config.py — Central configuration & auto-calibration engine
Industrial Ergonomics Platform (TuMeke-style)
FIXED: Cross-platform paths (Windows + Linux/Mac), output directory management
"""

from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
import math

AssessmentType = Literal["RULA", "REBA"]

# ─── Cross-platform output directory ─────────────────────────────────────────

def get_output_dir() -> Path:
    """Returns a writable output directory that works on Windows, Mac, and Linux."""
    base = Path.home() / "ErgoReports"
    base.mkdir(parents=True, exist_ok=True)
    return base

def get_output_path(filename: str) -> str:
    """Returns a full cross-platform path inside the ErgoReports directory."""
    return str(get_output_dir() / filename)

# ─── Auto-calibration ─────────────────────────────────────────────────────────

def auto_frame_skip(fps: float) -> int:
    if fps <= 15:  return 2
    elif fps <= 24: return 2
    elif fps <= 30: return 3
    elif fps <= 60: return 4
    else:           return 5

def auto_resolution(source_width: int) -> int:
    if source_width <= 480:  return source_width
    elif source_width <= 720: return 640
    else:                     return 720

def frames_for_seconds(fps: float, frame_skip: int, seconds: float) -> int:
    effective_fps = max(fps / frame_skip, 1.0)
    return max(1, int(math.ceil(effective_fps * seconds)))

# ─── System config dataclass ──────────────────────────────────────────────────

@dataclass
class SystemConfig:
    source_fps: float = 30.0
    source_width: int = 1280
    frame_skip: int = 3
    process_width: int = 640
    smoothing_window: int = 10
    spike_threshold: int = 2
    warning_seconds: float = 1.5
    high_risk_seconds: float = 2.0
    warning_score: int = 5
    high_risk_score: int = 6
    high_risk_pct_threshold: float = 30.0
    medium_risk_pct_threshold: float = 15.0
    model_complexity: int = 1
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.50
    output_video_path: str = ""
    report_path: str = ""

    def __post_init__(self):
        if not self.output_video_path:
            self.output_video_path = get_output_path("ergo_output.mp4")
        if not self.report_path:
            self.report_path = get_output_path("ergo_report.pdf")

    @property
    def effective_fps(self) -> float:
        return self.source_fps / self.frame_skip

    @property
    def warning_frames(self) -> int:
        return frames_for_seconds(self.source_fps, self.frame_skip, self.warning_seconds)

    @property
    def high_risk_frames(self) -> int:
        return frames_for_seconds(self.source_fps, self.frame_skip, self.high_risk_seconds)

    def calibrate(self, fps: float, width: int):
        self.source_fps    = max(fps, 1.0)
        self.source_width  = width
        self.frame_skip    = auto_frame_skip(fps)
        self.process_width = auto_resolution(width)

    def summary(self) -> dict:
        return {
            "source_fps":       self.source_fps,
            "frame_skip":       self.frame_skip,
            "effective_fps":    round(self.effective_fps, 1),
            "process_width":    self.process_width,
            "warning_frames":   self.warning_frames,
            "high_risk_frames": self.high_risk_frames,
        }

# ─── Risk level definitions ───────────────────────────────────────────────────

RULA_RISK = {
    1: ("Acceptable",       "#27ae60"),
    2: ("Acceptable",       "#27ae60"),
    3: ("Investigate",      "#f39c12"),
    4: ("Investigate",      "#f39c12"),
    5: ("Investigate Soon", "#e67e22"),
    6: ("High Risk",        "#e74c3c"),
    7: ("Very High Risk",   "#c0392b"),
}

REBA_RISK = {
    1:  ("Negligible", "#27ae60"),
    2:  ("Low",        "#2ecc71"),
    3:  ("Low",        "#2ecc71"),
    4:  ("Medium",     "#f39c12"),
    5:  ("Medium",     "#f39c12"),
    6:  ("Medium",     "#f39c12"),
    7:  ("High",       "#e74c3c"),
    8:  ("High",       "#e74c3c"),
    9:  ("High",       "#e74c3c"),
    10: ("Very High",  "#c0392b"),
    11: ("Very High",  "#c0392b"),
    12: ("Very High",  "#c0392b"),
    13: ("Very High",  "#c0392b"),
    14: ("Very High",  "#c0392b"),
    15: ("Very High",  "#c0392b"),
}

FINAL_RISK_COLORS = {
    "Low Risk":    "#27ae60",
    "Medium Risk": "#f39c12",
    "High Risk":   "#e74c3c",
}

# ─── Dynamic body-part recommendations ───────────────────────────────────────

BODY_PART_RECOMMENDATIONS = {
    "neck": {
        "High":   "Raise monitor/document to eye level to eliminate neck flexion.",
        "Medium": "Encourage neutral neck posture; check monitor distance and angle.",
    },
    "trunk": {
        "High":   "Raise workbench height or use adjustable sit-stand desk to eliminate bending.",
        "Medium": "Reduce forward lean; add lumbar support to seating.",
    },
    "upper_arm": {
        "High":   "Lower work surface or bring items closer to keep elbows at the side.",
        "Medium": "Adjust tool placement to keep elbows near the body.",
    },
    "lower_arm": {
        "High":   "Add arm rests or tool suspension to maintain 90° elbow angle.",
        "Medium": "Provide forearm support during sustained precision tasks.",
    },
    "wrist": {
        "High":   "Introduce bent-handle tools or reorient parts to keep wrist straight.",
        "Medium": "Reduce repetition frequency; add wrist rest where appropriate.",
    },
    "legs": {
        "High":   "Provide anti-fatigue matting, seating, or reduce standing duration.",
        "Medium": "Allow alternating sitting and standing throughout the shift.",
    },
}

RECOMMENDATIONS = {
    "RULA": {
        "Low Risk":    "Posture is generally acceptable. Schedule routine ergonomic reviews every 6 months.",
        "Medium Risk": "Workstation adjustments recommended. Review tool placement, seat height, and reach distances.",
        "High Risk":   "Immediate ergonomic intervention required. Redesign workstation, implement job rotation, and provide worker training.",
    },
    "REBA": {
        "Low Risk":    "Risk level is acceptable. Continue monitoring and apply good ergonomic practices.",
        "Medium Risk": "Further assessment needed. Consider changes to posture, task frequency, or load handling.",
        "High Risk":   "High risk of musculoskeletal disorder. Implement engineering controls immediately.",
    },
}
