"""
reba.py — REBA (Rapid Entire Body Assessment) scoring engine
Industrial Ergonomics Platform
FIXED: Added body_risk field + classify_reba_part function
Score range: 1–15
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from config import REBA_RISK


# ─── Body part risk classification (REBA) ────────────────────────────────────

def classify_reba_part(part: str, angle: float) -> str:
    """Classify individual body part risk for REBA (Low / Medium / High)."""
    if part == "neck":
        if angle <= 10:   return "Low"
        elif angle <= 20: return "Medium"
        else:             return "High"

    elif part == "trunk":
        if angle < 5:     return "Low"
        elif angle <= 20: return "Medium"
        else:             return "High"

    elif part == "upper_arm":
        if angle < 20:    return "Low"
        elif angle <= 45: return "Medium"
        else:             return "High"

    elif part == "lower_arm":
        flex = abs(180.0 - angle)
        if 60 <= flex <= 100: return "Low"
        elif 30 <= flex < 60: return "Medium"
        else:                 return "High"

    elif part == "wrist":
        dev = abs(angle - 90.0)
        if dev <= 15: return "Low"
        elif dev <= 30: return "Medium"
        else:          return "High"

    elif part == "legs":
        # knee_deg: 180=straight, lower=more bent
        flex = abs(180.0 - angle)
        if flex < 30:           return "Low"
        elif 30 <= flex <= 60:  return "Medium"
        else:                   return "High"

    return "Low"


# ─── Segment A scores (Trunk, Neck, Legs) ────────────────────────────────────

def score_neck_reba(deg: float) -> int:
    if deg <= 20: return 1
    else:         return 2

def score_trunk_reba(deg: float, lateral_deg: float = 0.0) -> int:
    if deg < 5:       base = 1
    elif deg <= 20:   base = 2
    elif deg <= 60:   base = 3
    else:             base = 4
    if lateral_deg > 10:
        base = min(base + 1, 5)
    return base

def score_legs_reba(knee_deg: float) -> int:
    base      = 1
    knee_flex = abs(180.0 - knee_deg)
    if 30 <= knee_flex <= 60:  return base + 1
    elif knee_flex > 60:       return base + 2
    return base


# ─── REBA Table A  [trunk-1][legs-1][neck-1] ─────────────────────────────────

TABLE_A_REBA = [
    [[1,2],[2,3],[3,3],[4,4]],
    [[2,3],[3,4],[4,5],[5,6]],
    [[3,4],[4,5],[5,6],[6,7]],
    [[4,5],[5,6],[6,7],[7,8]],
    [[6,7],[7,8],[8,9],[9,9]],
]

def lookup_table_a_reba(trunk: int, neck: int, legs: int) -> int:
    t = min(max(trunk-1, 0), 4)
    n = min(max(neck-1,  0), 1)
    l = min(max(legs-1,  0), 3)
    return TABLE_A_REBA[t][l][n]


# ─── Segment B scores (Upper arm, Lower arm, Wrist) ──────────────────────────

def score_upper_arm_reba(deg: float) -> int:
    if deg < 20:    return 1
    elif deg <= 45: return 2
    elif deg <= 90: return 3
    else:           return 4

def score_lower_arm_reba(deg: float) -> int:
    flex = abs(180.0 - deg)
    if 60 <= flex <= 100: return 1
    return 2

def score_wrist_reba(deg: float) -> int:
    dev = abs(deg - 90.0)
    if dev <= 15: return 1
    else:         return 2


# ─── REBA Table B  [upper_arm-1][lower_arm-1][wrist-1] ───────────────────────

TABLE_B_REBA = [
    [[1,2],[1,2]],
    [[1,3],[2,3]],
    [[3,4],[3,5]],
    [[4,5],[4,6]],
    [[6,7],[6,7]],
    [[7,8],[7,8]],
]

def lookup_table_b_reba(upper_arm: int, lower_arm: int, wrist: int) -> int:
    ua = min(max(upper_arm-1, 0), 5)
    la = min(max(lower_arm-1, 0), 1)
    w  = min(max(wrist-1,     0), 1)
    return TABLE_B_REBA[ua][la][w]


# ─── REBA Table C  [score_a-1][score_b-1] ────────────────────────────────────

TABLE_C_REBA = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7,  7,  7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7,  7,  8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8,  8,  8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9,  9,  9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10],
    [6, 6, 6, 7, 8, 8, 9, 9,10,10, 10, 11],
    [7, 7, 7, 8, 9, 9, 9,10,11,11, 11, 12],
    [8, 8, 8, 9,10,10,10,11,11,12, 12, 12],
    [9, 9, 9,10,10,11,11,12,12,13, 13, 13],
    [10,10,10,11,11,12,12,12,13,13, 14, 14],
    [11,11,11,12,12,13,13,13,14,14, 15, 15],
    [12,12,12,13,13,14,14,14,15,15, 15, 15],
]

def lookup_table_c_reba(score_a: int, score_b: int) -> int:
    a = min(max(score_a-1, 0), 11)
    b = min(max(score_b-1, 0), 11)
    return TABLE_C_REBA[a][b]


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class REBAResult:
    frame_idx:     int
    timestamp_sec: float
    angles:        Dict[str, float]

    s_neck:        int = 0
    s_trunk:       int = 0
    s_legs:        int = 0
    s_upper_arm:   int = 0
    s_lower_arm:   int = 0
    s_wrist:       int = 0

    score_a:       int = 0
    score_b:       int = 0

    raw_score:     int = 0
    final_score:   int = 0
    risk_level:    str = ""
    risk_color:    str = ""
    assessment:    str = "REBA"

    # FIXED: Added body_risk field matching RULAResult interface
    body_risk: Optional[Dict[str, str]] = None

    @property
    def max_score(self) -> int:
        return 15

    def to_dict(self) -> dict:
        return {
            "frame":            self.frame_idx,
            "time_sec":         round(self.timestamp_sec, 2),
            "neck_deg":         round(self.angles.get("neck", 0), 1),
            "trunk_deg":        round(self.angles.get("trunk", 0), 1),
            "upper_arm_deg":    round(self.angles.get("upper_arm", 0), 1),
            "lower_arm_deg":    round(self.angles.get("lower_arm", 0), 1),
            "wrist_deg":        round(self.angles.get("wrist", 0), 1),
            "knee_deg":         round(self.angles.get("knee", 0), 1),
            "score_a":          self.score_a,
            "score_b":          self.score_b,
            "reba_score":       self.final_score,
            "risk_level":       self.risk_level,
            # body_risk columns
            "neck_risk":        self.body_risk.get("neck", "")      if self.body_risk else "",
            "trunk_risk":       self.body_risk.get("trunk", "")     if self.body_risk else "",
            "upper_arm_risk":   self.body_risk.get("upper_arm", "") if self.body_risk else "",
            "lower_arm_risk":   self.body_risk.get("lower_arm", "") if self.body_risk else "",
            "wrist_risk":       self.body_risk.get("wrist", "")     if self.body_risk else "",
            "legs_risk":        self.body_risk.get("legs", "")      if self.body_risk else "",
        }


# ─── Scoring function ─────────────────────────────────────────────────────────

def compute_reba(frame_idx: int, timestamp_sec: float,
                 angles: Dict[str, float]) -> REBAResult:
    r = REBAResult(frame_idx=frame_idx, timestamp_sec=timestamp_sec, angles=angles)

    r.s_neck      = score_neck_reba(angles.get("neck", 0))
    r.s_trunk     = score_trunk_reba(angles.get("trunk", 0), angles.get("trunk_lateral", 0))
    r.s_legs      = score_legs_reba(angles.get("knee", 180))
    r.s_upper_arm = score_upper_arm_reba(angles.get("upper_arm", 0))
    r.s_lower_arm = score_lower_arm_reba(angles.get("lower_arm", 180))
    r.s_wrist     = score_wrist_reba(angles.get("wrist", 90))

    r.score_a = lookup_table_a_reba(r.s_trunk, r.s_neck, r.s_legs)
    r.score_b = lookup_table_b_reba(r.s_upper_arm, r.s_lower_arm, r.s_wrist)

    raw           = lookup_table_c_reba(r.score_a, r.score_b)
    r.raw_score   = min(max(raw, 1), 15)
    r.final_score = r.raw_score

    label, color = REBA_RISK.get(r.final_score, ("Very High", "#c0392b"))
    r.risk_level = label
    r.risk_color = color

    # FIXED: Populate body_risk (mirrors RULAResult interface)
    r.body_risk = {
        "neck":      classify_reba_part("neck",      angles.get("neck",      0)),
        "trunk":     classify_reba_part("trunk",     angles.get("trunk",     0)),
        "upper_arm": classify_reba_part("upper_arm", angles.get("upper_arm", 0)),
        "lower_arm": classify_reba_part("lower_arm", angles.get("lower_arm", 180)),
        "wrist":     classify_reba_part("wrist",     angles.get("wrist",     90)),
        "legs":      classify_reba_part("legs",      angles.get("knee",      180)),
    }

    return r
