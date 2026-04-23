"""
rula.py — RULA scoring engine (Enhanced)
Industrial Ergonomics Platform
+ Body Part Risk Breakdown (TuMeke-style)
"""

from dataclasses import dataclass
from typing import Dict
from config import RULA_RISK


# ─────────────────────────────────────────────────────────────────────────────
# 🆕 Body Part Risk Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_part_risk(part: str, angle: float) -> str:

    if part == "neck":
        if angle < 10: return "Low"
        elif angle < 20: return "Medium"
        else: return "High"

    elif part == "trunk":
        if angle < 10: return "Low"
        elif angle < 20: return "Medium"
        else: return "High"

    elif part == "upper_arm":
        if angle < 20: return "Low"
        elif angle < 45: return "Medium"
        else: return "High"

    elif part == "lower_arm":
        flex = abs(180 - angle)
        if 60 <= flex <= 100: return "Low"
        else: return "Medium"

    elif part == "wrist":
        dev = abs(angle - 90)
        if dev < 15: return "Low"
        elif dev < 30: return "Medium"
        else: return "High"

    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Individual segment scores (RULA core)
# ─────────────────────────────────────────────────────────────────────────────

def score_upper_arm(deg: float) -> int:
    if deg < 20: return 1
    elif deg <= 45: return 2
    elif deg <= 90: return 3
    else: return 4


def score_lower_arm(deg: float) -> int:
    flexion = abs(180.0 - deg)
    if 60 <= flexion <= 100:
        return 1
    return 2


def score_wrist(deg: float) -> int:
    deviation = abs(deg - 90.0)
    if deviation < 15: return 1
    elif deviation < 30: return 2
    else: return 3


def score_wrist_twist() -> int:
    return 1


def score_neck(deg: float) -> int:
    if deg < 10: return 1
    elif deg <= 20: return 2
    elif deg <= 30: return 3
    else: return 4


def score_trunk(deg: float) -> int:
    if deg < 10: return 1
    elif deg <= 20: return 2
    elif deg <= 60: return 3
    else: return 4


def score_legs() -> int:
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 📊 TABLE A
# ─────────────────────────────────────────────────────────────────────────────

TABLE_A = [
    [[[1,2],[2,2],[2,3]], [[2,2],[2,2],[3,3]]],
    [[[2,3],[3,3],[3,3]], [[2,3],[3,3],[3,4]]],
    [[[3,3],[3,3],[4,4]], [[3,3],[3,3],[4,4]]],
    [[[4,4],[4,4],[4,4]], [[4,4],[4,4],[4,5]]],
]

def lookup_table_a(ua, la, w, wt):
    return TABLE_A[min(ua-1,3)][min(la-1,1)][min(w-1,2)][min(wt-1,1)]


# ─────────────────────────────────────────────────────────────────────────────
# 📊 TABLE B
# ─────────────────────────────────────────────────────────────────────────────

TABLE_B = [
    [[1,3],[2,3],[3,4],[5,5]],
    [[2,3],[2,3],[4,5],[5,5]],
    [[3,3],[3,3],[4,5],[6,6]],
    [[5,5],[5,6],[6,7],[7,7]],
    [[7,7],[7,7],[7,8],[8,8]],
    [[8,8],[8,8],[8,8],[9,9]],
]

def lookup_table_b(neck, trunk, legs):
    return TABLE_B[min(neck-1,5)][min(trunk-1,3)][min(legs-1,1)]


# ─────────────────────────────────────────────────────────────────────────────
# 📊 TABLE C
# ─────────────────────────────────────────────────────────────────────────────

TABLE_C = [
    [1,2,3,3,4,5,5],
    [2,2,3,4,4,5,5],
    [3,3,3,4,4,5,6],
    [3,3,3,4,5,6,6],
    [4,4,4,5,6,7,7],
    [4,4,5,6,6,7,7],
    [5,5,6,6,7,7,7],
    [5,5,6,7,7,7,7],
]

def lookup_table_c(score_a, score_b):
    return TABLE_C[min(score_a-1,7)][min(score_b-1,6)]


# ─────────────────────────────────────────────────────────────────────────────
# 📦 Result Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RULAResult:
    frame_idx: int
    timestamp_sec: float
    angles: Dict[str, float]

    s_upper_arm: int = 0
    s_lower_arm: int = 0
    s_wrist: int = 0
    s_wrist_twist: int = 0
    s_neck: int = 0
    s_trunk: int = 0
    s_legs: int = 0

    score_a: int = 0
    score_b: int = 0

    raw_score: int = 0
    final_score: int = 0

    risk_level: str = ""
    risk_color: str = ""

    # 🆕 NEW
    body_risk: Dict[str, str] = None


    def to_dict(self):
        return {
            "frame": self.frame_idx,
            "time_sec": round(self.timestamp_sec, 2),

            "neck_deg": round(self.angles.get("neck", 0), 1),
            "trunk_deg": round(self.angles.get("trunk", 0), 1),
            "upper_arm_deg": round(self.angles.get("upper_arm", 0), 1),
            "lower_arm_deg": round(self.angles.get("lower_arm", 0), 1),
            "wrist_deg": round(self.angles.get("wrist", 0), 1),

            "rula_score": self.final_score,
            "risk_level": self.risk_level,

            # 🆕 NEW
            "neck_risk": self.body_risk.get("neck") if self.body_risk else "",
            "trunk_risk": self.body_risk.get("trunk") if self.body_risk else "",
            "upper_arm_risk": self.body_risk.get("upper_arm") if self.body_risk else "",
            "lower_arm_risk": self.body_risk.get("lower_arm") if self.body_risk else "",
            "wrist_risk": self.body_risk.get("wrist") if self.body_risk else "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_rula(frame_idx, timestamp_sec, angles):

    r = RULAResult(frame_idx=frame_idx, timestamp_sec=timestamp_sec, angles=angles)

    r.s_upper_arm = score_upper_arm(angles.get("upper_arm", 0))
    r.s_lower_arm = score_lower_arm(angles.get("lower_arm", 180))
    r.s_wrist = score_wrist(angles.get("wrist", 90))
    r.s_wrist_twist = score_wrist_twist()
    r.s_neck = score_neck(angles.get("neck", 0))
    r.s_trunk = score_trunk(angles.get("trunk", 0))
    r.s_legs = score_legs()

    r.score_a = lookup_table_a(r.s_upper_arm, r.s_lower_arm, r.s_wrist, r.s_wrist_twist)
    r.score_b = lookup_table_b(r.s_neck, r.s_trunk, r.s_legs)

    raw = lookup_table_c(r.score_a, r.score_b)
    r.raw_score = min(max(raw, 1), 7)
    r.final_score = r.raw_score

    label, color = RULA_RISK.get(r.final_score, ("Unknown", "#7f8c8d"))
    r.risk_level = label
    r.risk_color = color

    # 🆕 Body Breakdown
    r.body_risk = {
        "neck": classify_part_risk("neck", angles.get("neck", 0)),
        "trunk": classify_part_risk("trunk", angles.get("trunk", 0)),
        "upper_arm": classify_part_risk("upper_arm", angles.get("upper_arm", 0)),
        "lower_arm": classify_part_risk("lower_arm", angles.get("lower_arm", 180)),
        "wrist": classify_part_risk("wrist", angles.get("wrist", 90)),
    }

    return r