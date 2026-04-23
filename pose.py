"""
pose.py — MediaPipe Pose detection + joint angle computation
Industrial Ergonomics Platform
Shared module used by both RULA and REBA scoring engines.
"""

import cv2
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from typing import Optional, Dict, Tuple, NamedTuple
from config import SystemConfig

# ─── Landmark shortcuts ───────────────────────────────────────────────────────

LM = mp_pose.PoseLandmark

KEYPOINTS = {
    "nose":           LM.NOSE,
    "left_eye":       LM.LEFT_EYE,
    "right_eye":      LM.RIGHT_EYE,
    "left_ear":       LM.LEFT_EAR,
    "right_ear":      LM.RIGHT_EAR,
    "left_shoulder":  LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_elbow":     LM.LEFT_ELBOW,
    "right_elbow":    LM.RIGHT_ELBOW,
    "left_wrist":     LM.LEFT_WRIST,
    "right_wrist":    LM.RIGHT_WRIST,
    "left_hip":       LM.LEFT_HIP,
    "right_hip":      LM.RIGHT_HIP,
    "left_knee":      LM.LEFT_KNEE,
    "right_knee":     LM.RIGHT_KNEE,
    "left_ankle":     LM.LEFT_ANKLE,
    "right_ankle":    LM.RIGHT_ANKLE,
}

# ─── Geometry helpers ─────────────────────────────────────────────────────────

def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex B formed by rays BA and BC. Returns degrees [0, 180]."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def angle_with_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of vector (b - a) with the downward vertical. Returns degrees [0, 180]."""
    v = b - a
    vertical = np.array([0.0, 1.0])   # downward in image coords
    norm = np.linalg.norm(v) + 1e-9
    cosine = np.clip(np.dot(v / norm, vertical), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pt(landmarks, name: str, w: int, h: int) -> np.ndarray:
    lm = landmarks[KEYPOINTS[name]]
    return np.array([lm.x * w, lm.y * h], dtype=float)


def visibility(landmarks, name: str) -> float:
    return landmarks[KEYPOINTS[name]].visibility


# ─── Angle computation ────────────────────────────────────────────────────────

def compute_angles(landmarks, w: int, h: int) -> Dict[str, float]:
    """
    Compute all ergonomically-relevant body angles from MediaPipe landmarks.
    Uses the more visible side (left/right) for arm measurements.
    Returns a dict of angle values in degrees.
    """
    angles: Dict[str, float] = {}

    def p(name): return pt(landmarks, name, w, h)
    def vis(name): return visibility(landmarks, name)

    try:
        nose         = p("nose")
        l_shoulder   = p("left_shoulder");  r_shoulder = p("right_shoulder")
        l_elbow      = p("left_elbow");     r_elbow    = p("right_elbow")
        l_wrist      = p("left_wrist");     r_wrist    = p("right_wrist")
        l_hip        = p("left_hip");       r_hip      = p("right_hip")
        l_knee       = p("left_knee");      r_knee     = p("right_knee")
        l_ankle      = p("left_ankle");     r_ankle    = p("right_ankle")

        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip      = (l_hip      + r_hip)      / 2

        # ── Neck (head-to-spine angle) ────────────────────────────────────
        # Neck flexion: angle of head relative to vertical trunk
        neck_angle = angle_abc(nose, mid_shoulder, mid_hip)
        angles["neck"] = abs(180.0 - neck_angle)

        # ── Trunk flexion ─────────────────────────────────────────────────
        # Angle of spine from vertical
        angles["trunk"] = angle_with_vertical(mid_hip, mid_shoulder)

        # ── Trunk lateral bending ─────────────────────────────────────────
        # Shoulder tilt relative to hip line
        shoulder_vec = r_shoulder - l_shoulder
        hip_vec      = r_hip - l_hip
        angles["trunk_lateral"] = abs(
            np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]) -
                       np.arctan2(hip_vec[1], hip_vec[0]))
        )

        # ── Select dominant arm (higher visibility) ───────────────────────
        r_vis = (vis("right_shoulder") + vis("right_elbow") + vis("right_wrist")) / 3
        l_vis = (vis("left_shoulder")  + vis("left_elbow")  + vis("left_wrist"))  / 3
        use_right = r_vis >= l_vis
        shoulder = r_shoulder if use_right else l_shoulder
        elbow    = r_elbow    if use_right else l_elbow
        wrist    = r_wrist    if use_right else l_wrist
        hip_side = r_hip      if use_right else l_hip

        # ── Upper arm elevation ───────────────────────────────────────────
        # Angle from trunk to upper arm
        upper_arm_angle = angle_abc(hip_side, shoulder, elbow)
        angles["upper_arm"] = abs(upper_arm_angle)

        # ── Lower arm (elbow flexion) ─────────────────────────────────────
        angles["lower_arm"] = angle_abc(shoulder, elbow, wrist)

        # ── Wrist deviation ───────────────────────────────────────────────
        forearm_vec  = wrist - elbow
        forearm_norm = np.linalg.norm(forearm_vec) + 1e-9
        # deviation from the forearm axis
        vert = np.array([0.0, 1.0])
        wrist_angle = float(np.degrees(np.arccos(
            np.clip(np.dot(forearm_vec / forearm_norm, vert), -1.0, 1.0)
        )))
        angles["wrist"] = wrist_angle

        # ── Legs (for REBA) ───────────────────────────────────────────────
        r_kvis = vis("right_knee"); l_kvis = vis("left_knee")
        use_r_leg = r_kvis >= l_kvis
        hip_l  = r_hip   if use_r_leg else l_hip
        knee_l = r_knee  if use_r_leg else l_knee
        ankle_l= r_ankle if use_r_leg else l_ankle

        angles["knee"] = angle_abc(hip_l, knee_l, ankle_l)
        # Trunk-to-hip-to-knee → leg posture
        angles["leg_raise"] = angle_with_vertical(hip_l, knee_l)

    except Exception:
        pass  # Return partial dict — caller handles missing keys

    return angles


# ─── Pose detector ────────────────────────────────────────────────────────────

class PoseDetector:
    """
    MediaPipe Pose wrapper.
    One instance per video — NOT thread-safe across threads.
    """

    def __init__(self, cfg: SystemConfig):
        self._pose = mp_pose.Pose(
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            static_image_mode=False,
        )
        self._cfg = cfg

    def process(self, frame: np.ndarray) -> Tuple[Optional[object], Dict[str, float], np.ndarray]:
        """
        Process one BGR frame.
        Returns: (landmarks | None, angles_dict, annotated_frame)
        """
        h, w = frame.shape[:2]

        # Optional resize for performance
        proc_w = self._cfg.process_width
        if proc_w > 0 and w > proc_w:
            scale  = proc_w / w
            proc_h = int(h * scale)
            small  = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            small  = frame
            proc_h = h
            proc_w = w

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        if results.pose_landmarks:
            # Scale landmarks back to original frame for drawing
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            angles = compute_angles(results.pose_landmarks.landmark, proc_w, proc_h)
            return results.pose_landmarks.landmark, angles, annotated

        return None, {}, annotated

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
