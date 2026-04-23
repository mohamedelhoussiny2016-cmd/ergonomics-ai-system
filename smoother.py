"""
smoother.py — Score stability engine (TuMeke-style)
Industrial Ergonomics Platform

Implements:
  • Sliding window MEDIAN smoothing
  • Spike rejection (prevents single-frame noise from triggering alerts)
  • Time-based alert state machine (not frame-counting)
  • Final session risk classification
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np
import time

from config import SystemConfig

AlertState = Literal["OK", "WARNING", "HIGH_RISK"]


# ─── Score smoother ───────────────────────────────────────────────────────────

class ScoreSmoother:
    """
    Maintains a sliding window of raw scores and produces a stable smoothed score.
    Uses MEDIAN (not mean) to resist outliers.
    Applies spike rejection: if |new - last| > spike_threshold, clamp to last.
    """

    def __init__(self, window_size: int = 10, spike_threshold: int = 2):
        self._window         = deque(maxlen=window_size)
        self._spike_threshold = spike_threshold
        self._last_score: Optional[float] = None

    def update(self, raw_score: float) -> int:
        """Feed a raw score, return the smoothed integer score."""
        # Spike rejection
        if self._last_score is not None:
            if abs(raw_score - self._last_score) > self._spike_threshold:
                raw_score = self._last_score   # hold previous

        self._window.append(raw_score)
        self._last_score = raw_score

        smoothed = float(np.median(list(self._window)))
        return int(round(smoothed))

    def reset(self):
        self._window.clear()
        self._last_score = None

    def get_smoothed(self) -> Optional[int]:
        if not self._window:
            return None
        return int(round(float(np.median(list(self._window)))))


# ─── Time-based alert state machine ──────────────────────────────────────────

@dataclass
class AlertEvent:
    state:          AlertState
    triggered_at:   float          # wall-clock time
    frame_idx:      int
    timestamp_sec:  float
    score:          int
    duration_sec:   float = 0.0
    message:        str   = ""


class AlertStateMachine:
    """
    Time-based alert logic (NOT frame-count based).

    States:
      OK       → score < warning_score
      WARNING  → score >= warning_score for >= warning_seconds
      HIGH_RISK→ score >= high_risk_score for >= high_risk_seconds

    Hysteresis: requires score to stay BELOW threshold for
    at least 1 second before downgrading state.
    """

    def __init__(self, cfg: SystemConfig):
        self._cfg               = cfg
        self._state: AlertState = "OK"
        self._above_warning_since: Optional[float] = None
        self._above_high_since:    Optional[float] = None
        self._below_since:         Optional[float] = None
        self._events:              List[AlertEvent] = []
        self._last_event_wall:     float = 0.0

    @property
    def state(self) -> AlertState:
        return self._state

    def update(self, smoothed_score: int, frame_idx: int,
               timestamp_sec: float) -> Optional[AlertEvent]:
        """
        Feed the current smoothed score.
        Returns an AlertEvent if state changes, else None.
        """
        now = time.monotonic()
        cfg = self._cfg

        warning_thr   = cfg.warning_score
        high_risk_thr = cfg.high_risk_score
        warn_secs     = cfg.warning_seconds
        high_secs     = cfg.high_risk_seconds
        hysteresis    = 1.0   # seconds below threshold before downgrade

        # ── Track how long we've been above each threshold ─────────────────
        if smoothed_score >= high_risk_thr:
            if self._above_high_since is None:
                self._above_high_since = now
            self._above_warning_since = self._above_warning_since or now
            self._below_since = None

        elif smoothed_score >= warning_thr:
            self._above_high_since    = None
            if self._above_warning_since is None:
                self._above_warning_since = now
            self._below_since = None

        else:
            self._above_high_since    = None
            self._above_warning_since = None
            if self._below_since is None:
                self._below_since = now

        # ── Determine target state ─────────────────────────────────────────
        new_state: AlertState = "OK"

        if (self._above_high_since is not None and
                now - self._above_high_since >= high_secs):
            new_state = "HIGH_RISK"

        elif (self._above_warning_since is not None and
              now - self._above_warning_since >= warn_secs):
            new_state = "WARNING"

        else:
            # Hysteresis on downgrade
            if self._state != "OK":
                if self._below_since is None or now - self._below_since < hysteresis:
                    new_state = self._state   # stay in current elevated state

        # ── Fire event on state change ─────────────────────────────────────
        event: Optional[AlertEvent] = None
        if new_state != self._state:
            msg = {
                "OK":        "✅ Posture returned to acceptable level",
                "WARNING":   "⚠️ Warning: Sustained awkward posture detected",
                "HIGH_RISK": "🚨 HIGH RISK POSTURE DETECTED — Intervention Required",
            }[new_state]

            # Throttle: don't re-fire same HIGH_RISK event more than once per 5 s
            if new_state == "HIGH_RISK" and now - self._last_event_wall < 5.0:
                pass
            else:
                event = AlertEvent(
                    state=new_state,
                    triggered_at=now,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    score=smoothed_score,
                    message=msg,
                )
                self._events.append(event)
                self._last_event_wall = now

            self._state = new_state

        return event

    def get_events(self) -> List[AlertEvent]:
        return list(self._events)

    def reset(self):
        self._state = "OK"
        self._above_warning_since = None
        self._above_high_since    = None
        self._below_since         = None
        self._events.clear()
        self._last_event_wall = 0.0


# ─── Session-level statistics ─────────────────────────────────────────────────

@dataclass
class SessionStats:
    total_frames:         int   = 0
    detected_frames:      int   = 0
    scores:               List[int] = field(default_factory=list)
    smoothed_scores:      List[int] = field(default_factory=list)
    high_risk_frames:     int   = 0
    warning_frames:       int   = 0
    assessment_type:      str   = "RULA"
    duration_sec:         float = 0.0

    def add(self, raw: int, smoothed: int, high_thr: int, warn_thr: int):
        self.scores.append(raw)
        self.smoothed_scores.append(smoothed)
        if smoothed >= high_thr:
            self.high_risk_frames += 1
        elif smoothed >= warn_thr:
            self.warning_frames += 1

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.smoothed_scores)) if self.smoothed_scores else 0.0

    @property
    def max_score(self) -> int:
        return int(max(self.smoothed_scores)) if self.smoothed_scores else 0

    @property
    def pct_high_risk(self) -> float:
        n = len(self.smoothed_scores)
        return 100 * self.high_risk_frames / n if n > 0 else 0.0

    @property
    def pct_warning(self) -> float:
        n = len(self.smoothed_scores)
        return 100 * self.warning_frames / n if n > 0 else 0.0

    def final_classification(self, cfg: SystemConfig) -> str:
        """
        TuMeke-style final session risk classification
        based on % time in each risk zone.
        """
        if self.pct_high_risk >= cfg.high_risk_pct_threshold:
            return "High Risk"
        elif (self.pct_high_risk + self.pct_warning) >= cfg.medium_risk_pct_threshold:
            return "Medium Risk"
        else:
            return "Low Risk"

    def score_distribution(self, max_score: int = 7) -> dict:
        dist = {i: 0 for i in range(1, max_score + 1)}
        for s in self.smoothed_scores:
            k = min(max(s, 1), max_score)
            dist[k] = dist.get(k, 0) + 1
        return dist

    def get_smoothed_for_chart(self, downsample: int = 80) -> tuple:
        """Returns (times_list, scores_list) downsampled for plotting."""
        ss = self.smoothed_scores
        if not ss:
            return [], []
        step = max(1, len(ss) // downsample)
        idx  = list(range(0, len(ss), step))
        return idx, [ss[i] for i in idx]
