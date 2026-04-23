"""
processor.py — Core video processing engine
Industrial Ergonomics Platform

Generator-based architecture:
  for frame, result, stats in processor.stream(source):
      display(frame)

Handles auto-calibration, pose detection, scoring, smoothing,
alert state machine, output video writing.
"""

import cv2
import numpy as np
import threading
from typing import Generator, Optional, Tuple, List, Union
from dataclasses import dataclass

from config import SystemConfig, AssessmentType
from pose import PoseDetector
from rula import compute_rula, RULAResult
from reba import compute_reba, REBAResult
from smoother import ScoreSmoother, AlertStateMachine, SessionStats
from overlay import draw_hud, draw_no_detection

AnyResult = Union[RULAResult, REBAResult]


# ─── Video info ───────────────────────────────────────────────────────────────

@dataclass
class VideoInfo:
    fps:           float
    width:         int
    height:        int
    total_frames:  int
    source:        str

    @property
    def duration_sec(self) -> float:
        return self.total_frames / max(self.fps, 1.0)


def probe_video(source) -> Tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source!r}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src    = str(source) if not isinstance(source, int) else f"webcam:{source}"
    return cap, VideoInfo(fps=fps, width=width, height=height,
                          total_frames=total, source=src)


# ─── Frame result bundle ──────────────────────────────────────────────────────

@dataclass
class FrameBundle:
    """Everything the UI needs for one processed frame."""
    annotated_frame: np.ndarray
    result:          Optional[AnyResult]
    smoothed_score:  int
    alert_state:     str
    frame_idx:       int
    timestamp_sec:   float
    progress:        float          # 0.0 – 1.0
    stats:           SessionStats
    pose_detected:   bool


# ─── Main processor ───────────────────────────────────────────────────────────

class VideoProcessor:

    def __init__(self, cfg: SystemConfig, assessment: AssessmentType = "RULA"):
        self.cfg        = cfg
        self.assessment = assessment
        self._stop      = threading.Event()

    def stop(self):
        self._stop.set()

    def _make_writer(self, info: VideoInfo) -> Optional[cv2.VideoWriter]:
        try:
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = max(info.fps / self.cfg.frame_skip, 6.0)
            writer  = cv2.VideoWriter(
                self.cfg.output_video_path, fourcc, out_fps,
                (info.width, info.height))
            return writer if writer.isOpened() else None
        except Exception:
            return None

    def _score_frame(self, frame_idx: int, ts: float,
                     angles: dict) -> AnyResult:
        if self.assessment == "REBA":
            return compute_reba(frame_idx, ts, angles)
        return compute_rula(frame_idx, ts, angles)

    def stream(
        self,
        source: Union[str, int],
        save_output: bool = True,
        high_risk_frame_limit: int = 6,
    ) -> Generator[FrameBundle, None, None]:
        """
        Main generator. Yields FrameBundle for each processed frame.

        Usage:
            for bundle in processor.stream("video.mp4"):
                show(bundle.annotated_frame)
        """
        self._stop.clear()

        cap, info = probe_video(source)
        self.cfg.calibrate(info.fps, info.width)

        smoother = ScoreSmoother(
            window_size=self.cfg.smoothing_window,
            spike_threshold=self.cfg.spike_threshold,
        )
        alert_sm  = AlertStateMachine(self.cfg)
        stats     = SessionStats(assessment_type=self.assessment)
        writer    = self._make_writer(info) if save_output else None

        max_score = 15 if self.assessment == "REBA" else 7
        high_risk_thr = self.cfg.high_risk_score if self.assessment == "RULA" else 10
        warn_thr      = self.cfg.warning_score   if self.assessment == "RULA" else 7

        # For collecting high-risk sample frames
        self.high_risk_frames: List[np.ndarray] = []
        self.video_info = info
        self.session_stats = stats
        self.alert_events  = []

        frame_idx      = 0
        processed_idx  = 0
        last_smoothed  = 1

        with PoseDetector(self.cfg) as detector:
            try:
                while not self._stop.is_set():
                    ret, raw_frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    stats.total_frames += 1

                    # Skip frames for performance
                    if frame_idx % self.cfg.frame_skip != 0:
                        continue

                    processed_idx += 1
                    ts = frame_idx / max(info.fps, 1.0)

                    # Pose detection
                    landmarks, angles, annotated = detector.process(raw_frame)
                    pose_detected = landmarks is not None and bool(angles)

                    if pose_detected:
                        stats.detected_frames += 1
                        result = self._score_frame(frame_idx, ts, angles)
                        raw_score     = result.raw_score
                        smoothed      = smoother.update(raw_score)
                        last_smoothed = smoothed

                        # Update result with smoothed score
                        result.final_score = smoothed
                        from config import RULA_RISK, REBA_RISK
                        risk_map = REBA_RISK if self.assessment == "REBA" else RULA_RISK
                        label, color = risk_map.get(smoothed, ("Unknown", "#7f8c8d"))
                        result.risk_level = label
                        result.risk_color = color

                        # Session stats
                        stats.add(raw_score, smoothed, high_risk_thr, warn_thr)

                        # Alert state machine
                        alert_event = alert_sm.update(smoothed, frame_idx, ts)
                        if alert_event:
                            self.alert_events.append(alert_event)

                        # Collect high-risk frames for report
                        if (smoothed >= high_risk_thr and
                                len(self.high_risk_frames) < high_risk_frame_limit):
                            self.high_risk_frames.append(annotated.copy())

                    else:
                        result    = None
                        smoothed  = last_smoothed  # hold last known score

                    alert_state = alert_sm.state

                    # Draw HUD
                    if pose_detected:
                        out_frame = draw_hud(
                            annotated,
                            score=smoothed,
                            max_score=max_score,
                            risk_level=result.risk_level if result else "N/A",
                            assessment=self.assessment,
                            alert_state=alert_state,
                            timestamp_sec=ts,
                            frame_idx=frame_idx,
                            angles=angles,
                            avg_score=stats.avg_score,
                            pct_high=stats.pct_high_risk,
                        )
                    else:
                        out_frame = draw_no_detection(annotated, frame_idx)

                    # Write to output video (original resolution)
                    if writer:
                        resized_out = cv2.resize(out_frame, (info.width, info.height))
                        writer.write(resized_out)

                    progress = (frame_idx / info.total_frames
                                if info.total_frames > 0 else 0.0)

                    stats.duration_sec = ts

                    yield FrameBundle(
                        annotated_frame=out_frame,
                        result=result,
                        smoothed_score=smoothed,
                        alert_state=alert_state,
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        progress=min(progress, 1.0),
                        stats=stats,
                        pose_detected=pose_detected,
                    )

            finally:
                if writer:
                    writer.release()
                cap.release()

        self.session_stats = stats
        self.alert_sm      = alert_sm
