"""
Multimodal Signal Fusion Engine.
Fuses visual behavioral indicators and audio dynamics into unified engagement metrics
and temporal trend tracking across session intervals.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class MultimodalSnapshot:
    timestamp: float
    elapsed_seconds: float
    visual_score: float
    audio_score: float
    composite_engagement: float
    acoustic_state: str
    active_participant_count: int
    focus_percentage: float
    trend_flag: Optional[str] = None  # 'ATTENTION_DROP', 'INTERACTION_PEAK', 'PACING_ISSUE', None


class MultimodalFusionEngine:
    """
    Fuses visual and audio indicators with temporal smoothing (EMA)
    and interval aggregation to detect attention drops and session peaks.
    """

    def __init__(
        self,
        visual_weight: float = 0.60,
        audio_weight: float = 0.40,
        smoothing_alpha: float = 0.35,
        attention_drop_threshold: float = 45.0,
        attention_drop_duration_sec: float = 120.0,
        monologue_limit_sec: float = 900.0
    ):
        self.w_vis = visual_weight
        self.w_aud = audio_weight
        self.alpha = smoothing_alpha
        self.attention_drop_threshold = attention_drop_threshold
        self.drop_duration_sec = attention_drop_duration_sec
        self.monologue_limit_sec = monologue_limit_sec

        self.smoothed_engagement: Optional[float] = None
        self.history: List[MultimodalSnapshot] = []
        self.session_start_time: Optional[float] = None

        # Interval tracking (e.g. 1-minute bins)
        self.interval_seconds = 60.0
        self.interval_summaries: List[Dict] = []
        self.current_interval_buffer: List[MultimodalSnapshot] = []

    def start_session(self, start_time: Optional[float] = None):
        self.session_start_time = start_time or time.time()
        self.smoothed_engagement = None
        self.history.clear()
        self.interval_summaries.clear()
        self.current_interval_buffer.clear()

    def fuse_signals(
        self,
        visual_score: float,
        audio_score: float,
        acoustic_state: str,
        active_participants: int,
        focus_percentage: float,
        current_time: Optional[float] = None
    ) -> MultimodalSnapshot:
        """
        Calculates smoothed composite engagement and detects behavioral trends.
        """
        if current_time is None:
            current_time = time.time()
        if self.session_start_time is None:
            self.start_session(current_time)

        elapsed = current_time - self.session_start_time

        # Raw composite formula
        raw_composite = (self.w_vis * visual_score) + (self.w_aud * audio_score)
        # Cap between 0 and 100
        raw_composite = max(0.0, min(100.0, raw_composite))

        # Exponential Moving Average (EMA) smoothing
        if self.smoothed_engagement is None:
            self.smoothed_engagement = raw_composite
        else:
            self.smoothed_engagement = (self.alpha * raw_composite) + ((1.0 - self.alpha) * self.smoothed_engagement)

        trend_flag = self._evaluate_trends(elapsed, acoustic_state)

        snapshot = MultimodalSnapshot(
            timestamp=current_time,
            elapsed_seconds=round(elapsed, 1),
            visual_score=round(visual_score, 1),
            audio_score=round(audio_score, 1),
            composite_engagement=round(self.smoothed_engagement, 1),
            acoustic_state=acoustic_state,
            active_participant_count=active_participants,
            focus_percentage=round(focus_percentage, 1),
            trend_flag=trend_flag
        )

        self.history.append(snapshot)
        self.current_interval_buffer.append(snapshot)

        # Check if interval boundary reached (e.g. 60s)
        if len(self.current_interval_buffer) >= (self.interval_seconds):
            self._finalize_interval()

        return snapshot

    def _evaluate_trends(self, elapsed: float, current_acoustic: str) -> Optional[str]:
        """Detects sustained drops, monologue pacing fatigue, or interaction peaks."""
        if len(self.history) < 10:
            return None

        # Check for interaction peak
        if self.smoothed_engagement >= 75.0 and current_acoustic == "CLASS_DISCUSSION":
            return "INTERACTION_PEAK"

        # Check for sustained attention drop
        recent_window = [s for s in self.history if elapsed - s.elapsed_seconds <= self.drop_duration_sec]
        if len(recent_window) >= 15:
            avg_recent = sum(s.composite_engagement for s in recent_window) / len(recent_window)
            if avg_recent < self.attention_drop_threshold:
                return "ATTENTION_DROP"

        # Check for monologue pacing issue (> 15 minutes of continuous lecture monologue)
        monologue_window = [s for s in self.history if elapsed - s.elapsed_seconds <= self.monologue_limit_sec]
        if elapsed >= self.monologue_limit_sec and len(monologue_window) >= 30:
            lecture_ratio = sum(1 for s in monologue_window if s.acoustic_state == "LECTURE_MONOLOGUE") / len(monologue_window)
            if lecture_ratio > 0.85:
                return "PACING_ISSUE"

        return None

    def _finalize_interval(self):
        """Rolls up interval data into trend bins."""
        if not self.current_interval_buffer:
            return
        interval_idx = len(self.interval_summaries) + 1
        avg_eng = sum(s.composite_engagement for s in self.current_interval_buffer) / len(self.current_interval_buffer)
        avg_vis = sum(s.visual_score for s in self.current_interval_buffer) / len(self.current_interval_buffer)
        avg_aud = sum(s.audio_score for s in self.current_interval_buffer) / len(self.current_interval_buffer)

        self.interval_summaries.append({
            "interval_minute": interval_idx,
            "avg_engagement": round(avg_eng, 1),
            "avg_visual": round(avg_vis, 1),
            "avg_audio": round(avg_aud, 1),
            "sample_count": len(self.current_interval_buffer)
        })
        self.current_interval_buffer.clear()

    def get_session_analytics(self) -> Dict:
        """Computes comprehensive session-wide metrics."""
        if not self.history:
            return {
                "average_engagement": 0.0,
                "peak_engagement": 0.0,
                "lowest_engagement": 0.0,
                "total_snapshots": 0,
                "drop_events_count": 0,
                "peak_events_count": 0
            }

        scores = [s.composite_engagement for s in self.history]
        drops = sum(1 for s in self.history if s.trend_flag == "ATTENTION_DROP")
        peaks = sum(1 for s in self.history if s.trend_flag == "INTERACTION_PEAK")

        return {
            "average_engagement": round(sum(scores) / len(scores), 1),
            "peak_engagement": round(max(scores), 1),
            "lowest_engagement": round(min(scores), 1),
            "total_snapshots": len(self.history),
            "drop_events_count": drops,
            "peak_events_count": peaks,
            "intervals": self.interval_summaries
        }
