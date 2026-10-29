"""
Engagement and Session Dynamics Analysis - Vision Submodule.
Analyzes behavioral indicators: head pose, gaze orientation,
movement patterns, and inactivity duration.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import math


@dataclass
class VisualEngagementMetrics:
    participant_id: str
    gaze_forward_ratio: float      # 0.0 to 1.0 (looking at board/screen)
    movement_intensity: float       # Motion energy level
    inactivity_seconds: float       # Time without meaningful movement
    attention_state: str            # 'FOCUSED', 'DISTRACTED', 'RESTLESS', 'DROWSY'
    visual_score: float             # Normalized 0.0 to 100.0


class VisualEngagementDetector:
    """
    Evaluates participant engagement from visual behavior indicators.
    Works with both live video frames and feature coordinate vectors.
    """

    def __init__(
        self,
        max_yaw_attention: float = 30.0,
        max_pitch_attention: float = 25.0,
        inactivity_limit_seconds: float = 120.0,
        restlessness_threshold: float = 0.50
    ):
        self.max_yaw = max_yaw_attention
        self.max_pitch = max_pitch_attention
        self.inactivity_limit = inactivity_limit_seconds
        self.restlessness_threshold = restlessness_threshold

        # Track previous centers and timestamps for movement analysis
        # pid -> {'prev_center': (cx, cy), 'last_motion_time': float, 'last_seen': float}
        self.student_history: Dict[str, Dict] = {}

    def compute_head_pose_score(self, yaw_degrees: float, pitch_degrees: float) -> float:
        """
        Calculates gaze focus score based on head pose angle deviations from central axis.
        """
        yaw_penalty = min(1.0, abs(yaw_degrees) / (self.max_yaw * 1.5))
        pitch_penalty = min(1.0, abs(pitch_degrees) / (self.max_pitch * 1.5))
        combined_dev = (yaw_penalty * 0.6) + (pitch_penalty * 0.4)
        return max(0.0, 1.0 - combined_dev)

    def analyze_participant(
        self,
        participant_id: str,
        bbox: Tuple[int, int, int, int],
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        current_time: Optional[float] = None
    ) -> VisualEngagementMetrics:
        """
        Computes behavioral engagement indicators for a single student.
        """
        if current_time is None:
            current_time = time.time()

        x, y, w, h = bbox
        cx, cy = int(x + w / 2), int(y + h / 2)

        if participant_id not in self.student_history:
            self.student_history[participant_id] = {
                "prev_center": (cx, cy),
                "last_motion_time": current_time,
                "last_seen": current_time,
                "inactivity_seconds": 0.0,
                "motion_history": []
            }

        hist = self.student_history[participant_id]
        px, py = hist["prev_center"]

        # Calculate spatial displacement normalized by bounding box size
        norm_scale = max(1.0, (w + h) / 2.0)
        displacement = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / norm_scale

        # Update inactivity tracker
        motion_threshold = 0.03
        if displacement > motion_threshold:
            hist["last_motion_time"] = current_time
            inactivity = 0.0
        else:
            inactivity = current_time - hist["last_motion_time"]

        hist["prev_center"] = (cx, cy)
        hist["last_seen"] = current_time
        hist["inactivity_seconds"] = inactivity

        # Compute Gaze Focus Score (0.0 to 1.0)
        gaze_score = self.compute_head_pose_score(yaw_deg, pitch_deg)

        # Movement Quality Score
        if inactivity > self.inactivity_limit:
            state = "DROWSY"
            move_score = 0.25
        elif displacement > self.restlessness_threshold:
            state = "RESTLESS"
            move_score = 0.40
        elif gaze_score >= 0.70:
            state = "FOCUSED"
            move_score = 0.90
        else:
            state = "DISTRACTED"
            move_score = 0.50

        # Weighted Visual Score (0 - 100)
        score = ((gaze_score * 0.60) + (move_score * 0.40)) * 100.0

        # Inactivity penalty
        if inactivity > 60.0:
            penalty = min(30.0, (inactivity - 60.0) * 0.3)
            score = max(5.0, score - penalty)

        return VisualEngagementMetrics(
            participant_id=participant_id,
            gaze_forward_ratio=round(gaze_score, 3),
            movement_intensity=round(displacement, 3),
            inactivity_seconds=round(inactivity, 1),
            attention_state=state,
            visual_score=round(score, 1)
        )

    def analyze_classroom_batch(
        self,
        detections: List[Dict],
        current_time: Optional[float] = None
    ) -> Tuple[float, List[VisualEngagementMetrics]]:
        """
        Processes all participants in the classroom frame.
        Returns (average_visual_engagement, individual_metrics_list).
        """
        if not detections:
            return 0.0, []

        results = []
        for det in detections:
            pid = det.get("participant_id", "Unknown")
            bbox = det.get("bbox", (100, 100, 50, 50))
            yaw = det.get("yaw", 0.0)
            pitch = det.get("pitch", 0.0)

            metrics = self.analyze_participant(pid, bbox, yaw, pitch, current_time)
            results.append(metrics)

        avg_score = sum(m.visual_score for m in results) / len(results)
        return round(avg_score, 1), results
