"""
Attendance Tracking Module with Sustained Presence Verification.
Implements Scenario 1: Reduces false positives from momentary appearances
and automatically marks attendance based on continuous sustained presence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
from datetime import datetime


@dataclass
class ParticipantSessionRecord:
    participant_id: str
    first_seen: float
    last_seen: float
    total_present_seconds: float = 0.0
    detection_count: int = 0
    consecutive_present_seconds: float = 0.0
    status: str = "IN_PROGRESS"  # 'PRESENT', 'TRANSIENT_IGNORED', 'IN_PROGRESS', 'ABSENT'
    attendance_marked_at: Optional[float] = None
    timestamps: List[float] = field(default_factory=list)


class AttendanceTracker:
    """
    Tracks participants across classroom frames, calculating sustained presence.
    Filters out momentary transient passers-by.
    """

    def __init__(
        self,
        sustained_presence_threshold_seconds: float = 180.0,
        min_presence_ratio: float = 0.60,
        transient_filter_seconds: float = 30.0,
        sampling_interval_seconds: float = 1.0
    ):
        self.sustained_threshold = sustained_presence_threshold_seconds
        self.min_presence_ratio = min_presence_ratio
        self.transient_filter = transient_filter_seconds
        self.sampling_interval = sampling_interval_seconds

        self.session_start_time: Optional[float] = None
        self.session_end_time: Optional[float] = None
        self.records: Dict[str, ParticipantSessionRecord] = {}
        self.last_update_time: Optional[float] = None

    def start_session(self, start_time: Optional[float] = None):
        """Initializes a new classroom session."""
        self.session_start_time = start_time or time.time()
        self.last_update_time = self.session_start_time
        self.records.clear()

    def update(self, detected_participant_ids: List[str], current_time: Optional[float] = None):
        """
        Processes detections in the current time step.
        """
        if current_time is None:
            current_time = time.time()

        if self.session_start_time is None:
            self.start_session(current_time)

        dt = current_time - (self.last_update_time or current_time)
        if dt <= 0:
            dt = self.sampling_interval
        # Cap dt to avoid huge jumps on pauses
        dt = min(dt, 5.0)

        self.last_update_time = current_time

        # Update detected participants
        for pid in detected_participant_ids:
            if pid not in self.records:
                self.records[pid] = ParticipantSessionRecord(
                    participant_id=pid,
                    first_seen=current_time,
                    last_seen=current_time,
                    total_present_seconds=0.0,
                    detection_count=1,
                    timestamps=[current_time]
                )
            else:
                rec = self.records[pid]
                rec.last_seen = current_time
                rec.total_present_seconds += dt
                rec.detection_count += 1
                rec.timestamps.append(current_time)

                # Check if sustained presence threshold is met in real-time
                if rec.total_present_seconds >= self.sustained_threshold and rec.status != "PRESENT":
                    rec.status = "PRESENT"
                    rec.attendance_marked_at = current_time

    def evaluate_session(self, session_end_time: Optional[float] = None) -> List[Dict]:
        """
        Finalizes attendance decisions for all participants based on sustained presence rules.
        """
        if session_end_time is None:
            session_end_time = self.last_update_time or time.time()
        self.session_end_time = session_end_time

        session_duration = max(1.0, self.session_end_time - (self.session_start_time or self.session_end_time))

        final_records = []
        for pid, rec in self.records.items():
            presence_ratio = rec.total_present_seconds / session_duration

            # Sustained presence criteria:
            # 1. Total present seconds >= threshold OR presence_ratio >= min_presence_ratio
            # 2. Appearance > transient filter
            if rec.total_present_seconds < self.transient_filter:
                rec.status = "TRANSIENT_IGNORED"  # Passer-by filtered out
            elif rec.total_present_seconds >= self.sustained_threshold or presence_ratio >= self.min_presence_ratio:
                rec.status = "PRESENT"
                if rec.attendance_marked_at is None:
                    rec.attendance_marked_at = rec.last_seen
            else:
                rec.status = "ABSENT_INSUFFICIENT_PRESENCE"

            final_records.append({
                "participant_id": pid,
                "status": rec.status,
                "first_seen": datetime.fromtimestamp(rec.first_seen).strftime("%H:%M:%S"),
                "last_seen": datetime.fromtimestamp(rec.last_seen).strftime("%H:%M:%S"),
                "total_present_minutes": round(rec.total_present_seconds / 60.0, 2),
                "presence_percentage": round(min(100.0, presence_ratio * 100.0), 1),
                "sustained_verified": (rec.status == "PRESENT"),
                "attendance_timestamp": (
                    datetime.fromtimestamp(rec.attendance_marked_at).strftime("%H:%M:%S")
                    if rec.attendance_marked_at else "N/A"
                )
            })

        return final_records

    def get_realtime_status(self) -> Dict:
        """Returns instantaneous session attendance snapshot."""
        now = self.last_update_time or time.time()
        duration = max(1.0, now - (self.session_start_time or now))
        total_tracked = len(self.records)
        present_count = sum(1 for r in self.records.values() if r.status == "PRESENT")
        transient_count = sum(1 for r in self.records.values() if r.status == "TRANSIENT_IGNORED")

        return {
            "session_duration_minutes": round(duration / 60.0, 2),
            "total_tracked_candidates": total_tracked,
            "confirmed_present_count": present_count,
            "transient_filtered_count": transient_count,
            "attendance_rate_percent": round((present_count / total_tracked * 100.0) if total_tracked > 0 else 0.0, 1)
        }
