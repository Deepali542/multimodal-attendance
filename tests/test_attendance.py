"""
Unit Tests for Scenario 1: Sustained Attendance Verification.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.attendance_tracker import AttendanceTracker


class TestAttendanceTracker(unittest.TestCase):

    def setUp(self):
        # Sustained threshold: 100s, Transient filter: 20s, Session total: 200s
        self.tracker = AttendanceTracker(
            sustained_presence_threshold_seconds=100.0,
            min_presence_ratio=0.60,
            transient_filter_seconds=20.0
        )
        self.start_time = 1000.0
        self.tracker.start_session(self.start_time)

    def test_sustained_presence_marked_present(self):
        """Student present continuously should be marked PRESENT."""
        for t in range(0, 150):  # 150 seconds of presence
            self.tracker.update(["Student_Regular"], current_time=self.start_time + t)

        results = self.tracker.evaluate_session(session_end_time=self.start_time + 200)
        rec = next(r for r in results if r["participant_id"] == "Student_Regular")

        self.assertEqual(rec["status"], "PRESENT")
        self.assertTrue(rec["sustained_verified"])

    def test_transient_appearance_ignored(self):
        """Passer-by appearing for only 15 seconds should be TRANSIENT_IGNORED."""
        for t in range(10, 25):  # 15 seconds presence
            self.tracker.update(["Passer_By"], current_time=self.start_time + t)

        results = self.tracker.evaluate_session(session_end_time=self.start_time + 200)
        rec = next(r for r in results if r["participant_id"] == "Passer_By")

        self.assertEqual(rec["status"], "TRANSIENT_IGNORED")
        self.assertFalse(rec["sustained_verified"])

    def test_insufficient_presence_marked_absent(self):
        """Student present for 40s (above transient filter 20s, but below sustained threshold 100s & 60%)"""
        for t in range(0, 40):  # 40 seconds
            self.tracker.update(["Student_Brief"], current_time=self.start_time + t)

        results = self.tracker.evaluate_session(session_end_time=self.start_time + 200)
        rec = next(r for r in results if r["participant_id"] == "Student_Brief")

        self.assertEqual(rec["status"], "ABSENT_INSUFFICIENT_PRESENCE")
        self.assertFalse(rec["sustained_verified"])


if __name__ == "__main__":
    unittest.main()
