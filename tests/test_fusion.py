"""
Unit Tests for EPIC 4: Multimodal Signal Fusion.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_agent.multimodal_fusion import MultimodalFusionEngine


class TestMultimodalFusion(unittest.TestCase):

    def setUp(self):
        # 60% visual, 40% audio, 0.5 smoothing
        self.engine = MultimodalFusionEngine(
            visual_weight=0.60,
            audio_weight=0.40,
            smoothing_alpha=1.0,  # instant for deterministic unit testing
            attention_drop_threshold=40.0
        )
        self.engine.start_session(start_time=100.0)

    def test_weighted_fusion_calculation(self):
        """Checks if composite score matches 0.6 * vis + 0.4 * aud."""
        snapshot = self.engine.fuse_signals(
            visual_score=80.0,
            audio_score=50.0,
            acoustic_state="LECTURE_MONOLOGUE",
            active_participants=10,
            focus_percentage=90.0,
            current_time=101.0
        )
        # Expected: 0.6 * 80 + 0.4 * 50 = 48 + 20 = 68.0
        self.assertAlmostEqual(snapshot.composite_engagement, 68.0, places=1)

    def test_interaction_peak_detection(self):
        """High engagement during active discussion triggers INTERACTION_PEAK."""
        # Prime history
        for i in range(15):
            self.engine.fuse_signals(
                visual_score=90.0,
                audio_score=85.0,
                acoustic_state="CLASS_DISCUSSION",
                active_participants=12,
                focus_percentage=95.0,
                current_time=100.0 + i
            )

        snap = self.engine.history[-1]
        self.assertEqual(snap.trend_flag, "INTERACTION_PEAK")


if __name__ == "__main__":
    unittest.main()
