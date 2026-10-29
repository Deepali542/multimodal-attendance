"""
Unit Tests for Scenario 3: AI-Driven Pedagogical Suggestions.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_agent.reasoning_engine import PedagogicalReasoningAgent


class TestPedagogicalReasoningAgent(unittest.TestCase):

    def setUp(self):
        self.agent = PedagogicalReasoningAgent(enable_cloud_llm=False, cooldown_seconds=10.0)

    def test_attention_drop_intervention(self):
        """Low engagement should trigger ATTENTION_DROP suggestion with HIGH urgency."""
        suggestion = self.agent.evaluate_classroom_state(
            composite_engagement=32.0,
            visual_score=30.0,
            acoustic_state="QUIET_STUDY",
            trend_flag="ATTENTION_DROP",
            elapsed_seconds=120.0,
            active_student_count=15,
            current_time=100.0
        )
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion.trigger_category, "ATTENTION_DROP")
        self.assertEqual(suggestion.urgency_level, "HIGH")
        self.assertIn("OFFLINE_EXPERT_SYSTEM", suggestion.source)

    def test_cooldown_suppression(self):
        """Second call within cooldown window should return None to avoid teacher distraction."""
        # First call triggers suggestion
        self.agent.evaluate_classroom_state(
            composite_engagement=30.0,
            visual_score=25.0,
            acoustic_state="QUIET_STUDY",
            trend_flag="ATTENTION_DROP",
            elapsed_seconds=120.0,
            current_time=200.0
        )

        # Immediate follow-up within 5 seconds (cooldown is 10s)
        suppressed = self.agent.evaluate_classroom_state(
            composite_engagement=25.0,
            visual_score=20.0,
            acoustic_state="QUIET_STUDY",
            trend_flag="ATTENTION_DROP",
            elapsed_seconds=125.0,
            current_time=205.0
        )
        self.assertIsNone(suppressed)


if __name__ == "__main__":
    unittest.main()
