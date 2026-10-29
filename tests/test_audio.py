"""
Unit Tests for EPIC 3: Audio Signal Processing & Classifier.
"""

import unittest
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.audio_processor import AudioSignalProcessor, AudioFeatures
from src.audio.audio_classifier import ClassroomAcousticClassifier, ClassroomAcousticState


class TestAudioModule(unittest.TestCase):

    def setUp(self):
        self.processor = AudioSignalProcessor(sample_rate=16000, rms_speech_threshold=0.03)
        self.classifier = ClassroomAcousticClassifier(window_size=10)

    def test_silence_detection(self):
        """Zero samples should result in silent features."""
        silent_samples = [0.0] * 1600
        feat = self.processor.process_samples(silent_samples)
        self.assertFalse(feat.is_speech)
        self.assertEqual(feat.silence_ratio, 1.0)
        self.assertLess(feat.db_level, -60.0)

    def test_speech_detection(self):
        """Active sinusoidal wave above threshold should trigger speech detection."""
        active_samples = [0.2 * math.sin(i * 0.1) for i in range(1600)]
        feat = self.processor.process_samples(active_samples)
        self.assertTrue(feat.is_speech)
        self.assertGreater(feat.rms_energy, 0.03)
        self.assertGreater(feat.audio_engagement_score, 60.0)

    def test_acoustic_classification_discussion(self):
        """Alternating speech with natural pauses should classify as CLASS_DISCUSSION."""
        for i in range(10):
            # 50% speech, 40% silence pattern
            f = AudioFeatures(
                rms_energy=0.06,
                is_speech=(i % 2 == 0),
                db_level=-25.0,
                silence_ratio=0.35,
                dominant_frequency_hz=300.0,
                audio_engagement_score=85.0
            )
            self.classifier.add_feature(f)

        state = self.classifier.classify_current_state()
        self.assertEqual(state, ClassroomAcousticState.CLASS_DISCUSSION)


if __name__ == "__main__":
    unittest.main()
