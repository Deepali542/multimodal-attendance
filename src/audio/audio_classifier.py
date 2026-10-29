"""
Classroom Acoustic State Classifier.
Categorizes classroom dynamics: Lecture Monologue, Interactive Discussion,
Quiet Study / Inactive, or Ambient Noise / Disturbance.
"""

from enum import Enum
from typing import List
from src.audio.audio_processor import AudioFeatures


class ClassroomAcousticState(str, Enum):
    LECTURE_MONOLOGUE = "LECTURE_MONOLOGUE"
    CLASS_DISCUSSION = "CLASS_DISCUSSION"
    QUIET_STUDY = "QUIET_STUDY"
    NOISY_DISTURBANCE = "NOISY_DISTURBANCE"


class ClassroomAcousticClassifier:
    """
    Evaluates temporal trends of audio features across a rolling window
    (e.g., 30 to 60 seconds) to determine acoustic state.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history: List[AudioFeatures] = []

    def add_feature(self, feat: AudioFeatures):
        self.history.append(feat)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def classify_current_state(self) -> ClassroomAcousticState:
        """
        Infers acoustic state from accumulated window history.
        """
        if not self.history:
            return ClassroomAcousticState.QUIET_STUDY

        speech_count = sum(1 for f in self.history if f.is_speech)
        speech_ratio = speech_count / len(self.history)
        avg_rms = sum(f.rms_energy for f in self.history) / len(self.history)
        avg_silence = sum(f.silence_ratio for f in self.history) / len(self.history)

        # Disturbance / high ambient noise without structured pauses
        if avg_rms > 0.12 and speech_ratio > 0.85 and avg_silence < 0.10:
            return ClassroomAcousticState.NOISY_DISTURBANCE

        # Active Classroom Discussion: alternating speech with interactive rhythm
        if speech_ratio >= 0.40 and avg_silence >= 0.20 and avg_silence <= 0.60:
            return ClassroomAcousticState.CLASS_DISCUSSION

        # Dominant Lecture Monologue: steady speech with low student turn-taking
        if speech_ratio > 0.50 and avg_silence < 0.25:
            return ClassroomAcousticState.LECTURE_MONOLOGUE

        # Quiet study or silent attention
        if speech_ratio < 0.25:
            return ClassroomAcousticState.QUIET_STUDY

        return ClassroomAcousticState.LECTURE_MONOLOGUE
