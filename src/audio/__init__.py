"""
Audio Module for Acoustic Analysis and Dynamics Tracking.
"""

from src.audio.audio_processor import AudioSignalProcessor, AudioFeatures
from src.audio.audio_classifier import ClassroomAcousticClassifier, ClassroomAcousticState

__all__ = [
    "AudioSignalProcessor",
    "AudioFeatures",
    "ClassroomAcousticClassifier",
    "ClassroomAcousticState"
]
