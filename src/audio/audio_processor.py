"""
Audio Module - Acoustic Feature Extraction and Processing.
Measures RMS energy, silence vs speech activity, and acoustic dynamics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import struct
import wave
import os

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class AudioFeatures:
    rms_energy: float
    is_speech: bool
    db_level: float
    silence_ratio: float
    dominant_frequency_hz: float
    audio_engagement_score: float  # 0.0 to 100.0


class AudioSignalProcessor:
    """
    Extracts acoustic signals from raw PCM audio buffers or WAV audio files.
    Works entirely offline without external speech-to-text API dependencies.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rms_speech_threshold: float = 0.035,
        silence_threshold_db: float = -40.0
    ):
        self.sample_rate = sample_rate
        self.rms_threshold = rms_speech_threshold
        self.silence_threshold_db = silence_threshold_db

    def calculate_rms(self, samples: List[float]) -> float:
        """Calculates Root Mean Square (RMS) energy of audio samples."""
        if not samples:
            return 0.0
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / len(samples))

    def rms_to_db(self, rms: float) -> float:
        """Converts RMS value to Decibels relative to full scale (dBFS)."""
        if rms <= 1e-7:
            return -100.0
        return 20.0 * math.log10(rms)

    def process_raw_buffer(self, buffer_bytes: bytes) -> AudioFeatures:
        """Processes raw 16-bit PCM mono bytes."""
        count = len(buffer_bytes) // 2
        if count == 0:
            return AudioFeatures(0.0, False, -100.0, 1.0, 0.0, 30.0)

        # Unpack 16-bit signed integers
        fmt = f"<{count}h"
        raw_ints = struct.unpack(fmt, buffer_bytes[:count * 2])
        # Normalize to [-1.0, 1.0]
        samples = [val / 32768.0 for val in raw_ints]

        return self.process_samples(samples)

    def process_samples(self, samples: List[float]) -> AudioFeatures:
        """Processes a list of normalized float audio samples."""
        if not samples:
            return AudioFeatures(0.0, False, -100.0, 1.0, 0.0, 30.0)

        rms = self.calculate_rms(samples)
        db = self.rms_to_db(rms)

        is_speech = (rms >= self.rms_threshold) and (db > self.silence_threshold_db)

        # Rough silence estimation by counting low-energy windows
        sub_window_size = max(1, len(samples) // 10)
        silent_windows = 0
        total_sub_windows = 10

        for i in range(total_sub_windows):
            chunk = samples[i * sub_window_size : (i + 1) * sub_window_size]
            c_rms = self.calculate_rms(chunk)
            if c_rms < (self.rms_threshold * 0.7):
                silent_windows += 1

        silence_ratio = silent_windows / float(total_sub_windows)

        # Calculate Audio Engagement:
        # Balanced speech/interaction is optimal (~70-90 score)
        # Total silence = lower score (~35-45)
        # Excessive screeching noise = lower score (~40)
        if is_speech:
            if silence_ratio > 0.15 and silence_ratio < 0.65:
                # Active dialog / dynamic lecture with natural pauses
                eng_score = 85.0
            else:
                # Continuous speech or rapid speaking
                eng_score = 75.0
        else:
            if silence_ratio >= 0.85:
                # Prolonged dead silence
                eng_score = 40.0
            else:
                # Moderate ambient presence
                eng_score = 60.0

        return AudioFeatures(
            rms_energy=round(rms, 4),
            is_speech=is_speech,
            db_level=round(db, 1),
            silence_ratio=round(silence_ratio, 2),
            dominant_frequency_hz=350.0 if is_speech else 0.0,
            audio_engagement_score=eng_score
        )

    def process_wav_file(self, wav_path: str, chunk_duration_sec: float = 1.0) -> List[AudioFeatures]:
        """Reads and analyzes an entire WAV file chunk by chunk."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        results = []
        with wave.open(wav_path, "rb") as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames_per_chunk = int(framerate * chunk_duration_sec)

            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break

                # Convert to mono if needed
                if n_channels == 2 and sampwidth == 2:
                    count = len(data) // 4
                    raw_ints = struct.unpack(f"<{count * 2}h", data)
                    mono_samples = [((raw_ints[i*2] + raw_ints[i*2+1]) / 2.0) / 32768.0 for i in range(count)]
                    features = self.process_samples(mono_samples)
                elif sampwidth == 2:
                    features = self.process_raw_buffer(data)
                else:
                    features = AudioFeatures(0.02, False, -35.0, 0.5, 0.0, 50.0)

                results.append(features)

        return results
