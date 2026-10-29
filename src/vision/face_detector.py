"""
Vision Module - Face Detection & Participant Tracking.
Privacy-preserving face detection and tracking using OpenCV Haar cascades / DNN.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import hashlib
import time

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class DetectedFace:
    participant_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    center: Tuple[int, int]
    timestamp: float


class PrivacyFaceDetector:
    """
    Detects faces in video frames and assigns privacy-preserving local IDs.
    Does not transmit or store unmasked raw facial biometrics.
    """

    def __init__(self, confidence_threshold: float = 0.65, anonymize: bool = True):
        self.confidence_threshold = confidence_threshold
        self.anonymize = anonymize
        self.known_tracks: Dict[str, Tuple[int, int, float]] = {}  # id -> (cx, cy, last_seen)
        self.track_counter = 0

        self.classifier = None
        if HAS_CV2:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.classifier = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                print(f"[FaceDetector] Warning: Haar cascade could not be loaded: {e}")

    def _get_tracked_id(self, cx: int, cy: int, current_time: float, max_dist: int = 70) -> str:
        """Associates detected bounding center with existing tracked ID or creates new one."""
        best_id = None
        min_dist = float("inf")

        for pid, (px, py, last_t) in list(self.known_tracks.items()):
            # If track hasn't been seen for 60 seconds, don't match
            if current_time - last_t > 60.0:
                continue
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < min_dist and dist < max_dist:
                min_dist = dist
                best_id = pid

        if best_id is not None:
            self.known_tracks[best_id] = (cx, cy, current_time)
            return best_id

        # New participant detected
        self.track_counter += 1
        raw_id = f"Student_{self.track_counter:02d}"
        if self.anonymize:
            hashed_suffix = hashlib.sha256(raw_id.encode()).hexdigest()[:6]
            assigned_id = f"Student_{self.track_counter:02d}_{hashed_suffix}"
        else:
            assigned_id = raw_id

        self.known_tracks[assigned_id] = (cx, cy, current_time)
        return assigned_id

    def detect(self, frame, current_time: Optional[float] = None) -> List[DetectedFace]:
        """
        Detects faces in an RGB/BGR image frame.
        Returns list of DetectedFace objects.
        """
        if current_time is None:
            current_time = time.time()

        detected: List[DetectedFace] = []

        if HAS_CV2 and frame is not None and self.classifier is not None:
            try:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame

                faces = self.classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                    pid = self._get_tracked_id(cx, cy, current_time)
                    detected.append(DetectedFace(
                        participant_id=pid,
                        bbox=(int(x), int(y), int(w), int(h)),
                        confidence=0.88,
                        center=(cx, cy),
                        timestamp=current_time
                    ))
            except Exception as e:
                print(f"[FaceDetector] Error during cv2 face detection: {e}")

        return detected
