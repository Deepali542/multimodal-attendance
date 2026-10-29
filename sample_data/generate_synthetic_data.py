"""
Synthetic Data Generator for Classroom Simulation.
Generates realistic multi-student sessions with:
- 5 consistent students (present > 80% time)
- 2 transient passer-bys (present < 25 seconds)
- Attention drop period (minute 2 to 3)
- Interaction peak period (minute 4)
- Mixed acoustic states (Lecture -> Discussion -> Silence)
"""

import json
import os
import math
import wave
import struct

def generate_sample_session_data(filepath: str = "sample_data/sample_classroom_session.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    session_duration_sec = 300  # 5 minute demo session
    data = []

    # Student definitions:
    # Student_01 to Student_05: regular sustained students
    # Passerby_A: appears only at sec 40 to 60 (20s) -> should be TRANSIENT_IGNORED
    # Passerby_B: appears only at sec 180 to 195 (15s) -> should be TRANSIENT_IGNORED

    for sec in range(session_duration_sec):
        # Current time offset
        t = sec

        active_students = []

        # Sustained regular students
        for sid in range(1, 6):
            student_id = f"Student_{sid:02d}"

            # Simulate attention drop around sec 120-180
            if 120 <= t <= 180:
                yaw = 35.0 + 5.0 * math.sin(t * 0.1)  # Distracted looking away
                pitch = 28.0
            else:
                yaw = 5.0 * math.sin(t * 0.2 + sid)   # Attentive front facing
                pitch = 3.0 * math.cos(t * 0.1)

            active_students.append({
                "participant_id": student_id,
                "bbox": [100 * sid, 150, 60, 60],
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1)
            })

        # Transient passers-by
        if 40 <= t <= 60:
            active_students.append({
                "participant_id": "Visitor_Walking_Hallway",
                "bbox": [50, 100, 50, 50],
                "yaw": 45.0,
                "pitch": 0.0
            })
        if 180 <= t <= 195:
            active_students.append({
                "participant_id": "Late_Courier_Passerby",
                "bbox": [550, 110, 50, 50],
                "yaw": -50.0,
                "pitch": 10.0
            })

        # Audio dynamics
        if 120 <= t <= 180:
            # Drop period: silence / whisper
            rms = 0.015
            acoustic_state = "QUIET_STUDY"
            audio_score = 42.0
        elif 220 <= t <= 270:
            # Active Discussion Peak
            rms = 0.08
            acoustic_state = "CLASS_DISCUSSION"
            audio_score = 88.0
        else:
            # Lecture
            rms = 0.045
            acoustic_state = "LECTURE_MONOLOGUE"
            audio_score = 72.0

        data.append({
            "second": sec,
            "detected_students": active_students,
            "audio_rms": rms,
            "acoustic_state": acoustic_state,
            "audio_score": audio_score
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated synthetic session data at: {filepath}")
    return filepath


def generate_sample_wav(filepath: str = "sample_data/sample_classroom.wav", duration_sec: int = 10):
    """Generates a small test WAV audio file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sample_rate = 16000
    n_samples = sample_rate * duration_sec

    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        samples = []
        for i in range(n_samples):
            # 440 Hz tone with amplitude envelope
            val = int(32767.0 * 0.2 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            samples.append(struct.pack("<h", val))

        wf.writeframes(b"".join(samples))

    print(f"Generated sample WAV at: {filepath}")
    return filepath


if __name__ == "__main__":
    generate_sample_session_data()
    generate_sample_wav()
