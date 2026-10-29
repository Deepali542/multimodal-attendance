"""
Configuration Loader for Multimodal Attendance & Engagement Agent.
Provides strongly-typed configuration settings with automatic fallback.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Any

try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class AttendanceConfig:
    sustained_presence_threshold_seconds: float = 180.0
    min_presence_ratio: float = 0.60
    tracking_window_seconds: float = 10.0
    transient_appearance_filter_seconds: float = 30.0


@dataclass
class VisionConfig:
    detector_type: str = "haar"
    face_confidence_threshold: float = 0.65
    max_yaw_attention_degrees: float = 30.0
    max_pitch_attention_degrees: float = 25.0
    inactivity_threshold_seconds: float = 120.0
    restlessness_motion_threshold: float = 0.45


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_duration_seconds: float = 1.0
    energy_rms_speech_threshold: float = 0.035
    silence_threshold_db: float = -40.0
    chatter_noise_threshold: float = 0.15


@dataclass
class FusionConfig:
    visual_weight: float = 0.60
    audio_weight: float = 0.40
    smoothing_alpha: float = 0.35


@dataclass
class AIAgentConfig:
    attention_drop_threshold: float = 45.0
    attention_drop_duration_seconds: float = 120.0
    monologue_warning_seconds: float = 900.0
    high_energy_discussion_threshold: float = 75.0
    enable_generative_ai_cloud_hook: bool = False
    model_name: str = "gemini-1.5-flash"


@dataclass
class PrivacyConfig:
    anonymize_student_identities: bool = True
    hash_student_ids: bool = True
    save_raw_video_frames: bool = False


@dataclass
class AppConfig:
    project_name: str = "Multimodal Attendance & Engagement Agent"
    version: str = "1.0.0"
    offline_mode: bool = True
    log_level: str = "INFO"
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    ai_agent: AIAgentConfig = field(default_factory=AIAgentConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def load_config(config_path: str = None) -> AppConfig:
    """Loads configuration from yaml file or returns defaults."""
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "configs" / "config.yaml"

    cfg = AppConfig()

    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if "attendance" in data:
                cfg.attendance = AttendanceConfig(**data["attendance"])
            if "vision" in data:
                v = data["vision"]
                cfg.vision = VisionConfig(
                    detector_type=v.get("detector_type", "haar"),
                    face_confidence_threshold=v.get("face_confidence_threshold", 0.65),
                    max_yaw_attention_degrees=v.get("head_pose", {}).get("max_yaw_attention_degrees", 30.0),
                    max_pitch_attention_degrees=v.get("head_pose", {}).get("max_pitch_attention_degrees", 25.0),
                    inactivity_threshold_seconds=v.get("movement", {}).get("inactivity_threshold_seconds", 120.0),
                    restlessness_motion_threshold=v.get("movement", {}).get("restlessness_motion_threshold", 0.45)
                )
            if "audio" in data:
                cfg.audio = AudioConfig(**data["audio"])
            if "fusion" in data:
                cfg.fusion = FusionConfig(**data["fusion"])
            if "ai_agent" in data:
                cfg.ai_agent = AIAgentConfig(**data["ai_agent"])
            if "privacy" in data:
                cfg.privacy = PrivacyConfig(**data["privacy"])
            if "system" in data:
                s = data["system"]
                cfg.project_name = s.get("project_name", cfg.project_name)
                cfg.version = s.get("version", cfg.version)
                cfg.offline_mode = s.get("offline_mode", cfg.offline_mode)
                cfg.log_level = s.get("log_level", cfg.log_level)
        except Exception as e:
            print(f"[Config] Warning: Failed to load YAML config ({e}). Using robust defaults.")

    return cfg
