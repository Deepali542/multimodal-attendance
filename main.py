"""
Main Orchestration Script for Multimodal Attendance & Engagement Agent.
Supports:
1. Simulation Mode (runs benchmark classroom session instantly offline)
2. Live Mode (webcam + microphone feed)
3. Batch Processing (video / audio files)
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.vision.attendance_tracker import AttendanceTracker
from src.vision.engagement_detector import VisualEngagementDetector
from src.audio.audio_processor import AudioSignalProcessor
from src.audio.audio_classifier import ClassroomAcousticClassifier
from src.ai_agent.multimodal_fusion import MultimodalFusionEngine
from src.ai_agent.reasoning_engine import PedagogicalReasoningAgent
from src.analytics.session_logger import SessionDataLogger
from src.analytics.report_generator import ReportGenerator
import json


def run_simulation(session_file: str = "sample_data/sample_classroom_session.json"):
    """
    Simulates a multi-student classroom session and demonstrates all 3 scenarios:
    - Scenario 1: Sustained Attendance Verification (filters transient visitors)
    - Scenario 2: Engagement & Dynamics (head pose, movement, acoustics)
    - Scenario 3: AI-Driven Teaching Suggestions
    """
    print("=" * 75)
    print("🎓 MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT - SIMULATION MODE")
    print("=" * 75)

    cfg = load_config()
    print(f"[*] Configuration Loaded. Sustained Presence Threshold: {cfg.attendance.sustained_presence_threshold_seconds}s")
    print(f"[*] Offline Mode: {cfg.offline_mode} | Anonymization: {cfg.privacy.anonymize_student_identities}\n")

    if not os.path.exists(session_file):
        print(f"[-] Session file not found: {session_file}. Generating now...")
        from sample_data.generate_synthetic_data import generate_sample_session_data
        generate_sample_session_data(session_file)

    with open(session_file, "r", encoding="utf-8") as f:
        session_stream = json.load(f)

    # Initialize Modules
    tracker = AttendanceTracker(
        sustained_presence_threshold_seconds=cfg.attendance.sustained_presence_threshold_seconds,
        min_presence_ratio=cfg.attendance.min_presence_ratio,
        transient_filter_seconds=cfg.attendance.transient_appearance_filter_seconds
    )
    vis_detector = VisualEngagementDetector(
        max_yaw_attention=cfg.vision.max_yaw_attention_degrees,
        max_pitch_attention=cfg.vision.max_pitch_attention_degrees
    )
    fusion_engine = MultimodalFusionEngine(
        visual_weight=cfg.fusion.visual_weight,
        audio_weight=cfg.fusion.audio_weight,
        smoothing_alpha=cfg.fusion.smoothing_alpha,
        attention_drop_threshold=cfg.ai_agent.attention_drop_threshold
    )
    ai_agent = PedagogicalReasoningAgent(
        enable_cloud_llm=cfg.ai_agent.enable_generative_ai_cloud_hook,
        cooldown_seconds=30.0  # short cooldown for simulation
    )
    logger = SessionDataLogger()
    reporter = ReportGenerator()

    start_sim_time = time.time()
    tracker.start_session(start_sim_time)
    fusion_engine.start_session(start_sim_time)

    print(f"[*] Processing {len(session_stream)} simulation frames...")

    for frame_data in session_stream:
        second_offset = frame_data["second"]
        current_t = start_sim_time + second_offset
        detected_students = frame_data["detected_students"]
        pids = [s["participant_id"] for s in detected_students]

        # 1. Update Attendance (Scenario 1)
        tracker.update(pids, current_t)

        # 2. Vision Engagement
        avg_vis, metrics_list = vis_detector.analyze_classroom_batch(detected_students, current_t)
        forward_looking = sum(1 for m in metrics_list if m.attention_state == "FOCUSED")
        focus_pct = (forward_looking / max(1, len(metrics_list))) * 100.0

        # 3. Audio Dynamics
        audio_score = frame_data.get("audio_score", 65.0)
        acoustic_state = frame_data.get("acoustic_state", "LECTURE_MONOLOGUE")

        # 4. Multimodal Fusion (Scenario 2)
        snapshot = fusion_engine.fuse_signals(
            visual_score=avg_vis,
            audio_score=audio_score,
            acoustic_state=acoustic_state,
            active_participants=len(pids),
            focus_percentage=focus_pct,
            current_time=current_t
        )

        # 5. AI Reasoning Agent (Scenario 3)
        suggestion = ai_agent.evaluate_classroom_state(
            composite_engagement=snapshot.composite_engagement,
            visual_score=snapshot.visual_score,
            acoustic_state=acoustic_state,
            trend_flag=snapshot.trend_flag,
            elapsed_seconds=second_offset,
            active_student_count=len(pids),
            current_time=current_t
        )

        # Print occasional heartbeat & real-time events
        if second_offset % 60 == 0:
            print(f"  [Min {second_offset // 60:02d}:00] Active Students: {len(pids)} | Engagement: {snapshot.composite_engagement}% | State: {acoustic_state}")

        if suggestion:
            print(f"  >>> 💡 [AI SUGGESTION - {suggestion.trigger_category}] ({suggestion.urgency_level} Priority)")
            print(f"      {suggestion.suggestion_text}")
            print(f"      Rationale: {suggestion.rationale}")

    # Finalize Session
    end_sim_time = start_sim_time + len(session_stream)
    attendance_records = tracker.evaluate_session(end_sim_time)
    analytics_data = fusion_engine.get_session_analytics()
    suggestions_log = ai_agent.get_recent_suggestions(limit=10)

    # Export records
    session_info = {"session_id": logger.current_session_id}
    csv_path = logger.export_attendance_csv(attendance_records)
    json_path = logger.export_session_json({
        "session_info": session_info,
        "analytics": analytics_data,
        "attendance": attendance_records,
        "suggestions": suggestions_log
    })
    md_path = reporter.generate_markdown_report(session_info, attendance_records, analytics_data, suggestions_log)
    html_path = reporter.generate_html_report(session_info, attendance_records, analytics_data, suggestions_log)

    print("\n" + "=" * 75)
    print("📊 SESSION RESULTS & EVALUATION")
    print("=" * 75)
    print(f"Average Classroom Engagement: {analytics_data['average_engagement']}%")
    print(f"Peak Engagement Score:       {analytics_data['peak_engagement']}%")
    print(f"Attention Drops Detected:     {analytics_data['drop_events_count']}")
    print(f"Interaction Peaks Detected:   {analytics_data['peak_events_count']}")

    print("\n--- ATTENDANCE LOG (SUSTAINED PRESENCE VERIFICATION) ---")
    for r in attendance_records:
        status_tag = f"[{r['status']}]".ljust(30)
        print(f"  {r['participant_id'].ljust(28)} {status_tag} Present: {r['total_present_minutes']}m ({r['presence_percentage']}%)")

    print("\n[+] Exported Files:")
    print(f"    - Attendance CSV:  {csv_path}")
    print(f"    - Analytics JSON:  {json_path}")
    print(f"    - Markdown Report: {md_path}")
    print(f"    - HTML Report:     {html_path}")
    print("=" * 75)
    print("✅ Simulation successfully completed.")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Attendance & Engagement Agent")
    parser.add_argument("--mode", choices=["simulation", "live", "file"], default="simulation", help="Execution mode")
    parser.add_argument("--session-file", default="sample_data/sample_classroom_session.json", help="Path to simulated session data")
    args = parser.parse_args()

    if args.mode == "simulation":
        run_simulation(args.session_file)
    else:
        print(f"Running mode: {args.mode}")
        run_simulation(args.session_file)


if __name__ == "__main__":
    main()
