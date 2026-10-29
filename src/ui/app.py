"""
Interactive Web Dashboard for Multimodal Attendance & Engagement Agent.
Built with Streamlit. Provides real-time classroom telemetry,
attendance matrices, engagement dynamics graphs, and AI teaching suggestions.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from src.config import load_config
from src.vision.attendance_tracker import AttendanceTracker
from src.vision.engagement_detector import VisualEngagementDetector
from src.ai_agent.multimodal_fusion import MultimodalFusionEngine
from src.ai_agent.reasoning_engine import PedagogicalReasoningAgent
from src.analytics.report_generator import ReportGenerator


def run_streamlit_app():
    if not HAS_STREAMLIT:
        print("[!] Streamlit is not installed. Run 'pip install streamlit' or use main.py")
        return

    st.set_page_config(
        page_title="Classroom Intelligence Agent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎓 Multimodal Attendance & Engagement Agent")
    st.caption("AI-Powered Privacy-Preserving Classroom Intelligence System (Offline Capable)")

    cfg = load_config()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio("Session Mode", ["Benchmark Simulation", "Live Webcam & Mic (Beta)"])
    sustained_thresh = st.sidebar.slider("Sustained Presence (sec)", 30, 300, int(cfg.attendance.sustained_presence_threshold_seconds))
    attention_thresh = st.sidebar.slider("Attention Drop Threshold (%)", 30, 60, int(cfg.ai_agent.attention_drop_threshold))
    enable_cloud = st.sidebar.checkbox("Enable Google Cloud Gemini Hook", value=cfg.ai_agent.enable_generative_ai_cloud_hook)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Privacy Guardrails Active:**\n"
        "- Zero unmasked cloud transmission\n"
        "- Ephemeral face bounding tracking\n"
        "- Anonymized student identifiers"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Dynamics & AI Suggestions",
        "📋 Sustained Attendance Records",
        "📈 Temporal Engagement Analytics",
        "📄 Exported Reports"
    ])

    sample_json = ROOT_DIR / "sample_data" / "sample_classroom_session.json"
    if not sample_json.exists():
        from sample_data.generate_synthetic_data import generate_sample_session_data
        generate_sample_session_data(str(sample_json))

    with open(sample_json, "r", encoding="utf-8") as f:
        stream_data = json.load(f)

    with tab1:
        col_ctrl1, col_ctrl2 = st.columns([1, 4])
        with col_ctrl1:
            run_btn = st.button("▶️ Run Session Analysis", use_container_width=True)

        if run_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            kpi_cols = st.columns(4)

            tracker = AttendanceTracker(sustained_presence_threshold_seconds=sustained_thresh)
            vis_det = VisualEngagementDetector()
            fusion = MultimodalFusionEngine(attention_drop_threshold=attention_thresh)
            agent = PedagogicalReasoningAgent(enable_cloud_llm=enable_cloud, cooldown_seconds=20.0)

            t0 = time.time()
            tracker.start_session(t0)
            fusion.start_session(t0)

            chart_placeholder = st.empty()
            ai_box = st.empty()

            eng_history = []

            step_jump = max(1, len(stream_data) // 50)
            for i, frame in enumerate(stream_data[::step_jump]):
                sec = frame["second"]
                cur_t = t0 + sec
                pids = [s["participant_id"] for s in frame["detected_students"]]

                tracker.update(pids, cur_t)
                avg_vis, _ = vis_det.analyze_classroom_batch(frame["detected_students"], cur_t)
                snapshot = fusion.fuse_signals(
                    visual_score=avg_vis,
                    audio_score=frame.get("audio_score", 60.0),
                    acoustic_state=frame.get("acoustic_state", "LECTURE_MONOLOGUE"),
                    active_participants=len(pids),
                    focus_percentage=80.0,
                    current_time=cur_t
                )
                eng_history.append(snapshot.composite_engagement)

                sugg = agent.evaluate_classroom_state(
                    composite_engagement=snapshot.composite_engagement,
                    visual_score=snapshot.visual_score,
                    acoustic_state=frame.get("acoustic_state", "LECTURE_MONOLOGUE"),
                    trend_flag=snapshot.trend_flag,
                    elapsed_seconds=sec,
                    current_time=cur_t
                )

                progress_bar.progress(min(1.0, (i + 1) / (len(stream_data) // step_jump)))
                status_text.text(f"Processing Second {sec}/300 | Current Acoustic State: {frame.get('acoustic_state')}")

                with kpi_cols[0]:
                    st.metric("Live Engagement", f"{snapshot.composite_engagement:.1f}%")
                with kpi_cols[1]:
                    st.metric("Visual Focus", f"{snapshot.visual_score:.1f}%")
                with kpi_cols[2]:
                    st.metric("Active Faces", len(pids))
                with kpi_cols[3]:
                    st.metric("Acoustic State", frame.get("acoustic_state")[:14])

                if sugg:
                    with ai_box.container():
                        st.warning(f"**AI Suggestion ({sugg.trigger_category}):** {sugg.suggestion_text}\n\n*Rationale:* {sugg.rationale}")

            st.success("Session Analysis Complete!")
            st.session_state["evaluated_attendance"] = tracker.evaluate_session(t0 + 300)
            st.session_state["session_analytics"] = fusion.get_session_analytics()
            st.session_state["suggestions_log"] = agent.get_recent_suggestions(limit=10)

    with tab2:
        st.subheader("Scenario 1: Sustained Attendance Verification Matrix")
        st.markdown(
            "Attendance is determined by **sustained presence** across the session. "
            "Passers-by with brief appearances (< 30s) are automatically ignored."
        )
        if "evaluated_attendance" in st.session_state:
            st.dataframe(st.session_state["evaluated_attendance"], use_container_width=True)
        else:
            st.info("Click 'Run Session Analysis' in Tab 1 to generate live attendance.")

    with tab3:
        st.subheader("Scenario 2: Multimodal Trends & Dynamics")
        st.markdown("Temporal tracking of student engagement, attention drop detection, and discussion peaks.")
        if "session_analytics" in st.session_state:
            st.json(st.session_state["session_analytics"])
        else:
            st.info("Run session analysis to populate analytics.")

    with tab4:
        st.subheader("Scenario 3: AI Teaching Insights & Report Export")
        if "suggestions_log" in st.session_state:
            for s in st.session_state["suggestions_log"]:
                st.info(f"**[{s['time']}] {s['category']}** ({s['urgency']} Urgency)\n\n{s['suggestion']}\n\n*Reason:* {s['rationale']}")
        else:
            st.info("No suggestions generated yet.")


if __name__ == "__main__":
    run_streamlit_app()
