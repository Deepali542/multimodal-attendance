"""
AI Agent Prompt Templates and Heuristic Rule Bases.
Defines structured prompts for Generative AI (Google Cloud Gemini)
and contextual rule criteria for offline pedagogical assistance.
"""

GEMINI_TEACHING_AGENT_SYSTEM_PROMPT = """
You are an expert AI Pedagogical Assistant embedded in a smart classroom system.
Your goal is to support educators with real-time, non-intrusive decision support.

Context provided:
- Visual Engagement (% of students looking forward, movement quality)
- Audio Dynamics (acoustic state: lecture monologue, class discussion, quiet study, or noise)
- Sustained Attendance Count
- Session Duration and Temporal Trends (attention drop, interaction peak, pacing fatigue)

Rules:
1. Be concise, actionable, and encouraging (1-2 sentences max per suggestion).
2. Never enforce commands; use assistive phrasing ("Consider...", "You might want to...", "Good moment to...").
3. Focus on active learning techniques (polls, think-pair-share, targeted inquiry).
"""

OFFLINE_RECOMMENDATION_CATALOG = {
    "ATTENTION_DROP": [
        "📉 Classroom focus has slipped below 45% over the last few minutes. Consider pausing for a quick concept-check question or 30-second peer discussion.",
        "💡 Attention dip detected. Try switching the presentation medium (e.g., switch to whiteboard, live code demonstration, or short poll).",
        "🙋 Engagement is tapering off. Ask an open-ended question to a student in the middle rows to re-energize the room."
    ],
    "PACING_MONOLOGUE": [
        "⏳ You have been lecturing continuously for over 15 minutes. Consider breaking the monologue with a 'Think-Pair-Share' prompt.",
        "⏱️ Pacing check: High speech density with declining visual movement. Great moment to verify if students are following along.",
        "📝 Continuous monologue detected. Pause for 60 seconds to let students note down core takeaways and formulate questions."
    ],
    "INTERACTION_PEAK": [
        "🔥 Excellent interaction peak! Students are actively participating. Synthesize their main points before transitioning.",
        "✨ High classroom synergy detected. Document student ideas on the board to reinforce active contribution.",
        "👏 Discussion energy is at 80%+. Solid participation window—encourage a quieter student to share thoughts."
    ],
    "DROWSY_INACTIVITY": [
        "💤 Several participants are showing prolonged inactivity/low posture. A quick stand-up stretch or interactive quiz will restore alertness.",
        "☕ Low physical motion detected across rows. Consider a hands-on exercise or calling for group hands."
    ],
    "OPTIMAL_FLOW": [
        "✅ Classroom flow is well-balanced. Students are attentive and pacing is aligned with active learning targets.",
        "🎯 Steady engagement detected across both visual gaze and acoustic feedback."
    ]
}
