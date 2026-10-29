"""
AI-Driven Pedagogical Reasoning Engine.
Implements Scenario 3: Assistive decision-support agent providing
context-aware teaching suggestions based on detected engagement patterns.
Works 100% offline with optional Google Cloud Gemini Generative AI hook.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import time
import json
import os

from src.ai_agent.prompt_templates import (
    OFFLINE_RECOMMENDATION_CATALOG,
    GEMINI_TEACHING_AGENT_SYSTEM_PROMPT
)


@dataclass
class TeachingSuggestion:
    timestamp: float
    trigger_category: str  # 'ATTENTION_DROP', 'PACING_MONOLOGUE', 'INTERACTION_PEAK', 'DROWSY_INACTIVITY', 'OPTIMAL_FLOW'
    urgency_level: str     # 'LOW', 'MEDIUM', 'HIGH'
    suggestion_text: str
    rationale: str
    source: str            # 'OFFLINE_EXPERT_SYSTEM' or 'GOOGLE_CLOUD_GEMINI'


class PedagogicalReasoningAgent:
    """
    Evaluates multimodal classroom dynamics and outputs real-time, non-intrusive teaching advice.
    """

    def __init__(
        self,
        enable_cloud_llm: bool = False,
        gemini_api_key: Optional[str] = None,
        cooldown_seconds: float = 60.0
    ):
        self.enable_cloud_llm = enable_cloud_llm
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.cooldown_seconds = cooldown_seconds
        self.last_suggestion_time = 0.0
        self.suggestion_log: List[TeachingSuggestion] = []

    def evaluate_classroom_state(
        self,
        composite_engagement: float,
        visual_score: float,
        acoustic_state: str,
        trend_flag: Optional[str] = None,
        elapsed_seconds: float = 0.0,
        active_student_count: int = 0,
        current_time: Optional[float] = None
    ) -> Optional[TeachingSuggestion]:
        """
        Determines if a pedagogical intervention is warranted, respecting cooldown periods
        to avoid distracting the instructor.
        """
        if current_time is None:
            current_time = time.time()

        # Cooldown guard: Don't spam the teacher
        if (current_time - self.last_suggestion_time) < self.cooldown_seconds:
            return None

        trigger = None
        urgency = "LOW"
        rationale = ""

        if trend_flag == "ATTENTION_DROP" or composite_engagement < 42.0:
            trigger = "ATTENTION_DROP"
            urgency = "HIGH"
            rationale = f"Composite engagement dipped to {composite_engagement:.1f}% with low forward attention."
        elif trend_flag == "PACING_ISSUE" or (elapsed_seconds > 900 and acoustic_state == "LECTURE_MONOLOGUE"):
            trigger = "PACING_MONOLOGUE"
            urgency = "MEDIUM"
            rationale = "Continuous teacher monologue exceeding 15 minutes; student active interaction is low."
        elif trend_flag == "INTERACTION_PEAK" or (composite_engagement > 75.0 and acoustic_state == "CLASS_DISCUSSION"):
            trigger = "INTERACTION_PEAK"
            urgency = "LOW"
            rationale = "High student vocal participation and active visual synergy detected."
        elif visual_score < 35.0:
            trigger = "DROWSY_INACTIVITY"
            urgency = "MEDIUM"
            rationale = "Physical stillness or downward gaze detected across multiple student positions."
        elif elapsed_seconds > 0 and (elapsed_seconds % 600 < 5):
            # Checkpoint every 10 minutes if running fine
            trigger = "OPTIMAL_FLOW"
            urgency = "LOW"
            rationale = "Routine 10-minute session pacing checkpoint."

        if not trigger:
            return None

        # Generate recommendation
        suggestion_text = self._generate_advice(trigger, composite_engagement, acoustic_state)
        source = "OFFLINE_EXPERT_SYSTEM"

        # Check for Google Cloud Gemini Generative AI hook
        if self.enable_cloud_llm and self.gemini_api_key:
            cloud_advice = self._call_gemini_api(trigger, composite_engagement, acoustic_state)
            if cloud_advice:
                suggestion_text = cloud_advice
                source = "GOOGLE_CLOUD_GEMINI"

        suggestion = TeachingSuggestion(
            timestamp=current_time,
            trigger_category=trigger,
            urgency_level=urgency,
            suggestion_text=suggestion_text,
            rationale=rationale,
            source=source
        )

        self.last_suggestion_time = current_time
        self.suggestion_log.append(suggestion)
        return suggestion

    def _generate_advice(self, trigger: str, engagement: float, acoustic: str) -> str:
        """Pulls from contextually structured offline expert suggestions."""
        catalog = OFFLINE_RECOMMENDATION_CATALOG.get(trigger, OFFLINE_RECOMMENDATION_CATALOG["OPTIMAL_FLOW"])
        return random.choice(catalog)

    def _call_gemini_api(self, trigger: str, engagement: float, acoustic: str) -> Optional[str]:
        """
        Optional cloud generative AI integration for Google Cloud Vertex AI / Gemini API.
        Falls back to offline gracefully on network/auth failure.
        """
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            prompt = (
                f"{GEMINI_TEACHING_AGENT_SYSTEM_PROMPT}\n"
                f"Current Classroom State:\n"
                f"- Trigger: {trigger}\n"
                f"- Composite Engagement: {engagement:.1f}%\n"
                f"- Acoustic Dynamic: {acoustic}\n"
                f"Provide 1 concise, actionable teaching suggestion (under 30 words)."
            )
            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode())
                return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return None

    def get_recent_suggestions(self, limit: int = 5) -> List[Dict]:
        """Returns recent suggestions for UI display."""
        recent = self.suggestion_log[-limit:] if self.suggestion_log else []
        return [
            {
                "time": time.strftime("%H:%M:%S", time.localtime(s.timestamp)),
                "category": s.trigger_category,
                "urgency": s.urgency_level,
                "suggestion": s.suggestion_text,
                "rationale": s.rationale,
                "source": s.source
            }
            for s in reversed(recent)
        ]
