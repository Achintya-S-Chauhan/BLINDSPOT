from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any
from understanding import WorkflowUnderstanding, format_duration


@dataclass
class CompanionResponse:
    """
    Structured companion response representing a context-aware insight,
    recommendation, or summary for the user.
    """
    message: str
    category: str  # "focus", "resumption", "transition", "iteration", "idle", "general"
    confidence: int  # 1 (low) to 3 (high)
    supporting_context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize companion response into a dictionary for future AI/UI consumption."""
        return {
            "message": self.message,
            "category": self.category,
            "confidence": self.confidence,
            "supporting_context": self.supporting_context,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class CompanionAssistant:
    """
    Lightweight companion assistant that translates structured workflow
    understanding into concise, context-aware responses.
    """

    def respond(self, understanding: WorkflowUnderstanding) -> CompanionResponse:
        """
        Generate a structured companion response based on WorkflowUnderstanding.
        """
        act = understanding.current_activity
        dur = understanding.current_duration
        dur_str = format_duration(dur)
        pattern = understanding.pattern_type
        flow = understanding.recent_activities
        confidence = understanding.confidence

        supporting_context = {
            "current_activity": act,
            "current_window": understanding.current_window,
            "current_duration": dur,
            "pattern_type": pattern,
            "flow": flow,
            "transitions_count": understanding.recent_transitions_count,
        }

        # Case 0: Idle / No active context
        if pattern == "no_activity" or not act:
            return CompanionResponse(
                message="No active desktop activity detected right now.",
                category="idle",
                confidence=1,
                supporting_context=supporting_context,
                timestamp=understanding.timestamp,
            )

        # Case 1: Focused session
        if pattern == "focused_session":
            if dur >= 1800:  # 30+ minutes
                message = f"You've stayed deeply focused on {act} for {dur_str}."
            else:
                message = f"You've stayed focused on the same {act} activity."

            return CompanionResponse(
                message=message,
                category="focus",
                confidence=confidence,
                supporting_context=supporting_context,
                timestamp=understanding.timestamp,
            )

        # Case 2: Resumed task (A -> B -> A)
        if pattern == "resumed_task":
            interrupted_by = understanding.metadata.get("interrupted_by")
            if interrupted_by:
                if any(kw in interrupted_by.lower() for kw in ("browser", "browsing", "reading")):
                    message = f"You're back to your {act} task after a short research session."
                else:
                    message = f"You're back to your {act} task after a short {interrupted_by} session."
            else:
                message = f"You've returned to your {act} task."

            return CompanionResponse(
                message=message,
                category="resumption",
                confidence=confidence,
                supporting_context=supporting_context,
                timestamp=understanding.timestamp,
            )

        # Case 3: Workflow Shift (A -> B or A -> B -> C)
        if pattern == "workflow_shift":
            if len(flow) >= 2:
                prev_act = flow[-2]
                if any(kw in act.lower() for kw in ("browser", "browsing", "reading")):
                    message = f"Your workflow has shifted from {prev_act} into research."
                else:
                    message = f"Your workflow has shifted from {prev_act} into {act}."
            else:
                message = f"Your workflow has shifted to {act}."

            return CompanionResponse(
                message=message,
                category="transition",
                confidence=confidence,
                supporting_context=supporting_context,
                timestamp=understanding.timestamp,
            )

        # Case 4: Iterative workflow (A -> B -> A -> B)
        if pattern == "iterative_workflow":
            return CompanionResponse(
                message=f"You're actively iterating across tasks: {understanding.interpretation}.",
                category="iteration",
                confidence=confidence,
                supporting_context=supporting_context,
                timestamp=understanding.timestamp,
            )

        # Fallback general response
        return CompanionResponse(
            message=f"Currently active in {act}.",
            category="general",
            confidence=confidence,
            supporting_context=supporting_context,
            timestamp=understanding.timestamp,
        )
