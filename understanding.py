from dataclasses import dataclass, field
import time
from typing import Optional, List, Dict, Any
from context import DesktopContext
from history import ContextHistory, ContextRecord


def format_duration(seconds: float) -> str:
    """Format duration into readable string (e.g. 45s, 2m, 1h 5m)."""
    seconds = int(max(0.0, seconds))
    minutes = seconds // 60
    hours = minutes // 60

    if hours > 0:
        return f"{hours}h {minutes % 60}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{seconds}s"


@dataclass
class WorkflowUnderstanding:
    """
    Structured representation of current workflow context and recent activity pattern.
    """
    current_activity: Optional[str]
    current_window: Optional[str]
    current_duration: float
    recent_activities: List[str]
    recent_transitions_count: int
    pattern_type: str  # e.g. "focused_session", "resumed_task", "workflow_shift", "iterative_workflow", "no_activity"
    interpretation: str
    confidence: int  # 1 (low) to 3 (high)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize understanding into a structured dictionary for future AI consumption."""
        return {
            "current": {
                "activity": self.current_activity,
                "window_title": self.current_window,
                "duration": self.current_duration,
                "duration_formatted": format_duration(self.current_duration),
            },
            "recent": {
                "activities_flow": self.recent_activities,
                "transitions_count": self.recent_transitions_count,
            },
            "understanding": {
                "pattern_type": self.pattern_type,
                "interpretation": self.interpretation,
                "confidence": self.confidence,
            },
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ContextualUnderstandingAnalyzer:
    """
    Analyzes active DesktopContext and recent ContextHistory to deduce
    meaningful workflow patterns and interpretations.
    """

    def analyze(
        self,
        current_context: Optional[DesktopContext],
        context_start_time: Optional[float],
        history: ContextHistory,
        now: Optional[float] = None,
    ) -> WorkflowUnderstanding:
        """
        Derive structured workflow understanding from current context and recent history.
        """
        now = time.time() if now is None else now
        current_dur = max(0.0, now - context_start_time) if context_start_time else 0.0

        current_act = current_context.activity if current_context else None
        current_win = current_context.window_title if current_context else None
        current_conf = current_context.confidence if current_context else 1

        recent_records = history.get_recent(limit=10)
        recent_transitions_count = len(recent_records)

        # Build chronological activity sequence
        history_acts = [r.activity for r in recent_records]
        raw_flow = history_acts + ([current_act] if current_act else [])

        # Compress consecutive identical activities for flow summary (e.g. [A, A, B] -> [A, B])
        compressed_flow: List[str] = []
        for act in raw_flow:
            if not compressed_flow or compressed_flow[-1] != act:
                compressed_flow.append(act)

        # Case 0: No active context and no history
        if not current_act and not recent_records:
            return WorkflowUnderstanding(
                current_activity=None,
                current_window=None,
                current_duration=0.0,
                recent_activities=[],
                recent_transitions_count=0,
                pattern_type="no_activity",
                interpretation="No active desktop context detected.",
                confidence=1,
                timestamp=now,
            )

        # Case 1: Active context exists with no prior history
        if current_act and not recent_records:
            dur_str = format_duration(current_dur)
            return WorkflowUnderstanding(
                current_activity=current_act,
                current_window=current_win,
                current_duration=current_dur,
                recent_activities=[current_act],
                recent_transitions_count=0,
                pattern_type="focused_session",
                interpretation=f"Focused on {current_act} for {dur_str}.",
                confidence=current_conf,
                timestamp=now,
            )

        # Case 2: Prior history exists
        # Check if all recent history matches current activity
        distinct_acts = set(history_acts)
        if current_act and distinct_acts == {current_act}:
            total_dur = sum(r.duration for r in recent_records) + current_dur
            dur_str = format_duration(total_dur)
            return WorkflowUnderstanding(
                current_activity=current_act,
                current_window=current_win,
                current_duration=current_dur,
                recent_activities=[current_act],
                recent_transitions_count=recent_transitions_count,
                pattern_type="focused_session",
                interpretation=f"Continuously engaged in {current_act} for {dur_str}.",
                confidence=3,
                timestamp=now,
                metadata={"total_duration": total_dur},
            )

        # Check for Resumed Task: A -> B -> A (or A -> B -> C -> A)
        if current_act and len(recent_records) >= 2:
            last_record = recent_records[-1]
            prev_record = recent_records[-2]

            # Direct return: prev was current_act, last was something else
            if prev_record.activity == current_act and last_record.activity != current_act:
                intervening_act = last_record.activity
                intervening_dur = format_duration(last_record.duration)

                if "browser" in intervening_act or "browsing" in intervening_act or "reading" in intervening_act:
                    interp = f"Returned to {current_act} after a {intervening_dur} research session."
                else:
                    interp = f"Returned to {current_act} after a short {intervening_act} session ({intervening_dur})."

                return WorkflowUnderstanding(
                    current_activity=current_act,
                    current_window=current_win,
                    current_duration=current_dur,
                    recent_activities=compressed_flow,
                    recent_transitions_count=recent_transitions_count,
                    pattern_type="resumed_task",
                    interpretation=interp,
                    confidence=3,
                    timestamp=now,
                    metadata={"interrupted_by": intervening_act, "interruption_duration": last_record.duration},
                )

            # Return after 2 intervening steps: records[-3] was current_act
            if len(recent_records) >= 3 and recent_records[-3].activity == current_act and last_record.activity != current_act:
                interp = f"Resumed {current_act} after context switches across {prev_record.activity} and {last_record.activity}."
                return WorkflowUnderstanding(
                    current_activity=current_act,
                    current_window=current_win,
                    current_duration=current_dur,
                    recent_activities=compressed_flow,
                    recent_transitions_count=recent_transitions_count,
                    pattern_type="resumed_task",
                    interpretation=interp,
                    confidence=3,
                    timestamp=now,
                )

        # Check for Iterative Workflow: alternating between 2 activities (e.g. A -> B -> A -> B)
        if len(compressed_flow) >= 4:
            last_4 = compressed_flow[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                return WorkflowUnderstanding(
                    current_activity=current_act,
                    current_window=current_win,
                    current_duration=current_dur,
                    recent_activities=compressed_flow,
                    recent_transitions_count=recent_transitions_count,
                    pattern_type="iterative_workflow",
                    interpretation=f"Iterating between {last_4[0]} and {last_4[1]}.",
                    confidence=2,
                    timestamp=now,
                )

        # Case 3: Workflow Shift / Progression: A -> B or A -> B -> C
        if current_act and recent_records:
            last_record = recent_records[-1]
            if len(recent_records) >= 2:
                prev_record = recent_records[-2]
                interp = f"Transitioned from {prev_record.activity} via {last_record.activity} to {current_act}."
            else:
                dur_str = format_duration(last_record.duration)
                interp = f"Transitioned from {last_record.activity} ({dur_str}) to {current_act}."

            return WorkflowUnderstanding(
                current_activity=current_act,
                current_window=current_win,
                current_duration=current_dur,
                recent_activities=compressed_flow,
                recent_transitions_count=recent_transitions_count,
                pattern_type="workflow_shift",
                interpretation=interp,
                confidence=2,
                timestamp=now,
            )

        # Fallback general summary
        return WorkflowUnderstanding(
            current_activity=current_act,
            current_window=current_win,
            current_duration=current_dur,
            recent_activities=compressed_flow,
            recent_transitions_count=recent_transitions_count,
            pattern_type="general",
            interpretation=f"Active in {current_act or 'desktop'}." if current_act else "No active desktop context.",
            confidence=1,
            timestamp=now,
        )
