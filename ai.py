import os
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from context import DesktopContext
from history import ContextHistory
from understanding import WorkflowUnderstanding, format_duration


class MissingAPIKeyError(Exception):
    """Raised when the required Gemini API key is missing."""
    pass


class LLMProviderError(Exception):
    """Raised when an error occurs during LLM provider communication."""
    pass


@dataclass
class LLMContextPayload:
    """
    Compact, structured context payload prepared for LLM consumption.
    Distinguishes observed facts from heuristic inferences.
    """
    current_activity: Optional[str]
    current_window: Optional[str]
    current_duration_seconds: float
    current_duration_formatted: str
    recent_activities_flow: List[str]
    recent_episodes: List[Dict[str, Any]]
    workflow_pattern: str
    workflow_interpretation: str
    confidence: int
    current_ocr_snippet: Optional[str]
    user_query: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized dictionary representation of the context payload."""
        return {
            "current_context": {
                "activity": self.current_activity,
                "window_title": self.current_window,
                "duration_seconds": self.current_duration_seconds,
                "duration_formatted": self.current_duration_formatted,
                "ocr_snippet": self.current_ocr_snippet,
            },
            "recent_history": {
                "flow": self.recent_activities_flow,
                "episodes": self.recent_episodes,
            },
            "workflow_understanding": {
                "pattern": self.workflow_pattern,
                "interpretation": self.workflow_interpretation,
                "confidence": self.confidence,
            },
            "user_query": self.user_query,
            "timestamp": self.timestamp,
        }

    def to_prompt_text(self) -> str:
        """Format the payload into a clean, token-efficient prompt string."""
        lines = [
            "=== OBSERVED DESKTOP CONTEXT ===",
            f"Active Activity: {self.current_activity or 'None'}",
            f"Active Window:   {self.current_window or 'None'}",
            f"Active Duration: {self.current_duration_formatted}",
        ]

        if self.current_ocr_snippet:
            lines.append(f"Current Screen Text Snippet: {self.current_ocr_snippet}")

        lines.append("\n=== RECENT WORKFLOW HISTORY ===")
        if self.recent_episodes:
            for ep in self.recent_episodes:
                lines.append(
                    f"- {ep.get('activity')} ({ep.get('duration_formatted', '')}) on '{ep.get('window_title', '')}'"
                )
        else:
            lines.append("- No prior history recorded in this session.")

        lines.append(f"Flow Sequence: {' -> '.join(self.recent_activities_flow) if self.recent_activities_flow else 'None'}")

        lines.append("\n=== WORKFLOW UNDERSTANDING ===")
        lines.append(f"Pattern Type:   {self.workflow_pattern}")
        lines.append(f"Interpretation: {self.workflow_interpretation}")
        lines.append(f"Confidence:     {self.confidence}/3")

        lines.append(f"\n=== USER QUESTION ===\n{self.user_query}")
        return "\n".join(lines)


def format_context_payload(
    current_context: Optional[DesktopContext],
    context_start_time: Optional[float],
    history: ContextHistory,
    understanding: WorkflowUnderstanding,
    user_query: str,
    max_ocr_chars: int = 250,
    max_recent_episodes: int = 5,
    now: Optional[float] = None,
) -> LLMContextPayload:
    """
    Construct a compact, bounded LLMContextPayload from live BLINDSPOT state.
    """
    now = time.time() if now is None else now
    cur_dur = max(0.0, now - context_start_time) if context_start_time else 0.0

    cur_act = current_context.activity if current_context else None
    cur_win = current_context.window_title if current_context else None

    # Sanitize and truncate OCR text to prevent token explosion
    ocr_snippet = None
    if current_context and current_context.ocr_text:
        cleaned = " ".join(current_context.ocr_text.split())
        if cleaned:
            ocr_snippet = cleaned[:max_ocr_chars] + ("..." if len(cleaned) > max_ocr_chars else "")

    # Extract recent closed episodes (bounded)
    recent_records = history.get_recent(limit=max_recent_episodes)
    episodes = []
    for r in recent_records:
        episodes.append({
            "activity": r.activity,
            "window_title": r.window_title,
            "duration_seconds": r.duration,
            "duration_formatted": format_duration(r.duration),
        })

    return LLMContextPayload(
        current_activity=cur_act,
        current_window=cur_win,
        current_duration_seconds=cur_dur,
        current_duration_formatted=format_duration(cur_dur),
        recent_activities_flow=understanding.recent_activities,
        recent_episodes=episodes,
        workflow_pattern=understanding.pattern_type,
        workflow_interpretation=understanding.interpretation,
        confidence=understanding.confidence,
        current_ocr_snippet=ocr_snippet,
        user_query=user_query,
        timestamp=now,
    )


class LLMProvider:
    """Abstract interface for LLM providers."""

    def generate_response(self, system_instruction: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """
    Official Google Gemini API provider using the Interactions API over standard library HTTP.
    Reads API key securely from BLINDSPOT_GEMINI_API_KEY or GEMINI_API_KEY.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.6-flash", timeout: int = 15):
        self.api_key = api_key or os.environ.get("BLINDSPOT_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout

    def generate_response(self, system_instruction: str, user_prompt: str) -> str:
        if not self.api_key:
            raise MissingAPIKeyError(
                "Gemini API key is not configured.\n"
                "Please set the BLINDSPOT_GEMINI_API_KEY environment variable:\n"
                "  PowerShell: $env:BLINDSPOT_GEMINI_API_KEY='your_api_key'\n"
                "  CMD:        set BLINDSPOT_GEMINI_API_KEY=your_api_key\n"
                "  Bash:       export BLINDSPOT_GEMINI_API_KEY='your_api_key'"
            )

        clean_key = self.api_key.strip().strip('"\'')
        clean_model = self.model[7:] if self.model.startswith("models/") else self.model

        url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={clean_key}"

        request_body = {
            "model": clean_model,
            "input": user_prompt,
            "system_instruction": system_instruction,
            "generation_config": {
                "temperature": 0.2,
                "max_output_tokens": 800,
            },
        }

        data = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": clean_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            # ---- Shape 0: confirmed Interactions API steps[] shape ------
            # steps[*] where type == "model_output" → content[*].text
            steps = resp_data.get("steps")
            if isinstance(steps, list):
                texts = []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    if step.get("type") != "model_output":
                        continue
                    for part in step.get("content", []):
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
                result = "\n".join(t for t in texts if t.strip())
                if result.strip():
                    return result.strip()

            # ---- Shape 1: output is a plain string ----------------------
            out = resp_data.get("output")
            if isinstance(out, str) and out.strip():
                return out.strip()

            # ---- Shape 2: output is a list of content parts -------------
            if isinstance(out, list):
                texts = []
                for item in out:
                    if isinstance(item, dict):
                        if item.get("type", "text") == "text" and "text" in item:
                            texts.append(item["text"])
                        elif "text" in item and item.get("thought") is not True:
                            texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                result = "\n".join(t for t in texts if t.strip())
                if result.strip():
                    return result.strip()

            # ---- Shape 3: output is a dict with nested content or text --
            if isinstance(out, dict):
                if "text" in out and isinstance(out["text"], str) and out["text"].strip():
                    return out["text"].strip()
                parts = out.get("parts", [])
                texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p and p.get("thought") is not True]
                result = "\n".join(t for t in texts if t.strip())
                if result.strip():
                    return result.strip()

            # ---- Shape 4: top-level "text" ------------------------------
            top_text = resp_data.get("text")
            if isinstance(top_text, str) and top_text.strip():
                return top_text.strip()

            # ---- Shape 5: generateContent candidates[0].content.parts --
            candidates = resp_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
                result = "\n".join(t for t in texts if t.strip())
                if result.strip():
                    return result.strip()

            return "I observed your context, but could not parse the AI provider's response."

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            # Redact any accidental key instances from the error body
            if clean_key:
                err_msg = err_msg.replace(clean_key, "[REDACTED_API_KEY]")
            raise LLMProviderError(f"Gemini API HTTP {e.code}: {e.reason} -> {err_msg.strip()}")
        except urllib.error.URLError as e:
            raise LLMProviderError(f"Network error connecting to Gemini API: {e.reason}")
        except Exception as e:
            raise LLMProviderError(f"Unexpected error communicating with Gemini API: {e}")


class CompanionAI:
    """
    Context-aware AI Companion Core that answers user queries using
    actual observed desktop context and workflow understanding.
    """

    SYSTEM_INSTRUCTION = (
        "You are BLINDSPOT, an intelligent and context-aware desktop AI companion.\n"
        "You assist the user by understanding their current active workspace, recent activity history, "
        "and workflow patterns based on structured desktop perception.\n\n"
        "CORE RULES:\n"
        "1. Base your answer strictly on the OBSERVED FACTS provided in the context payload (active window, "
        "detected activities, durations, recent transitions, OCR snippet).\n"
        "2. Clearly distinguish between what BLINDSPOT directly observed versus logical inferences.\n"
        "3. NEVER invent, hallucinate, or assume activities or applications that BLINDSPOT did not observe.\n"
        "4. Be concise, direct, helpful, and speak in a friendly companion persona."
    )

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or GeminiProvider()

    def ask(
        self,
        current_context: Optional[DesktopContext],
        context_start_time: Optional[float],
        history: ContextHistory,
        understanding: WorkflowUnderstanding,
        user_query: str,
        now: Optional[float] = None,
    ) -> str:
        """
        Package structured live context and send query to the LLM provider.
        """
        payload = format_context_payload(
            current_context=current_context,
            context_start_time=context_start_time,
            history=history,
            understanding=understanding,
            user_query=user_query,
            now=now,
        )

        prompt_text = payload.to_prompt_text()

        try:
            return self.provider.generate_response(
                system_instruction=self.SYSTEM_INSTRUCTION,
                user_prompt=prompt_text,
            )
        except MissingAPIKeyError as e:
            return str(e)
        except LLMProviderError as e:
            return f"[AI Provider Error] {e}"
        except Exception as e:
            return f"[AI Error] An unexpected error occurred: {e}"
