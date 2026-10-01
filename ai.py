import os
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from context import DesktopContext
from history import ContextHistory
from understanding import WorkflowUnderstanding, format_duration
from conversation import ConversationSession, ConversationTurn
from tools import ToolRegistry, ToolResult


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
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
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
            "conversation_history": self.conversation_history,
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

        if self.conversation_history:
            lines.append("\n=== RECENT CONVERSATION ===")
            for turn in self.conversation_history:
                speaker = "User" if turn.get("role") == "user" else "BLINDSPOT"
                lines.append(f"{speaker}: {turn.get('content', '')}")

        lines.append(f"\n=== USER QUESTION ===\n{self.user_query}")
        return "\n".join(lines)


def format_context_payload(
    current_context: Optional[DesktopContext],
    context_start_time: Optional[float],
    history: ContextHistory,
    understanding: WorkflowUnderstanding,
    user_query: str,
    conversation: Optional[Union[ConversationSession, List[Dict[str, Any]]]] = None,
    max_ocr_chars: int = 250,
    max_recent_episodes: int = 5,
    max_conversation_turns: int = 10,
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

    # Extract recent conversation turns (bounded)
    conversation_turns: List[Dict[str, Any]] = []
    if isinstance(conversation, ConversationSession):
        conversation_turns = conversation.get_history_dicts(limit=max_conversation_turns)
    elif isinstance(conversation, list):
        conversation_turns = conversation[-max_conversation_turns:]

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
        conversation_history=conversation_turns,
        timestamp=now,
    )


@dataclass
class ToolCall:
    """Represents a tool or function call requested by the LLM."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


def parse_tool_calls(resp_data: Dict[str, Any]) -> List[ToolCall]:
    """
    Parse tool/function calls from Gemini response across Interactions API
    and generateContent payload shapes.
    """
    calls: List[ToolCall] = []

    if not isinstance(resp_data, dict):
        return calls

    # Shape A: Interactions API steps[]
    steps = resp_data.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type")
            if step_type in ("tool_call", "function_call"):
                call_info = step.get("tool_call") or step.get("function_call") or step.get("call") or step
                name = call_info.get("name") or call_info.get("function_name")
                args = call_info.get("arguments") or call_info.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                call_id = call_info.get("id") or step.get("id") or call_info.get("call_id") or step.get("call_id")
                if name:
                    calls.append(ToolCall(name=name, args=args, call_id=call_id))

            content = step.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        fc = part.get("functionCall") or part.get("function_call") or part.get("tool_call")
                        if isinstance(fc, dict) and fc.get("name"):
                            args = fc.get("args") or fc.get("arguments") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {"raw": args}
                            calls.append(ToolCall(name=fc["name"], args=args, call_id=fc.get("id")))

    # Shape B: candidates[0].content.parts (generateContent / standard Gemini)
    candidates = resp_data.get("candidates")
    if isinstance(candidates, list) and candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    fc = part.get("functionCall") or part.get("function_call")
                    if isinstance(fc, dict) and fc.get("name"):
                        args = fc.get("args") or fc.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {"raw": args}
                        calls.append(ToolCall(name=fc["name"], args=args, call_id=fc.get("id")))

    # Shape C: output list or dict
    out = resp_data.get("output")
    if isinstance(out, list):
        for item in out:
            if isinstance(item, dict):
                fc = item.get("functionCall") or item.get("function_call") or item.get("tool_call")
                if isinstance(fc, dict) and fc.get("name"):
                    args = fc.get("args") or fc.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    calls.append(ToolCall(name=fc["name"], args=args, call_id=fc.get("id")))
    elif isinstance(out, dict):
        fc = out.get("functionCall") or out.get("function_call") or out.get("tool_call")
        if isinstance(fc, dict) and fc.get("name"):
            args = fc.get("args") or fc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            calls.append(ToolCall(name=fc["name"], args=args, call_id=fc.get("id")))

    # Shape D: top-level tool_calls or function_calls
    direct_calls = resp_data.get("tool_calls") or resp_data.get("function_calls")
    if isinstance(direct_calls, list):
        for dc in direct_calls:
            if isinstance(dc, dict) and dc.get("name"):
                args = dc.get("args") or dc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                calls.append(ToolCall(name=dc["name"], args=args, call_id=dc.get("id")))

    return calls


def extract_text_response(resp_data: Dict[str, Any]) -> Optional[str]:
    """Extract plain text content from Gemini response across supported shapes."""
    if not isinstance(resp_data, dict):
        return None

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

    return None


def format_tool_results_for_interactions(
    tool_results: List[ToolResult],
    interaction_id: Optional[str] = None,
    clean_model: str = "gemini-3.6-flash",
    system_instruction: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build follow-up request body with tool results for Interactions API."""
    inputs = []
    for res in tool_results:
        item: Dict[str, Any] = {
            "type": "function_result",
            "name": res.tool_name,
            "result": res.output if res.success else {"error": res.error},
        }
        call_id = res.metadata.get("call_id")
        if call_id:
            item["call_id"] = call_id
        inputs.append(item)

    body: Dict[str, Any] = {
        "model": clean_model,
        "input": inputs,
        "generation_config": {
            "temperature": 0.2,
            "max_output_tokens": 800,
        },
    }
    if interaction_id:
        body["previous_interaction_id"] = interaction_id
    if system_instruction:
        body["system_instruction"] = system_instruction
    if tools:
        body["tools"] = tools
    return body


class LLMProvider:
    """Abstract interface for LLM providers."""

    def generate_response(
        self,
        system_instruction: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> str:
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

    def _send_http_request(self, request_body: Dict[str, Any], clean_key: str) -> Dict[str, Any]:
        """Send HTTP POST request to Gemini Interactions API and return parsed JSON."""
        url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={clean_key}"
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
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            if clean_key:
                err_msg = err_msg.replace(clean_key, "[REDACTED_API_KEY]")
            raise LLMProviderError(f"Gemini API HTTP {e.code}: {e.reason} -> {err_msg.strip()}")
        except urllib.error.URLError as e:
            raise LLMProviderError(f"Network error connecting to Gemini API: {e.reason}")
        except Exception as e:
            raise LLMProviderError(f"Unexpected error communicating with Gemini API: {e}")

    def generate_response(
        self,
        system_instruction: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_turns: int = 5,
    ) -> str:
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

        request_body: Dict[str, Any] = {
            "model": clean_model,
            "input": user_prompt,
            "system_instruction": system_instruction,
            "generation_config": {
                "temperature": 0.2,
                "max_output_tokens": 800,
            },
        }

        if tools:
            request_body["tools"] = tools

        # Execute interaction loop (handles potential tool call requests from Gemini)
        for _ in range(max_tool_turns):
            resp_data = self._send_http_request(request_body, clean_key)

            # Check if Gemini returned any tool calls
            tool_calls = parse_tool_calls(resp_data)
            if tool_calls and tool_registry is not None:
                tool_results = []
                for tc in tool_calls:
                    result = tool_registry.execute(tc.name, **tc.args)
                    if tc.call_id and "call_id" not in result.metadata:
                        result.metadata["call_id"] = tc.call_id
                    tool_results.append(result)

                interaction_id = resp_data.get("id")
                request_body = format_tool_results_for_interactions(
                    tool_results=tool_results,
                    interaction_id=interaction_id,
                    clean_model=clean_model,
                    system_instruction=system_instruction,
                    tools=tools,
                )
                continue

            # No tool calls: extract text response
            text = extract_text_response(resp_data)
            if text:
                return text

            return "I observed your context, but could not parse the AI provider's response."

        return "Maximum tool calling iterations exceeded."


class CompanionAI:
    """
    Context-aware AI Companion Core that answers user queries using
    actual observed desktop context, workflow understanding, and tools.
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
        "4. Be concise, direct, helpful, and speak in a friendly companion persona.\n"
        "5. When recent conversation is provided, maintain conversational continuity and understand "
        "follow-up questions or references to earlier exchanges.\n"
        "6. Use registered tools when you need additional or refreshed information about the user's desktop state."
    )

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        conversation: Optional[ConversationSession] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.provider = provider or GeminiProvider()
        self.conversation = conversation if conversation is not None else ConversationSession()
        self.tools = tool_registry if tool_registry is not None else ToolRegistry()

    def clear_conversation(self) -> None:
        """Reset the active conversation session."""
        self.conversation.clear()

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name using the companion's ToolRegistry."""
        return self.tools.execute(name, **kwargs)

    def ask(
        self,
        current_context: Optional[DesktopContext],
        context_start_time: Optional[float],
        history: ContextHistory,
        understanding: WorkflowUnderstanding,
        user_query: str,
        conversation: Optional[ConversationSession] = None,
        now: Optional[float] = None,
    ) -> str:
        """
        Package structured live context, history, and conversation turns,
        send query to the LLM provider, and record the dialogue turns.
        """
        active_session = conversation if conversation is not None else self.conversation

        payload = format_context_payload(
            current_context=current_context,
            context_start_time=context_start_time,
            history=history,
            understanding=understanding,
            user_query=user_query,
            conversation=active_session,
            now=now,
        )

        prompt_text = payload.to_prompt_text()

        try:
            import inspect
            tool_schemas = self.tools.get_tool_schemas() if len(self.tools) > 0 else None
            sig = inspect.signature(self.provider.generate_response)
            kwargs: Dict[str, Any] = {
                "system_instruction": self.SYSTEM_INSTRUCTION,
                "user_prompt": prompt_text,
            }
            if "tools" in sig.parameters:
                kwargs["tools"] = tool_schemas
            if "tool_registry" in sig.parameters:
                kwargs["tool_registry"] = self.tools

            response = self.provider.generate_response(**kwargs)
            # Record user turn and assistant reply in conversation session
            active_session.add_user_message(user_query)
            active_session.add_assistant_message(response)
            return response
        except MissingAPIKeyError as e:
            return str(e)
        except LLMProviderError as e:
            return f"[AI Provider Error] {e}"
        except Exception as e:
            return f"[AI Error] An unexpected error occurred: {e}"

    TASK_PLANNING_SYSTEM_INSTRUCTION = (
        "You are BLINDSPOT's task planning engine.\n"
        "Convert the user's desktop task request into an explicit, sequential list of registered tool calls.\n"
        "RULES:\n"
        "1. You must ONLY choose tools from the provided schemas.\n"
        "2. Return ONLY a valid JSON array of objects, with NO surrounding markdown or commentary.\n"
        "3. Each object must have:\n"
        "   - 'tool_name': name of the registered tool\n"
        "   - 'args': dictionary of argument values\n"
        "   - 'description': brief summary of what this step accomplishes\n"
        "4. Maximum 5 steps.\n"
        "5. Do NOT invent tools or parameters that are not in the schemas.\n"
        "6. If the request cannot be fulfilled, return []."
    )

    def plan_task(self, user_request: str) -> "Task":
        """
        Convert a user's multi-step request into a structured Task with validated steps.
        Uses Gemini if available, with deterministic fallback for standard commands.
        Rejects invalid plans cleanly rather than executing partial or corrupted plans.
        """
        from task import create_task, validate_task, TaskStatus, MAX_TASK_STEPS
        from tools import validate_tool_args
        import uuid

        tid = f"task_{uuid.uuid4().hex[:8]}"

        if not isinstance(user_request, str) or not user_request.strip():
            return create_task(
                user_request=user_request or "",
                steps_data=[],
                task_id=tid,
                status=TaskStatus.FAILED,
                error="User task request cannot be empty.",
            )

        schemas = self.tools.get_tool_schemas() if len(self.tools) > 0 else []
        if not schemas:
            return create_task(
                user_request=user_request,
                steps_data=[],
                task_id=tid,
                status=TaskStatus.FAILED,
                error="No tools are registered in ToolRegistry.",
            )

        llm_response = None
        parsed = None
        plan_err = None

        # 1. Try prompting the LLM provider
        try:
            planning_prompt = (
                f"User task request: {user_request}\n\n"
                f"Available tool schemas:\n{json.dumps(schemas, indent=2)}\n\n"
                "Return the sequential tool steps as a JSON array:"
            )
            kwargs: Dict[str, Any] = {
                "system_instruction": self.TASK_PLANNING_SYSTEM_INSTRUCTION,
                "user_prompt": planning_prompt,
            }
            llm_response = self.provider.generate_response(**kwargs)
            parsed = self._extract_steps_json(llm_response)
            if parsed is None and llm_response:
                plan_err = "Plan rejected: Failed to extract valid JSON plan from model response."
        except Exception as e:
            parsed = None
            plan_err = f"Plan rejected: Provider error ({e})"

        # 2. If the LLM returned parsed step data, validate it strictly
        if parsed is not None:
            if not isinstance(parsed, list):
                return create_task(
                    user_request=user_request,
                    steps_data=[],
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    error="Plan rejected: Expected JSON array of steps.",
                )

            if len(parsed) == 0:
                return create_task(
                    user_request=user_request,
                    steps_data=[],
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    error="Plan rejected: Model returned an empty plan.",
                )

            if len(parsed) > MAX_TASK_STEPS:
                return create_task(
                    user_request=user_request,
                    steps_data=[],
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    error=f"Plan rejected: Model plan exceeds maximum limit of {MAX_TASK_STEPS} steps (got {len(parsed)}).",
                )

            steps_data: List[Dict[str, Any]] = []
            seen_steps = []
            for idx, step in enumerate(parsed, start=1):
                if not isinstance(step, dict):
                    return create_task(
                        user_request=user_request,
                        steps_data=[],
                        task_id=tid,
                        status=TaskStatus.FAILED,
                        error=f"Plan rejected: Step {idx} is not a valid dictionary object.",
                    )

                tool_name = step.get("tool_name")
                if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
                    return create_task(
                        user_request=user_request,
                        steps_data=[],
                        task_id=tid,
                        status=TaskStatus.FAILED,
                        error=f"Plan rejected: Step {idx} is missing a valid 'tool_name'.",
                    )

                tool_name_clean = tool_name.strip()
                tool = self.tools.get(tool_name_clean)
                if not tool:
                    return create_task(
                        user_request=user_request,
                        steps_data=[],
                        task_id=tid,
                        status=TaskStatus.FAILED,
                        error=f"Plan rejected: Model requested unknown tool '{tool_name_clean}'.",
                    )

                args = step.get("args")
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    return create_task(
                        user_request=user_request,
                        steps_data=[],
                        task_id=tid,
                        status=TaskStatus.FAILED,
                        error=f"Plan rejected: Step {idx} ({tool_name_clean}) arguments must be a dictionary.",
                    )

                is_valid_args, arg_err = validate_tool_args(tool, args)
                if not is_valid_args:
                    return create_task(
                        user_request=user_request,
                        steps_data=[],
                        task_id=tid,
                        status=TaskStatus.FAILED,
                        error=f"Plan rejected: Invalid arguments for tool '{tool_name_clean}': {arg_err}",
                    )

                # Duplicate consecutive step check
                step_sig = (tool_name_clean, json.dumps(args, sort_keys=True))
                if seen_steps and seen_steps[-1] == step_sig:
                    return create_task(
                        user_request=user_request,
                        steps_data=[],
                        task_id=tid,
                        status=TaskStatus.FAILED,
                        error=f"Plan rejected: Model plan contains duplicate consecutive step '{tool_name_clean}'.",
                    )
                seen_steps.append(step_sig)

                steps_data.append({
                    "tool_name": tool_name_clean,
                    "args": args,
                    "description": step.get("description", f"Execute {tool_name_clean}"),
                })

            task = create_task(user_request=user_request, steps_data=steps_data, task_id=tid)
            is_valid, val_err = validate_task(task, self.tools, MAX_TASK_STEPS)
            if not is_valid:
                task.status = TaskStatus.FAILED
                task.error = f"Plan rejected: {val_err}"
                task.steps = []
            return task

        # 3. Deterministic fallback if provider was unavailable or did not return JSON
        fallback_steps = self._deterministic_task_fallback(user_request)
        if fallback_steps:
            task = create_task(user_request=user_request, steps_data=fallback_steps, task_id=tid)
            is_valid, val_err = validate_task(task, self.tools, MAX_TASK_STEPS)
            if is_valid:
                return task

        # 4. If neither worked, return a failed task cleanly
        return create_task(
            user_request=user_request,
            steps_data=[],
            task_id=tid,
            status=TaskStatus.FAILED,
            error=plan_err or "Could not plan task from user request.",
        )

    def _extract_steps_json(self, text: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extract and parse a JSON array of step dictionaries from model text.
        Handles markdown fences, preamble/postamble explanatory text, and nested step objects.
        Returns list of step dicts if parseable, else None.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        import re

        # Strategy 1: Look for markdown code blocks ```json ... ``` or ``` ... ```
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        for block in code_blocks:
            candidate = block.strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    for key in ("steps", "plan", "tools", "actions", "task"):
                        if isinstance(data.get(key), list):
                            return data[key]
            except Exception:
                pass

        # Strategy 2: Look for outermost JSON array [...]
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                data = json.loads(text[start_arr : end_arr + 1])
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        # Strategy 3: Look for outermost JSON object {...}
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                data = json.loads(text[start_obj : end_obj + 1])
                if isinstance(data, dict):
                    for key in ("steps", "plan", "tools", "actions", "task"):
                        if isinstance(data.get(key), list):
                            return data[key]
            except Exception:
                pass

        return None

    def _deterministic_task_fallback(self, user_request: str) -> List[Dict[str, Any]]:
        """Rule-based fallback for standard desktop action patterns."""
        from tools import ALLOWED_APPLICATIONS
        req_lower = user_request.lower().strip()
        steps = []
        opened_app = None

        # Check for open / launch application
        for app in ALLOWED_APPLICATIONS:
            if f"open {app}" in req_lower or f"launch {app}" in req_lower or req_lower == f"open {app}":
                if self.tools.get("open_application"):
                    steps.append({
                        "tool_name": "open_application",
                        "args": {"app_name": app},
                        "description": f"Open {app.capitalize()}",
                    })
                    opened_app = app
                break

        # Check for focus / switch to application
        focused_app = None
        for app in ALLOWED_APPLICATIONS:
            if f"focus {app}" in req_lower or f"switch to {app}" in req_lower:
                if self.tools.get("focus_application"):
                    steps.append({
                        "tool_name": "focus_application",
                        "args": {"window_title_fragment": app.capitalize()},
                        "description": f"Focus {app.capitalize()}",
                    })
                    focused_app = app
                break

        # Check for "focus it" referencing the opened app
        if not focused_app and opened_app and ("focus it" in req_lower or "focus" in req_lower or "bring to front" in req_lower):
            if self.tools.get("focus_application"):
                steps.append({
                    "tool_name": "focus_application",
                    "args": {"window_title_fragment": opened_app.capitalize()},
                    "description": f"Focus {opened_app.capitalize()}",
                })

        # Check for inspection commands
        if not steps:
            if "context" in req_lower and self.tools.get("get_current_context"):
                steps.append({
                    "tool_name": "get_current_context",
                    "args": {"include_ocr": True},
                    "description": "Inspect current desktop context",
                })
            elif ("activity" in req_lower or "history" in req_lower) and self.tools.get("get_recent_activity"):
                steps.append({
                    "tool_name": "get_recent_activity",
                    "args": {"limit": 5},
                    "description": "Inspect recent activity history",
                })

        return steps


