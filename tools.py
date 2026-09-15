from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from context import DesktopContext
from history import ContextHistory
from understanding import format_duration
from conversation import ConversationSession


class DuplicateToolError(Exception):
    """Raised when registering a tool whose name is already registered."""
    pass


@dataclass
class ToolResult:
    """
    Structured result returned by tool execution.
    Provides uniform status, output payload, error message, and metadata.
    """
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool result to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BaseTool(ABC):
    """
    Abstract base class for provider-agnostic BLINDSPOT tools.
    Encapsulates name, description, JSON schema parameters, safety classification, and execution logic.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        is_read_only: bool = True,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.is_read_only = is_read_only

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool logic and return a structured ToolResult."""
        raise NotImplementedError

    def to_schema(self) -> Dict[str, Any]:
        """Return schema representation suitable for LLM function/tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """
    Thread-safe registry for managing and executing tools.
    Includes duplicate detection, schema inspection, and safety boundary enforcement.
    """

    def __init__(self, allow_actions: bool = False):
        self.allow_actions = allow_actions
        self._tools: Dict[str, BaseTool] = {}
        self._lock = threading.Lock()

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises DuplicateToolError if name already exists."""
        with self._lock:
            if tool.name in self._tools:
                raise DuplicateToolError(f"A tool named '{tool.name}' is already registered.")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Return list of all registered tools."""
        with self._lock:
            return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return list of tool schemas for LLM provider consumption."""
        with self._lock:
            return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name with safety checks and error handling.
        Returns structured ToolResult indicating success or failure.
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered.",
            )

        # Safety boundary check: block action tools when actions are not allowed
        if not tool.is_read_only and not self.allow_actions:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is a mutating action tool, but action execution is disabled by the safety boundary.",
            )

        try:
            result = tool.execute(**kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(tool_name=name, success=True, output=result)
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Error executing tool '{name}': {e}",
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)


# ---------------- SAFE READ-ONLY TOOLS ----------------

class GetCurrentContextTool(BaseTool):
    """Safe read-only tool to inspect the active desktop context."""

    def __init__(self, state_getter: Optional[Callable[[], Dict[str, Any]]] = None):
        super().__init__(
            name="get_current_context",
            description="Retrieve the current active desktop context including active window, detected activity, duration, and OCR text snippet.",
            parameters={
                "type": "object",
                "properties": {
                    "include_ocr": {
                        "type": "boolean",
                        "description": "Whether to include the OCR text snippet in the output.",
                        "default": True,
                    }
                },
            },
            is_read_only=True,
        )
        self.state_getter = state_getter

    def execute(
        self,
        include_ocr: bool = True,
        state: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
        **kwargs,
    ) -> ToolResult:
        st = state or (self.state_getter() if self.state_getter else {})
        cur_ctx: Optional[DesktopContext] = st.get("current_context")
        start_time: Optional[float] = st.get("context_start_time")

        current_time = time.time() if now is None else now
        duration = max(0.0, current_time - start_time) if start_time else 0.0

        if not cur_ctx:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={
                    "has_active_context": False,
                    "message": "No active desktop context detected.",
                },
            )

        ocr_snippet = None
        if include_ocr and cur_ctx.ocr_text:
            cleaned = " ".join(cur_ctx.ocr_text.split())
            ocr_snippet = cleaned[:250] + ("..." if len(cleaned) > 250 else "")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output={
                "has_active_context": True,
                "activity": cur_ctx.activity,
                "window_title": cur_ctx.window_title,
                "confidence": cur_ctx.confidence,
                "duration_seconds": duration,
                "duration_formatted": format_duration(duration),
                "ocr_snippet": ocr_snippet,
            },
        )


class GetRecentActivityTool(BaseTool):
    """Safe read-only tool to inspect recent workflow episodes and transitions."""

    def __init__(
        self,
        state_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        history: Optional[ContextHistory] = None,
    ):
        super().__init__(
            name="get_recent_activity",
            description="Retrieve recent workflow episodes and activity transitions from desktop history.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent episodes to return.",
                        "default": 5,
                    }
                },
            },
            is_read_only=True,
        )
        self.state_getter = state_getter
        self.history = history

    def execute(
        self,
        limit: int = 5,
        state: Optional[Dict[str, Any]] = None,
        history: Optional[ContextHistory] = None,
        **kwargs,
    ) -> ToolResult:
        st = state or (self.state_getter() if self.state_getter else {})
        hist: Optional[ContextHistory] = history or self.history or st.get("history")

        if not hist:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={"episodes": [], "count": 0, "message": "No history available."},
            )

        records = hist.get_recent(limit=limit)
        episodes = []
        flow = []
        for r in records:
            episodes.append({
                "activity": r.activity,
                "window_title": r.window_title,
                "duration_seconds": r.duration,
                "duration_formatted": format_duration(r.duration),
                "start_time": r.start_time,
                "end_time": r.end_time,
            })
            if not flow or flow[-1] != r.activity:
                flow.append(r.activity)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output={
                "episodes": episodes,
                "flow": flow,
                "count": len(episodes),
            },
        )


class GetConversationHistoryTool(BaseTool):
    """Safe read-only tool to inspect recent dialogue turns."""

    def __init__(
        self,
        state_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        conversation: Optional[ConversationSession] = None,
    ):
        super().__init__(
            name="get_conversation_history",
            description="Retrieve recent conversational dialogue turns between user and BLINDSPOT.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent dialogue turns to return.",
                        "default": 10,
                    }
                },
            },
            is_read_only=True,
        )
        self.state_getter = state_getter
        self.conversation = conversation

    def execute(
        self,
        limit: int = 10,
        state: Optional[Dict[str, Any]] = None,
        conversation: Optional[ConversationSession] = None,
        **kwargs,
    ) -> ToolResult:
        st = state or (self.state_getter() if self.state_getter else {})
        conv: Optional[ConversationSession] = conversation or self.conversation or st.get("conversation")

        if not conv:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={"turns": [], "count": 0, "message": "No conversation session available."},
            )

        turns = conv.get_history_dicts(limit=limit)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={
                "turns": turns,
                "count": len(turns),
            },
        )


def create_default_tool_registry(
    state: Optional[Dict[str, Any]] = None,
    conversation: Optional[ConversationSession] = None,
    history: Optional[ContextHistory] = None,
    allow_actions: bool = False,
) -> ToolRegistry:
    """
    Factory function to create a ToolRegistry populated with the default safe read-only tools,
    wired to BLINDSPOT's live state.
    """
    registry = ToolRegistry(allow_actions=allow_actions)
    state_getter = (lambda: state) if state is not None else None

    registry.register(GetCurrentContextTool(state_getter=state_getter))
    registry.register(GetRecentActivityTool(state_getter=state_getter, history=history))
    registry.register(GetConversationHistoryTool(state_getter=state_getter, conversation=conversation))

    return registry
