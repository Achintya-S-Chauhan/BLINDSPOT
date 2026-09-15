from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from context import DesktopContext
from history import ContextHistory
from understanding import format_duration
from conversation import ConversationSession


# ============================================================
# SAFETY CONSTANTS
# ============================================================

# Tool names that are unconditionally blocked regardless of registry state.
# This prevents Gemini from requesting dangerous operations even if a tool
# with that name were ever mistakenly registered.
BLOCKED_DANGEROUS_TOOL_NAMES: frozenset = frozenset({
    # Shell / process execution
    "run_shell", "exec_shell", "execute_shell", "shell", "bash", "cmd",
    "run_command", "exec_command", "execute_command", "subprocess", "popen",
    # Python eval/exec
    "eval_python", "exec_python", "run_python", "execute_code",
    # File modification / deletion
    "delete_file", "remove_file", "write_file", "overwrite_file",
    "create_file", "truncate_file", "modify_file",
    # Credential / password handling
    "get_password", "store_password", "read_credentials", "send_credentials",
    "login", "authenticate",
    # Messaging / email
    "send_email", "send_message", "send_sms", "send_notification",
    # System control
    "shutdown", "restart", "reboot", "hibernate", "sleep_system",
    "kill_process", "terminate_process",
    # Arbitrary executable paths
    "run_executable", "launch_executable", "execute_path",
})


# ============================================================
# ENUMS / CONSTANTS
# ============================================================

class ActionRisk(enum.Enum):
    """Risk classification for tools.

    READ_ONLY      — No system state is changed. Approved automatically.
    LOW_RISK_ACTION — Reversible desktop action. Requires explicit approval.
    HIGH_RISK_ACTION — Irreversible or destructive. Blocked by default.
    """
    READ_ONLY = "read_only"
    LOW_RISK_ACTION = "low_risk_action"
    HIGH_RISK_ACTION = "high_risk_action"


# ============================================================
# EXCEPTIONS
# ============================================================

class DuplicateToolError(Exception):
    """Raised when registering a tool whose name is already registered."""
    pass


class PermissionDeniedError(Exception):
    """Raised when an action is denied by the permission policy."""
    pass


# ============================================================
# PERMISSION SYSTEM
# ============================================================

@dataclass
class ActionRequest:
    """Describes a pending action that requires approval before execution."""
    tool_name: str
    risk: ActionRisk
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    request_id: str = field(default_factory=lambda: str(time.time()))


# Callable type: (ActionRequest) -> bool
ApprovalCallback = Callable[[ActionRequest], bool]


def _auto_approve(_request: ActionRequest) -> bool:
    """Default approver: always grants permission (used for READ_ONLY)."""
    return True


def _auto_deny(_request: ActionRequest) -> bool:
    """Default approver: always denies permission (used for HIGH_RISK_ACTION)."""
    return False


class PermissionPolicy:
    """
    Controls which actions are permitted based on ActionRisk level.

    The policy maps each risk level to an ApprovalCallback.
    Callers may inject custom callbacks for testing or GUI integration.

    Default behaviour
    -----------------
    READ_ONLY        → auto-approve
    LOW_RISK_ACTION  → auto-deny  (caller must inject a real approval mechanism)
    HIGH_RISK_ACTION → auto-deny  (unconditionally blocked)
    """

    def __init__(
        self,
        read_only_approver: Optional[ApprovalCallback] = None,
        low_risk_approver: Optional[ApprovalCallback] = None,
        high_risk_approver: Optional[ApprovalCallback] = None,
    ):
        self._approvers: Dict[ActionRisk, ApprovalCallback] = {
            ActionRisk.READ_ONLY: read_only_approver or _auto_approve,
            ActionRisk.LOW_RISK_ACTION: low_risk_approver or _auto_deny,
            ActionRisk.HIGH_RISK_ACTION: high_risk_approver or _auto_deny,
        }

    def request_approval(self, request: ActionRequest) -> bool:
        """
        Ask the appropriate approver for the given request.
        Returns True if approved, False if denied.
        HIGH_RISK_ACTION is structurally blocked regardless of the injected approver.
        """
        if request.risk == ActionRisk.HIGH_RISK_ACTION:
            return False  # Structural safety: high-risk is always blocked
        approver = self._approvers.get(request.risk, _auto_deny)
        return approver(request)

    def set_approver(self, risk: ActionRisk, callback: ApprovalCallback) -> None:
        """Replace the approver for the given risk level (used for testing / GUI injection)."""
        if risk == ActionRisk.HIGH_RISK_ACTION:
            raise ValueError("Cannot override the HIGH_RISK_ACTION approver — it is unconditionally blocked.")
        self._approvers[risk] = callback


# Singleton default policy (can be replaced at the application layer)
DEFAULT_POLICY = PermissionPolicy()


# ============================================================
# RESULT
# ============================================================

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


# ============================================================
# BASE TOOL
# ============================================================

class BaseTool(ABC):
    """
    Abstract base class for provider-agnostic BLINDSPOT tools.
    Encapsulates name, description, JSON schema parameters, risk classification,
    and execution logic.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        risk: ActionRisk = ActionRisk.READ_ONLY,
        # Legacy compatibility: allow is_read_only kwarg
        is_read_only: Optional[bool] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}

        # Honour legacy is_read_only kwarg if risk was not explicitly set
        if is_read_only is not None:
            if is_read_only:
                self.risk = ActionRisk.READ_ONLY
            else:
                self.risk = ActionRisk.LOW_RISK_ACTION
        else:
            self.risk = risk

    @property
    def is_read_only(self) -> bool:
        """Backward-compatible property: True only for READ_ONLY tools."""
        return self.risk == ActionRisk.READ_ONLY

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

    def action_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the tool's action risk and classification."""
        return {
            "name": self.name,
            "risk": self.risk.value,
            "is_read_only": self.is_read_only,
            "description": self.description,
        }


# ============================================================
# TOOL REGISTRY
# ============================================================

class ToolRegistry:
    """
    Thread-safe registry for managing and executing tools.
    Includes duplicate detection, schema inspection, permission checking,
    and safety boundary enforcement.
    """

    def __init__(
        self,
        allow_actions: bool = False,
        permission_policy: Optional[PermissionPolicy] = None,
    ):
        self.allow_actions = allow_actions
        if permission_policy is not None:
            self._policy = permission_policy
        elif allow_actions:
            # When allow_actions=True, auto-approve LOW_RISK_ACTION as well
            self._policy = PermissionPolicy(low_risk_approver=_auto_approve)
        else:
            self._policy = PermissionPolicy()
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

        Safety flow:
          0. Reject outright if name is on the BLOCKED_DANGEROUS_TOOL_NAMES list.
          1. Confirm tool exists in the registry.
          2. Reject if registry has allow_actions=False and tool is not READ_ONLY.
          3. Reject if tool is HIGH_RISK_ACTION (unconditionally blocked).
          4. Build an ActionRequest and ask the PermissionPolicy for approval.
          5. Execute the tool only if approved.
          6. Catch and wrap any OS/execution exceptions as structured ToolResult.
        """
        # Step 0: Hard safety block — dangerous names never reach the registry lookup
        if name.lower() in BLOCKED_DANGEROUS_TOOL_NAMES:
            return ToolResult(
                tool_name=name,
                success=False,
                error=(
                    f"Tool '{name}' is categorically blocked by BLINDSPOT safety policy. "
                    "Shell commands, arbitrary code execution, file modification, credential "
                    "handling, messaging, and system control are never permitted."
                ),
                metadata={"blocked_by": "BLOCKED_DANGEROUS_TOOL_NAMES"},
            )

        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered.",
            )

        # Step 2: Legacy safety boundary check
        if not tool.is_read_only and not self.allow_actions:
            return ToolResult(
                tool_name=name,
                success=False,
                error=(
                    f"Tool '{name}' is a mutating action tool, but action execution is "
                    "disabled by the safety boundary."
                ),
            )

        # Step 3: HIGH_RISK_ACTION is unconditionally blocked
        if tool.risk == ActionRisk.HIGH_RISK_ACTION:
            return ToolResult(
                tool_name=name,
                success=False,
                error=(
                    f"Tool '{name}' is classified as HIGH_RISK_ACTION and is unconditionally "
                    "blocked by BLINDSPOT safety policy."
                ),
                metadata={"risk": tool.risk.value, "blocked_by": "HIGH_RISK_ACTION_policy"},
            )

        # Step 4: Permission policy check
        request = ActionRequest(
            tool_name=name,
            risk=tool.risk,
            args=kwargs,
            description=tool.description,
        )
        if not self._policy.request_approval(request):
            return ToolResult(
                tool_name=name,
                success=False,
                error=(
                    f"Permission denied for tool '{name}' "
                    f"(risk={tool.risk.value}). Request was not approved."
                ),
                metadata={"risk": tool.risk.value, "denied_by": "PermissionPolicy"},
            )

        # Step 5 & 6: Execute with structured OS failure handling
        try:
            result = tool.execute(**kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(tool_name=name, success=True, output=result)
        except PermissionError as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"OS permission error during '{name}': {e}",
                metadata={"exception_type": "PermissionError"},
            )
        except OSError as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"OS error during '{name}': {e}",
                metadata={"exception_type": "OSError"},
            )
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Error executing tool '{name}': {e}",
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)


# ============================================================
# SAFE READ-ONLY TOOLS
# ============================================================

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
            risk=ActionRisk.READ_ONLY,
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
            risk=ActionRisk.READ_ONLY,
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
            risk=ActionRisk.READ_ONLY,
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


# ============================================================
# VALIDATION HELPERS
# ============================================================

# Allowlisted application logical names (maps to WindowsDesktopIO.APP_PATHS)
ALLOWED_APPLICATIONS: List[str] = [
    "notepad",
    "calculator",
    "paint",
    "wordpad",
    "explorer",
]

# Allowed mouse buttons
ALLOWED_MOUSE_BUTTONS: List[str] = ["left", "right", "middle"]

# Screen bounds (conservative safe zone — tools reject coordinates outside this)
SCREEN_MAX_X: int = 7680   # Up to 8K wide
SCREEN_MAX_Y: int = 4320   # Up to 8K tall
SCREEN_MIN_COORD: int = 0

# Maximum text length that can be typed in one call
MAX_TYPE_TEXT_LENGTH: int = 500

# Allowed single key names (subset of pyautogui keys)
ALLOWED_KEYS: List[str] = [
    "enter", "return", "tab", "escape", "esc", "space", "backspace", "delete",
    "up", "down", "left", "right",
    "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "capslock", "numlock", "scrolllock",
    "insert",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

# Allowed modifier keys for hotkeys
ALLOWED_HOTKEY_MODIFIERS: List[str] = ["ctrl", "alt", "shift", "win", "command"]

# Complete set of keys valid inside a hotkey combination
ALLOWED_HOTKEY_KEYS: List[str] = ALLOWED_KEYS + ALLOWED_HOTKEY_MODIFIERS


def _validate_app_name(app_name: str) -> Optional[str]:
    """Return error string if app_name is not on the allowlist, else None."""
    if not isinstance(app_name, str) or not app_name.strip():
        return "app_name must be a non-empty string."
    if app_name.lower() not in ALLOWED_APPLICATIONS:
        return (
            f"Application '{app_name}' is not in the allowed list. "
            f"Allowed: {', '.join(ALLOWED_APPLICATIONS)}."
        )
    return None


def _validate_coordinates(x: Any, y: Any) -> Optional[str]:
    """Return error string if coordinates are out of bounds, else None."""
    if not isinstance(x, int) or not isinstance(y, int):
        return f"x and y must be integers, got x={type(x).__name__}, y={type(y).__name__}."
    if not (SCREEN_MIN_COORD <= x <= SCREEN_MAX_X):
        return f"x={x} is outside valid screen bounds [0, {SCREEN_MAX_X}]."
    if not (SCREEN_MIN_COORD <= y <= SCREEN_MAX_Y):
        return f"y={y} is outside valid screen bounds [0, {SCREEN_MAX_Y}]."
    return None


def _validate_mouse_button(button: Any) -> Optional[str]:
    """Return error string if button name is invalid, else None."""
    if button not in ALLOWED_MOUSE_BUTTONS:
        return f"button='{button}' is not allowed. Allowed: {', '.join(ALLOWED_MOUSE_BUTTONS)}."
    return None


def _validate_key(key: Any) -> Optional[str]:
    """Return error string if key name is invalid, else None."""
    if not isinstance(key, str) or not key.strip():
        return "key must be a non-empty string."
    if key.lower() not in ALLOWED_KEYS:
        return (
            f"Key '{key}' is not in the allowed key list. "
            "Only standard alphanumeric and navigation keys are permitted."
        )
    return None


def _validate_hotkey_keys(keys: tuple) -> Optional[str]:
    """Return error string if any key in the hotkey combination is invalid, else None."""
    if not keys:
        return "hotkey requires at least one key."
    if len(keys) > 5:
        return f"hotkey combination is too long ({len(keys)} keys). Maximum is 5."
    bad = [k for k in keys if k.lower() not in ALLOWED_HOTKEY_KEYS]
    if bad:
        return (
            f"Hotkey keys {bad} are not allowed. "
            "Only standard keys and modifiers (ctrl, alt, shift, win) are permitted."
        )
    return None


def _validate_text(text: Any) -> Optional[str]:
    """Return error string if text is invalid for typing, else None."""
    if not isinstance(text, str):
        return f"text must be a string, got {type(text).__name__}."
    if len(text) == 0:
        return "text must not be empty."
    if len(text) > MAX_TYPE_TEXT_LENGTH:
        return (
            f"text length ({len(text)}) exceeds maximum allowed length ({MAX_TYPE_TEXT_LENGTH})."
        )
    return None


# ============================================================
# ACTION TOOLS
# ============================================================

class OpenApplicationTool(BaseTool):
    """LOW_RISK action: Open an allowlisted desktop application."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="open_application",
            description=(
                f"Open an allowlisted desktop application. "
                f"Allowed apps: {', '.join(ALLOWED_APPLICATIONS)}. "
                "Never opens arbitrary executables or shell commands."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": (
                            f"Logical name of the application to open. "
                            f"Must be one of: {', '.join(ALLOWED_APPLICATIONS)}."
                        ),
                    }
                },
                "required": ["app_name"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(self, app_name: str = "", **kwargs) -> ToolResult:
        err = _validate_app_name(app_name)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.open_application(app_name.lower())
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


class FocusApplicationTool(BaseTool):
    """LOW_RISK action: Bring a window to the foreground by title fragment."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="focus_application",
            description="Bring a window to the foreground by matching a fragment of its title.",
            parameters={
                "type": "object",
                "properties": {
                    "window_title_fragment": {
                        "type": "string",
                        "description": "Substring of the target window title (case-insensitive match).",
                    }
                },
                "required": ["window_title_fragment"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(self, window_title_fragment: str = "", **kwargs) -> ToolResult:
        if not isinstance(window_title_fragment, str) or not window_title_fragment.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="window_title_fragment must be a non-empty string.",
            )
        if len(window_title_fragment) > 200:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="window_title_fragment is too long (max 200 characters).",
            )
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.focus_application(window_title_fragment)
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


class MoveMouseTool(BaseTool):
    """LOW_RISK action: Move the mouse pointer to screen coordinates."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="move_mouse",
            description="Move the mouse pointer to the specified screen coordinates (x, y).",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in pixels."},
                    "y": {"type": "integer", "description": "Y coordinate in pixels."},
                    "duration": {
                        "type": "number",
                        "description": "Movement duration in seconds (default 0.2).",
                        "default": 0.2,
                    },
                },
                "required": ["x", "y"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(self, x: int = 0, y: int = 0, duration: float = 0.2, **kwargs) -> ToolResult:
        err = _validate_coordinates(x, y)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        if not isinstance(duration, (int, float)) or duration < 0 or duration > 10:
            return ToolResult(
                tool_name=self.name, success=False,
                error="duration must be a number in [0, 10] seconds.",
            )
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.move_mouse(x, y, duration=float(duration))
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


class ClickTool(BaseTool):
    """LOW_RISK action: Click a mouse button at screen coordinates."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="click",
            description=(
                "Click a mouse button at the specified screen coordinates. "
                f"Allowed buttons: {', '.join(ALLOWED_MOUSE_BUTTONS)}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in pixels."},
                    "y": {"type": "integer", "description": "Y coordinate in pixels."},
                    "button": {
                        "type": "string",
                        "description": "Mouse button: 'left', 'right', or 'middle'.",
                        "default": "left",
                    },
                    "clicks": {
                        "type": "integer",
                        "description": "Number of clicks (1 for single, 2 for double).",
                        "default": 1,
                    },
                },
                "required": ["x", "y"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(
        self,
        x: int = 0,
        y: int = 0,
        button: str = "left",
        clicks: int = 1,
        **kwargs,
    ) -> ToolResult:
        err = _validate_coordinates(x, y)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        err = _validate_mouse_button(button)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        if not isinstance(clicks, int) or clicks < 1 or clicks > 3:
            return ToolResult(
                tool_name=self.name, success=False,
                error="clicks must be an integer in [1, 3].",
            )
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.click(x, y, button=button, clicks=clicks)
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


class TypeTextTool(BaseTool):
    """LOW_RISK action: Type text into the currently focused control."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="type_text",
            description=(
                f"Type text into the currently focused control (max {MAX_TYPE_TEXT_LENGTH} characters). "
                "Never types passwords or sensitive credentials."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": f"Text to type (max {MAX_TYPE_TEXT_LENGTH} characters).",
                    },
                    "interval": {
                        "type": "number",
                        "description": "Delay between keystrokes in seconds (default 0.02).",
                        "default": 0.02,
                    },
                },
                "required": ["text"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(self, text: str = "", interval: float = 0.02, **kwargs) -> ToolResult:
        err = _validate_text(text)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        if not isinstance(interval, (int, float)) or interval < 0 or interval > 1:
            return ToolResult(
                tool_name=self.name, success=False,
                error="interval must be a number in [0, 1] seconds.",
            )
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.type_text(text, interval=float(interval))
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


class PressKeyTool(BaseTool):
    """LOW_RISK action: Press a single keyboard key."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="press_key",
            description=(
                "Press a single keyboard key. "
                "Only standard alphanumeric and navigation keys are permitted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Key name (e.g. 'enter', 'tab', 'escape', 'f5', 'a', '1'). "
                            "Must be a key in the allowed key list."
                        ),
                    }
                },
                "required": ["key"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(self, key: str = "", **kwargs) -> ToolResult:
        err = _validate_key(key)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.press_key(key.lower())
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


class HotkeyTool(BaseTool):
    """LOW_RISK action: Press a keyboard shortcut (hotkey combination)."""

    def __init__(self, desktop_io=None):
        super().__init__(
            name="hotkey",
            description=(
                "Press a keyboard shortcut (e.g. ctrl+c, ctrl+z, alt+tab). "
                "Only combinations of allowed keys and modifiers (ctrl, alt, shift, win) are permitted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Ordered list of key names forming the shortcut "
                            "(e.g. ['ctrl', 'c'] for copy)."
                        ),
                    }
                },
                "required": ["keys"],
            },
            risk=ActionRisk.LOW_RISK_ACTION,
        )
        self._io = desktop_io

    def execute(self, keys: Optional[List[str]] = None, **kwargs) -> ToolResult:
        if keys is None:
            keys = []
        if not isinstance(keys, list):
            return ToolResult(
                tool_name=self.name, success=False,
                error="keys must be a list of key name strings.",
            )
        keys_tuple = tuple(k.lower() if isinstance(k, str) else k for k in keys)
        err = _validate_hotkey_keys(keys_tuple)
        if err:
            return ToolResult(tool_name=self.name, success=False, error=err)
        io = self._io
        if io is None:
            from desktop_io import WindowsDesktopIO
            io = WindowsDesktopIO()
        result = io.hotkey(*keys_tuple)
        return ToolResult(
            tool_name=self.name,
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
        )


# ============================================================
# FACTORY
# ============================================================

def create_default_tool_registry(
    state: Optional[Dict[str, Any]] = None,
    conversation: Optional[ConversationSession] = None,
    history: Optional[ContextHistory] = None,
    allow_actions: bool = False,
    permission_policy: Optional[PermissionPolicy] = None,
    desktop_io=None,
    include_action_tools: bool = False,
) -> ToolRegistry:
    """
    Factory function to create a ToolRegistry populated with the default safe read-only tools,
    wired to BLINDSPOT's live state.

    If include_action_tools=True, also registers the desktop action tools.
    Action tools still require allow_actions=True and permission approval to execute.
    """
    registry = ToolRegistry(allow_actions=allow_actions, permission_policy=permission_policy)
    state_getter = (lambda: state) if state is not None else None

    registry.register(GetCurrentContextTool(state_getter=state_getter))
    registry.register(GetRecentActivityTool(state_getter=state_getter, history=history))
    registry.register(GetConversationHistoryTool(state_getter=state_getter, conversation=conversation))

    if include_action_tools:
        registry.register(OpenApplicationTool(desktop_io=desktop_io))
        registry.register(FocusApplicationTool(desktop_io=desktop_io))
        registry.register(MoveMouseTool(desktop_io=desktop_io))
        registry.register(ClickTool(desktop_io=desktop_io))
        registry.register(TypeTextTool(desktop_io=desktop_io))
        registry.register(PressKeyTool(desktop_io=desktop_io))
        registry.register(HotkeyTool(desktop_io=desktop_io))

    return registry

