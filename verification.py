"""
BLINDSPOT — Task Step Verification Abstraction (Phase 7C).

Provides:
- VerificationStatus: Enum representing verified, failed, or not_verifiable.
- StepVerificationResult: Structured outcome of post-execution step verification.
- DesktopObserver: Abstract perception observation interface.
- DefaultContextObserver: Uses existing perception architecture without duplicating OCR/window detection.
- MockDesktopObserver: Deterministic mock perception observer for tests.
- TaskVerifier: Logic to verify open_application and focus_application against desktop context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
from typing import Dict, Any, Optional, Callable, List
from context import DesktopContext, RawObservation, InterpretedContext


class VerificationStatus(enum.Enum):
    """Status of post-execution verification for a task step."""
    VERIFIED = "verified"
    FAILED = "failed"
    NOT_VERIFIABLE = "not_verifiable"


@dataclass
class StepVerificationResult:
    """
    Structured outcome of post-execution verification for a task step.

    Fields:
    - status: VerificationStatus (VERIFIED, FAILED, NOT_VERIFIABLE)
    - message: Human-readable message explaining the verification outcome.
    - details: Dictionary containing observed window title, activity, etc.
    """
    status: VerificationStatus
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verification result to dictionary representation."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


# ============================================================
# OBSERVATION INTERFACE
# ============================================================

class DesktopObserver(ABC):
    """
    Abstract interface for observing the current desktop environment.
    Enables pluggable perception observation without duplicating perception code.
    """
    @abstractmethod
    def observe(self) -> DesktopContext:
        """Obtain current desktop context."""
        ...


class DefaultContextObserver(DesktopObserver):
    """
    Concrete observer that obtains current desktop context using existing
    perception architecture without duplicating OCR, screen capture, or window detection.

    Sources checked in priority order:
    1. Custom context_provider callable, if provided.
    2. Live state dictionary (e.g. state['current_context']), if provided.
    3. Direct query to existing perception modules (eyes.window.get_active_window(),
       eyes.interpreter.interpret_context()).
    """
    def __init__(
        self,
        context_provider: Optional[Callable[[], Optional[DesktopContext]]] = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        self._provider = context_provider
        self._state = state

    def observe(self) -> DesktopContext:
        # 1. Custom provider
        if self._provider is not None:
            try:
                ctx = self._provider()
                if ctx is not None:
                    return ctx
            except Exception:
                pass

        # 2. State dictionary
        if self._state is not None:
            ctx = self._state.get("current_context")
            if ctx is not None:
                return ctx

        # 3. Existing perception modules
        try:
            from eyes.window import get_active_window, get_active_window_rect
            from eyes.interpreter import interpret_context

            window_title = get_active_window() or ""
            rect = get_active_window_rect()
            activity, confidence = interpret_context(window_title, "")
            obs = RawObservation(
                window_title=window_title,
                ocr_text="",
                window_rect=rect,
            )
            interp = InterpretedContext(
                activity=activity,
                confidence=confidence,
            )
            return DesktopContext(
                observation=obs,
                interpretation=interp,
            )
        except Exception:
            return DesktopContext(
                observation=RawObservation(window_title="", ocr_text=""),
                interpretation=InterpretedContext(activity="using an application", confidence=1),
            )


class MockDesktopObserver(DesktopObserver):
    """
    Deterministic mock observer for testing verification logic without touching the desktop.
    """
    def __init__(self, context: Optional[DesktopContext] = None, window_title: str = ""):
        self._context = context
        self._window_title = window_title

    def set_window(self, window_title: str, activity: str = "using an application", confidence: int = 2) -> None:
        self._window_title = window_title
        obs = RawObservation(window_title=window_title, ocr_text="")
        interp = InterpretedContext(activity=activity, confidence=confidence)
        self._context = DesktopContext(observation=obs, interpretation=interp)

    def observe(self) -> DesktopContext:
        if self._context is not None:
            return self._context
        obs = RawObservation(window_title=self._window_title, ocr_text="")
        interp = InterpretedContext(activity="using an application", confidence=2)
        return DesktopContext(observation=obs, interpretation=interp)


# ============================================================
# TASK STEP VERIFIER
# ============================================================

class TaskVerifier:
    """
    Verifies the outcome of individual task steps against observed desktop context.
    """

    APP_ALIASES: Dict[str, List[str]] = {
        "notepad": ["notepad"],
        "calculator": ["calculator", "calc"],
        "paint": ["paint"],
        "wordpad": ["wordpad"],
        "explorer": ["explorer", "file explorer"],
    }

    def __init__(self, observer: Optional[Any] = None):
        if observer is None:
            self._observer: DesktopObserver = DefaultContextObserver()
        elif hasattr(observer, "observe"):
            self._observer = observer
        elif callable(observer):
            class _CallableObserver(DesktopObserver):
                def observe(self) -> DesktopContext:
                    res = observer()
                    if isinstance(res, DesktopContext):
                        return res
                    obs = RawObservation(window_title=str(res or ""), ocr_text="")
                    interp = InterpretedContext(activity="using an application", confidence=2)
                    return DesktopContext(observation=obs, interpretation=interp)
            self._observer = _CallableObserver()
        else:
            self._observer = DefaultContextObserver()

    @property
    def observer(self) -> DesktopObserver:
        return self._observer

    def _get_context(self, tool: Optional[Any] = None) -> DesktopContext:
        # Check if the tool has an active_window on its MockDesktopIO
        if tool is not None:
            io = getattr(tool, "_io", None)
            if io is not None and hasattr(io, "active_window") and io.active_window is not None:
                # If using DefaultContextObserver, prefer mock active_window over live desktop
                if isinstance(self._observer, DefaultContextObserver):
                    obs = RawObservation(window_title=io.active_window, ocr_text="")
                    interp = InterpretedContext(activity="using an application", confidence=2)
                    return DesktopContext(observation=obs, interpretation=interp)
        return self._observer.observe()

    def verify_step(self, step: Any, tool: Optional[Any] = None) -> StepVerificationResult:
        """
        Verify the post-execution state of a task step.
        """
        tool_name = getattr(step, "tool_name", "")
        if tool_name == "open_application":
            return self._verify_open_application(step, tool)
        elif tool_name == "focus_application":
            return self._verify_focus_application(step, tool)
        else:
            return StepVerificationResult(
                status=VerificationStatus.NOT_VERIFIABLE,
                message=f"Action '{tool_name}' does not support reliable post-execution desktop verification.",
                details={"tool_name": tool_name},
            )

    def _verify_open_application(self, step: Any, tool: Optional[Any] = None) -> StepVerificationResult:
        args = getattr(step, "args", {}) or {}
        app_name = (args.get("app_name") or "").strip().lower()
        if not app_name:
            return StepVerificationResult(
                status=VerificationStatus.FAILED,
                message="Verification failed: missing 'app_name' argument for open_application.",
                details={"args": args},
            )

        context = self._get_context(tool)
        observed_title = (context.window_title or "").strip()
        observed_lower = observed_title.lower()

        aliases = self.APP_ALIASES.get(app_name, [app_name])
        matched = any(alias in observed_lower for alias in aliases)

        if matched:
            return StepVerificationResult(
                status=VerificationStatus.VERIFIED,
                message=f"Application '{app_name}' verified active on desktop (observed window: '{observed_title}').",
                details={
                    "app_name": app_name,
                    "observed_window": observed_title,
                    "activity": context.activity,
                },
            )
        else:
            return StepVerificationResult(
                status=VerificationStatus.FAILED,
                message=(
                    f"Verification failed for open_application: expected '{app_name}' active on desktop, "
                    f"but observed window '{observed_title}'."
                ),
                details={
                    "expected_app": app_name,
                    "observed_window": observed_title,
                    "activity": context.activity,
                },
            )

    def _verify_focus_application(self, step: Any, tool: Optional[Any] = None) -> StepVerificationResult:
        args = getattr(step, "args", {}) or {}
        fragment = (args.get("window_title_fragment") or "").strip()
        if not fragment:
            return StepVerificationResult(
                status=VerificationStatus.FAILED,
                message="Verification failed: missing 'window_title_fragment' argument for focus_application.",
                details={"args": args},
            )

        context = self._get_context(tool)
        observed_title = (context.window_title or "").strip()

        if fragment.lower() in observed_title.lower():
            return StepVerificationResult(
                status=VerificationStatus.VERIFIED,
                message=f"Window focus verified for fragment '{fragment}' (observed active window: '{observed_title}').",
                details={
                    "fragment": fragment,
                    "observed_window": observed_title,
                    "activity": context.activity,
                },
            )
        else:
            return StepVerificationResult(
                status=VerificationStatus.FAILED,
                message=(
                    f"Verification failed for focus_application: active window '{observed_title}' "
                    f"does not match fragment '{fragment}'."
                ),
                details={
                    "fragment": fragment,
                    "observed_window": observed_title,
                    "activity": context.activity,
                },
            )
