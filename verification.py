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
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
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

    def observe_fresh(self) -> DesktopContext:
        """Obtain instantaneous desktop context without relying on cached perception state."""
        return self.observe()


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
        prefer_fresh_window: bool = False,
    ):
        self._provider = context_provider
        self._state = state
        self._prefer_fresh_window = prefer_fresh_window

    def observe(self) -> DesktopContext:
        if self._prefer_fresh_window:
            return self.observe_fresh()

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
        return self._query_perception_modules()

    def observe_fresh(self) -> DesktopContext:
        """Obtain instantaneous desktop context using live OS window query."""
        # 1. Custom provider takes highest precedence
        if self._provider is not None:
            try:
                ctx = self._provider()
                if ctx is not None:
                    return ctx
            except Exception:
                pass

        # 2. Query live OS window detection module directly
        try:
            from eyes.window import get_active_window, get_active_window_rect
            from eyes.interpreter import interpret_context

            window_title = get_active_window()
            if window_title is not None and str(window_title).strip():
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
            pass

        # 3. Safe empty fallback — do not use cached state['current_context'] for fresh observation
        return DesktopContext(
            observation=RawObservation(window_title="", ocr_text=""),
            interpretation=InterpretedContext(activity="unknown", confidence=0),
        )

    def _query_perception_modules(self) -> DesktopContext:
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

    def observe_fresh(self) -> DesktopContext:
        return self.observe()


# ============================================================
# TASK STEP VERIFIER
# ============================================================

class TaskVerifier:
    """
    Verifies the outcome of individual task steps against observed desktop context.
    Uses bounded polling to accommodate asynchronous desktop application launches.
    """

    APP_ALIASES: Dict[str, List[str]] = {
        "notepad": ["notepad"],
        "calculator": ["calculator", "calc"],
        "paint": ["paint"],
        "wordpad": ["wordpad"],
        "explorer": ["explorer", "file explorer"],
    }

    def __init__(
        self,
        observer: Optional[Any] = None,
        poll_timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ):
        if observer is None:
            self._observer: DesktopObserver = DefaultContextObserver(prefer_fresh_window=True)
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
            self._observer = DefaultContextObserver(prefer_fresh_window=True)

        if poll_timeout is not None:
            self.poll_timeout: float = max(0.0, float(poll_timeout))
        elif isinstance(self._observer, DefaultContextObserver) and self._observer._provider is None:
            self.poll_timeout = 2.5
        else:
            self.poll_timeout = 0.0

        self.poll_interval: float = max(0.001, float(poll_interval))

    @property
    def observer(self) -> DesktopObserver:
        return self._observer

    def _get_fresh_context(self) -> DesktopContext:
        if hasattr(self._observer, "observe_fresh") and callable(self._observer.observe_fresh):
            return self._observer.observe_fresh()
        return self._observer.observe()

    def _poll_for_window(
        self,
        predicate: Callable[[str], bool],
    ) -> Tuple[bool, DesktopContext]:
        start = time.time()
        timeout = self.poll_timeout
        interval = self.poll_interval

        while True:
            try:
                ctx = self._get_fresh_context()
            except Exception:
                ctx = DesktopContext(
                    observation=RawObservation(window_title="", ocr_text=""),
                    interpretation=InterpretedContext(activity="unknown", confidence=0),
                )

            title = (ctx.window_title or "").strip()
            if predicate(title):
                return True, ctx

            if timeout <= 0.0 or (time.time() - start) >= timeout:
                return False, ctx

            time.sleep(min(interval, max(0.0, timeout - (time.time() - start))))

    def verify_step(self, step: Any, tool: Optional[Any] = None) -> StepVerificationResult:
        """
        Verify the post-execution state of a task step against observed desktop state.
        Never fabricates verification status from mock tool attributes.
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

        aliases = self.APP_ALIASES.get(app_name, [app_name])

        def _matches(observed_title: str) -> bool:
            obs_lower = observed_title.lower()
            return any(alias in obs_lower for alias in aliases)

        matched, context = self._poll_for_window(_matches)
        observed_title = (context.window_title or "").strip()

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

        frag_lower = fragment.lower()

        def _matches(observed_title: str) -> bool:
            return frag_lower in observed_title.lower()

        matched, context = self._poll_for_window(_matches)
        observed_title = (context.window_title or "").strip()

        if matched:
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
