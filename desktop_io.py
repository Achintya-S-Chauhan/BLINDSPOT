"""
desktop_io.py — Controlled Desktop I/O Abstraction for BLINDSPOT.

Provides a clean, mockable interface for performing safe, low-level desktop
interactions on Windows. Action tools MUST use this abstraction rather than
calling Windows APIs directly.

Architecture
------------
DesktopIO (abstract base)
  ├── WindowsDesktopIO   — Real Windows implementation via pyautogui / pygetwindow
  └── MockDesktopIO      — In-process mock for tests; records calls, never touches hardware

All methods return DesktopIOResult so that callers always receive a structured
success/failure response rather than raising exceptions.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# RESULT TYPE
# ============================================================

@dataclass
class DesktopIOResult:
    """Structured result from a DesktopIO operation."""
    success: bool
    operation: str
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ============================================================
# ABSTRACT INTERFACE
# ============================================================

class DesktopIO(ABC):
    """
    Abstract interface for desktop I/O operations.
    All concrete implementations MUST honour the contract defined here.
    """

    @abstractmethod
    def open_application(self, app_name: str) -> DesktopIOResult:
        """
        Launch an application by its logical name (e.g. 'notepad', 'calculator').
        The implementation is responsible for mapping logical names to actual paths.
        """
        ...

    @abstractmethod
    def focus_application(self, window_title_fragment: str) -> DesktopIOResult:
        """
        Bring a window whose title contains `window_title_fragment` to the foreground.
        """
        ...

    @abstractmethod
    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> DesktopIOResult:
        """Move the mouse pointer to screen coordinates (x, y)."""
        ...

    @abstractmethod
    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> DesktopIOResult:
        """
        Click the specified mouse button at screen coordinates (x, y).
        `button` must be one of: 'left', 'right', 'middle'.
        """
        ...

    @abstractmethod
    def type_text(self, text: str, interval: float = 0.02) -> DesktopIOResult:
        """Type `text` into the currently focused control."""
        ...

    @abstractmethod
    def press_key(self, key: str) -> DesktopIOResult:
        """
        Press a single keyboard key identified by its pyautogui name
        (e.g. 'enter', 'tab', 'escape', 'f5').
        """
        ...

    @abstractmethod
    def hotkey(self, *keys: str) -> DesktopIOResult:
        """
        Press a keyboard shortcut defined by a sequence of key names
        (e.g. hotkey('ctrl', 'c') for copy).
        """
        ...


# ============================================================
# WINDOWS IMPLEMENTATION
# ============================================================

class WindowsDesktopIO(DesktopIO):
    """
    Real Windows implementation of DesktopIO using pyautogui and pygetwindow.

    pyautogui provides mouse/keyboard control.
    pygetwindow provides window focus management.

    If neither library is installed, all methods return a structured error
    rather than crashing BLINDSPOT — graceful degradation.
    """

    # Logical name → Windows executable mapping (allowlist is enforced at tool layer)
    APP_PATHS: Dict[str, str] = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "wordpad": "wordpad.exe",
        "explorer": "explorer.exe",
    }

    def _try_import_pyautogui(self):
        try:
            import pyautogui
            return pyautogui
        except ImportError:
            return None

    def _try_import_win32(self):
        try:
            import win32gui
            import win32con
            return win32gui, win32con
        except ImportError:
            return None, None

    def _try_import_pygetwindow(self):
        try:
            import pygetwindow as gw
            return gw
        except ImportError:
            return None

    def open_application(self, app_name: str) -> DesktopIOResult:
        import subprocess
        exe = self.APP_PATHS.get(app_name.lower())
        if not exe:
            return DesktopIOResult(
                success=False,
                operation="open_application",
                error=f"Application '{app_name}' is not in the allowed application list.",
                metadata={"requested": app_name},
            )
        try:
            subprocess.Popen([exe], shell=False)
            return DesktopIOResult(
                success=True,
                operation="open_application",
                output=f"Launched '{app_name}' ({exe}).",
                metadata={"app_name": app_name, "exe": exe},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="open_application",
                error=f"Failed to launch '{app_name}': {e}",
                metadata={"app_name": app_name},
            )

    def focus_application(self, window_title_fragment: str) -> DesktopIOResult:
        fragment_clean = (window_title_fragment or "").strip()
        if not fragment_clean:
            return DesktopIOResult(
                success=False,
                operation="focus_application",
                error="window_title_fragment must be a non-empty string.",
                metadata={"fragment": window_title_fragment},
            )

        win32gui_mod, win32con_mod = self._try_import_win32()
        if win32gui_mod is not None:
            fragment_lower = fragment_clean.lower()
            matching_windows: List[Tuple[int, str]] = []

            def _enum_callback(hwnd: int, _lparam: Any) -> bool:
                try:
                    if win32gui_mod.IsWindowVisible(hwnd):
                        title = win32gui_mod.GetWindowText(hwnd)
                        if title and fragment_lower in title.lower():
                            matching_windows.append((hwnd, title))
                except Exception:
                    pass
                return True

            try:
                win32gui_mod.EnumWindows(_enum_callback, None)
            except Exception as e:
                return DesktopIOResult(
                    success=False,
                    operation="focus_application",
                    error=f"Failed to enumerate windows: {e}",
                    metadata={"fragment": window_title_fragment},
                )

            if not matching_windows:
                return DesktopIOResult(
                    success=False,
                    operation="focus_application",
                    error=f"No window found with title containing '{window_title_fragment}'.",
                    metadata={"fragment": window_title_fragment},
                )

            target_hwnd, target_title = matching_windows[0]

            try:
                # Restore window if minimized
                if win32gui_mod.IsIconic(target_hwnd):
                    win32gui_mod.ShowWindow(target_hwnd, win32con_mod.SW_RESTORE)
                else:
                    win32gui_mod.ShowWindow(target_hwnd, win32con_mod.SW_SHOW)
            except Exception:
                pass

            def _is_foreground_confirmed() -> bool:
                try:
                    return win32gui_mod.GetForegroundWindow() == target_hwnd
                except Exception:
                    return False

            focused = False
            last_err = None

            # Attempt 1: win32gui.SetForegroundWindow
            try:
                ret = win32gui_mod.SetForegroundWindow(target_hwnd)
                # Never trust API return value alone; verify window actually became foreground
                if ret is not False and ret != 0:
                    focused = _is_foreground_confirmed()
                else:
                    focused = _is_foreground_confirmed()
            except Exception as e:
                last_err = e
                focused = _is_foreground_confirmed()

            # Attempt 2: ctypes user32.SetForegroundWindow fallback if not yet confirmed focused
            if not focused:
                try:
                    import ctypes
                    c_ret = ctypes.windll.user32.SetForegroundWindow(target_hwnd)
                    # Check return value AND verify window actually became foreground
                    if c_ret:
                        focused = _is_foreground_confirmed()
                    else:
                        focused = _is_foreground_confirmed()
                except Exception as e:
                    if last_err is None:
                        last_err = e
                    focused = _is_foreground_confirmed()

            if focused:
                return DesktopIOResult(
                    success=True,
                    operation="focus_application",
                    output=f"Focused window: '{target_title}'.",
                    metadata={"window_title": target_title, "hwnd": target_hwnd},
                )
            else:
                err_detail = f": {last_err}" if last_err else " (SetForegroundWindow returned failure and window was not active)"
                return DesktopIOResult(
                    success=False,
                    operation="focus_application",
                    error=f"Failed to focus window '{target_title}'{err_detail}",
                    metadata={"fragment": window_title_fragment, "target_title": target_title, "hwnd": target_hwnd},
                )

        # Fallback to pygetwindow if win32gui is not available
        gw = self._try_import_pygetwindow()
        if gw is None:
            return DesktopIOResult(
                success=False,
                operation="focus_application",
                error="Neither win32gui nor pygetwindow is available. Cannot focus windows.",
            )
        try:
            windows = gw.getWindowsWithTitle(window_title_fragment)
            if not windows:
                return DesktopIOResult(
                    success=False,
                    operation="focus_application",
                    error=f"No window found with title containing '{window_title_fragment}'.",
                    metadata={"fragment": window_title_fragment},
                )
            win = windows[0]
            win.activate()
            return DesktopIOResult(
                success=True,
                operation="focus_application",
                output=f"Focused window: '{win.title}'.",
                metadata={"window_title": win.title},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="focus_application",
                error=f"Failed to focus window '{window_title_fragment}': {e}",
            )

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> DesktopIOResult:
        pag = self._try_import_pyautogui()
        if pag is None:
            return DesktopIOResult(
                success=False,
                operation="move_mouse",
                error="pyautogui is not installed. Cannot control the mouse.",
            )
        try:
            pag.moveTo(x, y, duration=duration)
            return DesktopIOResult(
                success=True,
                operation="move_mouse",
                output=f"Mouse moved to ({x}, {y}).",
                metadata={"x": x, "y": y, "duration": duration},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="move_mouse",
                error=f"Failed to move mouse to ({x}, {y}): {e}",
            )

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> DesktopIOResult:
        pag = self._try_import_pyautogui()
        if pag is None:
            return DesktopIOResult(
                success=False,
                operation="click",
                error="pyautogui is not installed. Cannot perform mouse clicks.",
            )
        try:
            pag.click(x, y, clicks=clicks, button=button)
            return DesktopIOResult(
                success=True,
                operation="click",
                output=f"Clicked {button} button {clicks}x at ({x}, {y}).",
                metadata={"x": x, "y": y, "button": button, "clicks": clicks},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="click",
                error=f"Failed to click at ({x}, {y}): {e}",
            )

    def type_text(self, text: str, interval: float = 0.02) -> DesktopIOResult:
        pag = self._try_import_pyautogui()
        if pag is None:
            return DesktopIOResult(
                success=False,
                operation="type_text",
                error="pyautogui is not installed. Cannot type text.",
            )
        try:
            pag.typewrite(text, interval=interval)
            return DesktopIOResult(
                success=True,
                operation="type_text",
                output=f"Typed {len(text)} characters.",
                metadata={"length": len(text)},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="type_text",
                error=f"Failed to type text: {e}",
            )

    def press_key(self, key: str) -> DesktopIOResult:
        pag = self._try_import_pyautogui()
        if pag is None:
            return DesktopIOResult(
                success=False,
                operation="press_key",
                error="pyautogui is not installed. Cannot press keys.",
            )
        try:
            pag.press(key)
            return DesktopIOResult(
                success=True,
                operation="press_key",
                output=f"Pressed key '{key}'.",
                metadata={"key": key},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="press_key",
                error=f"Failed to press key '{key}': {e}",
            )

    def hotkey(self, *keys: str) -> DesktopIOResult:
        pag = self._try_import_pyautogui()
        if pag is None:
            return DesktopIOResult(
                success=False,
                operation="hotkey",
                error="pyautogui is not installed. Cannot press hotkeys.",
            )
        try:
            pag.hotkey(*keys)
            combo = "+".join(keys)
            return DesktopIOResult(
                success=True,
                operation="hotkey",
                output=f"Pressed hotkey '{combo}'.",
                metadata={"keys": list(keys)},
            )
        except Exception as e:
            return DesktopIOResult(
                success=False,
                operation="hotkey",
                error=f"Failed to press hotkey '{'+'.join(keys)}': {e}",
            )


# ============================================================
# MOCK IMPLEMENTATION (tests + dry-run)
# ============================================================

@dataclass
class MockCall:
    """Record of a single call made on MockDesktopIO."""
    method: str
    args: Tuple
    kwargs: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class MockDesktopIO(DesktopIO):
    """
    In-process mock that records all calls without touching real hardware.
    Suitable for unit tests and dry-run mode.

    Inject `fail_operations` with operation names to simulate OS failures.
    Inspect `calls` after execution to verify what was invoked.
    """

    def __init__(
        self,
        fail_operations: Optional[List[str]] = None,
        initial_active_window: Optional[str] = None,
    ):
        self.calls: List[MockCall] = []
        self._fail_ops: set = set(fail_operations or [])
        self.active_window: Optional[str] = initial_active_window

    def _record(self, method: str, *args, **kwargs) -> DesktopIOResult:
        self.calls.append(MockCall(method=method, args=args, kwargs=kwargs))
        if method in self._fail_ops:
            return DesktopIOResult(
                success=False,
                operation=method,
                error=f"[MOCK] Simulated OS failure for operation '{method}'.",
                metadata={"simulated_failure": True},
            )
        return DesktopIOResult(
            success=True,
            operation=method,
            output=f"[MOCK] {method} executed successfully.",
            metadata={"mock": True, "args": args, "kwargs": kwargs},
        )

    def open_application(self, app_name: str) -> DesktopIOResult:
        res = self._record("open_application", app_name)
        if res.success:
            self.active_window = app_name
        return res

    def focus_application(self, window_title_fragment: str) -> DesktopIOResult:
        res = self._record("focus_application", window_title_fragment)
        if res.success:
            self.active_window = window_title_fragment
        return res

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> DesktopIOResult:
        return self._record("move_mouse", x, y, duration=duration)

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> DesktopIOResult:
        return self._record("click", x, y, button=button, clicks=clicks)

    def type_text(self, text: str, interval: float = 0.02) -> DesktopIOResult:
        return self._record("type_text", text, interval=interval)

    def press_key(self, key: str) -> DesktopIOResult:
        return self._record("press_key", key)

    def hotkey(self, *keys: str) -> DesktopIOResult:
        return self._record("hotkey", *keys)

    def reset(self) -> None:
        """Clear recorded calls (useful between test cases)."""
        self.calls.clear()
        self.active_window = None

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_call(self, method: Optional[str] = None) -> Optional[MockCall]:
        """Return the last recorded call, optionally filtered by method name."""
        if method:
            filtered = [c for c in self.calls if c.method == method]
            return filtered[-1] if filtered else None
        return self.calls[-1] if self.calls else None


# ============================================================
# BACKEND FACTORY & SELECTION HELPERS
# ============================================================

def get_default_desktop_io(mock: bool = False) -> DesktopIO:
    """
    Select and instantiate the appropriate DesktopIO backend.

    Selection rules:
    - If `mock` is True: returns MockDesktopIO().
    - On Windows (sys.platform == 'win32'): returns WindowsDesktopIO().
    - On non-Windows platforms (e.g. Linux, macOS): returns MockDesktopIO()
      to preserve cross-platform compatibility without crashing.
    """
    if mock:
        return MockDesktopIO()
    if sys.platform == "win32":
        return WindowsDesktopIO()
    return MockDesktopIO()


def get_backend_label(desktop_io: Any) -> str:
    """Return a human-readable label ('REAL WINDOWS I/O' or 'MOCK I/O') for the active backend."""
    if isinstance(desktop_io, MockDesktopIO):
        return "MOCK I/O"
    elif isinstance(desktop_io, WindowsDesktopIO):
        return "REAL WINDOWS I/O"
    return "UNKNOWN I/O"
