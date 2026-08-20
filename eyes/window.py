import sys
import ctypes
from ctypes import wintypes
import win32gui

# Ensure the process is DPI-aware so Win32 window coordinates match MSS physical pixel coordinates
try:
    if sys.platform == "win32":
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def get_active_window() -> str:
    """Return the title of the active foreground window."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd and win32gui.IsWindow(hwnd):
            return win32gui.GetWindowText(hwnd)
        return ""
    except Exception:
        return ""


def get_active_window_rect() -> dict | None:
    """
    Return the bounding rectangle dict {"left": int, "top": int, "width": int, "height": int}
    of the active foreground window.
    Returns None if no active window, if minimized (iconic), or if rectangle dimensions are invalid.
    """
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd or not win32gui.IsWindow(hwnd):
            return None

        # Minimized windows should not be captured
        if win32gui.IsIconic(hwnd):
            return None

        # Prefer DwmGetWindowAttribute (DWMWA_EXTENDED_FRAME_BOUNDS = 9) to obtain true visible bounds
        # without the legacy invisible window resize border padding on Windows 10/11
        rect = wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )

        if res == 0:
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        else:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)

        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        return {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
    except Exception:
        return None