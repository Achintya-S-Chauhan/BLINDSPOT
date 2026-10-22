"""
tests/test_backend_selection.py — Phase 8C Stage 3 Tests: Runtime Backend Selection.

Covers:
1. get_default_desktop_io selection rules on Windows (default WindowsDesktopIO, explicit MockDesktopIO).
2. Cross-platform safety on non-Windows platforms (default MockDesktopIO, no Windows-specific crashes).
3. get_backend_label representation ('REAL WINDOWS I/O' vs 'MOCK I/O').
4. initialize_system integration with backend selection and state tracking.
5. CLI argument parsing (--mock-io, --ui) in main.py.
6. Standalone UI entry point (ui.py main()) CLI argument handling.
7. UI badge and footer label display of active backend in BlindspotMainWindow.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk

from desktop_io import (
    DesktopIO,
    WindowsDesktopIO,
    MockDesktopIO,
    get_default_desktop_io,
    get_backend_label,
)
from main import initialize_system
from ui import UIStateBridge, BlindspotMainWindow


class TestBackendSelection(unittest.TestCase):
    """Unit tests for DesktopIO backend selection factory and labels."""

    def test_windows_default_selects_windows_desktop_io(self):
        """01. On Windows, get_default_desktop_io(mock=False) returns WindowsDesktopIO."""
        with patch("sys.platform", "win32"):
            io = get_default_desktop_io(mock=False)
            self.assertIsInstance(io, WindowsDesktopIO)

    def test_windows_explicit_mock_selects_mock_desktop_io(self):
        """02. On Windows, get_default_desktop_io(mock=True) returns MockDesktopIO."""
        with patch("sys.platform", "win32"):
            io = get_default_desktop_io(mock=True)
            self.assertIsInstance(io, MockDesktopIO)

    def test_non_windows_default_selects_mock_desktop_io(self):
        """03. On non-Windows platforms, get_default_desktop_io(mock=False) safely returns MockDesktopIO."""
        for platform in ("linux", "darwin", "freebsd"):
            with patch("sys.platform", platform):
                io = get_default_desktop_io(mock=False)
                self.assertIsInstance(io, MockDesktopIO)

    def test_non_windows_explicit_mock_selects_mock_desktop_io(self):
        """04. On non-Windows platforms, get_default_desktop_io(mock=True) returns MockDesktopIO."""
        with patch("sys.platform", "linux"):
            io = get_default_desktop_io(mock=True)
            self.assertIsInstance(io, MockDesktopIO)

    def test_backend_label_formatting(self):
        """05. get_backend_label accurately returns 'REAL WINDOWS I/O' or 'MOCK I/O'."""
        mock_io = MockDesktopIO()
        win_io = WindowsDesktopIO()

        self.assertEqual(get_backend_label(mock_io), "MOCK I/O")
        self.assertEqual(get_backend_label(win_io), "REAL WINDOWS I/O")
        self.assertEqual(get_backend_label(None), "UNKNOWN I/O")


class TestSystemInitializationBackend(unittest.TestCase):
    """Tests for initialize_system desktop_io wiring."""

    def test_initialize_system_default_windows(self):
        """06. initialize_system on Windows creates WindowsDesktopIO by default."""
        with patch("sys.platform", "win32"):
            components = initialize_system()
            io = components["desktop_io"]
            self.assertIsInstance(io, WindowsDesktopIO)
            self.assertEqual(components["state"]["desktop_io"], io)

    def test_initialize_system_explicit_mock_flag(self):
        """07. initialize_system(mock_io=True) creates MockDesktopIO on any platform."""
        with patch("sys.platform", "win32"):
            components = initialize_system(mock_io=True)
            io = components["desktop_io"]
            self.assertIsInstance(io, MockDesktopIO)
            self.assertEqual(components["state"]["desktop_io"], io)

    def test_initialize_system_injected_io_preserved(self):
        """08. Explicitly passed desktop_io is preserved regardless of platform."""
        custom_io = MockDesktopIO()
        components = initialize_system(desktop_io=custom_io)
        self.assertIs(components["desktop_io"], custom_io)
        self.assertIs(components["state"]["desktop_io"], custom_io)


class TestCliBackendSelection(unittest.TestCase):
    """Tests for CLI argument parsing for backend selection in main.py and ui.py."""

    @patch("main.initialize_system")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_main_cli_default_uses_windows_backend(self, mock_thread, mock_monitor, mock_init):
        """09. python main.py defaults to mock_io=False."""
        from main import main as main_entry

        mock_init.return_value = {
            "state": {"last_context": None, "context_start_time": None, "activity_time": {}},
            "state_lock": MagicMock(),
            "analyzer": MagicMock(),
            "assistant": MagicMock(),
            "companion_ai": MagicMock(),
            "tool_registry": MagicMock(),
            "task_executor": MagicMock(),
            "stop_event": MagicMock(),
            "desktop_io": WindowsDesktopIO(),
        }

        with patch("sys.argv", ["main.py"]), patch("builtins.input", side_effect=EOFError):
            try:
                main_entry()
            except (EOFError, StopIteration):
                pass

        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        self.assertFalse(kwargs.get("mock_io"))

    @patch("main.initialize_system")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_main_cli_mock_io_flag(self, mock_thread, mock_monitor, mock_init):
        """10. python main.py --mock-io passes mock_io=True."""
        from main import main as main_entry

        mock_init.return_value = {
            "state": {"last_context": None, "context_start_time": None, "activity_time": {}},
            "state_lock": MagicMock(),
            "analyzer": MagicMock(),
            "assistant": MagicMock(),
            "companion_ai": MagicMock(),
            "tool_registry": MagicMock(),
            "task_executor": MagicMock(),
            "stop_event": MagicMock(),
            "desktop_io": MockDesktopIO(),
        }

        with patch("sys.argv", ["main.py", "--mock-io"]), patch("builtins.input", side_effect=EOFError):
            try:
                main_entry()
            except (EOFError, StopIteration):
                pass

        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        self.assertTrue(kwargs.get("mock_io"))

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_main_cli_ui_with_mock_io(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """11. python main.py --ui --mock-io passes mock_io=True and hands io to run_ui."""
        from main import main as main_entry

        mock_io = MockDesktopIO()
        mock_init.return_value = {
            "state": {},
            "state_lock": MagicMock(),
            "analyzer": MagicMock(),
            "assistant": MagicMock(),
            "companion_ai": MagicMock(),
            "tool_registry": MagicMock(),
            "task_executor": MagicMock(),
            "stop_event": MagicMock(),
            "desktop_io": mock_io,
        }

        with patch("sys.argv", ["main.py", "--ui", "--mock-io"]):
            main_entry()

        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        self.assertTrue(kwargs.get("mock_io"))

        mock_run_ui.assert_called_once()
        _, ui_kwargs = mock_run_ui.call_args
        self.assertIs(ui_kwargs.get("desktop_io"), mock_io)

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_ui_standalone_cli_mock_io(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """12. python ui.py --mock-io passes mock_io=True and hands io to run_ui."""
        from ui import main as ui_entry

        mock_io = MockDesktopIO()
        mock_init.return_value = {
            "state": {},
            "state_lock": MagicMock(),
            "analyzer": MagicMock(),
            "assistant": MagicMock(),
            "companion_ai": MagicMock(),
            "tool_registry": MagicMock(),
            "task_executor": MagicMock(),
            "stop_event": MagicMock(),
            "desktop_io": mock_io,
        }

        with patch("sys.argv", ["ui.py", "--mock-io"]):
            ui_entry()

        mock_init.assert_called_once_with(mock_io=True)
        mock_run_ui.assert_called_once()
        _, ui_kwargs = mock_run_ui.call_args
        self.assertIs(ui_kwargs.get("desktop_io"), mock_io)


class TestUIBackendDisplay(unittest.TestCase):
    """Tests for UI label and badge display of the active backend."""

    def test_ui_labels_real_windows_io(self):
        """13. BlindspotMainWindow displays '[REAL WINDOWS I/O]' badge and footer with WindowsDesktopIO."""
        root = tk.Tk()
        root.withdraw()
        try:
            state = {"is_running": True}
            lock = MagicMock()
            win_io = WindowsDesktopIO()
            bridge = UIStateBridge(
                state=state,
                state_lock=lock,
                desktop_io=win_io,
            )
            app = BlindspotMainWindow(root=root, bridge=bridge)

            self.assertEqual(app.backend_label, "REAL WINDOWS I/O")
            self.assertEqual(app.backend_badge.cget("text"), "[REAL WINDOWS I/O]")
            self.assertEqual(app.footer_backend_label.cget("text"), "Backend: REAL WINDOWS I/O")
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_ui_labels_mock_io(self):
        """14. BlindspotMainWindow displays '[MOCK I/O]' badge and footer with MockDesktopIO."""
        root = tk.Tk()
        root.withdraw()
        try:
            state = {"is_running": True}
            lock = MagicMock()
            mock_io = MockDesktopIO()
            bridge = UIStateBridge(
                state=state,
                state_lock=lock,
                desktop_io=mock_io,
            )
            app = BlindspotMainWindow(root=root, bridge=bridge)

            self.assertEqual(app.backend_label, "MOCK I/O")
            self.assertEqual(app.backend_badge.cget("text"), "[MOCK I/O]")
            self.assertEqual(app.footer_backend_label.cget("text"), "Backend: MOCK I/O")
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
