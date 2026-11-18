"""
tests/test_startup.py — Desktop-First Architecture & UI Startup Regression Tests.

Verifies:
1. Default launch (`python main.py`) launches the Tkinter desktop UI as the primary product interface.
2. Explicit UI flag (`python main.py --ui`) launches the desktop UI identically.
3. Terminal CLI mode is preserved via `--cli` and `--no-ui` flags.
4. Backend selection (`--mock-io`) propagates cleanly to both initialize_system and run_ui.
5. Background dashboard API (`--api`, `--dashboard`) attaches engine state and coexists with UI.
6. Clean shutdown: window closure stops monitor and closes active context.
7. Engine ownership: single perception engine shared between UI and dashboard without duplicates.
8. State synchronization: UIStateBridge and dashboard API reflect identical running/stopped state.
9. Help flag (`--help`, `-h`) displays usage without starting threads.

All tests use mocked Tkinter and backend dependencies for deterministic, headless execution.
"""

import sys
import unittest
import threading
import time
from unittest.mock import MagicMock, patch

from desktop_io import MockDesktopIO, WindowsDesktopIO
from ui import UIStateBridge
from task import Task, TaskStep, TaskStatus


class DummyCompanionAI:
    def __init__(self, answer="Test answer"):
        self.answer = answer
        self.last_query = None

    def ask(self, **kwargs):
        self.last_query = kwargs.get("user_query")
        return self.answer

    def plan_task(self, query):
        t = Task(user_request=query, task_id="t1")
        t.steps = [TaskStep(step_id=1, tool_name="open_application", args={"app_name": "notepad"})]
        return t

    def clear_conversation(self):
        pass


class TestDesktopStartup(unittest.TestCase):
    """Regression test suite for desktop-first launch and engine coexistence."""

    def _make_mock_components(self, mock_io=False):
        io = MockDesktopIO() if mock_io else WindowsDesktopIO()
        lock = threading.Lock()
        stop_event = threading.Event()
        history = MagicMock()
        history.get_recent.return_value = []
        state = {
            "last_window": None,
            "last_context": None,
            "last_confidence": 0,
            "context_start_time": None,
            "activity_time": {},
            "history": history,
            "is_running": True,
            "desktop_io": io,
        }
        return {
            "state": state,
            "state_lock": lock,
            "analyzer": MagicMock(),
            "assistant": MagicMock(),
            "companion_ai": DummyCompanionAI(),
            "tool_registry": MagicMock(),
            "task_executor": MagicMock(),
            "stop_event": stop_event,
            "desktop_io": io,
        }

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_01_default_launch_starts_desktop_ui(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """Default launch (`python main.py`) launches desktop UI with all injected components."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py"]):
            main_entry()

        mock_init.assert_called_once()
        _, init_kwargs = mock_init.call_args
        self.assertFalse(init_kwargs.get("mock_io"))
        # In UI mode, low_risk_approver is None at init so UI attaches its own dialog approver
        self.assertIsNone(init_kwargs.get("low_risk_approver"))

        mock_run_ui.assert_called_once()
        _, ui_kwargs = mock_run_ui.call_args
        self.assertIs(ui_kwargs.get("state"), components["state"])
        self.assertIs(ui_kwargs.get("state_lock"), components["state_lock"])
        self.assertIs(ui_kwargs.get("companion_ai"), components["companion_ai"])
        self.assertIs(ui_kwargs.get("tool_registry"), components["tool_registry"])
        self.assertIs(ui_kwargs.get("task_executor"), components["task_executor"])
        self.assertIs(ui_kwargs.get("desktop_io"), components["desktop_io"])
        self.assertIs(ui_kwargs.get("analyzer"), components["analyzer"])

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_02_explicit_ui_flag_starts_desktop_ui(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """Explicit `--ui` flag launches desktop UI."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py", "--ui"]):
            main_entry()

        mock_run_ui.assert_called_once()

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_03_cli_flag_launches_terminal_command_loop(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """Explicit `--cli` flag enters CLI command loop and uses CLI approver."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py", "--cli"]), patch("builtins.input", side_effect=EOFError):
            try:
                main_entry()
            except (EOFError, StopIteration):
                pass

        mock_run_ui.assert_not_called()
        mock_init.assert_called_once()
        _, init_kwargs = mock_init.call_args
        self.assertIsNotNone(init_kwargs.get("low_risk_approver"))

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_04_no_ui_flag_launches_terminal_command_loop(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """Explicit `--no-ui` flag enters CLI command loop without launching UI."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py", "--no-ui"]), patch("builtins.input", side_effect=EOFError):
            try:
                main_entry()
            except (EOFError, StopIteration):
                pass

        mock_run_ui.assert_not_called()

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_05_mock_io_flag_propagates_to_ui(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """Flag `--mock-io` passes mock_io=True to initialize_system and hands MockDesktopIO to run_ui."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=True)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py", "--mock-io"]):
            main_entry()

        mock_init.assert_called_once()
        _, init_kwargs = mock_init.call_args
        self.assertTrue(init_kwargs.get("mock_io"))

        mock_run_ui.assert_called_once()
        _, ui_kwargs = mock_run_ui.call_args
        self.assertIsInstance(ui_kwargs.get("desktop_io"), MockDesktopIO)

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    @patch("api.attach_engine_state")
    @patch("api.start_api_server_background")
    def test_06_api_flag_starts_background_server_and_ui(
        self, mock_start_api, mock_attach, mock_thread, mock_monitor, mock_run_ui, mock_init
    ):
        """Flag `--api` starts background FastAPI server attached to engine and launches desktop UI."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py", "--api"]):
            main_entry()

        mock_attach.assert_called_once_with(
            components["state"],
            components["state_lock"],
            components["analyzer"],
        )
        mock_start_api.assert_called_once_with(host="127.0.0.1", port=8000)
        mock_run_ui.assert_called_once()

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    @patch("api.attach_engine_state")
    @patch("api.start_api_server_background")
    def test_07_dashboard_flag_alias_starts_api_and_ui(
        self, mock_start_api, mock_attach, mock_thread, mock_monitor, mock_run_ui, mock_init
    ):
        """Flag `--dashboard` behaves identically to `--api`."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        with patch("sys.argv", ["main.py", "--dashboard"]):
            main_entry()

        mock_attach.assert_called_once()
        mock_start_api.assert_called_once()
        mock_run_ui.assert_called_once()

    @patch("builtins.print")
    def test_08_help_flag_displays_usage(self, mock_print):
        """Flags `-h` and `--help` print usage and return without starting components."""
        from main import main as main_entry

        with patch("sys.argv", ["main.py", "--help"]):
            main_entry()

        printed_text = " ".join(str(call[0][0]) for call in mock_print.call_args_list if call[0])
        self.assertIn("BLINDSPOT", printed_text)
        self.assertIn("--cli", printed_text)
        self.assertIn("--api", printed_text)

    @patch("main.initialize_system")
    @patch("ui.run_ui")
    @patch("main.monitor")
    @patch("threading.Thread")
    def test_09_window_closure_clean_shutdown(self, mock_thread, mock_monitor, mock_run_ui, mock_init):
        """Window closure in run_ui triggers stop_event.set() and closes active context."""
        from main import main as main_entry

        components = self._make_mock_components(mock_io=False)
        mock_init.return_value = components

        # Simulate run_ui completing (user closed window)
        mock_run_ui.return_value = None

        with patch("sys.argv", ["main.py"]):
            main_entry()

        # stop_event must be set after run_ui exits
        self.assertTrue(components["stop_event"].is_set())

    def test_10_ui_state_bridge_injected_components(self):
        """UIStateBridge correctly receives injected companion_ai, analyzer, and dispatches ask()."""
        components = self._make_mock_components(mock_io=False)
        companion_ai = components["companion_ai"]
        analyzer = components["analyzer"]

        bridge = UIStateBridge(
            state=components["state"],
            state_lock=components["state_lock"],
            companion_ai=companion_ai,
            tool_registry=components["tool_registry"],
            task_executor=components["task_executor"],
            desktop_io=components["desktop_io"],
            analyzer=analyzer,
        )

        self.assertIs(bridge.companion_ai, companion_ai)
        self.assertIs(bridge.analyzer, analyzer)

        res = bridge.ask("what am I working on?")
        self.assertTrue(res["success"])
        self.assertEqual(res["answer"], "Test answer")
        self.assertEqual(companion_ai.last_query, "what am I working on?")

    def test_11_engine_state_shared_between_ui_and_dashboard(self):
        """Desktop engine state is shared with api.py without duplicate engines."""
        from api import attach_engine_state, get_engine
        components = self._make_mock_components(mock_io=False)

        attach_engine_state(
            components["state"],
            components["state_lock"],
            components["analyzer"],
        )

        ret_state, ret_lock, ret_analyzer = get_engine()
        self.assertIs(ret_state, components["state"])
        self.assertIs(ret_lock, components["state_lock"])
        self.assertIs(ret_analyzer, components["analyzer"])

    def test_12_ui_and_dashboard_monitoring_toggle_synchronization(self):
        """Toggling perception from UIStateBridge or dashboard updates the exact same state."""
        from api import attach_engine_state, toggle_engine
        components = self._make_mock_components(mock_io=False)
        attach_engine_state(components["state"], components["state_lock"], components["analyzer"])

        bridge = UIStateBridge(
            state=components["state"],
            state_lock=components["state_lock"],
            companion_ai=components["companion_ai"],
        )

        # Initially running
        self.assertTrue(bridge.get_status_snapshot()["is_running"])

        # Dashboard pauses perception
        toggle_res = toggle_engine({"running": False})
        self.assertFalse(toggle_res["is_running"])
        # Desktop UI immediately observes paused state
        self.assertFalse(bridge.get_status_snapshot()["is_running"])

        # Desktop UI resumes perception
        bridge.toggle_running()
        self.assertTrue(bridge.get_status_snapshot()["is_running"])

        # Dashboard pauses with toggle (no explicit param)
        toggle_res2 = toggle_engine({})
        self.assertFalse(toggle_res2["is_running"])
        self.assertFalse(bridge.get_status_snapshot()["is_running"])


if __name__ == "__main__":
    unittest.main()
