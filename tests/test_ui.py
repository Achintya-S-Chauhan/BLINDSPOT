"""
BLINDSPOT - Phase 8A: UI Foundation Tests.

Tests:
- UIStateBridge snapshot retrieval, running state toggle, and safe task planning
- Validation and safety guarantees (no real execution from UI)
- BlindspotMainWindow widget creation, view updates, error handling, and clean destruction
- Backend initialization and thread lifecycle controls
"""

import unittest
import threading
import time
import tkinter as tk
from unittest.mock import MagicMock, patch

from ui import UIStateBridge, BlindspotMainWindow
from main import initialize_system, monitor
from task import Task, TaskStep, TaskStatus
from tools import ToolRegistry, BaseTool, ToolResult, ActionRisk


class MockTestTool(BaseTool):
    def __init__(self, name="mock_tool", risk=ActionRisk.READ_ONLY):
        super().__init__(name=name, description="Mock test tool", risk=risk)

    def execute(self, **kwargs):
        return ToolResult(tool_name=self.name, success=True, output="mock ok")


class DummyCompanionAI:
    def __init__(self, plan_return=None, should_raise=False):
        self.plan_return = plan_return
        self.should_raise = should_raise
        self.last_planned_query = None

    def plan_task(self, query: str) -> Task:
        self.last_planned_query = query
        if self.should_raise:
            raise RuntimeError("Simulated AI planning crash")
        if self.plan_return is not None:
            return self.plan_return
        # Default simple valid 2-step task
        t = Task(user_request=query, task_id="task_test123")
        t.steps = [
            TaskStep(step_id=1, tool_name="mock_tool_a", args={}, description="Step 1", depends_on=[]),
            TaskStep(step_id=2, tool_name="mock_tool_b", args={}, description="Step 2", depends_on=[1]),
        ]
        return t


class TestUIStateBridge(unittest.TestCase):
    def setUp(self):
        self.state = {
            "last_window": "Code.exe",
            "last_context": "coding",
            "last_confidence": 3,
            "context_start_time": time.time() - 45.0,
            "last_ocr_time": time.time() - 2.0,
            "is_running": True,
        }
        self.lock = threading.Lock()
        self.registry = ToolRegistry(allow_actions=True)
        self.registry.register(MockTestTool("mock_tool_a"))
        self.registry.register(MockTestTool("mock_tool_b"))
        self.mock_ai = DummyCompanionAI()
        self.bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=self.mock_ai,
            tool_registry=self.registry,
        )

    def test_01_get_status_snapshot(self):
        snapshot = self.bridge.get_status_snapshot()
        self.assertTrue(snapshot["is_running"])
        self.assertEqual(snapshot["active_window"], "Code.exe")
        self.assertEqual(snapshot["active_context"], "coding")
        self.assertEqual(snapshot["confidence"], 3)
        self.assertGreaterEqual(snapshot["duration_seconds"], 45.0)
        self.assertIn("45s", snapshot["duration_formatted"])

    def test_02_snapshot_defaults_when_empty(self):
        empty_state = {}
        bridge = UIStateBridge(state=empty_state, state_lock=self.lock)
        snapshot = bridge.get_status_snapshot()
        self.assertTrue(snapshot["is_running"])
        self.assertEqual(snapshot["active_window"], "No active window detected")
        self.assertEqual(snapshot["active_context"], "No active context")
        self.assertEqual(snapshot["confidence"], 0)
        self.assertEqual(snapshot["duration_seconds"], 0.0)

    def test_03_toggle_running(self):
        self.assertTrue(self.bridge.get_status_snapshot()["is_running"])
        new_val = self.bridge.toggle_running()
        self.assertFalse(new_val)
        self.assertFalse(self.bridge.get_status_snapshot()["is_running"])

        # Toggle back
        new_val = self.bridge.toggle_running()
        self.assertTrue(new_val)
        self.assertTrue(self.bridge.get_status_snapshot()["is_running"])

    def test_04_set_running(self):
        self.bridge.set_running(False)
        self.assertFalse(self.bridge.get_status_snapshot()["is_running"])
        self.bridge.set_running(True)
        self.assertTrue(self.bridge.get_status_snapshot()["is_running"])

    def test_05_plan_task_empty_query(self):
        res = self.bridge.plan_task("")
        self.assertFalse(res["success"])
        self.assertIn("cannot be empty", res["error"])

        res2 = self.bridge.plan_task("   ")
        self.assertFalse(res2["success"])
        self.assertIn("cannot be empty", res2["error"])

    def test_06_plan_task_no_ai_configured(self):
        bridge = UIStateBridge(state=self.state, state_lock=self.lock, companion_ai=None)
        res = bridge.plan_task("Test task")
        self.assertFalse(res["success"])
        self.assertIn("Companion AI is not configured", res["error"])

    def test_07_plan_task_successful_formulation(self):
        res = self.bridge.plan_task("Organize files")
        self.assertTrue(res["success"])
        self.assertEqual(res["task_id"], "task_test123")
        self.assertEqual(res["total_steps"], 2)
        self.assertEqual(len(res["steps"]), 2)
        self.assertEqual(res["steps"][0]["depends_on"], [])
        self.assertEqual(res["steps"][1]["depends_on"], [1])
        # Safety guarantee check: execution must be explicitly disabled
        self.assertFalse(res["execution_enabled"])
        self.assertIn("PHASE 8A SAFETY CONTROL", res["execution_notice"])

    def test_08_plan_task_ai_rejection(self):
        failed_task = Task(user_request="bad query", task_id="t_fail", status=TaskStatus.FAILED, error="Unsupported task")
        ai = DummyCompanionAI(plan_return=failed_task)
        bridge = UIStateBridge(state=self.state, state_lock=self.lock, companion_ai=ai, tool_registry=self.registry)
        res = bridge.plan_task("bad query")
        self.assertFalse(res["success"])
        self.assertIn("Unsupported task", res["error"])

    def test_09_plan_task_validation_rejection(self):
        # Task with unregistered tool
        bad_task = Task(user_request="unregistered tool task", task_id="t_bad")
        bad_task.steps = [
            TaskStep(step_id=1, tool_name="nonexistent_tool", args={}, description="Step 1"),
        ]
        ai = DummyCompanionAI(plan_return=bad_task)
        bridge = UIStateBridge(state=self.state, state_lock=self.lock, companion_ai=ai, tool_registry=self.registry)
        res = bridge.plan_task("unregistered tool task")
        self.assertFalse(res["success"])
        self.assertIn("Task validation rejected", res["error"])

    def test_10_plan_task_exception_handling(self):
        ai = DummyCompanionAI(should_raise=True)
        bridge = UIStateBridge(state=self.state, state_lock=self.lock, companion_ai=ai, tool_registry=self.registry)
        res = bridge.plan_task("trigger exception")
        self.assertFalse(res["success"])
        self.assertIn("AI planning error", res["error"])


class TestBlindspotMainWindow(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Keep window hidden during test suite

        self.state = {
            "last_window": "VSCode",
            "last_context": "coding",
            "last_confidence": 3,
            "context_start_time": time.time() - 30.0,
            "is_running": True,
        }
        self.lock = threading.Lock()
        self.registry = ToolRegistry(allow_actions=True)
        self.registry.register(MockTestTool("mock_tool_a"))
        self.registry.register(MockTestTool("mock_tool_b"))
        self.mock_ai = DummyCompanionAI()
        self.bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=self.mock_ai,
            tool_registry=self.registry,
        )
        self.app = BlindspotMainWindow(root=self.root, bridge=self.bridge, poll_interval_ms=1000)

    def tearDown(self):
        self.app.on_close()

    def test_11_window_initialization(self):
        self.assertIn("BLINDSPOT", self.root.title())
        self.assertEqual(self.app.val_window.cget("text"), "VSCode")
        self.assertEqual(self.app.val_activity.cget("text"), "coding")
        self.assertEqual(self.app.val_confidence.cget("text"), "3/3")
        self.assertEqual(self.app.status_badge.cget("text"), "● RUNNING")
        self.assertEqual(self.app.toggle_btn.cget("text"), "Pause Perception")

    def test_12_update_view_running_to_stopped(self):
        snapshot_stopped = {
            "is_running": False,
            "active_window": "Browser",
            "active_context": "research",
            "confidence": 2,
            "duration_formatted": "1m 15s",
        }
        self.app.update_view(snapshot_stopped)
        self.assertEqual(self.app.status_badge.cget("text"), "○ STOPPED")
        self.assertEqual(self.app.toggle_btn.cget("text"), "Resume Perception")
        self.assertEqual(self.app.val_window.cget("text"), "Browser")
        self.assertEqual(self.app.val_activity.cget("text"), "research")
        self.assertEqual(self.app.val_confidence.cget("text"), "2/3")
        self.assertEqual(self.app.val_duration.cget("text"), "1m 15s")

    def test_13_toggle_button_interaction(self):
        self.assertTrue(self.bridge.get_status_snapshot()["is_running"])
        self.app._on_toggle_clicked()
        self.assertFalse(self.bridge.get_status_snapshot()["is_running"])
        self.assertEqual(self.app.status_badge.cget("text"), "○ STOPPED")
        self.assertIn("Perception monitoring paused", self.app.footer_label.cget("text"))

        self.app._on_toggle_clicked()
        self.assertTrue(self.bridge.get_status_snapshot()["is_running"])
        self.assertEqual(self.app.status_badge.cget("text"), "● RUNNING")
        self.assertIn("Perception monitoring resumed", self.app.footer_label.cget("text"))

    def test_14_status_message_and_error_display(self):
        self.app.set_status_message("Custom info notice")
        self.assertEqual(self.app.footer_label.cget("text"), "Custom info notice")

        self.app.set_status_error("Something went wrong")
        self.assertEqual(self.app.footer_label.cget("text"), "Error: Something went wrong")

    def test_15_plan_task_ui_interaction_empty_query(self):
        self.app.task_entry.delete(0, tk.END)
        self.app._on_plan_clicked()
        self.assertIn("Please enter a task request", self.app.footer_label.cget("text"))

    def test_16_handle_plan_result_success(self):
        mock_result = {
            "success": True,
            "task_id": "task_ui_001",
            "user_request": "Inspect desktop",
            "total_steps": 1,
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "mock_tool",
                    "args": {"mode": "fast"},
                    "description": "Inspect window",
                    "depends_on": [],
                }
            ],
            "execution_notice": "[PHASE 8A SAFETY CONTROL] Action execution disabled.",
        }
        self.app._handle_plan_result(mock_result)
        out = self.app.plan_output.get("1.0", tk.END)
        self.assertIn("task_ui_001", out)
        self.assertIn("mock_tool(mode='fast')", out)
        self.assertIn("PHASE 8A SAFETY CONTROL", out)
        self.assertIn("Plan formulated successfully", self.app.footer_label.cget("text"))

    def test_17_handle_plan_result_rejection(self):
        mock_result = {
            "success": False,
            "error": "Dangerous tool request blocked by policy",
        }
        self.app._handle_plan_result(mock_result)
        out = self.app.plan_output.get("1.0", tk.END)
        self.assertIn("[PLAN REJECTED]", out)
        self.assertIn("Dangerous tool request blocked", out)
        self.assertIn("Error: Dangerous tool request blocked", self.app.footer_label.cget("text"))

    def test_18_clear_button_interaction(self):
        self.app.task_entry.insert(0, "Some input")
        self.app._set_output_text("Some plan text")
        self.app._on_clear_clicked()
        self.assertEqual(self.app.task_entry.get(), "")
        self.assertIn("Type a task request above", self.app.plan_output.get("1.0", tk.END))
        self.assertIn("Input cleared", self.app.footer_label.cget("text"))


class TestSystemLifecycleIntegration(unittest.TestCase):
    def test_19_initialize_system(self):
        components = initialize_system()
        self.assertIn("state", components)
        self.assertIn("state_lock", components)
        self.assertIn("analyzer", components)
        self.assertIn("assistant", components)
        self.assertIn("companion_ai", components)
        self.assertIn("tool_registry", components)
        self.assertIn("task_executor", components)
        self.assertIn("verifier", components)
        self.assertIn("stop_event", components)

        # Check default running flag
        self.assertTrue(components["state"]["is_running"])

    def test_20_monitor_clean_shutdown_via_stop_event(self):
        state = {"is_running": False, "last_window": None}
        lock = threading.Lock()
        stop_event = threading.Event()

        t = threading.Thread(target=monitor, args=(state, lock, stop_event), daemon=True)
        t.start()
        self.assertTrue(t.is_alive())

        # Signal stop event
        stop_event.set()
        t.join(timeout=1.5)
        self.assertFalse(t.is_alive())


class TestPlanningRejectionRegression(unittest.TestCase):
    """Regression tests for task planning rejection and UI preview display."""

    def test_21_open_whatsapp_planning_rejection_reproduced(self):
        """Entering 'open whatsapp' must reject cleanly because whatsapp is not allowlisted."""
        components = initialize_system()
        companion_ai = components["companion_ai"]
        task = companion_ai.plan_task("open whatsapp")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(len(task.steps), 0)
        self.assertIn("whatsapp", task.error.lower())
        self.assertIn("allowed applications", task.error.lower())

    def test_22_open_notepad_planning_success(self):
        """Entering 'open notepad' must formulate a valid open_application plan."""
        components = initialize_system()
        companion_ai = components["companion_ai"]
        task = companion_ai.plan_task("open notepad")
        self.assertNotEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(len(task.steps), 1)
        self.assertEqual(task.steps[0].tool_name, "open_application")
        self.assertEqual(task.steps[0].args.get("app_name"), "notepad")

    def test_23_ui_displays_rejected_plan_error_not_empty(self):
        """Rejected plans must display structured error guidance in plan preview, not empty."""
        root = tk.Tk()
        root.withdraw()
        try:
            components = initialize_system()
            bridge = UIStateBridge(
                state=components["state"],
                state_lock=components["state_lock"],
                companion_ai=components["companion_ai"],
                tool_registry=components["tool_registry"],
            )
            app = BlindspotMainWindow(root=root, bridge=bridge)
            app.task_entry.insert(0, "open whatsapp")

            # Execute plan formulation
            res = bridge.plan_task("open whatsapp")
            self.assertFalse(res["success"])
            self.assertIn("whatsapp", res["error"].lower())

            # Handle result in UI
            app._handle_plan_result(res)

            output_text = app.plan_output.get("1.0", tk.END)
            self.assertIn("[PLAN REJECTED]", output_text)
            self.assertIn("whatsapp", output_text.lower())
            self.assertIn("allowed applications", output_text.lower())
            self.assertIn("notepad, calculator, paint", output_text.lower())
            self.assertNotEqual(output_text.strip(), "")
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_24_ui_planning_indicator_not_empty_during_planning(self):
        """Clicking Plan Task must show immediate feedback, not leaving preview blank."""
        root = tk.Tk()
        root.withdraw()
        try:
            components = initialize_system()
            bridge = UIStateBridge(
                state=components["state"],
                state_lock=components["state_lock"],
                companion_ai=components["companion_ai"],
                tool_registry=components["tool_registry"],
            )
            app = BlindspotMainWindow(root=root, bridge=bridge)
            app.task_entry.insert(0, "open notepad")
            app._on_plan_clicked()

            output_text = app.plan_output.get("1.0", tk.END)
            self.assertIn("[PLANNING TASK]", output_text)
            self.assertIn("open notepad", output_text)
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
