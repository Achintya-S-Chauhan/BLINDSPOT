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
from task import Task, TaskStep, TaskStatus, StepStatus, TaskExecutor, FailureType
from tools import ToolRegistry, BaseTool, ToolResult, ActionRisk, ActionRequest, PermissionPolicy, _auto_deny
from verification import TaskVerifier, DesktopObserver, StepVerificationResult, VerificationStatus
from recovery import RetryPolicy, RecoveryRecord, RecoveryDecision
from context import DesktopContext


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


class MockFailingVerifier:
    """Verifier mock that always fails verification for testing."""
    def verify_step(self, step, tool=None):
        return StepVerificationResult(
            status=VerificationStatus.FAILED,
            message="Simulated verification mismatch: expected target window not found.",
            details={"observed": "WrongWindow"},
        )


class MockVerifyingVerifier:
    """Verifier mock that verifies specific tools."""
    def verify_step(self, step, tool=None):
        if step.tool_name in ("open_application", "focus_application", "mock_action"):
            return StepVerificationResult(
                status=VerificationStatus.VERIFIED,
                message=f"Action '{step.tool_name}' verified active on desktop.",
                details={"observed": "VerifiedDesktop"},
            )
        return StepVerificationResult(
            status=VerificationStatus.NOT_VERIFIABLE,
            message=f"Action '{step.tool_name}' does not support desktop verification.",
        )


class TestPhase8BControlledExecution(unittest.TestCase):
    """Integration and safety tests for Phase 8B Controlled Desktop Execution."""

    def setUp(self):
        self.state = {
            "last_window": "VSCode",
            "last_context": "coding",
            "last_confidence": 3,
            "is_running": True,
        }
        self.lock = threading.Lock()
        self.registry = ToolRegistry(allow_actions=True)
        self.registry.register(MockTestTool("mock_tool_a", risk=ActionRisk.READ_ONLY))
        self.registry.register(MockTestTool("mock_tool_b", risk=ActionRisk.READ_ONLY))
        self.registry.register(MockTestTool("mock_action", risk=ActionRisk.LOW_RISK_ACTION))

        self.verifier = MockVerifyingVerifier()
        self.retry_policy = RetryPolicy(max_retries=2)
        self.executor = TaskExecutor(
            tool_registry=self.registry,
            verifier=self.verifier,
            retry_policy=self.retry_policy,
        )
        self.mock_ai = DummyCompanionAI()
        self.bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=self.mock_ai,
            tool_registry=self.registry,
            task_executor=self.executor,
        )

    def test_25_plan_approval_and_invalidation(self):
        """Plan must be explicitly approved, and can be invalidated cleanly."""
        res = self.bridge.plan_task("Sample task")
        self.assertTrue(res["success"])
        self.assertIsNotNone(self.bridge.pending_task)
        self.assertFalse(self.bridge.is_plan_approved)

        # Approve plan
        app_res = self.bridge.approve_plan()
        self.assertTrue(app_res["success"])
        self.assertTrue(self.bridge.is_plan_approved)

        # Invalidate plan
        self.bridge.invalidate_plan()
        self.assertIsNone(self.bridge.pending_task)
        self.assertFalse(self.bridge.is_plan_approved)

        # Approving when no plan exists fails gracefully
        fail_app = self.bridge.approve_plan()
        self.assertFalse(fail_app["success"])
        self.assertIn("No valid plan available", fail_app["error"])

    def test_26_execution_requires_explicit_plan_approval(self):
        """Task execution must be refused unless plan has explicit user approval."""
        res = self.bridge.plan_task("Sample task")
        self.assertTrue(res["success"])
        self.assertFalse(self.bridge.is_plan_approved)

        # Attempt to execute unapproved plan
        exec_res = self.bridge.execute_approved_task()
        self.assertFalse(exec_res["success"])
        self.assertIn("Plan has not been approved", exec_res["error"])

        # Approve and then execute
        self.bridge.approve_plan()
        self.assertTrue(self.bridge.is_plan_approved)
        exec_ok = self.bridge.execute_approved_task()
        self.assertTrue(exec_ok["success"])
        self.assertEqual(exec_ok["status"], "completed")
        self.assertEqual(exec_ok["executed_steps"], 2)

    def test_27_plan_modification_invalidates_approval_in_ui(self):
        """Editing task request input in the UI invalidates existing approval."""
        root = tk.Tk()
        root.withdraw()
        try:
            app = BlindspotMainWindow(root=root, bridge=self.bridge)
            app.task_entry.insert(0, "Inspect desktop")

            # Plan task
            res = self.bridge.plan_task("Inspect desktop")
            app._handle_plan_result(res)
            self.assertEqual(str(app.approve_btn.cget("state")), "normal")
            self.assertEqual(str(app.execute_btn.cget("state")), "disabled")

            # Approve plan
            app._on_approve_clicked()
            self.assertEqual(str(app.approve_btn.cget("state")), "disabled")
            self.assertEqual(str(app.execute_btn.cget("state")), "normal")
            self.assertTrue(self.bridge.is_plan_approved)

            # Modify input text
            app.task_entry.insert(tk.END, " extra text")
            app._on_input_changed()

            # Must invalidate approval immediately
            self.assertFalse(self.bridge.is_plan_approved)
            self.assertIsNone(self.bridge.pending_task)
            self.assertEqual(str(app.approve_btn.cget("state")), "disabled")
            self.assertEqual(str(app.execute_btn.cget("state")), "disabled")
            self.assertIn("invalidated", app.footer_label.cget("text"))

            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_28_controlled_execution_success_with_task_executor(self):
        """Approved task executes through TaskExecutor with step verifications recorded."""
        self.bridge.plan_task("Test task execution")
        self.bridge.approve_plan()

        res = self.bridge.execute_approved_task()
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["total_steps"], 2)
        self.assertEqual(res["executed_steps"], 2)
        self.assertEqual(res["successful_steps"], 2)
        self.assertGreaterEqual(len(res["steps"]), 2)

        # Plan must be consumed (cannot re-execute without formulating a new plan)
        self.assertIsNone(self.bridge.pending_task)
        self.assertFalse(self.bridge.is_plan_approved)

    def test_29_permission_policy_denial_halts_and_blocks_task(self):
        """Action tool denied by permission policy halts execution in BLOCKED state."""
        # Custom task with a low-risk action step and a dependent step
        action_task = Task(user_request="run action", task_id="t_action")
        action_task.steps = [
            TaskStep(step_id=1, tool_name="mock_action", args={}, description="Step 1", depends_on=[]),
            TaskStep(step_id=2, tool_name="mock_tool_b", args={}, description="Step 2", depends_on=[1]),
        ]
        ai = DummyCompanionAI(plan_return=action_task)
        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=ai,
            tool_registry=self.registry,
            task_executor=self.executor,
            action_approver=lambda req: False,  # Deny permission
        )

        plan_res = bridge.plan_task("run action")
        self.assertTrue(plan_res["success"])
        bridge.approve_plan()

        exec_res = bridge.execute_approved_task()
        self.assertFalse(exec_res["success"])
        self.assertEqual(exec_res["status"], "blocked")
        self.assertEqual(exec_res["failure_type"], "permission_denied")
        self.assertEqual(exec_res["executed_steps"], 1)
        self.assertEqual(exec_res["skipped_steps"], 1)
        self.assertEqual(exec_res["steps"][0]["status"], "blocked")

    def test_30_permission_policy_approval_proceeds_and_verifies(self):
        """Action tool approved by permission policy executes and records verification."""
        action_task = Task(user_request="run approved action", task_id="t_action_ok")
        action_task.steps = [
            TaskStep(step_id=1, tool_name="mock_action", args={}, description="Step 1", depends_on=[]),
        ]
        ai = DummyCompanionAI(plan_return=action_task)
        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=ai,
            tool_registry=self.registry,
            task_executor=self.executor,
            action_approver=lambda req: True,  # Grant permission
        )

        bridge.plan_task("run approved action")
        bridge.approve_plan()

        exec_res = bridge.execute_approved_task()
        self.assertTrue(exec_res["success"])
        self.assertEqual(exec_res["status"], "completed")
        self.assertEqual(exec_res["steps"][0]["verification"]["status"], "verified")
        self.assertIn("verified active on desktop", exec_res["steps"][0]["verification"]["message"])

    def test_31_duplicate_and_simultaneous_execution_prevention(self):
        """Simultaneous planning or duplicate executions are rejected."""
        self.bridge.plan_task("Task 1")
        self.bridge.approve_plan()

        # Simulate execution in progress
        self.bridge.is_executing = True

        # Attempt duplicate execution
        dup_res = self.bridge.execute_approved_task()
        self.assertFalse(dup_res["success"])
        self.assertIn("already in progress", dup_res["error"])

        # Attempt planning during execution
        plan_res = self.bridge.plan_task("Task 2")
        self.assertFalse(plan_res["success"])
        self.assertIn("currently in progress", plan_res["error"])

        # Attempt approval during execution
        app_res = self.bridge.approve_plan()
        self.assertFalse(app_res["success"])
        self.assertIn("while execution is in progress", app_res["error"])

        self.bridge.is_executing = False

    def test_32_verification_failure_displayed_accurately(self):
        """Verification failure halts task, records failure_type, and is formatted in UI."""
        failing_executor = TaskExecutor(
            tool_registry=self.registry,
            verifier=MockFailingVerifier(),
            retry_policy=RetryPolicy(enabled=False, max_retries=0),
        )
        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=self.mock_ai,
            tool_registry=self.registry,
            task_executor=failing_executor,
        )

        bridge.plan_task("Fail verify task")
        bridge.approve_plan()

        res = bridge.execute_approved_task()
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["failure_type"], "verification_failure")

        # Test UI rendering of verification failure
        root = tk.Tk()
        root.withdraw()
        try:
            app = BlindspotMainWindow(root=root, bridge=bridge)
            app._handle_execution_result(res)
            out = app.plan_output.get("1.0", tk.END)
            self.assertIn("[TASK EXECUTION FAILED — POST-EXECUTION VERIFICATION FAILED]", out)
            self.assertIn("Verification: [FAILED]", out)
            self.assertNotIn("ALL STEPS VERIFIED", out)
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_33_retry_and_recovery_displayed_in_ui(self):
        """Transient step failure that recovers on retry is reflected in UI."""
        call_count = [0]

        class FlakyTool(BaseTool):
            def __init__(self):
                super().__init__(name="flaky_tool", description="Flaky test tool", risk=ActionRisk.READ_ONLY)
            def execute(self, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return ToolResult(tool_name=self.name, success=False, error="Transient error")
                return ToolResult(tool_name=self.name, success=True, output="Recovered")

        reg = ToolRegistry(allow_actions=True)
        reg.register(FlakyTool())
        flaky_task = Task(user_request="run flaky", task_id="t_flaky")
        flaky_task.steps = [TaskStep(step_id=1, tool_name="flaky_tool", args={}, description="Step 1")]
        ai = DummyCompanionAI(plan_return=flaky_task)

        executor = TaskExecutor(
            tool_registry=reg,
            verifier=False,
            retry_policy=RetryPolicy(safe_tools={"flaky_tool"}, max_retries=2),
        )
        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=ai,
            tool_registry=reg,
            task_executor=executor,
        )
        bridge.plan_task("run flaky")
        bridge.approve_plan()

        res = bridge.execute_approved_task()
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["retry_attempts"], 1)
        self.assertEqual(res["steps"][0]["attempts"], 2)

        # Check UI output
        root = tk.Tk()
        root.withdraw()
        try:
            app = BlindspotMainWindow(root=root, bridge=bridge)
            app._handle_execution_result(res)
            out = app.plan_output.get("1.0", tk.END)
            self.assertIn("Attempts: 2", out)
            self.assertIn("1 retried", out)
            app.on_close()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_34_ui_cannot_bypass_execution_pipeline(self):
        """UI planning does not execute tools, and execution only runs through TaskExecutor."""
        mock_tool_exec = MagicMock()
        tool = MockTestTool("mock_tool_tracked")
        tool.execute = mock_tool_exec
        reg = ToolRegistry(allow_actions=True)
        reg.register(tool)

        task = Task(user_request="test tracked", task_id="t_tracked")
        task.steps = [TaskStep(step_id=1, tool_name="mock_tool_tracked", args={}, description="Step 1")]
        ai = DummyCompanionAI(plan_return=task)

        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=ai,
            tool_registry=reg,
            task_executor=TaskExecutor(tool_registry=reg, verifier=False),
        )

        # 1. Planning phase must NEVER execute tools
        bridge.plan_task("test tracked")
        mock_tool_exec.assert_not_called()

        # 2. Approving plan must NEVER execute tools
        bridge.approve_plan()
        mock_tool_exec.assert_not_called()

        # 3. Only execute_approved_task executes tools
        mock_tool_exec.return_value = ToolResult(tool_name="mock_tool_tracked", success=True)
        bridge.execute_approved_task()
        mock_tool_exec.assert_called_once()

    def test_35_window_close_unblocks_pending_approval(self):
        """Closing the window while an approval is pending unblocks thread safely."""
        root = tk.Tk()
        root.withdraw()
        try:
            app = BlindspotMainWindow(root=root, bridge=self.bridge)
            req = ActionRequest(
                tool_name="mock_action",
                risk=ActionRisk.LOW_RISK_ACTION,
                description="Test action request",
            )

            ready_evt = threading.Event()
            app._on_approval_waiting = ready_evt.set
            app.root.after = MagicMock()

            # Start worker thread requesting approval
            approval_result = [None]
            def _request_worker():
                approval_result[0] = app._ui_action_approver(req)

            t = threading.Thread(target=_request_worker, daemon=True)
            t.start()

            # Ensure the worker has reached the approval synchronization point before closing
            self.assertTrue(
                ready_evt.wait(timeout=2.0),
                "Worker thread failed to reach approval registration point",
            )
            with app._approval_lock:
                self.assertIsNotNone(app._active_approval_event)

            # Close window
            app.on_close()
            t.join(timeout=1.0)
            self.assertFalse(t.is_alive())
            self.assertFalse(approval_result[0])
        finally:
            try:
                root.destroy()
            except Exception:
                pass


    def test_36_approver_exception_fails_closed_in_execution(self):
        """If an action approver raises an unhandled exception, it must fail closed and block the task."""
        def faulty_approver(request):
            raise RuntimeError("Unexpected approver crash")

        action_task = Task(user_request="run faulty action", task_id="t_faulty")
        action_task.steps = [
            TaskStep(step_id=1, tool_name="mock_action", args={}, description="Step 1"),
        ]
        ai = DummyCompanionAI(plan_return=action_task)
        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=ai,
            tool_registry=self.registry,
            task_executor=self.executor,
            action_approver=faulty_approver,
        )

        bridge.plan_task("run faulty action")
        bridge.approve_plan()

        res = bridge.execute_approved_task()
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "blocked")
        self.assertEqual(res["failure_type"], "permission_denied")
        self.assertEqual(res["steps"][0]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
