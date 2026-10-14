"""
BLINDSPOT — Phase 7C: Context-Aware Task Verification Tests.

Covers:
1. VerificationStatus and StepVerificationResult data model and serialization.
2. Perception observation interface (DesktopObserver, DefaultContextObserver, MockDesktopObserver).
3. Verification of open_application and focus_application (success & failure).
4. Explicit not_verifiable status for unverified desktop actions and read-only queries.
5. TaskExecutor integration: recording verification results and halting on verification failure.
6. Execution failure and permission denial halting execution before verification.
7. Multi-step sequential execution with verification.
8. Preservation of safety constraints (dangerous tools, 5-step limit, task validation).
"""

import unittest
from context import DesktopContext, RawObservation, InterpretedContext
from desktop_io import MockDesktopIO
from task import (
    Task,
    TaskStep,
    TaskStatus,
    StepStatus,
    TaskExecutor,
    create_task,
    validate_task,
    MAX_TASK_STEPS,
)
from tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ActionRisk,
    PermissionPolicy,
    ActionRequest,
    create_default_tool_registry,
)
from verification import (
    VerificationStatus,
    StepVerificationResult,
    DesktopObserver,
    DefaultContextObserver,
    MockDesktopObserver,
    TaskVerifier,
)


class TestVerificationDataModel(unittest.TestCase):
    """Test VerificationStatus enum and StepVerificationResult dataclass."""

    def test_01_verification_status_values(self):
        """01. VerificationStatus has verified, failed, and not_verifiable values."""
        self.assertEqual(VerificationStatus.VERIFIED.value, "verified")
        self.assertEqual(VerificationStatus.FAILED.value, "failed")
        self.assertEqual(VerificationStatus.NOT_VERIFIABLE.value, "not_verifiable")

    def test_02_step_verification_result_to_dict(self):
        """02. StepVerificationResult serializes cleanly to a dictionary."""
        res = StepVerificationResult(
            status=VerificationStatus.VERIFIED,
            message="Application 'notepad' verified active.",
            details={"window_title": "Untitled - Notepad", "app_name": "notepad"},
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "verified")
        self.assertEqual(d["message"], "Application 'notepad' verified active.")
        self.assertEqual(d["details"]["window_title"], "Untitled - Notepad")


class TestObservationInterface(unittest.TestCase):
    """Test observation interface that reuses existing perception architecture."""

    def test_03_mock_desktop_observer(self):
        """03. MockDesktopObserver returns deterministic context snapshots."""
        obs = MockDesktopObserver(window_title="Calculator")
        ctx = obs.observe()
        self.assertIsInstance(ctx, DesktopContext)
        self.assertEqual(ctx.window_title, "Calculator")

        obs.set_window("Notepad", activity="writing", confidence=3)
        ctx2 = obs.observe()
        self.assertEqual(ctx2.window_title, "Notepad")
        self.assertEqual(ctx2.activity, "writing")
        self.assertEqual(ctx2.confidence, 3)

    def test_04_default_context_observer_with_state(self):
        """04. DefaultContextObserver pulls from live state when available."""
        raw = RawObservation(window_title="Visual Studio Code", ocr_text="def test(): pass")
        interp = InterpretedContext(activity="coding", confidence=3)
        mock_ctx = DesktopContext(observation=raw, interpretation=interp)

        state = {"current_context": mock_ctx}
        observer = DefaultContextObserver(state=state)
        observed = observer.observe()

        self.assertEqual(observed.window_title, "Visual Studio Code")
        self.assertEqual(observed.activity, "coding")

    def test_05_default_context_observer_with_provider_callable(self):
        """05. DefaultContextObserver pulls from custom provider callable."""
        raw = RawObservation(window_title="Edge Browser", ocr_text="")
        interp = InterpretedContext(activity="browsing", confidence=2)
        mock_ctx = DesktopContext(observation=raw, interpretation=interp)

        observer = DefaultContextObserver(context_provider=lambda: mock_ctx)
        observed = observer.observe()
        self.assertEqual(observed.window_title, "Edge Browser")


class TestTaskVerifierLogic(unittest.TestCase):
    """Unit tests for TaskVerifier step verification logic."""

    def setUp(self):
        self.mock_observer = MockDesktopObserver()
        self.verifier = TaskVerifier(observer=self.mock_observer)

    def test_06_verify_open_application_success(self):
        """06. open_application verifies successfully when active window matches."""
        self.mock_observer.set_window("Untitled - Notepad")
        step = TaskStep(step_id=1, tool_name="open_application", args={"app_name": "notepad"})
        res = self.verifier.verify_step(step)

        self.assertEqual(res.status, VerificationStatus.VERIFIED)
        self.assertIn("verified active", res.message)
        self.assertEqual(res.details["app_name"], "notepad")

    def test_07_verify_open_application_alias_calc(self):
        """07. open_application matches canonical application aliases like calc/calculator."""
        self.mock_observer.set_window("Calculator")
        step = TaskStep(step_id=1, tool_name="open_application", args={"app_name": "calculator"})
        res = self.verifier.verify_step(step)
        self.assertEqual(res.status, VerificationStatus.VERIFIED)

    def test_08_verify_open_application_failure(self):
        """08. open_application fails verification when active window does not match."""
        self.mock_observer.set_window("Terminal")
        step = TaskStep(step_id=1, tool_name="open_application", args={"app_name": "notepad"})
        res = self.verifier.verify_step(step)

        self.assertEqual(res.status, VerificationStatus.FAILED)
        self.assertIn("expected 'notepad'", res.message)
        self.assertEqual(res.details["observed_window"], "Terminal")

    def test_09_verify_open_application_missing_args(self):
        """09. open_application fails verification if app_name is empty."""
        step = TaskStep(step_id=1, tool_name="open_application", args={})
        res = self.verifier.verify_step(step)
        self.assertEqual(res.status, VerificationStatus.FAILED)

    def test_10_verify_focus_application_success(self):
        """10. focus_application verifies successfully when title contains fragment."""
        self.mock_observer.set_window("Document - WordPad")
        step = TaskStep(step_id=1, tool_name="focus_application", args={"window_title_fragment": "wordpad"})
        res = self.verifier.verify_step(step)

        self.assertEqual(res.status, VerificationStatus.VERIFIED)
        self.assertIn("Window focus verified", res.message)

    def test_11_verify_focus_application_failure(self):
        """11. focus_application fails verification when active window does not contain fragment."""
        self.mock_observer.set_window("Untitled - Paint")
        step = TaskStep(step_id=1, tool_name="focus_application", args={"window_title_fragment": "WordPad"})
        res = self.verifier.verify_step(step)

        self.assertEqual(res.status, VerificationStatus.FAILED)
        self.assertIn("does not match fragment", res.message)

    def test_12_unsupported_actions_marked_not_verifiable(self):
        """12. Actions without reliable verification are explicitly marked NOT_VERIFIABLE."""
        unsupported_tools = ["move_mouse", "click", "type_text", "press_key", "hotkey", "get_current_context"]
        for tool_name in unsupported_tools:
            step = TaskStep(step_id=1, tool_name=tool_name, args={})
            res = self.verifier.verify_step(step)
            self.assertEqual(
                res.status,
                VerificationStatus.NOT_VERIFIABLE,
                f"Tool '{tool_name}' must be marked NOT_VERIFIABLE",
            )
            self.assertIn("does not support reliable", res.message)


class TestTaskExecutorVerificationIntegration(unittest.TestCase):
    """Integration tests for TaskExecutor verifying scenarios from Phase 7C prompt."""

    def setUp(self):
        self.mock_io = MockDesktopIO()
        self.registry = create_default_tool_registry(
            allow_actions=True,
            permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
            desktop_io=self.mock_io,
            include_action_tools=True,
        )
        self.mock_observer = MockDesktopObserver()
        self.verifier = TaskVerifier(observer=self.mock_observer)
        self.executor = TaskExecutor(tool_registry=self.registry, verifier=self.verifier)

    def test_13_scenario_1_read_only_multi_step_task_completes(self):
        """Scenario 1: Read-only multi-step task completes (marked NOT_VERIFIABLE, not falsely failed)."""
        task = create_task(
            user_request="Inspect current context and activity",
            steps_data=[
                {"tool_name": "get_current_context", "args": {}},
                {"tool_name": "get_recent_activity", "args": {}},
            ],
            task_id="ro_task",
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 2)
        # Both steps marked not_verifiable, neither failed
        self.assertEqual(task.steps[0].verification.status, VerificationStatus.NOT_VERIFIABLE)
        self.assertEqual(task.steps[1].verification.status, VerificationStatus.NOT_VERIFIABLE)

    def test_14_scenario_2_approved_desktop_action_executes_and_verifies(self):
        """Scenario 2: Approved desktop action executes and its verification is recorded."""
        self.mock_observer.set_window("Untitled - Notepad")
        task = create_task(
            user_request="Open notepad",
            steps_data=[
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
            ],
            task_id="action_task",
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.verified_steps, 1)

        v = task.steps[0].verification
        self.assertIsNotNone(v)
        self.assertEqual(v.status, VerificationStatus.VERIFIED)
        self.assertIn("verified active", v.message)

        # Check serialization contains verification
        d = task.steps[0].to_dict()
        self.assertEqual(d["verification"]["status"], "verified")

    def test_15_scenario_3_verification_failure_stops_subsequent_steps(self):
        """Scenario 3: Verification failure stops subsequent steps immediately."""
        # Observer sees "Calculator", which mismatches step 1's "notepad"
        self.mock_observer.set_window("Calculator")
        task = create_task(
            user_request="Open notepad and then move mouse",
            steps_data=[
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
                {"tool_name": "move_mouse", "args": {"x": 100, "y": 200}},
            ],
            task_id="fail_verify_task",
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 0)
        self.assertEqual(summary.failed_step_id, 1)
        self.assertIn("verification failed", summary.error.lower())

        # Step 1 failed at verification
        step_1 = task.steps[0]
        self.assertEqual(step_1.status, StepStatus.FAILED)
        self.assertEqual(step_1.verification.status, VerificationStatus.FAILED)

        # Step 2 was never executed
        step_2 = task.steps[1]
        self.assertEqual(step_2.status, StepStatus.PENDING)
        self.assertIsNone(step_2.result)
        self.assertIsNone(step_2.verification)

    def test_16_scenario_4_permission_denial_stops_before_verification(self):
        """Scenario 4: Permission denial stops execution before verification is invoked."""
        denying_policy = PermissionPolicy(low_risk_approver=lambda req: False)
        reg = create_default_tool_registry(
            allow_actions=True,
            permission_policy=denying_policy,
            desktop_io=self.mock_io,
            include_action_tools=True,
        )
        executor = TaskExecutor(tool_registry=reg, verifier=self.verifier)

        task = create_task(
            user_request="Open notepad then move mouse",
            steps_data=[
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
                {"tool_name": "move_mouse", "args": {"x": 10, "y": 20}},
            ],
            task_id="deny_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.BLOCKED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 0)

        # Verification was NEVER invoked for denied step
        self.assertIsNone(task.steps[0].verification)
        self.assertEqual(task.steps[0].status, StepStatus.BLOCKED)
        self.assertEqual(task.steps[1].status, StepStatus.PENDING)

    def test_17_scenario_5_execution_failure_stops_before_verification(self):
        """Scenario 5: Execution failure stops execution before verification is invoked."""
        failing_io = MockDesktopIO(fail_operations=["open_application"])
        reg = create_default_tool_registry(
            allow_actions=True,
            permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
            desktop_io=failing_io,
            include_action_tools=True,
        )
        executor = TaskExecutor(tool_registry=reg, verifier=self.verifier)

        task = create_task(
            user_request="Open notepad then move mouse",
            steps_data=[
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
                {"tool_name": "move_mouse", "args": {"x": 50, "y": 50}},
            ],
            task_id="exec_fail_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 0)

        # Verification was NEVER invoked for execution failure
        self.assertIsNone(task.steps[0].verification)
        self.assertEqual(task.steps[0].status, StepStatus.FAILED)
        self.assertEqual(task.steps[1].status, StepStatus.PENDING)

    def test_18_scenario_6_unsupported_verification_is_not_verifiable(self):
        """Scenario 6: Unsupported verification is marked not_verifiable, not falsely successful."""
        task = create_task(
            user_request="Move mouse",
            steps_data=[
                {"tool_name": "move_mouse", "args": {"x": 100, "y": 200}},
            ],
            task_id="unsupported_task",
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.verified_steps, 0)  # NOT counted as verified!

        v = task.steps[0].verification
        self.assertIsNotNone(v)
        self.assertEqual(v.status, VerificationStatus.NOT_VERIFIABLE)

    def test_19_scenario_7_safety_limits_intact(self):
        """Scenario 7: Dangerous tools, task validation, and 5-step limit remain intact."""
        # Dangerous tool rejection
        task_dangerous = create_task(
            user_request="Run shell",
            steps_data=[{"tool_name": "run_shell", "args": {"command": "dir"}}],
            task_id="dangerous_task",
        )
        is_valid, err = validate_task(task_dangerous, self.registry)
        self.assertFalse(is_valid)
        self.assertIn("dangerous tool", err)

        # 5-step limit
        steps_6 = [{"tool_name": "get_current_context", "args": {}} for _ in range(6)]
        task_6 = create_task(user_request="6 steps", steps_data=steps_6, task_id="too_long")
        is_valid_6, err_6 = validate_task(task_6, self.registry, max_steps=5)
        self.assertFalse(is_valid_6)
        self.assertIn("exceeds maximum allowed limit", err_6)

    def test_20_sequential_verification_multi_step_approved(self):
        """20. Multi-step approved actions with verification succeed sequentially."""
        # Use an observer that simulates the active window changing as steps run
        windows = ["Untitled - Notepad", "Untitled - Notepad"]

        def _changing_observer():
            win = windows.pop(0) if windows else "Untitled - Notepad"
            obs = RawObservation(window_title=win, ocr_text="")
            interp = InterpretedContext(activity="writing", confidence=2)
            return DesktopContext(observation=obs, interpretation=interp)

        verifier = TaskVerifier(observer=_changing_observer)
        executor = TaskExecutor(tool_registry=self.registry, verifier=verifier)

        task = create_task(
            user_request="Open notepad and focus it",
            steps_data=[
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
                {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}},
            ],
            task_id="seq_verify_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 2)
        self.assertEqual(summary.verified_steps, 2)
        self.assertEqual(task.steps[0].verification.status, VerificationStatus.VERIFIED)
        self.assertEqual(task.steps[1].verification.status, VerificationStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main()
