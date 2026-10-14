"""
BLINDSPOT — Phase 7D: Task Recovery & Execution Resilience Tests.

Covers:
1. FailureType and RecoveryDecision enums and RecoveryRecord serialization.
2. Bounded RetryPolicy rules (safe-only, forbidden tools, bounded limits).
3. Transient failure recovery on eligible safe tools (read-only and idempotent focus).
4. Retry exhaustion and accurate reporting of failure classification and attempts.
5. Immediate halt without retries for non-idempotent actions (click, typing, keys, mouse, open).
6. Immediate halt without retries for permission denial.
7. Verification failure recovery for idempotent vs non-retryable actions.
8. Preservation of completed steps during retries (completed steps never re-run).
9. Accurate tracking of skipped subsequent steps upon halt.
10. Preservation of validation and safety boundaries.
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
)
from tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ActionRisk,
    PermissionPolicy,
    create_default_tool_registry,
)
from verification import (
    TaskVerifier,
    MockDesktopObserver,
    VerificationStatus,
)
from recovery import (
    FailureType,
    RecoveryDecision,
    RecoveryRecord,
    RetryPolicy,
    DEFAULT_SAFE_RETRY_TOOLS,
    FORBIDDEN_RETRY_TOOLS,
    MAX_ALLOWABLE_RETRIES,
)


class MockFlakyTool(BaseTool):
    """
    Test tool that fails a configurable number of times before succeeding.
    Records every call in execution_log.
    """
    def __init__(
        self,
        name: str,
        fail_times: int,
        execution_log: list,
        risk: ActionRisk = ActionRisk.READ_ONLY,
    ):
        super().__init__(name=name, description=f"Flaky tool {name}", risk=risk)
        self.fail_times = fail_times
        self.call_count = 0
        self.execution_log = execution_log

    def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        self.execution_log.append((self.name, self.call_count, kwargs))
        if self.call_count <= self.fail_times:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Transient failure (call {self.call_count}/{self.fail_times})",
            )
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Success on call {self.call_count}",
        )


class TestRecoveryDataModelAndPolicy(unittest.TestCase):
    """Test failure classification, recovery records, and retry policy safety bounds."""

    def test_01_failure_type_and_recovery_decision_values(self):
        """01. Enums contain required classifications and decisions."""
        self.assertEqual(FailureType.NONE.value, "none")
        self.assertEqual(FailureType.VALIDATION_ERROR.value, "validation_error")
        self.assertEqual(FailureType.PERMISSION_DENIED.value, "permission_denied")
        self.assertEqual(FailureType.EXECUTION_ERROR.value, "execution_error")
        self.assertEqual(FailureType.VERIFICATION_FAILURE.value, "verification_failure")
        self.assertEqual(FailureType.RETRY_EXHAUSTED.value, "retry_exhausted")

        self.assertEqual(RecoveryDecision.NONE.value, "none")
        self.assertEqual(RecoveryDecision.RETRY.value, "retry")
        self.assertEqual(RecoveryDecision.ABORT.value, "abort")

    def test_02_recovery_record_serialization(self):
        """02. RecoveryRecord serializes to dictionary representation."""
        rec = RecoveryRecord(
            failure_type=FailureType.EXECUTION_ERROR,
            decision=RecoveryDecision.RETRY,
            attempt=1,
            max_retries=2,
            error="Connection timeout",
            retryable=True,
            message="Retrying step",
        )
        d = rec.to_dict()
        self.assertEqual(d["failure_type"], "execution_error")
        self.assertEqual(d["decision"], "retry")
        self.assertEqual(d["attempt"], 1)
        self.assertEqual(d["max_retries"], 2)
        self.assertTrue(d["retryable"])

    def test_03_retry_policy_bounds_max_retries(self):
        """03. RetryPolicy enforces upper ceiling on allowable retries."""
        policy_high = RetryPolicy(max_retries=10)
        self.assertLessEqual(policy_high.max_retries, MAX_ALLOWABLE_RETRIES)

        policy_neg = RetryPolicy(max_retries=-1)
        self.assertEqual(policy_neg.max_retries, 0)

    def test_04_forbidden_tools_never_retry(self):
        """04. Clicks, typing, keys, mouse, and app launch are categorically forbidden from retrying."""
        policy = RetryPolicy(max_retries=2)
        for tool_name in ["click", "type_text", "press_key", "hotkey", "move_mouse", "open_application"]:
            self.assertFalse(
                policy.is_safe_to_retry(tool_name),
                f"Tool '{tool_name}' must be forbidden from retry",
            )
            decision, rec = policy.evaluate(tool_name, attempt=1, failure_type=FailureType.EXECUTION_ERROR)
            self.assertEqual(decision, RecoveryDecision.ABORT)
            self.assertFalse(rec.retryable)

    def test_05_permission_denial_and_validation_never_retry(self):
        """05. Permission denial and validation errors are never retried."""
        policy = RetryPolicy(max_retries=2)
        self.assertFalse(
            policy.is_safe_to_retry("get_current_context", failure_type=FailureType.PERMISSION_DENIED)
        )
        self.assertFalse(
            policy.is_safe_to_retry("get_current_context", failure_type=FailureType.VALIDATION_ERROR)
        )

        decision, rec = policy.evaluate(
            "get_current_context", attempt=1, failure_type=FailureType.PERMISSION_DENIED
        )
        self.assertEqual(decision, RecoveryDecision.ABORT)
        self.assertEqual(rec.failure_type, FailureType.PERMISSION_DENIED)

    def test_06_safe_tools_eligible_for_retry(self):
        """06. Only explicit safe tools (read-only queries and idempotent focus) are retryable."""
        policy = RetryPolicy(max_retries=2)
        for safe_tool in ["get_current_context", "get_recent_activity", "get_conversation_history", "focus_application"]:
            self.assertTrue(
                policy.is_safe_to_retry(safe_tool),
                f"Tool '{safe_tool}' should be eligible for retry",
            )


class TestTaskExecutionRecovery(unittest.TestCase):
    """Integration tests for TaskExecutor recovery decisions and bounded retries."""

    def setUp(self):
        self.execution_log = []
        self.mock_io = MockDesktopIO()
        self.registry = create_default_tool_registry(
            allow_actions=True,
            permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
            desktop_io=self.mock_io,
            include_action_tools=True,
        )
        self.mock_observer = MockDesktopObserver()
        self.verifier = TaskVerifier(observer=self.mock_observer)
        self.retry_policy = RetryPolicy(max_retries=2)
        self.executor = TaskExecutor(
            tool_registry=self.registry,
            verifier=self.verifier,
            retry_policy=self.retry_policy,
        )

    def test_07_transient_failure_recovers_on_safe_read_only_tool(self):
        """07. Safe read-only tool retries after transient failure and succeeds on attempt 2."""
        flaky_ro = MockFlakyTool("get_current_context", fail_times=1, execution_log=self.execution_log)
        # Register in a clean registry
        reg = ToolRegistry(allow_actions=True)
        reg.register(flaky_ro)

        executor = TaskExecutor(tool_registry=reg, retry_policy=self.retry_policy)
        task = create_task(
            user_request="Read context with transient failure",
            steps_data=[{"tool_name": "get_current_context", "args": {}}],
            task_id="recover_ro_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.retry_attempts, 1)
        self.assertEqual(task.steps[0].attempts, 2)
        self.assertEqual(task.steps[0].status, StepStatus.COMPLETED)
        self.assertEqual(flaky_ro.call_count, 2)

    def test_08_retry_exhaustion_on_persistent_failure(self):
        """08. Persistent failure exhausts allowed retries, marks RETRY_EXHAUSTED, and skips later steps."""
        # Fails 5 times (exceeds max_retries=2)
        flaky_ro = MockFlakyTool("get_current_context", fail_times=5, execution_log=self.execution_log)
        dummy_tool = MockFlakyTool("focus_application", fail_times=0, execution_log=self.execution_log)

        reg = ToolRegistry(allow_actions=True)
        reg.register(flaky_ro)
        reg.register(dummy_tool)

        executor = TaskExecutor(
            tool_registry=reg,
            retry_policy=RetryPolicy(max_retries=2),
        )
        task = create_task(
            user_request="Read context then focus app",
            steps_data=[
                {"tool_name": "get_current_context", "args": {}},
                {"tool_name": "focus_application", "args": {}},
            ],
            task_id="exhaust_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 0)
        self.assertEqual(summary.failed_step_id, 1)
        self.assertEqual(summary.failure_type, "retry_exhausted")
        self.assertEqual(summary.retry_attempts, 2)  # retried twice, 3 attempts total
        self.assertEqual(task.steps[0].attempts, 3)

        # Later step must have been skipped
        self.assertEqual(summary.skipped_steps, 1)
        self.assertEqual(summary.skipped_step_ids, [2])
        self.assertEqual(task.steps[1].status, StepStatus.PENDING)
        self.assertEqual(dummy_tool.call_count, 0)

    def test_09_non_idempotent_actions_halt_immediately_without_retrying(self):
        """09. Potentially destructive or non-idempotent actions (click, type_text, move_mouse) never retry."""
        failing_io = MockDesktopIO(fail_operations=["click", "type_text", "move_mouse"])
        reg = create_default_tool_registry(
            allow_actions=True,
            permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
            desktop_io=failing_io,
            include_action_tools=True,
        )
        executor = TaskExecutor(tool_registry=reg, retry_policy=self.retry_policy)

        for non_idempotent_tool in ["click", "type_text", "move_mouse"]:
            args = {"x": 10, "y": 20} if non_idempotent_tool in ("click", "move_mouse") else {"text": "hi"}
            task = create_task(
                user_request=f"Execute {non_idempotent_tool}",
                steps_data=[
                    {"tool_name": non_idempotent_tool, "args": args},
                    {"tool_name": "get_current_context", "args": {}},
                ],
                task_id=f"no_retry_{non_idempotent_tool}",
            )
            summary = executor.execute(task)

            self.assertEqual(summary.status, TaskStatus.FAILED)
            self.assertEqual(summary.executed_steps, 1)
            self.assertEqual(summary.retry_attempts, 0)  # ZERO retries
            self.assertEqual(task.steps[0].attempts, 1)  # EXACTLY 1 attempt
            self.assertEqual(task.steps[0].recovery.retryable, False)
            self.assertEqual(summary.skipped_steps, 1)

    def test_10_permission_denial_halts_immediately_without_retrying(self):
        """10. Permission denial halts immediately with zero retries."""
        denying_policy = PermissionPolicy(low_risk_approver=lambda req: False)
        reg = create_default_tool_registry(
            allow_actions=True,
            permission_policy=denying_policy,
            desktop_io=self.mock_io,
            include_action_tools=True,
        )
        executor = TaskExecutor(tool_registry=reg, retry_policy=self.retry_policy)

        task = create_task(
            user_request="Denied action",
            steps_data=[
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
                {"tool_name": "get_current_context", "args": {}},
            ],
            task_id="deny_halt_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.BLOCKED)
        self.assertEqual(summary.failure_type, "permission_denied")
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.retry_attempts, 0)
        self.assertEqual(task.steps[0].attempts, 1)
        self.assertEqual(summary.skipped_steps, 1)
        self.assertEqual(summary.skipped_step_ids, [2])

    def test_11_verification_failure_recovery_on_idempotent_focus(self):
        """11. Verification failure on idempotent focus_application retries and verifies on attempt 2."""
        # Simulated observer: first call returns WrongWindow, second call returns Notepad
        observed_windows = ["WrongWindow", "Document - Notepad"]

        def _changing_observer():
            win = observed_windows.pop(0) if observed_windows else "Document - Notepad"
            obs = RawObservation(window_title=win, ocr_text="")
            interp = InterpretedContext(activity="writing", confidence=2)
            return DesktopContext(observation=obs, interpretation=interp)

        verifier = TaskVerifier(observer=_changing_observer)
        executor = TaskExecutor(
            tool_registry=self.registry,
            verifier=verifier,
            retry_policy=self.retry_policy,
        )

        task = create_task(
            user_request="Focus notepad",
            steps_data=[
                {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}},
            ],
            task_id="focus_retry_verify",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.verified_steps, 1)
        self.assertEqual(summary.retry_attempts, 1)
        self.assertEqual(task.steps[0].attempts, 2)
        self.assertEqual(task.steps[0].verification.status, VerificationStatus.VERIFIED)

    def test_12_verification_failure_exhaustion_on_focus(self):
        """12. Persistent verification failure on focus_application exhausts retries and halts cleanly."""
        # Observer always sees WrongWindow
        observer = MockDesktopObserver(window_title="WrongWindow")
        verifier = TaskVerifier(observer=observer)
        executor = TaskExecutor(
            tool_registry=self.registry,
            verifier=verifier,
            retry_policy=RetryPolicy(max_retries=2),
        )

        task = create_task(
            user_request="Focus notepad with failure",
            steps_data=[
                {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}},
                {"tool_name": "get_current_context", "args": {}},
            ],
            task_id="focus_exhaust_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.failure_type, "retry_exhausted")
        self.assertEqual(summary.retry_attempts, 2)
        self.assertEqual(task.steps[0].attempts, 3)
        self.assertEqual(summary.skipped_steps, 1)

    def test_13_completed_steps_not_reexecuted_during_retry(self):
        """13. Completed steps are never re-executed when a later step retries."""
        step1_calls = []
        step2_calls = []

        class Step1Tool(BaseTool):
            def __init__(self):
                super().__init__(name="step1_tool", description="", risk=ActionRisk.READ_ONLY)
            def execute(self, **kwargs):
                step1_calls.append(True)
                return ToolResult(tool_name=self.name, success=True, output="Step 1 OK")

        class Step2FlakyTool(BaseTool):
            def __init__(self):
                super().__init__(name="get_current_context", description="", risk=ActionRisk.READ_ONLY)
            def execute(self, **kwargs):
                step2_calls.append(True)
                if len(step2_calls) == 1:
                    return ToolResult(tool_name=self.name, success=False, error="Fail on 1")
                return ToolResult(tool_name=self.name, success=True, output="Step 2 OK")

        reg = ToolRegistry(allow_actions=True)
        reg.register(Step1Tool())
        reg.register(Step2FlakyTool())

        executor = TaskExecutor(
            tool_registry=reg,
            retry_policy=RetryPolicy(max_retries=2),
        )
        task = create_task(
            user_request="Step 1 then Step 2",
            steps_data=[
                {"tool_name": "step1_tool", "args": {}},
                {"tool_name": "get_current_context", "args": {}},
            ],
            task_id="preserve_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.successful_steps, 2)
        self.assertEqual(summary.retry_attempts, 1)

        # Step 1 was executed EXACTLY ONCE
        self.assertEqual(len(step1_calls), 1)
        # Step 2 was executed TWICE (failed once, succeeded on retry)
        self.assertEqual(len(step2_calls), 2)
        # Step 1 status remained COMPLETED
        self.assertEqual(task.steps[0].status, StepStatus.COMPLETED)
        self.assertEqual(task.steps[1].status, StepStatus.COMPLETED)

    def test_14_disabled_retry_policy_halts_on_first_failure(self):
        """14. When retry policy is disabled, eligible safe tools halt on the very first failure."""
        flaky_ro = MockFlakyTool("get_current_context", fail_times=1, execution_log=self.execution_log)
        reg = ToolRegistry(allow_actions=True)
        reg.register(flaky_ro)

        disabled_policy = RetryPolicy(enabled=False, max_retries=0)
        executor = TaskExecutor(tool_registry=reg, retry_policy=disabled_policy)

        task = create_task(
            user_request="No retry task",
            steps_data=[{"tool_name": "get_current_context", "args": {}}],
            task_id="no_retry_task",
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.retry_attempts, 0)
        self.assertEqual(flaky_ro.call_count, 1)

    def test_15_validation_rejection_marks_skipped_steps(self):
        """15. Pre-execution task validation rejection classifies failure and marks all steps skipped."""
        task_dangerous = create_task(
            user_request="Execute bash",
            steps_data=[{"tool_name": "bash", "args": {}}],
            task_id="danger_task",
        )
        summary = self.executor.execute(task_dangerous, pre_validate=True)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.failure_type, "validation_error")
        self.assertEqual(summary.executed_steps, 0)
        self.assertEqual(summary.skipped_steps, 1)
        self.assertEqual(summary.skipped_step_ids, [1])


if __name__ == "__main__":
    unittest.main()
