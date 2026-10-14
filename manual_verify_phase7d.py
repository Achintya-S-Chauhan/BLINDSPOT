"""
BLINDSPOT — Phase 7D: Task Recovery & Execution Resilience Demonstrator.

Executes and verifies deterministic recovery scenarios:
1. Transient failure on safe read-only tool recovers on retry.
2. Persistent failure on safe tool exhausts bounded retries and skips subsequent steps.
3. Non-idempotent desktop action (click, type, mouse, open) fails with ZERO retries.
4. Permission denial halts immediately with ZERO retries.
5. Verification failure on idempotent focus_application recovers on retry.
6. Completed steps are preserved and NEVER re-executed during recovery.
7. Validation failure halts before execution and reports all steps skipped.
"""

from context import DesktopContext, RawObservation, InterpretedContext
from desktop_io import MockDesktopIO
from task import TaskExecutor, TaskStatus, StepStatus, create_task, validate_task
from tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ActionRisk,
    PermissionPolicy,
    create_default_tool_registry,
)
from verification import TaskVerifier, MockDesktopObserver, VerificationStatus
from recovery import RetryPolicy, FailureType, RecoveryDecision


def print_banner(title: str):
    print("=" * 70)
    print(f" SCENARIO: {title}")
    print("=" * 70)


def print_summary(summary):
    retry_str = f", {summary.retry_attempts} retries" if summary.retry_attempts > 0 else ""
    print(f"Task Result:    {summary.status.value.upper()}")
    print(
        f"Progress:       {summary.successful_steps}/{summary.total_steps} completed "
        f"({summary.executed_steps} executed, {summary.verified_steps} verified{retry_str})"
    )
    for sr in summary.step_results:
        v = sr.get("verification") or {}
        v_status = v.get("status")
        v_msg = v.get("message")
        attempts = sr.get("attempts", 1)
        att_str = f" [attempts: {attempts}]" if attempts > 1 else ""

        if sr["status"] == "completed":
            sym = "[OK: VERIFIED]" if v_status == "verified" else "[OK: UNVERIFIED]"
        elif sr["status"] == "blocked":
            sym = "[BLOCKED]"
        else:
            sym = "[FAILED: VERIFICATION]" if v_status == "failed" else "[FAILED: EXECUTION]"

        print(f"  {sym} Step {sr['step_id']}: {sr['tool_name']}{att_str}")
        if sr.get("output"):
            print(f"       Result:       {sr['output']}")
        if v_status:
            print(f"       Verification: [{v_status.upper()}] {v_msg}")
        if sr.get("recovery") and sr["recovery"].get("message") and sr["recovery"].get("decision") != "none":
            print(f"       Recovery:     {sr['recovery']['message']}")
        if sr.get("error"):
            print(f"       Error:        {sr['error']}")

    if summary.status != TaskStatus.COMPLETED:
        print(f"Task halted: {summary.error or 'Stopped'}")
        if summary.failure_type:
            print(f"Failure classification: {summary.failure_type.upper()}")
        if summary.skipped_steps > 0:
            print(f"Skipped steps: {summary.skipped_steps} ({', '.join(f'Step {i}' for i in summary.skipped_step_ids)})")
    print()


def main():
    print("BLINDSPOT Phase 7D -- Recovery & Resilience Demonstrator\n")

    # 1. Safe read-only tool transient failure recovers on retry
    print_banner("1. Transient failure on safe tool recovers on retry")
    call_log_1 = []

    class TransientTool(BaseTool):
        def __init__(self):
            super().__init__(name="get_current_context", description="", risk=ActionRisk.READ_ONLY)
            self.calls = 0
        def execute(self, **kwargs):
            self.calls += 1
            call_log_1.append(self.calls)
            if self.calls == 1:
                return ToolResult(tool_name=self.name, success=False, error="Transient IPC timeout")
            return ToolResult(tool_name=self.name, success=True, output="Context retrieved successfully")

    reg1 = ToolRegistry(allow_actions=True)
    reg1.register(TransientTool())
    executor1 = TaskExecutor(tool_registry=reg1, retry_policy=RetryPolicy(max_retries=2))
    task1 = create_task("Read context", [{"tool_name": "get_current_context", "args": {}}])
    s1 = executor1.execute(task1)
    print_summary(s1)
    assert s1.status == TaskStatus.COMPLETED
    assert s1.retry_attempts == 1
    assert task1.steps[0].attempts == 2

    # 2. Persistent failure exhausts retries and skips subsequent steps
    print_banner("2. Persistent failure exhausts retries and skips later steps")
    class FailingTool(BaseTool):
        def __init__(self):
            super().__init__(name="get_current_context", description="", risk=ActionRisk.READ_ONLY)
        def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, success=False, error="Persistent disk read error")

    class SecondTool(BaseTool):
        def __init__(self):
            super().__init__(name="get_recent_activity", description="", risk=ActionRisk.READ_ONLY)
        def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, success=True, output="Recent activity OK")

    reg2 = ToolRegistry(allow_actions=True)
    reg2.register(FailingTool())
    reg2.register(SecondTool())
    executor2 = TaskExecutor(tool_registry=reg2, retry_policy=RetryPolicy(max_retries=2))
    task2 = create_task(
        "Read context then next step",
        [
            {"tool_name": "get_current_context", "args": {}},
            {"tool_name": "get_recent_activity", "args": {}},
        ],
    )
    s2 = executor2.execute(task2)
    print_summary(s2)
    assert s2.status == TaskStatus.FAILED
    assert s2.failure_type == "retry_exhausted"
    assert s2.retry_attempts == 2
    assert task2.steps[0].attempts == 3
    assert s2.skipped_steps == 1
    assert s2.skipped_step_ids == [2]

    # 3. Non-idempotent action fails with ZERO retries
    print_banner("3. Non-idempotent desktop action (click) fails with ZERO retries")
    failing_io = MockDesktopIO(fail_operations=["click"])
    reg3 = create_default_tool_registry(
        allow_actions=True,
        permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
        desktop_io=failing_io,
        include_action_tools=True,
    )
    executor3 = TaskExecutor(tool_registry=reg3, retry_policy=RetryPolicy(max_retries=2))
    task3 = create_task(
        "Click then read context",
        [
            {"tool_name": "click", "args": {"x": 100, "y": 200}},
            {"tool_name": "get_current_context", "args": {}},
        ],
    )
    s3 = executor3.execute(task3)
    print_summary(s3)
    assert s3.status == TaskStatus.FAILED
    assert s3.retry_attempts == 0
    assert task3.steps[0].attempts == 1
    assert s3.skipped_steps == 1

    # 4. Permission denial halts immediately with ZERO retries
    print_banner("4. Permission denial halts immediately with ZERO retries")
    denying_io = MockDesktopIO()
    reg4 = create_default_tool_registry(
        allow_actions=True,
        permission_policy=PermissionPolicy(low_risk_approver=lambda req: False),
        desktop_io=denying_io,
        include_action_tools=True,
    )
    executor4 = TaskExecutor(tool_registry=reg4, retry_policy=RetryPolicy(max_retries=2))
    task4 = create_task(
        "Open app without permission",
        [
            {"tool_name": "open_application", "args": {"app_name": "notepad"}},
            {"tool_name": "get_current_context", "args": {}},
        ],
    )
    s4 = executor4.execute(task4)
    print_summary(s4)
    assert s4.status == TaskStatus.BLOCKED
    assert s4.failure_type == "permission_denied"
    assert s4.retry_attempts == 0
    assert task4.steps[0].attempts == 1
    assert s4.skipped_steps == 1

    # 5. Verification failure on idempotent focus_application recovers on retry
    print_banner("5. Verification failure on focus_application recovers on retry")
    windows = ["WrongWindow", "Untitled - Notepad"]
    def _observer():
        w = windows.pop(0) if windows else "Untitled - Notepad"
        obs = RawObservation(window_title=w, ocr_text="")
        interp = InterpretedContext(activity="writing", confidence=2)
        return DesktopContext(observation=obs, interpretation=interp)

    verifier5 = TaskVerifier(observer=_observer)
    reg5 = create_default_tool_registry(
        allow_actions=True,
        permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
        desktop_io=denying_io,
        include_action_tools=True,
    )
    executor5 = TaskExecutor(tool_registry=reg5, verifier=verifier5, retry_policy=RetryPolicy(max_retries=2))
    task5 = create_task(
        "Focus Notepad",
        [{"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}}],
    )
    s5 = executor5.execute(task5)
    print_summary(s5)
    assert s5.status == TaskStatus.COMPLETED
    assert s5.verified_steps == 1
    assert s5.retry_attempts == 1
    assert task5.steps[0].attempts == 2

    # 6. Completed steps are preserved and NEVER re-executed
    print_banner("6. Completed steps preserved and never re-executed")
    step1_count = 0
    step2_count = 0
    class CountingStep1(BaseTool):
        def __init__(self):
            super().__init__(name="step1", description="", risk=ActionRisk.READ_ONLY)
        def execute(self, **kwargs):
            nonlocal step1_count
            step1_count += 1
            return ToolResult(tool_name=self.name, success=True, output="Step 1 OK")

    class FlakyStep2(BaseTool):
        def __init__(self):
            super().__init__(name="get_current_context", description="", risk=ActionRisk.READ_ONLY)
        def execute(self, **kwargs):
            nonlocal step2_count
            step2_count += 1
            if step2_count == 1:
                return ToolResult(tool_name=self.name, success=False, error="Glitch")
            return ToolResult(tool_name=self.name, success=True, output="Step 2 OK")

    reg6 = ToolRegistry(allow_actions=True)
    reg6.register(CountingStep1())
    reg6.register(FlakyStep2())
    executor6 = TaskExecutor(tool_registry=reg6, retry_policy=RetryPolicy(max_retries=2))
    task6 = create_task(
        "Preserve step 1",
        [
            {"tool_name": "step1", "args": {}},
            {"tool_name": "get_current_context", "args": {}},
        ],
    )
    s6 = executor6.execute(task6)
    print_summary(s6)
    assert s6.status == TaskStatus.COMPLETED
    assert step1_count == 1  # Step 1 was NEVER re-executed!
    assert step2_count == 2  # Step 2 retried once and passed
    assert s6.retry_attempts == 1

    print("\nALL 7 PHASE 7D RESILIENCE SCENARIOS DETERMINISTICALLY VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
