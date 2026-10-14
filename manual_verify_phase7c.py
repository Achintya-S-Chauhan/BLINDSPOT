"""
BLINDSPOT — Phase 7C: Deterministic Verification Demonstrator Script.

Executes and verifies all 7 Phase 7C scenarios using deterministic mocks:
1. Read-only multi-step task completes.
2. Approved desktop action executes and its verification is recorded.
3. Verification failure stops subsequent steps.
4. Permission denial stops execution before verification.
5. Execution failure stops execution before verification.
6. Unsupported verification is marked not_verifiable, not falsely successful.
7. Existing dangerous-tool rejection, task validation, and five-step limit remain intact.

Note: Runs deterministically via MockDesktopIO and MockDesktopObserver.
Does not claim or depend on real Windows desktop state.
"""

import sys
from task import TaskExecutor, TaskStatus, StepStatus, create_task, validate_task
from tools import (
    create_default_tool_registry,
    PermissionPolicy,
    ActionRisk,
    BLOCKED_DANGEROUS_TOOL_NAMES,
)
from desktop_io import MockDesktopIO
from verification import (
    TaskVerifier,
    MockDesktopObserver,
    VerificationStatus,
)


def print_banner(title: str):
    print("=" * 70)
    print(f" SCENARIO: {title}")
    print("=" * 70)


def print_summary(summary):
    print(f"Task Result:    {summary.status.value.upper()}")
    print(
        f"Progress:       {summary.successful_steps}/{summary.total_steps} completed "
        f"({summary.executed_steps} executed, {summary.verified_steps} verified)"
    )
    for sr in summary.step_results:
        v = sr.get("verification") or {}
        v_status = v.get("status")
        v_msg = v.get("message")

        if sr["status"] == "completed":
            sym = "[OK: VERIFIED]" if v_status == "verified" else "[OK: UNVERIFIED]"
        elif sr["status"] == "blocked":
            sym = "[BLOCKED]"
        else:
            sym = "[FAILED: VERIFICATION]" if v_status == "failed" else "[FAILED: EXECUTION]"

        print(f"  {sym} Step {sr['step_id']}: {sr['tool_name']}")
        if sr.get("output"):
            print(f"       Result:       {sr['output']}")
        if v_status:
            print(f"       Verification: [{v_status.upper()}] {v_msg}")
        if sr.get("error"):
            print(f"       Error:        {sr['error']}")
    if summary.error and summary.status != TaskStatus.COMPLETED:
        print(f"Task halted: {summary.error}")
    print()


def main():
    print("BLINDSPOT Phase 7C -- Verification Demonstrator\n")
    all_passed = True

    # Scenario 1: Read-only multi-step task completes
    print_banner("1. Read-only multi-step task completes")
    mock_io = MockDesktopIO()
    reg = create_default_tool_registry(
        allow_actions=True,
        desktop_io=mock_io,
        include_action_tools=True,
    )
    verifier = TaskVerifier(observer=MockDesktopObserver())
    executor = TaskExecutor(tool_registry=reg, verifier=verifier)
    task1 = create_task(
        user_request="Read context and activity",
        steps_data=[
            {"tool_name": "get_current_context", "args": {}},
            {"tool_name": "get_recent_activity", "args": {}},
        ],
        task_id="s1_task",
    )
    s1 = executor.execute(task1)
    print_summary(s1)
    assert s1.status == TaskStatus.COMPLETED
    assert s1.successful_steps == 2
    assert all(sr["verification"]["status"] == "not_verifiable" for sr in s1.step_results)

    # Scenario 2: Approved desktop action executes and its verification is recorded
    print_banner("2. Approved desktop action executes and verification is recorded")
    observer2 = MockDesktopObserver(window_title="Untitled - Notepad")
    verifier2 = TaskVerifier(observer=observer2)
    reg2 = create_default_tool_registry(
        allow_actions=True,
        permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
        desktop_io=mock_io,
        include_action_tools=True,
    )
    executor2 = TaskExecutor(tool_registry=reg2, verifier=verifier2)
    task2 = create_task(
        user_request="Open notepad",
        steps_data=[{"tool_name": "open_application", "args": {"app_name": "notepad"}}],
        task_id="s2_task",
    )
    s2 = executor2.execute(task2)
    print_summary(s2)
    assert s2.status == TaskStatus.COMPLETED
    assert s2.verified_steps == 1
    assert s2.step_results[0]["verification"]["status"] == "verified"

    # Scenario 3: Verification failure stops subsequent steps
    print_banner("3. Verification failure stops subsequent steps")
    observer3 = MockDesktopObserver(window_title="Calculator")  # Mismatch for notepad!
    verifier3 = TaskVerifier(observer=observer3)
    executor3 = TaskExecutor(tool_registry=reg2, verifier=verifier3)
    task3 = create_task(
        user_request="Open notepad then move mouse",
        steps_data=[
            {"tool_name": "open_application", "args": {"app_name": "notepad"}},
            {"tool_name": "move_mouse", "args": {"x": 100, "y": 200}},
        ],
        task_id="s3_task",
    )
    s3 = executor3.execute(task3)
    print_summary(s3)
    assert s3.status == TaskStatus.FAILED
    assert s3.executed_steps == 1
    assert s3.successful_steps == 0
    assert s3.step_results[0]["verification"]["status"] == "failed"

    # Scenario 4: Permission denial stops execution before verification
    print_banner("4. Permission denial stops execution before verification")
    reg4 = create_default_tool_registry(
        allow_actions=True,
        permission_policy=PermissionPolicy(low_risk_approver=lambda req: False),
        desktop_io=mock_io,
        include_action_tools=True,
    )
    executor4 = TaskExecutor(tool_registry=reg4, verifier=verifier2)
    task4 = create_task(
        user_request="Open notepad then move mouse",
        steps_data=[
            {"tool_name": "open_application", "args": {"app_name": "notepad"}},
            {"tool_name": "move_mouse", "args": {"x": 100, "y": 200}},
        ],
        task_id="s4_task",
    )
    s4 = executor4.execute(task4)
    print_summary(s4)
    assert s4.status == TaskStatus.BLOCKED
    assert s4.executed_steps == 1
    assert s4.step_results[0]["verification"] is None

    # Scenario 5: Execution failure stops execution before verification
    print_banner("5. Execution failure stops execution before verification")
    failing_io = MockDesktopIO(fail_operations=["open_application"])
    reg5 = create_default_tool_registry(
        allow_actions=True,
        permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
        desktop_io=failing_io,
        include_action_tools=True,
    )
    executor5 = TaskExecutor(tool_registry=reg5, verifier=verifier2)
    task5 = create_task(
        user_request="Open notepad then move mouse",
        steps_data=[
            {"tool_name": "open_application", "args": {"app_name": "notepad"}},
            {"tool_name": "move_mouse", "args": {"x": 100, "y": 200}},
        ],
        task_id="s5_task",
    )
    s5 = executor5.execute(task5)
    print_summary(s5)
    assert s5.status == TaskStatus.FAILED
    assert s5.executed_steps == 1
    assert s5.step_results[0]["verification"] is None

    # Scenario 6: Unsupported verification marked not_verifiable, not falsely successful
    print_banner("6. Unsupported verification is marked not_verifiable")
    task6 = create_task(
        user_request="Move mouse pointer",
        steps_data=[{"tool_name": "move_mouse", "args": {"x": 300, "y": 400}}],
        task_id="s6_task",
    )
    s6 = executor2.execute(task6)
    print_summary(s6)
    assert s6.status == TaskStatus.COMPLETED
    assert s6.successful_steps == 1
    assert s6.verified_steps == 0
    assert s6.step_results[0]["verification"]["status"] == "not_verifiable"

    # Scenario 7: Dangerous-tool rejection, validation, and 5-step limit intact
    print_banner("7. Dangerous-tool rejection, validation, and 5-step limit intact")
    task7a = create_task(
        user_request="Execute bash",
        steps_data=[{"tool_name": "bash", "args": {}}],
        task_id="s7a_task",
    )
    valid_a, err_a = validate_task(task7a, reg2)
    print(f"Dangerous tool rejection: Valid={valid_a}, Error='{err_a}'")
    assert not valid_a
    assert "dangerous tool" in err_a

    steps_7b = [{"tool_name": "get_current_context", "args": {}} for _ in range(6)]
    task7b = create_task(user_request="Six steps", steps_data=steps_7b, task_id="s7b_task")
    valid_b, err_b = validate_task(task7b, reg2)
    print(f"5-step limit enforcement: Valid={valid_b}, Error='{err_b}'")
    assert not valid_b
    assert "exceeds maximum allowed limit" in err_b

    print("\nALL 7 PHASE 7C SCENARIOS DETERMINISTICALLY VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
