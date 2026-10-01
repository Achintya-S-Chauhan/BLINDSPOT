"""
BLINDSPOT — Phase 7A: Task Execution Tests.

Covers:
- Task and TaskStep creation and serialization
- Sequential step execution through ToolRegistry
- Successful completion of multi-step tasks
- Early termination on failed step (no further steps run)
- Early termination on blocked/denied step (PermissionPolicy enforcement)
- Maximum-step boundary enforcement (MAX_TASK_STEPS)
- ToolRegistry-only execution (unregistered tools rejected)
- Dangerous tool name blocking in tasks
- CompanionAI.plan_task() integration (both LLM JSON output & deterministic fallback)
"""

import unittest
from unittest.mock import MagicMock
from task import (
    Task,
    TaskStep,
    TaskStatus,
    StepStatus,
    TaskExecutor,
    create_task,
    MAX_TASK_STEPS,
    validate_task,
)
from tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ActionRisk,
    PermissionPolicy,
    ActionRequest,
    BLOCKED_DANGEROUS_TOOL_NAMES,
    validate_tool_args,
)
from desktop_io import MockDesktopIO
from ai import CompanionAI, LLMProvider


class MockStepTool(BaseTool):
    """Simple test tool that records its execution order and arguments."""

    def __init__(self, name: str, execution_log: list, should_fail: bool = False, risk: ActionRisk = ActionRisk.READ_ONLY):
        super().__init__(name=name, description=f"Mock tool {name}", risk=risk)
        self.execution_log = execution_log
        self.should_fail = should_fail

    def execute(self, **kwargs) -> ToolResult:
        self.execution_log.append((self.name, kwargs))
        if self.should_fail:
            return ToolResult(tool_name=self.name, success=False, error=f"{self.name} deliberate failure")
        return ToolResult(tool_name=self.name, success=True, output=f"{self.name} success")


class TestTaskExecutionFoundation(unittest.TestCase):

    def setUp(self):
        self.execution_log = []
        self.registry = ToolRegistry(allow_actions=True)
        self.tool_a = MockStepTool("tool_a", self.execution_log)
        self.tool_b = MockStepTool("tool_b", self.execution_log)
        self.tool_c = MockStepTool("tool_c", self.execution_log)
        self.registry.register(self.tool_a)
        self.registry.register(self.tool_b)
        self.registry.register(self.tool_c)
        self.executor = TaskExecutor(tool_registry=self.registry, max_steps=5)

    def test_01_task_and_step_creation(self):
        """01. Task and TaskStep initialize with proper defaults and serialization."""
        task = create_task(
            user_request="Inspect context and activity",
            steps_data=[
                {"tool_name": "tool_a", "args": {"key": "val1"}, "description": "Step 1"},
                {"tool_name": "tool_b", "args": {}, "description": "Step 2"},
            ],
            task_id="test_task_1",
        )
        self.assertEqual(task.task_id, "test_task_1")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[0].step_id, 1)
        self.assertEqual(task.steps[0].tool_name, "tool_a")
        self.assertEqual(task.steps[0].args, {"key": "val1"})
        self.assertEqual(task.steps[0].status, StepStatus.PENDING)

        d = task.to_dict()
        self.assertEqual(d["task_id"], "test_task_1")
        self.assertEqual(d["status"], "pending")
        self.assertEqual(d["total_steps"], 2)
        self.assertEqual(d["completed_steps_count"], 0)

    def test_02_sequential_execution_success(self):
        """02. Task steps execute in strictly sequential order to completion."""
        task = create_task(
            user_request="Run A then B then C",
            steps_data=[
                {"tool_name": "tool_a", "args": {"order": 1}},
                {"tool_name": "tool_b", "args": {"order": 2}},
                {"tool_name": "tool_c", "args": {"order": 3}},
            ],
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.total_steps, 3)
        self.assertEqual(summary.executed_steps, 3)
        self.assertEqual(summary.successful_steps, 3)
        self.assertIsNone(summary.error)
        self.assertIsNone(summary.failed_step_id)

        # Verify strict order
        expected_log = [
            ("tool_a", {"order": 1}),
            ("tool_b", {"order": 2}),
            ("tool_c", {"order": 3}),
        ]
        self.assertEqual(self.execution_log, expected_log)
        self.assertEqual(len(task.completed_steps), 3)

    def test_03_executor_stops_immediately_on_failed_step(self):
        """03. Execution halts immediately on a failing step; subsequent steps are never called."""
        failing_tool = MockStepTool("failing_tool", self.execution_log, should_fail=True)
        self.registry.register(failing_tool)

        task = create_task(
            user_request="Run A, then fail, then C",
            steps_data=[
                {"tool_name": "tool_a", "args": {"n": 1}},
                {"tool_name": "failing_tool", "args": {"n": 2}},
                {"tool_name": "tool_c", "args": {"n": 3}},
            ],
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.failed_step_id, 2)
        self.assertIn("deliberate failure", summary.error)

        # Tool C must NEVER have been called
        called_tools = [name for name, _ in self.execution_log]
        self.assertEqual(called_tools, ["tool_a", "failing_tool"])
        self.assertNotIn("tool_c", called_tools)

        # Step 3 must still be PENDING
        self.assertEqual(task.steps[2].status, StepStatus.PENDING)

    def test_04_executor_stops_immediately_on_blocked_permission(self):
        """04. If permission is denied for a step, task status is BLOCKED and execution halts."""
        # Policy that denies all actions
        denying_policy = PermissionPolicy(low_risk_approver=lambda req: False)
        registry = ToolRegistry(allow_actions=True, permission_policy=denying_policy)

        action_tool = MockStepTool("action_tool", self.execution_log, risk=ActionRisk.LOW_RISK_ACTION)
        registry.register(action_tool)
        registry.register(self.tool_c)

        executor = TaskExecutor(tool_registry=registry)
        task = create_task(
            user_request="Do action then C",
            steps_data=[
                {"tool_name": "action_tool", "args": {}},
                {"tool_name": "tool_c", "args": {}},
            ],
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.BLOCKED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 0)
        self.assertEqual(summary.failed_step_id, 1)
        self.assertIn("Permission denied", summary.error)

        # Tool C must never execute
        self.assertEqual(len(self.execution_log), 0)
        self.assertEqual(task.steps[1].status, StepStatus.PENDING)

    def test_05_maximum_step_boundary_enforced(self):
        """05. Tasks with more steps than max_steps are rejected before execution."""
        task = create_task(
            user_request="Too many steps",
            steps_data=[{"tool_name": "tool_a"}] * 6,
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 0)
        self.assertIn("exceeds maximum allowed limit", summary.error)
        self.assertEqual(len(self.execution_log), 0)

    def test_06_tool_registry_only_execution_unregistered_rejected(self):
        """06. Steps attempting to invoke unregistered tools fail cleanly and halt task."""
        task = create_task(
            user_request="Run unregistered tool",
            steps_data=[
                {"tool_name": "tool_a"},
                {"tool_name": "nonexistent_secret_tool"},
                {"tool_name": "tool_c"},
            ],
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.failed_step_id, 2)
        self.assertIn("is not registered in ToolRegistry", summary.error)
        self.assertEqual(len(self.execution_log), 1)  # Only tool_a executed

    def test_07_blocked_dangerous_names_in_tasks(self):
        """07. Attempting dangerous tools like 'run_shell' or 'delete_file' is hard blocked."""
        task = create_task(
            user_request="Run dangerous shell command",
            steps_data=[
                {"tool_name": "run_shell", "args": {"cmd": "dir"}},
            ],
        )
        summary = self.executor.execute(task)

        # It fails because run_shell is not registered and categorically blocked
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertIn("not registered", summary.error)
        self.assertEqual(len(self.execution_log), 0)

    def test_08_permission_enforcement_with_approval(self):
        """08. Low-risk action tool with granted permission executes successfully."""
        approving_policy = PermissionPolicy(low_risk_approver=lambda req: True)
        registry = ToolRegistry(allow_actions=True, permission_policy=approving_policy)
        action_tool = MockStepTool("approved_action", self.execution_log, risk=ActionRisk.LOW_RISK_ACTION)
        registry.register(action_tool)

        executor = TaskExecutor(tool_registry=registry)
        task = create_task(
            user_request="Run approved action",
            steps_data=[{"tool_name": "approved_action", "args": {"target": "ok"}}],
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(self.execution_log, [("approved_action", {"target": "ok"})])

    def test_09_companion_ai_plan_task_fallback(self):
        """09. CompanionAI plans standard tasks deterministically via fallback."""
        from tools import create_default_tool_registry
        reg = create_default_tool_registry(
            allow_actions=True,
            desktop_io=MockDesktopIO(),
            include_action_tools=True,
        )

        class DummyOfflineProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return "Not valid JSON"

        ai = CompanionAI(provider=DummyOfflineProvider(), tool_registry=reg)
        task = ai.plan_task("Open Notepad and focus it")

        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[0].tool_name, "open_application")
        self.assertEqual(task.steps[0].args, {"app_name": "notepad"})
        self.assertEqual(task.steps[1].tool_name, "focus_application")
        self.assertEqual(task.steps[1].args, {"window_title_fragment": "Notepad"})

    def test_10_companion_ai_plan_task_llm_json(self):
        """10. CompanionAI parses structured JSON tool steps returned by LLM provider."""
        from tools import create_default_tool_registry
        reg = create_default_tool_registry(
            allow_actions=True,
            desktop_io=MockDesktopIO(),
            include_action_tools=True,
        )

        llm_json = """
        ```json
        [
          {"tool_name": "open_application", "args": {"app_name": "calculator"}, "description": "Open Calc"},
          {"tool_name": "focus_application", "args": {"window_title_fragment": "Calculator"}, "description": "Focus Calc"}
        ]
        ```
        """

        class MockPlanProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return llm_json

        ai = CompanionAI(provider=MockPlanProvider(), tool_registry=reg)
        task = ai.plan_task("Open calculator and focus it")

        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[0].tool_name, "open_application")
        self.assertEqual(task.steps[0].args, {"app_name": "calculator"})
        self.assertEqual(task.steps[1].tool_name, "focus_application")

    def test_11_full_task_execution_with_mock_desktop_io(self):
        """11. End-to-end task planning and execution with MockDesktopIO and approval."""
        from tools import create_default_tool_registry
        mock_io = MockDesktopIO()
        reg = create_default_tool_registry(
            allow_actions=True,
            permission_policy=PermissionPolicy(low_risk_approver=lambda req: True),
            desktop_io=mock_io,
            include_action_tools=True,
        )
        ai = CompanionAI(tool_registry=reg)
        task = ai.plan_task("Open Notepad and focus it")

        executor = TaskExecutor(tool_registry=reg)
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 2)
        called_methods = [c.method for c in mock_io.calls]
        self.assertIn("open_application", called_methods)
        self.assertIn("focus_application", called_methods)


class TestTaskReliabilityPhase7B(unittest.TestCase):
    """
    Focused reliability tests for Phase 7B:
    - Valid plan parsing
    - Markdown JSON handling
    - Malformed JSON handling
    - Unknown tool handling
    - Invalid arguments handling
    - >5 steps handling
    - Duplicate steps handling
    - Pre-execution task validation failure
    - Catches dangerous tools
    - Successful multi-step execution & accurate final summary
    - Early stop on failed step
    - Early stop on blocked step
    - Deterministic execution states
    """

    def setUp(self):
        self.mock_io = MockDesktopIO()
        self.policy = PermissionPolicy(low_risk_approver=lambda req: True)
        from tools import create_default_tool_registry
        self.registry = create_default_tool_registry(
            allow_actions=True,
            permission_policy=self.policy,
            desktop_io=self.mock_io,
            include_action_tools=True,
        )
        self.executor = TaskExecutor(tool_registry=self.registry, pre_validate=True)

    def test_plan_task_valid_parsing(self):
        """01. Valid JSON plan from LLM is correctly parsed into a Task."""
        valid_json = """
        [
            {"tool_name": "get_current_context", "args": {}, "description": "Check current context"},
            {"tool_name": "get_recent_activity", "args": {"limit": 5}, "description": "Get recent activities"}
        ]
        """
        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return valid_json

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("What is my context and recent activity?")

        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[0].tool_name, "get_current_context")
        self.assertEqual(task.steps[1].tool_name, "get_recent_activity")
        self.assertEqual(task.steps[1].args, {"limit": 5})

    def test_plan_task_markdown_wrapped_json(self):
        """02. Markdown-wrapped JSON with explanation is parsed cleanly without error."""
        markdown_json = """
        Here is the recommended task plan for your request:

        ```json
        [
            {"tool_name": "open_application", "args": {"app_name": "notepad"}, "description": "Launch notepad"},
            {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}, "description": "Bring notepad to front"}
        ]
        ```

        Make sure Notepad is installed before executing.
        """
        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return markdown_json

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("Open notepad")

        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[0].tool_name, "open_application")
        self.assertEqual(task.steps[0].args, {"app_name": "notepad"})
        self.assertEqual(task.steps[1].tool_name, "focus_application")

    def test_plan_task_malformed_json_clean_rejection(self):
        """03. Malformed JSON returns cleanly rejected task with FAILED status and no execution."""
        bad_json = "I cannot fulfill this request because [unclosed json array without braces"

        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return bad_json

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("Open something broken")

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("valid json", task.error.lower())
        self.assertEqual(len(task.steps), 0)

        # Executing a pre-failed task returns 0 executed steps
        summary = self.executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 0)

    def test_plan_task_unknown_tool_rejected(self):
        """04. Plans containing unregistered tools are rejected cleanly during planning."""
        plan_with_unknown = """
        [
            {"tool_name": "launch_rocket", "args": {"destination": "mars"}, "description": "Fly away"}
        ]
        """
        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return plan_with_unknown

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("Launch rocket")

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertTrue("unknown tool" in task.error.lower() or "unregistered" in task.error.lower())

    def test_plan_task_invalid_arguments_rejected(self):
        """05. Plans with arguments violating tool constraints are cleanly rejected."""
        # 'malicious_app' is not in ALLOWED_APPLICATIONS
        plan_bad_args = """
        [
            {"tool_name": "open_application", "args": {"app_name": "malicious_trojan.exe"}}
        ]
        """
        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return plan_bad_args

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("Run trojan")

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("invalid arguments", task.error.lower())

    def test_plan_task_more_than_5_steps_rejected(self):
        """06. Plans with >5 steps are rejected during planning."""
        plan_6_steps = """
        [
            {"tool_name": "get_current_context", "args": {}},
            {"tool_name": "get_recent_activity", "args": {"limit": 1}},
            {"tool_name": "get_recent_activity", "args": {"limit": 2}},
            {"tool_name": "get_recent_activity", "args": {"limit": 3}},
            {"tool_name": "get_recent_activity", "args": {"limit": 4}},
            {"tool_name": "get_recent_activity", "args": {"limit": 5}}
        ]
        """
        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return plan_6_steps

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("Do 6 things")

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("maximum limit", task.error.lower())

    def test_plan_task_duplicate_consecutive_steps_rejected(self):
        """07. Consecutive identical steps are detected and rejected as unnecessary duplicates."""
        dup_plan = """
        [
            {"tool_name": "open_application", "args": {"app_name": "notepad"}},
            {"tool_name": "open_application", "args": {"app_name": "notepad"}}
        ]
        """
        class MockProvider(LLMProvider):
            def generate_response(self, system_instruction: str, user_prompt: str, **kwargs) -> str:
                return dup_plan

        ai = CompanionAI(provider=MockProvider(), tool_registry=self.registry)
        task = ai.plan_task("Open notepad twice")

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("duplicate", task.error)

    def test_task_validation_failure_prevents_any_execution(self):
        """08. Pre-execution task validation gate halts invalid tasks with 0 executed steps."""
        task = create_task(
            user_request="Unvalidated task",
            steps_data=[
                {"tool_name": "get_current_context", "args": {}},
                {"tool_name": "nonexistent_fake_tool", "args": {}},
            ],
        )
        is_valid, err = validate_task(task, self.registry)
        self.assertFalse(is_valid)
        self.assertIn("nonexistent_fake_tool", err)

        summary = self.executor.execute(task, pre_validate=True)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.executed_steps, 0)
        self.assertIn("Task validation failed", summary.error)

    def test_validate_task_catches_dangerous_tool(self):
        """09. Dangerous tool names like run_shell or delete_file are caught by validate_task."""
        for dangerous_name in ("run_shell", "delete_file", "write_file", "reboot"):
            task = create_task(
                user_request=f"Try {dangerous_name}",
                steps_data=[{"tool_name": dangerous_name, "args": {}}],
            )
            is_valid, err = validate_task(task, self.registry)
            self.assertFalse(is_valid)
            self.assertIn(dangerous_name, err)
            self.assertIn("dangerous tool", err)

    def test_successful_multi_step_execution_and_summary(self):
        """10. Deterministic execution produces accurate final summary with all metadata."""
        task = create_task(
            user_request="Get context and open notepad",
            steps_data=[
                {"tool_name": "get_current_context", "args": {}, "description": "Context check"},
                {"tool_name": "open_application", "args": {"app_name": "notepad"}, "description": "Launch notepad"},
            ],
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.total_steps, 2)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 2)
        self.assertIsNone(summary.error)
        self.assertIsNone(summary.failed_step_id)
        self.assertGreaterEqual(summary.duration_seconds, 0.0)
        self.assertEqual(len(summary.step_results), 2)
        self.assertEqual(summary.step_results[0]["status"], "completed")
        self.assertEqual(summary.step_results[1]["status"], "completed")

    def test_failed_step_halts_subsequent_execution(self):
        """11. When a step fails, subsequent steps never execute and summary reflects failure."""
        log = []
        mock_tool_ok = MockStepTool("mock_ok", log, should_fail=False)
        mock_tool_fail = MockStepTool("mock_fail", log, should_fail=True)
        mock_tool_unreached = MockStepTool("mock_unreached", log, should_fail=False)

        self.registry.register(mock_tool_ok)
        self.registry.register(mock_tool_fail)
        self.registry.register(mock_tool_unreached)

        task = create_task(
            user_request="Step with failure",
            steps_data=[
                {"tool_name": "mock_ok"},
                {"tool_name": "mock_fail"},
                {"tool_name": "mock_unreached"},
            ],
        )
        summary = self.executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.total_steps, 3)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.failed_step_id, 2)
        self.assertIn("mock_fail deliberate failure", summary.error)

        called = [entry[0] for entry in log]
        self.assertEqual(called, ["mock_ok", "mock_fail"])
        self.assertNotIn("mock_unreached", called)
        self.assertEqual(task.steps[2].status, StepStatus.PENDING)

    def test_blocked_step_halts_and_records_blocked_status(self):
        """12. Denied permission sets task status to BLOCKED and stops immediately."""
        denying_policy = PermissionPolicy(low_risk_approver=lambda req: False)
        from tools import create_default_tool_registry
        reg = create_default_tool_registry(
            allow_actions=True,
            permission_policy=denying_policy,
            desktop_io=self.mock_io,
            include_action_tools=True,
        )
        executor = TaskExecutor(tool_registry=reg, pre_validate=True)

        task = create_task(
            user_request="Open notepad without approval",
            steps_data=[
                {"tool_name": "get_current_context", "args": {}},
                {"tool_name": "open_application", "args": {"app_name": "notepad"}},
                {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}},
            ],
        )
        summary = executor.execute(task)

        self.assertEqual(summary.status, TaskStatus.BLOCKED)
        self.assertEqual(summary.total_steps, 3)
        self.assertEqual(summary.executed_steps, 2)
        self.assertEqual(summary.successful_steps, 1)
        self.assertEqual(summary.failed_step_id, 2)
        self.assertIn("Permission denied", summary.error)
        self.assertEqual(task.steps[2].status, StepStatus.PENDING)

    def test_deterministic_state_transitions(self):
        """13. Task and step state transitions are explicit, sequential, and deterministic."""
        task = create_task(
            user_request="State transition check",
            steps_data=[
                {"tool_name": "get_current_context", "args": {}},
            ],
        )
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.steps[0].status, StepStatus.PENDING)

        summary = self.executor.execute(task)

        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.steps[0].status, StepStatus.COMPLETED)
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.completed_at)
        self.assertGreaterEqual(task.completed_at, task.started_at)
        self.assertIsNotNone(task.steps[0].started_at)
        self.assertIsNotNone(task.steps[0].completed_at)


if __name__ == "__main__":
    unittest.main()

