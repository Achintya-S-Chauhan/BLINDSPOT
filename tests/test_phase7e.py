"""
BLINDSPOT - Phase 7E: Dependency-Aware Task Planning & Execution Tests.

Covers:
- Dependency graph validation (missing refs, self-deps, cycles, duplicates)
- Topological ordering and execution prerequisite enforcement
- Failure propagation through dependency chains
- Backward compatibility with dependency-free plans
- AI plan parsing and safety boundaries
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
    validate_task,
    topological_sort_steps,
    _validate_no_dependency_cycle,
    MAX_TASK_STEPS,
)
from tools import (
    ToolRegistry,
    BaseTool,
    ToolResult,
    ActionRisk,
    PermissionPolicy,
    BLOCKED_DANGEROUS_TOOL_NAMES,
)
from recovery import FailureType, RetryPolicy
from verification import TaskVerifier, MockDesktopObserver
from ai import CompanionAI, LLMProvider


class MockStepTool(BaseTool):
    def __init__(self, name, execution_log, should_fail=False, should_block=False, risk=ActionRisk.READ_ONLY):
        super().__init__(name=name, description=f"Mock tool {name}", risk=risk)
        self.execution_log = execution_log
        self.should_fail = should_fail
        self.should_block = should_block

    def execute(self, **kwargs):
        self.execution_log.append(self.name)
        if self.should_block:
            return ToolResult(tool_name=self.name, success=False, error=f"Permission denied for {self.name}", metadata={"denied_by": "policy"})
        if self.should_fail:
            return ToolResult(tool_name=self.name, success=False, error=f"{self.name} deliberate failure")
        return ToolResult(tool_name=self.name, success=True, output=f"{self.name} ok")


def _make_registry(*tools):
    reg = ToolRegistry(allow_actions=True)
    for t in tools:
        reg.register(t)
    return reg


def _make_executor(registry, verifier=False, retry_policy=False):
    return TaskExecutor(tool_registry=registry, verifier=verifier, retry_policy=retry_policy)


class MockLLMProvider(LLMProvider):
    def __init__(self, response="[]"):
        self._response = response

    def generate_response(self, **kwargs):
        return self._response


class TestDependencyValidation(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.tool_a = MockStepTool("tool_a", self.log)
        self.tool_b = MockStepTool("tool_b", self.log)
        self.tool_c = MockStepTool("tool_c", self.log)
        self.tool_d = MockStepTool("tool_d", self.log)
        self.registry = _make_registry(self.tool_a, self.tool_b, self.tool_c, self.tool_d)

    def test_01_valid_linear_chain(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [2]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertTrue(ok, err)

    def test_02_valid_branching_graph(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_d", "args": {}, "depends_on": [2, 3]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertTrue(ok, err)

    def test_03_missing_prerequisite_id(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [99]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("99", err)
        self.assertIn("does not exist", err)

    def test_04_self_dependency(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [2]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("self-dependency", err)

    def test_05_dependency_cycle(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
            {"tool_name": "tool_c", "args": {}},
        ])
        task.steps[0].depends_on = [3]
        task.steps[1].depends_on = [1]
        task.steps[2].depends_on = [2]
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("cycle", err.lower())

    def test_06_duplicate_step_ids(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
        ])
        task.steps[1].step_id = 1
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("Duplicate step_id", err)

    def test_07_duplicate_dependency_references(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1, 1]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("duplicate dependency reference", err.lower())

    def test_08_malformed_dependency_non_integer(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}, "depends_on": ["step_1"]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("integer", err.lower())

    def test_09_five_step_limit_respected(self):
        task = create_task("test", [{"tool_name": "tool_a", "args": {}}] * 6)
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("exceeds maximum", err)

    def test_10_backward_compatibility_no_deps(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
        ])
        self.assertEqual(task.steps[0].depends_on, [])
        self.assertEqual(task.steps[1].depends_on, [])
        ok, err = validate_task(task, self.registry)
        self.assertTrue(ok, err)

    def test_10b_depends_on_non_list_rejected(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": "1"},
            {"tool_name": "tool_b", "args": {}},
        ])
        self.assertEqual(task.steps[0].depends_on, "1")
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("must be a list", err.lower())

    def test_10d_boolean_dependency_rejected(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [True]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("integer", err.lower())

    def test_10e_null_depends_on_rejected(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}, "depends_on": None},
        ])
        self.assertIsNone(task.steps[0].depends_on)
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("must be a list", err.lower())

    def test_10c_serialization_includes_depends_on(self):
        task = create_task("test", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
        ])
        d = task.steps[1].to_dict()
        self.assertIn("depends_on", d)
        self.assertEqual(d["depends_on"], [1])


class TestTopologicalSort(unittest.TestCase):
    def _steps(self, spec):
        return [TaskStep(step_id=sid, tool_name=f"t{sid}", depends_on=deps) for sid, deps in spec]

    def test_topo_no_deps_preserves_order(self):
        steps = self._steps([(1, []), (2, []), (3, [])])
        ordered = topological_sort_steps(steps)
        self.assertEqual([s.step_id for s in ordered], [1, 2, 3])

    def test_topo_linear_chain(self):
        steps = self._steps([(1, []), (2, [1]), (3, [2])])
        ordered = topological_sort_steps(steps)
        self.assertEqual([s.step_id for s in ordered], [1, 2, 3])

    def test_topo_diamond(self):
        steps = self._steps([(1, []), (2, [1]), (3, [1]), (4, [2, 3])])
        ordered = topological_sort_steps(steps)
        ids = [s.step_id for s in ordered]
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 4)
        self.assertIn(2, ids)
        self.assertIn(3, ids)

    def test_topo_reversed_input_reordered(self):
        steps = self._steps([(1, []), (2, [3]), (3, [])])
        ordered = topological_sort_steps(steps)
        ids = [s.step_id for s in ordered]
        self.assertLess(ids.index(3), ids.index(2))


class TestDependencyExecution(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.tool_a = MockStepTool("tool_a", self.log)
        self.tool_b = MockStepTool("tool_b", self.log)
        self.tool_c = MockStepTool("tool_c", self.log)
        self.tool_d = MockStepTool("tool_d", self.log)
        self.registry = _make_registry(self.tool_a, self.tool_b, self.tool_c, self.tool_d)
        self.executor = _make_executor(self.registry)

    def test_11_prerequisites_run_before_dependents(self):
        task = create_task("chain", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [2]},
        ])
        summary = self.executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(self.log, ["tool_a", "tool_b", "tool_c"])

    def test_12_deterministic_execution_ordering(self):
        task = create_task("independent", [
            {"tool_name": "tool_c", "args": {}, "depends_on": []},
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": []},
        ])
        summary = self.executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(self.log, ["tool_c", "tool_a", "tool_b"])

    def test_13_dependent_blocked_when_prerequisite_fails(self):
        failing_a = MockStepTool("tool_a", self.log, should_fail=True)
        reg = _make_registry(failing_a, self.tool_b)
        executor = _make_executor(reg)
        task = create_task("fail-chain", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertNotIn("tool_b", self.log)

    def test_14_dependent_blocked_when_prerequisite_blocked(self):
        blocking_a = MockStepTool("tool_a", self.log, should_block=True, risk=ActionRisk.LOW_RISK_ACTION)
        reg = _make_registry(blocking_a, self.tool_b)
        executor = _make_executor(reg)
        task = create_task("block-chain", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
        ])
        summary = executor.execute(task)
        self.assertIn(summary.status, (TaskStatus.FAILED, TaskStatus.BLOCKED))
        self.assertNotIn("tool_b", self.log)

    def test_15_verification_failure_prevents_dependent_execution(self):
        observer = MockDesktopObserver()
        observer.set_window("SomethingElse")
        verifier = TaskVerifier(observer=observer)
        from desktop_io import MockDesktopIO
        from tools import OpenApplicationTool, FocusApplicationTool
        mock_io = MockDesktopIO()
        mock_io.active_window = None
        open_tool = OpenApplicationTool(desktop_io=mock_io)
        focus_tool = FocusApplicationTool(desktop_io=mock_io)
        reg = ToolRegistry(allow_actions=True)
        reg.register(open_tool)
        reg.register(focus_tool)
        executor = TaskExecutor(tool_registry=reg, verifier=verifier, retry_policy=False)
        task = create_task("verify-chain", [
            {"tool_name": "open_application", "args": {"app_name": "notepad"}, "depends_on": []},
            {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}, "depends_on": [1]},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        for sr in summary.step_results:
            self.assertNotEqual(sr["tool_name"], "focus_application")

    def test_16_retry_exhaustion_prevents_dependent(self):
        class AlwaysFailTool(BaseTool):
            def __init__(self, log):
                super().__init__("get_current_context", "ctx", ActionRisk.READ_ONLY)
                self.log = log
            def execute(self, **kwargs):
                self.log.append("get_current_context")
                return ToolResult(tool_name=self.name, success=False, error="transient failure")

        class AlwaysSucceedTool(BaseTool):
            def __init__(self, log):
                super().__init__("get_recent_activity", "activity", ActionRisk.READ_ONLY)
                self.log = log
            def execute(self, **kwargs):
                self.log.append("get_recent_activity")
                return ToolResult(tool_name=self.name, success=True, output="ok")

        reg = ToolRegistry(allow_actions=True)
        reg.register(AlwaysFailTool(self.log))
        reg.register(AlwaysSucceedTool(self.log))
        retry = RetryPolicy(max_retries=1)
        executor = TaskExecutor(tool_registry=reg, verifier=False, retry_policy=retry)
        task = create_task("retry-chain", [
            {"tool_name": "get_current_context", "args": {"include_ocr": False}, "depends_on": []},
            {"tool_name": "get_recent_activity", "args": {"limit": 1}, "depends_on": [1]},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertNotIn("get_recent_activity", self.log)

    def test_17_completed_steps_not_reexecuted(self):
        task = create_task("no-rerun", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [2]},
        ])
        summary = self.executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(self.log.count("tool_a"), 1)
        self.assertEqual(self.log.count("tool_b"), 1)
        self.assertEqual(self.log.count("tool_c"), 1)

    def test_18_skipped_step_reporting(self):
        failing_a = MockStepTool("tool_a", self.log, should_fail=True)
        reg = _make_registry(failing_a, self.tool_b, self.tool_c)
        executor = _make_executor(reg)
        task = create_task("skip-report", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
            {"tool_name": "tool_c", "args": {}},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(summary.skipped_steps, 2)
        self.assertIn(2, summary.skipped_step_ids)
        self.assertIn(3, summary.skipped_step_ids)

    def test_19_independent_steps_do_not_bypass_fail_fast(self):
        failing_a = MockStepTool("tool_a", self.log, should_fail=True)
        reg = _make_registry(failing_a, self.tool_b, self.tool_c)
        executor = _make_executor(reg)
        task = create_task("fail-fast", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
            {"tool_name": "tool_c", "args": {}},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertNotIn("tool_b", self.log)
        self.assertNotIn("tool_c", self.log)

    def test_20_invalid_graph_zero_tool_calls(self):
        task = create_task("cyclic-test", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
        ])
        task.steps[0].depends_on = [2]
        task.steps[1].depends_on = [1]
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("cycle", err.lower())
        # Execute with task already marked failed
        task2 = create_task("cyclic-exec", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
        ])
        task2.steps[0].depends_on = [2]
        task2.steps[1].depends_on = [1]
        task2.status = TaskStatus.FAILED
        task2.error = "Dependency cycle"
        executor = TaskExecutor(tool_registry=self.registry, verifier=False, retry_policy=False)
        summary = executor.execute(task2)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(self.log, [])

    def test_20b_invalid_graph_pre_validate_blocks_execution(self):
        task = create_task("invalid-pre", [
            {"tool_name": "tool_a", "args": {}},
            {"tool_name": "tool_b", "args": {}},
        ])
        task.steps[0].depends_on = [99]
        executor = TaskExecutor(tool_registry=self.registry, verifier=False, retry_policy=False)
        summary = executor.execute(task, pre_validate=True)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(self.log, [])

    def test_20c_pre_blocked_task_preserves_blocked_state(self):
        task = create_task(
            "blocked-pre",
            [{"tool_name": "tool_a", "args": {}}],
            status=TaskStatus.BLOCKED,
            error="Blocked by policy beforehand",
        )
        executor = TaskExecutor(tool_registry=self.registry, verifier=False, retry_policy=False)
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.BLOCKED)
        self.assertEqual(summary.failure_type, FailureType.PERMISSION_DENIED.value)
        self.assertEqual(summary.executed_steps, 0)
        self.assertEqual(summary.skipped_steps, 1)
        self.assertEqual(self.log, [])


class TestAIDependencyIntegration(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.tool_a = MockStepTool("tool_a", self.log)
        self.tool_b = MockStepTool("tool_b", self.log)
        self.registry = _make_registry(self.tool_a, self.tool_b)

    def _make_ai(self, response):
        return CompanionAI(provider=MockLLMProvider(response), tool_registry=self.registry)

    def test_21_valid_depends_on_parsed(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "description": "Step A", "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "description": "Step B", "depends_on": [1]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertNotEqual(task.status, TaskStatus.FAILED, task.error)
        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[1].depends_on, [1])

    def test_22_invalid_model_dependencies_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [99]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("out-of-range", task.error.lower())

    def test_22b_non_integer_dep_in_model_output_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": ["step_1"]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("non-integer", task.error.lower())

    def test_22c_self_dep_in_model_output_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [2]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("self-dependency", task.error.lower())

    def test_22d_duplicate_dep_in_model_output_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1, 1]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("duplicate dependency reference", task.error.lower())

    def test_22e_non_list_depends_on_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": "1"},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("must be a list", task.error.lower())

    def test_22f_boolean_dep_in_model_output_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [True]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("non-integer", task.error.lower())

    def test_23_existing_plans_without_deps_work(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "description": "Step A"},
            {"tool_name": "tool_b", "args": {}, "description": "Step B"},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertNotEqual(task.status, TaskStatus.FAILED, task.error)
        self.assertEqual(len(task.steps), 2)
        self.assertEqual(task.steps[0].depends_on, [])
        self.assertEqual(task.steps[1].depends_on, [])

    def test_24_dangerous_tools_remain_rejected(self):
        import json
        dangerous_name = next(iter(BLOCKED_DANGEROUS_TOOL_NAMES))
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": dangerous_name, "args": {}, "depends_on": [1]},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("do something dangerous")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("tool", task.error.lower())

    def test_25_permission_denial_not_bypassed_by_deps(self):
        blocking_tool = MockStepTool("tool_a", self.log, should_block=True, risk=ActionRisk.LOW_RISK_ACTION)
        reg = _make_registry(blocking_tool, self.tool_b)
        executor = _make_executor(reg)
        task = create_task("perm-check", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
        ])
        summary = executor.execute(task)
        self.assertIn(summary.status, (TaskStatus.FAILED, TaskStatus.BLOCKED))
        self.assertNotIn("tool_b", self.log)

    def test_26_retry_forbidden_tools_not_retried_in_dep_chain(self):
        from tools import OpenApplicationTool
        from desktop_io import MockDesktopIO
        mock_io = MockDesktopIO()
        mock_io.active_window = "Notepad"
        open_tool = OpenApplicationTool(desktop_io=mock_io)
        reg = ToolRegistry(allow_actions=True)
        reg.register(open_tool)
        retry = RetryPolicy()
        executor = TaskExecutor(tool_registry=reg, verifier=False, retry_policy=retry)
        task = create_task("retry-check", [
            {"tool_name": "open_application", "args": {"app_name": "notepad"}, "depends_on": []},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(task.steps[0].attempts, 1)

    def test_26b_null_depends_on_rejected(self):
        import json
        plan = json.dumps([
            {"tool_name": "tool_a", "args": {}, "depends_on": None},
            {"tool_name": "tool_b", "args": {}, "depends_on": None},
        ])
        ai = self._make_ai(plan)
        task = ai.plan_task("run a then b")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("must be a list", task.error.lower())


class TestDependencyEndToEnd(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.tool_a = MockStepTool("tool_a", self.log)
        self.tool_b = MockStepTool("tool_b", self.log)
        self.tool_c = MockStepTool("tool_c", self.log)
        self.tool_d = MockStepTool("tool_d", self.log)
        self.registry = _make_registry(self.tool_a, self.tool_b, self.tool_c, self.tool_d)
        self.executor = _make_executor(self.registry)

    def test_e2e_diamond_all_succeed(self):
        task = create_task("diamond", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_d", "args": {}, "depends_on": [2, 3]},
        ])
        ok, err = validate_task(task, self.registry)
        self.assertTrue(ok, err)
        summary = self.executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.COMPLETED)
        self.assertEqual(summary.successful_steps, 4)
        self.assertEqual(self.log[0], "tool_a")
        self.assertEqual(self.log[-1], "tool_d")

    def test_e2e_chain_fails_midway_blocks_rest(self):
        failing_b = MockStepTool("tool_b", self.log, should_fail=True)
        reg = _make_registry(self.tool_a, failing_b, self.tool_c)
        executor = _make_executor(reg)
        task = create_task("mid-fail", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [2]},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertIn("tool_a", self.log)
        self.assertIn("tool_b", self.log)
        self.assertNotIn("tool_c", self.log)

    def test_e2e_validate_before_any_execution(self):
        task = create_task("self-dep", [{"tool_name": "tool_a", "args": {}}])
        task.steps[0].depends_on = [1]
        ok, err = validate_task(task, self.registry)
        self.assertFalse(ok)
        self.assertIn("self-dependency", err)

    def test_e2e_blocked_dep_step_marked_blocked(self):
        failing_a = MockStepTool("tool_a", self.log, should_fail=True)
        reg = _make_registry(failing_a, self.tool_b)
        executor = _make_executor(reg)
        task = create_task("blocked-dep-status", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        step_b = task.steps[1]
        self.assertEqual(step_b.status, StepStatus.BLOCKED)
        self.assertIn("Prerequisite", step_b.error)

    def test_27_retry_resets_verification_if_second_attempt_fails_execution(self):
        class FlakyTool(BaseTool):
            def __init__(self):
                super().__init__(name="focus_application", description="flaky tool", risk=ActionRisk.LOW_RISK_ACTION)
                self.calls = 0
            def execute(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    # Attempt 1: succeeds tool, but verification will fail
                    return ToolResult(tool_name=self.name, success=True, output="ok")
                else:
                    # Attempt 2: tool execution itself fails
                    return ToolResult(tool_name=self.name, success=False, error="Tool crashed on retry")

        observer = MockDesktopObserver()
        observer.set_window("UnrelatedWindow")
        verifier = TaskVerifier(observer=observer)
        reg = ToolRegistry(allow_actions=True)
        reg.register(FlakyTool())
        retry = RetryPolicy(max_retries=1)
        executor = TaskExecutor(tool_registry=reg, verifier=verifier, retry_policy=retry)
        task = create_task("flaky-test", [
            {"tool_name": "focus_application", "args": {"window_title_fragment": "Notepad"}, "depends_on": []},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        step = task.steps[0]
        self.assertEqual(step.attempts, 2)
        # On attempt 2, execution failed, so verification must be None (not stale from attempt 1)
        self.assertIsNone(step.verification)

    def test_e2e_transitive_blocked_steps_all_marked(self):
        failing_a = MockStepTool("tool_a", self.log, should_fail=True)
        reg = _make_registry(failing_a, self.tool_b, self.tool_c)
        executor = _make_executor(reg)
        task = create_task("transitive-block", [
            {"tool_name": "tool_a", "args": {}, "depends_on": []},
            {"tool_name": "tool_b", "args": {}, "depends_on": [1]},
            {"tool_name": "tool_c", "args": {}, "depends_on": [2]},
        ])
        summary = executor.execute(task)
        self.assertEqual(summary.status, TaskStatus.FAILED)
        self.assertEqual(task.steps[0].status, StepStatus.FAILED)
        self.assertEqual(task.steps[1].status, StepStatus.BLOCKED)
        self.assertEqual(task.steps[2].status, StepStatus.BLOCKED)
        self.assertEqual(summary.executed_steps, 1)
        self.assertEqual(summary.skipped_steps, 2)
        self.assertEqual(summary.skipped_step_ids, [2, 3])


if __name__ == "__main__":
    unittest.main()
