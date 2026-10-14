"""
BLINDSPOT — Phase 7A–7E: Task Execution Foundation with Dependency-Aware Planning.

Provides structured task representations, task steps, and sequential execution
through ToolRegistry with strict safety boundaries.

Phase 7E additions:
- TaskStep.depends_on: explicit list of prerequisite step IDs.
- validate_task: dependency graph validation (missing refs, self-deps, cycles,
  duplicates) before any tool invocation.
- topological_sort_steps: deterministic step ordering respecting declared deps.
- TaskExecutor: prerequisite enforcement; steps with unsatisfied deps are BLOCKED.
"""

from dataclasses import dataclass, field
import enum
import json
import threading
import time
import uuid
from typing import Optional, List, Dict, Any, Tuple, Set
from tools import ToolRegistry, ToolResult
from verification import StepVerificationResult, VerificationStatus, TaskVerifier
from recovery import FailureType, RecoveryDecision, RecoveryRecord, RetryPolicy


# Maximum allowable steps in a single task to prevent runaway chains
MAX_TASK_STEPS: int = 5


class TaskStatus(enum.Enum):
    """Lifecycle status of a Task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class StepStatus(enum.Enum):
    """Lifecycle status of an individual TaskStep."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskStep:
    """
    Explicit registered tool invocation within a task.
    No hidden execution; every step is inspected and recorded.

    Phase 7E: depends_on lists prerequisite step IDs that must have
    reached StepStatus.COMPLETED before this step may execute.
    An empty list (default) means no prerequisites — backward compatible.
    """
    step_id: int
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    depends_on: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None
    verification: Optional[StepVerificationResult] = None
    recovery: Optional[RecoveryRecord] = None
    attempts: int = 1
    max_retries: int = 0
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task step to dictionary representation."""
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Task:
    """
    Structured representation of an AI-requested or user-requested task.
    """
    task_id: str
    user_request: str
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def completed_steps(self) -> List[TaskStep]:
        """Return list of steps that successfully completed."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    @property
    def failed_step(self) -> Optional[TaskStep]:
        """Return the first failed or blocked step, if any."""
        for s in self.steps:
            if s.status in (StepStatus.FAILED, StepStatus.BLOCKED):
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps),
            "completed_steps_count": len(self.completed_steps),
        }


@dataclass
class TaskExecutionSummary:
    """
    Structured summary returned upon task completion or halt.
    """
    task_id: str
    status: TaskStatus
    user_request: str
    total_steps: int
    executed_steps: int
    successful_steps: int
    failed_step_id: Optional[int] = None
    failure_type: Optional[str] = None
    retry_attempts: int = 0
    skipped_steps: int = 0
    skipped_step_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def verified_steps(self) -> int:
        """Return count of steps that were successfully verified."""
        return sum(
            1 for sr in self.step_results
            if sr.get("verification") and sr["verification"].get("status") == "verified"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution summary to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "user_request": self.user_request,
            "total_steps": self.total_steps,
            "executed_steps": self.executed_steps,
            "successful_steps": self.successful_steps,
            "verified_steps": self.verified_steps,
            "failed_step_id": self.failed_step_id,
            "failure_type": self.failure_type,
            "retry_attempts": self.retry_attempts,
            "skipped_steps": self.skipped_steps,
            "skipped_step_ids": self.skipped_step_ids,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "step_results": self.step_results,
        }


def validate_task(
    task: Task,
    tool_registry: ToolRegistry,
    max_steps: int = MAX_TASK_STEPS,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a task and all its steps prior to execution.
    Returns (True, None) if valid, or (False, error_message).

    Validation rules:
    1. Task must not be None and must have task_id and user_request.
    2. Step count must be between 1 and max_steps.
    3. Each step must have step_id > 0, valid tool_name, and dict args.
    4. Tool name must NOT be in BLOCKED_DANGEROUS_TOOL_NAMES.
    5. Tool must be registered in the provided ToolRegistry.
    6. All required parameters defined in tool's schema must be present in step.args.
    7. Argument types must match basic expected JSON schema types.
    8. No duplicate identical consecutive steps.
    9. (Phase 7E) Dependency graph must be valid:
       - depends_on must be a list of integers.
       - All referenced step IDs must exist in the task.
       - No step may depend on itself.
       - No duplicate entries within depends_on of a single step.
       - No duplicate step IDs across all steps.
       - The dependency graph must be acyclic (no cycles).
    """
    from tools import BLOCKED_DANGEROUS_TOOL_NAMES, validate_tool_args

    if not isinstance(task, Task):
        return False, "Invalid task object: must be an instance of Task."

    if not task.task_id or not isinstance(task.task_id, str):
        return False, "Task missing valid task_id."

    if not isinstance(task.steps, list) or len(task.steps) == 0:
        return False, "Task contains no steps to execute."

    if len(task.steps) > max_steps:
        return False, f"Task step count ({len(task.steps)}) exceeds maximum allowed limit of {max_steps} steps."

    # --- Phase 7E: Duplicate step_id check ---
    seen_step_ids: Set[int] = set()
    for idx, step in enumerate(task.steps, start=1):
        if not isinstance(step, TaskStep):
            return False, f"Step {idx} is not a valid TaskStep object."
        if step.step_id in seen_step_ids:
            return False, f"Duplicate step_id {step.step_id} found (at position {idx})."
        seen_step_ids.add(step.step_id)

    all_step_ids = seen_step_ids

    seen_steps: List[Tuple[str, str]] = []
    for idx, step in enumerate(task.steps, start=1):
        tool_name = step.tool_name
        if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
            return False, f"Step {idx} missing valid tool_name."

        tool_name_clean = tool_name.strip()
        tool_name_lower = tool_name_clean.lower()
        if tool_name_lower in BLOCKED_DANGEROUS_TOOL_NAMES:
            return False, (
                f"Step {idx} requests dangerous tool '{tool_name_clean}' which is categorically "
                "blocked by BLINDSPOT safety policy."
            )

        tool = tool_registry.get(tool_name_clean)
        if not tool:
            return False, f"Step {idx} references unregistered tool '{tool_name_clean}'."

        if not isinstance(step.args, dict):
            return False, f"Step {idx} arguments must be a dictionary, got {type(step.args).__name__}."

        is_valid_args, arg_err = validate_tool_args(tool, step.args)
        if not is_valid_args:
            return False, f"Step {idx} ({tool_name_clean}) invalid arguments: {arg_err}"

        # Duplicate consecutive step check
        step_sig = (tool_name_clean, json.dumps(step.args, sort_keys=True))
        if seen_steps and seen_steps[-1] == step_sig:
            return False, f"Step {idx} ({tool_name_clean}) is an unnecessary duplicate of preceding step."
        seen_steps.append(step_sig)

        deps = step.depends_on
        if not isinstance(deps, list) or isinstance(deps, (str, bytes)):
            return False, (
                f"Step {idx} (id={step.step_id}): 'depends_on' must be a list, "
                f"got {type(deps).__name__}."
            )
        seen_dep_ids: Set[int] = set()
        for dep_id in deps:
            if not isinstance(dep_id, int) or isinstance(dep_id, bool):
                return False, (
                    f"Step {idx} (id={step.step_id}): dependency ID must be an integer, "
                    f"got {type(dep_id).__name__} ({dep_id!r})."
                )
            if dep_id == step.step_id:
                return False, (
                    f"Step {idx} (id={step.step_id}) declares a self-dependency."
                )
            if dep_id not in all_step_ids:
                return False, (
                    f"Step {idx} (id={step.step_id}) depends on step {dep_id} "
                    "which does not exist in this task."
                )
            if dep_id in seen_dep_ids:
                return False, (
                    f"Step {idx} (id={step.step_id}) has duplicate dependency reference {dep_id}."
                )
            seen_dep_ids.add(dep_id)

    # --- Phase 7E: Cycle detection (Kahn's algorithm) ---
    ok, cycle_err = _validate_no_dependency_cycle(task.steps)
    if not ok:
        return False, cycle_err

    return True, None


def _validate_no_dependency_cycle(steps: List["TaskStep"]) -> Tuple[bool, Optional[str]]:
    """
    Detect dependency cycles using Kahn's topological sort algorithm.
    Returns (True, None) if acyclic, or (False, error_message) if a cycle exists.
    """
    step_ids = {s.step_id for s in steps}
    # Build adjacency list: dep_id -> [steps that depend on dep_id]
    # and in-degree map
    in_degree: Dict[int, int] = {s.step_id: 0 for s in steps}
    successors: Dict[int, List[int]] = {s.step_id: [] for s in steps}
    for step in steps:
        for dep_id in step.depends_on:
            in_degree[step.step_id] += 1
            successors[dep_id].append(step.step_id)

    queue = [sid for sid in in_degree if in_degree[sid] == 0]
    queue.sort()  # deterministic
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for succ in sorted(successors[node]):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if visited != len(steps):
        # Find the step IDs involved in the cycle for a useful message
        in_cycle = [sid for sid, deg in in_degree.items() if deg > 0]
        return False, (
            f"Dependency cycle detected among step IDs: {sorted(in_cycle)}. "
            "All task steps must form a directed acyclic graph."
        )
    return True, None


def topological_sort_steps(steps: List["TaskStep"]) -> List["TaskStep"]:
    """
    Return steps in a valid execution order respecting declared dependencies.
    Preserves original list order among steps with equal dependency rank
    (stable topological sort using Kahn's algorithm).

    Precondition: steps form a valid DAG (call validate_task first).
    """
    in_degree: Dict[int, int] = {s.step_id: 0 for s in steps}
    successors: Dict[int, List[int]] = {s.step_id: [] for s in steps}
    id_to_step: Dict[int, "TaskStep"] = {s.step_id: s for s in steps}
    # Build using original list positions for stable ordering
    original_order: Dict[int, int] = {s.step_id: i for i, s in enumerate(steps)}

    for step in steps:
        for dep_id in step.depends_on:
            in_degree[step.step_id] += 1
            successors[dep_id].append(step.step_id)

    # Priority queue ordered by (in_degree==0), then original index for stability
    available = sorted(
        [sid for sid, deg in in_degree.items() if deg == 0],
        key=lambda sid: original_order[sid],
    )
    result: List["TaskStep"] = []
    while available:
        node = available.pop(0)
        result.append(id_to_step[node])
        newly_ready = []
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                newly_ready.append(succ)
        # Insert newly-ready successors in original-order position
        newly_ready.sort(key=lambda sid: original_order[sid])
        available = sorted(
            available + newly_ready,
            key=lambda sid: original_order[sid],
        )
    return result


class TaskExecutor:
    """
    Deterministic task execution engine for BLINDSPOT.

    Safety Guarantees:
    1. Pre-execution task validation: invalid plans never reach tool execution.
    2. Steps are executed strictly sequentially with single-execution locking.
    3. Tool execution ALWAYS goes through ToolRegistry — no bypass, no arbitrary functions.
    4. Respects PermissionPolicy (auto-approve read-only, explicit CLI approval for actions).
    5. Execution stops IMMEDIATELY on any failed, denied, or blocked step.
    6. Enforces MAX_TASK_STEPS (rejects tasks exceeding limit).
    7. Non-recursive: executor cannot trigger new tasks.
    8. Deterministic state transitions:
       PENDING -> RUNNING -> COMPLETED
       PENDING -> RUNNING -> FAILED
       PENDING -> RUNNING -> BLOCKED
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_steps: int = MAX_TASK_STEPS,
        pre_validate: bool = False,
        verifier: Optional[Any] = None,
        retry_policy: Optional[Any] = None,
    ):
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.pre_validate = pre_validate
        if verifier is None:
            self.verifier: Optional[TaskVerifier] = TaskVerifier()
        elif verifier is False:
            self.verifier = None
        else:
            self.verifier = verifier

        if retry_policy is None:
            self.retry_policy: RetryPolicy = RetryPolicy()
        elif retry_policy is False:
            self.retry_policy = RetryPolicy(enabled=False, max_retries=0)
        else:
            self.retry_policy = retry_policy

        self._lock = threading.Lock()

    def execute(self, task: Task, pre_validate: Optional[bool] = None) -> TaskExecutionSummary:
        """
        Execute the ordered steps of a task sequentially through ToolRegistry.
        Stops immediately on error, failure, or permission denial.
        """
        with self._lock:
            now = time.time()

            # Check if task already in terminal failed/blocked state BEFORE overwriting status.
            # This handles tasks that were pre-rejected by plan_task() or validate_task().
            if task.status in (TaskStatus.FAILED, TaskStatus.BLOCKED) or (
                task.error and not task.steps
            ):
                task.completed_at = now
                terminal_status = (
                    task.status
                    if task.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)
                    else TaskStatus.FAILED
                )
                term_failure_type = (
                    FailureType.PERMISSION_DENIED.value
                    if terminal_status == TaskStatus.BLOCKED
                    else FailureType.VALIDATION_ERROR.value
                )
                return TaskExecutionSummary(
                    task_id=task.task_id,
                    status=terminal_status,
                    user_request=task.user_request,
                    total_steps=len(task.steps),
                    executed_steps=0,
                    successful_steps=0,
                    failed_step_id=task.steps[0].step_id if task.steps else None,
                    failure_type=term_failure_type,
                    retry_attempts=0,
                    skipped_steps=len(task.steps),
                    skipped_step_ids=[s.step_id for s in task.steps],
                    error=task.error or "Task failed prior to execution",
                    duration_seconds=0.0,
                )

            task.started_at = now
            task.status = TaskStatus.RUNNING

            # Check step count boundary
            if len(task.steps) > self.max_steps:
                task.status = TaskStatus.FAILED
                task.error = f"Task step count ({len(task.steps)}) exceeds maximum allowed limit of {self.max_steps} steps."
                task.completed_at = now
                return TaskExecutionSummary(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    user_request=task.user_request,
                    total_steps=len(task.steps),
                    executed_steps=0,
                    successful_steps=0,
                    failed_step_id=task.steps[0].step_id if task.steps else None,
                    failure_type=FailureType.VALIDATION_ERROR.value,
                    retry_attempts=0,
                    skipped_steps=len(task.steps),
                    skipped_step_ids=[s.step_id for s in task.steps],
                    error=task.error,
                    duration_seconds=0.0,
                )

            # Pre-execution Validation Gate
            should_validate = self.pre_validate if pre_validate is None else pre_validate
            if should_validate:
                is_valid, val_err = validate_task(task, self.tool_registry, self.max_steps)
                if not is_valid:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task validation failed: {val_err}"
                    task.completed_at = time.time()
                    return TaskExecutionSummary(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        user_request=task.user_request,
                        total_steps=len(task.steps),
                        executed_steps=0,
                        successful_steps=0,
                        failed_step_id=task.steps[0].step_id if task.steps else None,
                        failure_type=FailureType.VALIDATION_ERROR.value,
                        retry_attempts=0,
                        skipped_steps=len(task.steps),
                        skipped_step_ids=[s.step_id for s in task.steps],
                        error=task.error,
                        duration_seconds=task.completed_at - now,
                    )

            if not task.steps:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                return TaskExecutionSummary(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    user_request=task.user_request,
                    total_steps=0,
                    executed_steps=0,
                    successful_steps=0,
                    duration_seconds=task.completed_at - now,
                )

            executed_count = 0
            successful_count = 0
            total_retries = 0
            failed_step_id: Optional[int] = None
            task_failure_type: Optional[str] = None
            step_results: List[Dict[str, Any]] = []

            # Phase 7E: Compute topological execution order.
            # topological_sort_steps requires a valid DAG (pre-validation ensures this).
            ordered_steps = topological_sort_steps(task.steps)

            # Track which step_ids have completed successfully for prerequisite checks.
            completed_ids: Set[int] = set()

            for step in ordered_steps:
                # --- Phase 7E: Prerequisite enforcement ---
                # A step may only run when ALL its prerequisites are COMPLETED.
                # Any other terminal state (FAILED, BLOCKED) halts this step.
                unsatisfied_prereqs = [dep for dep in step.depends_on if dep not in completed_ids]
                if unsatisfied_prereqs:
                    # Check whether any prerequisite reached a non-COMPLETED terminal state.
                    id_to_step_map = {s.step_id: s for s in task.steps}
                    failed_prereqs = [
                        dep for dep in step.depends_on
                        if id_to_step_map.get(dep, None) is not None
                        and id_to_step_map[dep].status in (StepStatus.FAILED, StepStatus.BLOCKED)
                    ]
                    if failed_prereqs:
                        # A prerequisite failed — block this step and halt.
                        step.status = StepStatus.BLOCKED
                        step.completed_at = time.time()
                        step.error = (
                            f"Prerequisite step(s) {failed_prereqs} did not complete successfully; "
                            "this step is blocked."
                        )
                        step.recovery = RecoveryRecord(
                            failure_type=FailureType.VALIDATION_ERROR,
                            decision=RecoveryDecision.ABORT,
                            attempt=0,
                            max_retries=0,
                            error=step.error,
                            retryable=False,
                            message="Step blocked because a prerequisite step failed or was blocked.",
                        )
                        if task.status == TaskStatus.RUNNING:
                            task.status = TaskStatus.FAILED
                            task.error = step.error
                        if failed_step_id is None:
                            failed_step_id = step.step_id
                            task_failure_type = FailureType.VALIDATION_ERROR.value
                        step_results.append(step.to_dict())
                        # Preserve fail-fast: stop executing further steps.
                        break
                    else:
                        # Prerequisites are pending (execution halted upstream — should not happen
                        # after a break, but guard defensively).
                        break

                step.started_at = time.time()
                step.status = StepStatus.RUNNING
                executed_count += 1

                # Safety Check: Tool must be registered in ToolRegistry
                tool = self.tool_registry.get(step.tool_name)
                if not tool:
                    err_msg = f"Tool '{step.tool_name}' is not registered in ToolRegistry."
                    step.status = StepStatus.FAILED
                    step.error = err_msg
                    step.completed_at = time.time()
                    step.attempts = 1
                    step.recovery = RecoveryRecord(
                        failure_type=FailureType.VALIDATION_ERROR,
                        decision=RecoveryDecision.ABORT,
                        attempt=1,
                        max_retries=0,
                        error=err_msg,
                        retryable=False,
                        message="Unregistered tool cannot be executed or retried.",
                    )
                    task.status = TaskStatus.FAILED
                    task.error = f"Step {step.step_id} failed: {err_msg}"
                    failed_step_id = step.step_id
                    task_failure_type = FailureType.VALIDATION_ERROR.value
                    step_results.append(step.to_dict())
                    break

                is_retryable = self.retry_policy.is_safe_to_retry(step.tool_name, tool=tool)
                max_retries = self.retry_policy.max_retries if is_retryable else 0
                step.max_retries = max_retries

                attempt = 0
                step_success = False

                while True:
                    attempt += 1
                    step.attempts = attempt
                    step.verification = None  # Reset verification so previous failed/stale attempts do not persist

                    # Execute tool strictly through ToolRegistry
                    res: ToolResult = self.tool_registry.execute(step.tool_name, **step.args)
                    step.result = res
                    step.completed_at = time.time()

                    if not res.success:
                        # Distinguish permission denial / safety block from general execution errors
                        is_blocked = (
                            res.metadata.get("denied_by") is not None
                            or res.metadata.get("blocked_by") is not None
                            or "permission denied" in (res.error or "").lower()
                            or "safety policy" in (res.error or "").lower()
                            or "safety boundary" in (res.error or "").lower()
                        )
                        if is_blocked:
                            step.status = StepStatus.BLOCKED
                            task.status = TaskStatus.BLOCKED
                            step.error = res.error
                            task.error = f"Step {step.step_id} ({step.tool_name}) stopped: {res.error}"
                            failed_step_id = step.step_id
                            task_failure_type = FailureType.PERMISSION_DENIED.value
                            step.recovery = RecoveryRecord(
                                failure_type=FailureType.PERMISSION_DENIED,
                                decision=RecoveryDecision.ABORT,
                                attempt=attempt,
                                max_retries=max_retries,
                                error=res.error,
                                retryable=False,
                                message="Action blocked by permission policy or safety boundary; retries prohibited.",
                            )
                            step_results.append(step.to_dict())
                            break
                        else:
                            # Tool execution failure - evaluate retry policy
                            decision, rec = self.retry_policy.evaluate(
                                tool_name=step.tool_name,
                                attempt=attempt,
                                failure_type=FailureType.EXECUTION_ERROR,
                                error=res.error or "",
                                tool=tool,
                            )
                            step.recovery = rec
                            if decision == RecoveryDecision.RETRY:
                                total_retries += 1
                                continue
                            else:
                                step.status = StepStatus.FAILED
                                task.status = TaskStatus.FAILED
                                step.error = res.error
                                task.error = f"Step {step.step_id} ({step.tool_name}) stopped: {res.error}"
                                failed_step_id = step.step_id
                                task_failure_type = rec.failure_type.value
                                step_results.append(step.to_dict())
                                break
                    else:
                        # Verification Gate: verify step if verifier is configured
                        if self.verifier:
                            v_res = self.verifier.verify_step(step, tool=tool)
                            step.verification = v_res
                            if v_res.status == VerificationStatus.FAILED:
                                decision, rec = self.retry_policy.evaluate(
                                    tool_name=step.tool_name,
                                    attempt=attempt,
                                    failure_type=FailureType.VERIFICATION_FAILURE,
                                    error=v_res.message or "",
                                    tool=tool,
                                )
                                step.recovery = rec
                                if decision == RecoveryDecision.RETRY:
                                    total_retries += 1
                                    continue
                                else:
                                    step.status = StepStatus.FAILED
                                    task.status = TaskStatus.FAILED
                                    step.error = f"Verification failed: {v_res.message}"
                                    task.error = f"Step {step.step_id} ({step.tool_name}) verification failed: {v_res.message}"
                                    failed_step_id = step.step_id
                                    task_failure_type = rec.failure_type.value
                                    step_results.append(step.to_dict())
                                    break

                        # Step successfully completed and verified
                        step.status = StepStatus.COMPLETED
                        successful_count += 1
                        step_success = True
                        completed_ids.add(step.step_id)  # Phase 7E: track completed
                        step.recovery = RecoveryRecord(
                            failure_type=FailureType.NONE,
                            decision=RecoveryDecision.NONE,
                            attempt=attempt,
                            max_retries=max_retries,
                            error=None,
                            retryable=False,
                            message="Step completed successfully.",
                        )
                        step_results.append(step.to_dict())
                        break

                if not step_success:
                    # Halt task execution immediately; subsequent steps remain PENDING
                    break

            # --- Phase 7E: Post-loop BLOCKED marking ---
            # Any PENDING step whose prerequisite has FAILED or BLOCKED status is
            # explicitly marked BLOCKED to distinguish it from a step skipped by fail-fast.
            # We iterate in topological order so transitive dependencies are marked in one pass.
            id_to_step_post = {s.step_id: s for s in task.steps}
            for step in ordered_steps:
                if step.status != StepStatus.PENDING:
                    continue
                failed_prereqs_post = [
                    dep for dep in step.depends_on
                    if id_to_step_post.get(dep) is not None
                    and id_to_step_post[dep].status in (StepStatus.FAILED, StepStatus.BLOCKED)
                ]
                if failed_prereqs_post:
                    step.status = StepStatus.BLOCKED
                    step.completed_at = time.time()
                    step.error = (
                        f"Prerequisite step(s) {failed_prereqs_post} did not complete successfully; "
                        "this step is blocked."
                    )
                    step.recovery = RecoveryRecord(
                        failure_type=FailureType.VALIDATION_ERROR,
                        decision=RecoveryDecision.ABORT,
                        attempt=0,
                        max_retries=0,
                        error=step.error,
                        retryable=False,
                        message="Step blocked because a prerequisite step failed or was blocked.",
                    )

            task.completed_at = time.time()
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.COMPLETED

            skipped_steps_list = [s for s in task.steps if s.started_at is None]
            skipped_count = len(skipped_steps_list)
            skipped_ids = [s.step_id for s in skipped_steps_list]

            return TaskExecutionSummary(
                task_id=task.task_id,
                status=task.status,
                user_request=task.user_request,
                total_steps=len(task.steps),
                executed_steps=executed_count,
                successful_steps=successful_count,
                failed_step_id=failed_step_id,
                failure_type=task_failure_type,
                retry_attempts=total_retries,
                skipped_steps=skipped_count,
                skipped_step_ids=skipped_ids,
                error=task.error,
                duration_seconds=task.completed_at - now,
                step_results=step_results,
            )


def create_task(
    user_request: str,
    steps_data: List[Dict[str, Any]],
    task_id: Optional[str] = None,
    status: Optional[TaskStatus] = None,
    error: Optional[str] = None,
) -> Task:
    """
    Factory to construct a Task with validated TaskSteps.

    Phase 7E: If a step dict contains 'depends_on', it is preserved on the step
    for strict validation by validate_task(). If omitted or None, it defaults to []
    for backward compatibility with existing plans.
    """
    tid = task_id or f"task_{uuid.uuid4().hex[:8]}"
    steps: List[TaskStep] = []
    for idx, sdata in enumerate(steps_data, start=1):
        # If 'depends_on' omitted, default to empty list (backward compatibility).
        # If present with explicit null, keep as None so validation rejects it.
        if "depends_on" in sdata:
            deps = sdata["depends_on"]  # may be None or malformed; validation will handle
        else:
            deps = []
        step = TaskStep(
            step_id=idx,
            tool_name=sdata.get("tool_name", ""),
            args=sdata.get("args") or {},
            description=sdata.get("description", ""),
            depends_on=deps,
        )
        steps.append(step)

    initial_status = status
    if initial_status is None:
        initial_status = TaskStatus.FAILED if error else TaskStatus.PENDING

    return Task(
        task_id=tid,
        user_request=user_request,
        steps=steps,
        status=initial_status,
        error=error,
    )

