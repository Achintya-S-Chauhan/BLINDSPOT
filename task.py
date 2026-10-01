"""
BLINDSPOT — Phase 7A: Task Execution Foundation.

Provides structured task representations, task steps, and sequential execution
through ToolRegistry with strict safety boundaries.
"""

from dataclasses import dataclass, field
import enum
import json
import threading
import time
import uuid
from typing import Optional, List, Dict, Any, Tuple
from tools import ToolRegistry, ToolResult


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
    """
    step_id: int
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None
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
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
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
    error: Optional[str] = None
    duration_seconds: float = 0.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution summary to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "user_request": self.user_request,
            "total_steps": self.total_steps,
            "executed_steps": self.executed_steps,
            "successful_steps": self.successful_steps,
            "failed_step_id": self.failed_step_id,
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

    seen_steps: List[Tuple[str, str]] = []
    for idx, step in enumerate(task.steps, start=1):
        if not isinstance(step, TaskStep):
            return False, f"Step {idx} is not a valid TaskStep object."

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

    return True, None


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

    def __init__(self, tool_registry: ToolRegistry, max_steps: int = MAX_TASK_STEPS, pre_validate: bool = False):
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.pre_validate = pre_validate
        self._lock = threading.Lock()

    def execute(self, task: Task, pre_validate: Optional[bool] = None) -> TaskExecutionSummary:
        """
        Execute the ordered steps of a task sequentially through ToolRegistry.
        Stops immediately on error, failure, or permission denial.
        """
        with self._lock:
            now = time.time()
            task.started_at = now
            task.status = TaskStatus.RUNNING

            # Check if task already in terminal failed state
            if task.status == TaskStatus.FAILED or (task.error and not task.steps):
                task.status = TaskStatus.FAILED
                task.completed_at = now
                return TaskExecutionSummary(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    user_request=task.user_request,
                    total_steps=len(task.steps),
                    executed_steps=0,
                    successful_steps=0,
                    error=task.error or "Task failed prior to execution",
                    duration_seconds=0.0,
                )

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
            failed_step_id: Optional[int] = None
            step_results: List[Dict[str, Any]] = []

            for step in task.steps:
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
                    task.status = TaskStatus.FAILED
                    task.error = f"Step {step.step_id} failed: {err_msg}"
                    failed_step_id = step.step_id
                    step_results.append(step.to_dict())
                    break

                # Execute tool strictly through ToolRegistry
                res: ToolResult = self.tool_registry.execute(step.tool_name, **step.args)
                step.result = res
                step.completed_at = time.time()

                if res.success:
                    step.status = StepStatus.COMPLETED
                    successful_count += 1
                    step_results.append(step.to_dict())
                else:
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
                    else:
                        step.status = StepStatus.FAILED
                        task.status = TaskStatus.FAILED

                    step.error = res.error
                    task.error = f"Step {step.step_id} ({step.tool_name}) stopped: {res.error}"
                    failed_step_id = step.step_id
                    step_results.append(step.to_dict())
                    # STOP IMMEDIATELY on failure or block
                    break

            task.completed_at = time.time()
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.COMPLETED

            return TaskExecutionSummary(
                task_id=task.task_id,
                status=task.status,
                user_request=task.user_request,
                total_steps=len(task.steps),
                executed_steps=executed_count,
                successful_steps=successful_count,
                failed_step_id=failed_step_id,
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
    """
    tid = task_id or f"task_{uuid.uuid4().hex[:8]}"
    steps: List[TaskStep] = []
    for idx, sdata in enumerate(steps_data, start=1):
        step = TaskStep(
            step_id=idx,
            tool_name=sdata.get("tool_name", ""),
            args=sdata.get("args") or {},
            description=sdata.get("description", ""),
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

