"""
BLINDSPOT — Phase 7A: Task Execution Foundation.

Provides structured task representations, task steps, and sequential execution
through ToolRegistry with strict safety boundaries.
"""

from dataclasses import dataclass, field
import enum
import time
import uuid
from typing import Optional, List, Dict, Any
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


class TaskExecutor:
    """
    Deterministic task execution engine for BLINDSPOT.

    Safety Guarantees:
    1. Steps are executed strictly sequentially.
    2. Tool execution ALWAYS goes through ToolRegistry — no bypass, no arbitrary functions.
    3. Respects PermissionPolicy (auto-approve read-only, explicit CLI approval for actions).
    4. Execution stops IMMEDIATELY on any failed, denied, or blocked step.
    5. Enforces MAX_TASK_STEPS (rejects tasks exceeding limit).
    6. Non-recursive: executor cannot trigger new tasks.
    """

    def __init__(self, tool_registry: ToolRegistry, max_steps: int = MAX_TASK_STEPS):
        self.tool_registry = tool_registry
        self.max_steps = max_steps

    def execute(self, task: Task) -> TaskExecutionSummary:
        """
        Execute the ordered steps of a task sequentially through ToolRegistry.
        Stops immediately on error, failure, or permission denial.
        """
        now = time.time()
        task.started_at = now
        task.status = TaskStatus.RUNNING

        # Safety Check: Maximum step boundary
        if len(task.steps) > self.max_steps:
            err_msg = (
                f"Task rejected: step count ({len(task.steps)}) exceeds maximum allowed "
                f"limit of {self.max_steps} steps."
            )
            task.status = TaskStatus.FAILED
            task.error = err_msg
            task.completed_at = time.time()
            return TaskExecutionSummary(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                user_request=task.user_request,
                total_steps=len(task.steps),
                executed_steps=0,
                successful_steps=0,
                error=err_msg,
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

            # Execute tool through ToolRegistry (which enforces permissions & dangerous tool blocks)
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

    return Task(
        task_id=tid,
        user_request=user_request,
        steps=steps,
        status=TaskStatus.PENDING,
    )
