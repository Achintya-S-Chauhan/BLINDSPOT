"""
BLINDSPOT — Task Recovery & Execution Resilience (Phase 7D).

Provides:
- FailureType: Classification of failures (validation, permission, execution, verification, retry_exhausted).
- RecoveryDecision: Action to take upon failure (retry, abort, none).
- RecoveryRecord: Detailed record of recovery decisions and attempt state for each step.
- RetryPolicy: Bounded, deterministic retry policy strictly restricted to operations proven safe to retry.
"""

from dataclasses import dataclass
import enum
from typing import Optional, Dict, Any, Tuple, Iterable
from tools import BaseTool, ActionRisk


class FailureType(enum.Enum):
    """Classification of failure during task execution."""
    NONE = "none"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_DENIED = "permission_denied"
    EXECUTION_ERROR = "execution_error"
    VERIFICATION_FAILURE = "verification_failure"
    RETRY_EXHAUSTED = "retry_exhausted"


class RecoveryDecision(enum.Enum):
    """Action to take following step execution outcome."""
    NONE = "none"
    RETRY = "retry"
    ABORT = "abort"


@dataclass
class RecoveryRecord:
    """
    Structured outcome of failure classification and recovery decision for a task step.
    """
    failure_type: FailureType = FailureType.NONE
    decision: RecoveryDecision = RecoveryDecision.NONE
    attempt: int = 1
    max_retries: int = 0
    error: Optional[str] = None
    retryable: bool = False
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize recovery record to dictionary representation."""
        return {
            "failure_type": self.failure_type.value,
            "decision": self.decision.value,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "error": self.error,
            "retryable": self.retryable,
            "message": self.message,
        }


# ============================================================
# RETRY SAFETY RULES & POLICY
# ============================================================

DEFAULT_MAX_RETRIES: int = 2
MAX_ALLOWABLE_RETRIES: int = 3  # Hard ceiling to prevent runaway retry chains

# Operations proven safe to retry: read-only queries and idempotent window activation
DEFAULT_SAFE_RETRY_TOOLS: frozenset = frozenset({
    "get_current_context",
    "get_recent_activity",
    "get_conversation_history",
    "focus_application",
})

# Operations that duplicate effects or modify desktop state and must NEVER be retried
FORBIDDEN_RETRY_TOOLS: frozenset = frozenset({
    "click",
    "type_text",
    "press_key",
    "hotkey",
    "move_mouse",
    "open_application",
})


class RetryPolicy:
    """
    Deterministic, bounded retry policy for BLINDSPOT task execution.

    Safety Guarantees:
    1. Only operations proven safe to retry (read-only queries and idempotent focus) are eligible.
    2. Never retries clicks, typing, key presses, hotkeys, or mouse movements.
    3. Never retries permission denials or task validation errors.
    4. Deterministic and strictly bounded by max_retries <= MAX_ALLOWABLE_RETRIES (no infinite loops).
    5. Non-recursive: does not spawn new tasks or subagents.
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        safe_tools: Optional[Iterable[str]] = None,
        enabled: bool = True,
    ):
        bounded = min(max(0, max_retries), MAX_ALLOWABLE_RETRIES)
        self.max_retries = bounded
        self.enabled = enabled
        self.safe_tools = frozenset(safe_tools) if safe_tools is not None else DEFAULT_SAFE_RETRY_TOOLS

    def is_safe_to_retry(
        self,
        tool_name: str,
        tool: Optional[BaseTool] = None,
        failure_type: FailureType = FailureType.NONE,
    ) -> bool:
        """
        Determine if an operation is safe to retry.
        Returns True ONLY for operations proven safe to retry.
        """
        if not self.enabled:
            return False

        # Permission denial and validation failures are NEVER retryable
        if failure_type in (FailureType.PERMISSION_DENIED, FailureType.VALIDATION_ERROR):
            return False

        # Explicitly forbidden tools are categorically blocked from retry
        if tool_name in FORBIDDEN_RETRY_TOOLS:
            return False

        # Only tools explicitly in the safe set are eligible
        if tool_name in self.safe_tools:
            return True

        return False

    def evaluate(
        self,
        tool_name: str,
        attempt: int,
        failure_type: FailureType,
        error: str = "",
        tool: Optional[BaseTool] = None,
    ) -> Tuple[RecoveryDecision, RecoveryRecord]:
        """
        Evaluate whether a failed step should be retried or aborted.
        Returns (RecoveryDecision, RecoveryRecord).
        """
        retryable = self.is_safe_to_retry(tool_name, tool=tool, failure_type=failure_type)

        if not retryable:
            msg = (
                f"Step '{tool_name}' failed ({failure_type.value}); "
                "operation does not support safe retries."
            )
            rec = RecoveryRecord(
                failure_type=failure_type,
                decision=RecoveryDecision.ABORT,
                attempt=attempt,
                max_retries=self.max_retries,
                error=error,
                retryable=False,
                message=msg,
            )
            return RecoveryDecision.ABORT, rec

        if attempt > self.max_retries:
            msg = (
                f"Step '{tool_name}' exhausted all {self.max_retries} allowed retries "
                f"({attempt} attempts executed)."
            )
            rec = RecoveryRecord(
                failure_type=FailureType.RETRY_EXHAUSTED,
                decision=RecoveryDecision.ABORT,
                attempt=attempt,
                max_retries=self.max_retries,
                error=error,
                retryable=True,
                message=msg,
            )
            return RecoveryDecision.ABORT, rec

        msg = (
            f"Step '{tool_name}' failed ({failure_type.value}); "
            f"initiating retry {attempt}/{self.max_retries}."
        )
        rec = RecoveryRecord(
            failure_type=failure_type,
            decision=RecoveryDecision.RETRY,
            attempt=attempt,
            max_retries=self.max_retries,
            error=error,
            retryable=True,
            message=msg,
        )
        return RecoveryDecision.RETRY, rec
