"""
BLINDSPOT - Phase 8B: Controlled Desktop Execution Integration.

Provides:
- UIStateBridge: Thread-safe data and control bridge connecting UI to backend state,
  perception pipeline, task planning, and controlled task execution.
- BlindspotMainWindow: Native Windows desktop UI using standard library Tkinter/ttk.
  Displays running/stopped state, active application, detected context, and a safe,
  two-stage controlled task execution interface (plan review -> explicit plan approval -> execution).
"""

import sys
import time
import threading
import queue
from typing import Dict, Any, Optional, Callable, List

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from task import (
    Task,
    TaskStep,
    TaskStatus,
    StepStatus,
    TaskExecutionSummary,
    validate_task,
    FailureType,
)
from tools import (
    ActionRisk,
    ActionRequest,
    ToolResult,
    PermissionPolicy,
    _auto_deny,
)
from verification import VerificationStatus
from recovery import RecoveryRecord, FailureType, RecoveryDecision
from understanding import ContextualUnderstandingAnalyzer
from main import format_time


class UIStateBridge:
    """
    Thread-safe bridge between BLINDSPOT backend state, task planning,
    controlled task execution, and companion chat Q&A.
    Decoupled from Tk widgets to allow full headless testing.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        state_lock: threading.Lock,
        companion_ai: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        task_executor: Optional[Any] = None,
        action_approver: Optional[Callable[[ActionRequest], bool]] = None,
        desktop_io: Optional[Any] = None,
        analyzer: Optional[Any] = None,
    ):
        self.state = state
        self.state_lock = state_lock
        self.companion_ai = companion_ai
        self.tool_registry = tool_registry
        self.task_executor = task_executor
        self.action_approver = action_approver
        self.desktop_io = desktop_io or (state.get("desktop_io") if state else None)
        self.analyzer = analyzer if analyzer is not None else ContextualUnderstandingAnalyzer()

        # Phase 8B Task & Execution state
        self.pending_task: Optional[Task] = None
        self.is_plan_approved: bool = False
        self.is_planning: bool = False
        self.is_executing: bool = False
        self.last_execution_summary: Optional[TaskExecutionSummary] = None

        # Phase 9A Companion Chat state
        self.is_asking: bool = False

        if action_approver and self.tool_registry and hasattr(self.tool_registry, "_policy"):
            self.tool_registry._policy.set_approver(ActionRisk.LOW_RISK_ACTION, action_approver)

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Ask a natural-language question about current desktop context.
        Reuses CompanionAI.ask() and the exact context wiring from the CLI.
        Guards against duplicate submissions.
        """
        if self.is_asking:
            return {
                "success": False,
                "error": "A conversational query is already in progress. Please wait.",
            }

        q_clean = query.strip()
        if not q_clean:
            return {
                "success": False,
                "error": "Question cannot be empty.",
            }

        if self.companion_ai is None:
            return {
                "success": False,
                "error": "Companion AI is not configured. Q&A unavailable.",
            }

        self.is_asking = True
        try:
            now = time.time()
            with self.state_lock:
                cur_ctx = self.state.get("current_context")
                start_time = self.state.get("context_start_time")
                hist = self.state.get("history")
                if self.analyzer is not None:
                    understanding = self.analyzer.analyze(
                        current_context=cur_ctx,
                        context_start_time=start_time,
                        history=hist,
                        now=now,
                    )
                else:
                    understanding = None

            answer = self.companion_ai.ask(
                current_context=cur_ctx,
                context_start_time=start_time,
                history=hist,
                understanding=understanding,
                user_query=q_clean,
                now=now,
            )
            return {
                "success": True,
                "query": q_clean,
                "answer": answer,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"AI Q&A error: {str(e)}",
            }
        finally:
            self.is_asking = False

    def clear_conversation(self) -> None:
        """Reset the active conversation session history."""
        if self.companion_ai is not None and hasattr(self.companion_ai, "clear_conversation"):
            self.companion_ai.clear_conversation()

    def get_conversation_turns(self) -> List[Dict[str, Any]]:
        """Return serialized list of recorded conversation turns."""
        if self.companion_ai is not None and hasattr(self.companion_ai, "conversation"):
            conv = self.companion_ai.conversation
            if conv is not None and hasattr(conv, "get_history_dicts"):
                return conv.get_history_dicts()
        return []

    def set_action_approver(self, approver: Callable[[ActionRequest], bool]) -> None:
        """Register a runtime callback for LOW_RISK_ACTION approval."""
        self.action_approver = approver
        if self.tool_registry and hasattr(self.tool_registry, "_policy"):
            self.tool_registry._policy.set_approver(ActionRisk.LOW_RISK_ACTION, approver)

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Safely fetch a consistent snapshot of the current desktop perception state."""
        now = time.time()
        with self.state_lock:
            is_running = self.state.get("is_running", True)
            active_window = self.state.get("last_window") or "No active window detected"
            active_context = self.state.get("last_context") or "No active context"
            confidence = self.state.get("last_confidence", 0)
            context_start_time = self.state.get("context_start_time")
            last_ocr_time = self.state.get("last_ocr_time")

            duration = 0.0
            if context_start_time is not None:
                duration = max(0.0, now - context_start_time)

        return {
            "is_running": is_running,
            "active_window": active_window,
            "active_context": active_context,
            "confidence": confidence,
            "context_start_time": context_start_time,
            "duration_seconds": duration,
            "duration_formatted": format_time(duration),
            "last_ocr_time": last_ocr_time,
        }

    def set_running(self, running: bool) -> None:
        """Set the perception monitor running state."""
        with self.state_lock:
            self.state["is_running"] = bool(running)

    def toggle_running(self) -> bool:
        """Toggle the running state and return the new value."""
        with self.state_lock:
            new_val = not self.state.get("is_running", True)
            self.state["is_running"] = new_val
            return new_val

    def plan_task(self, task_request: str) -> Dict[str, Any]:
        """
        Formulate and validate a task plan from user input.
        Phase 8B: Stores validated plan as pending_task awaiting explicit user approval.
        """
        if self.is_executing:
            return {
                "success": False,
                "error": "Task execution is currently in progress. Cannot formulate a new plan.",
            }

        req_clean = task_request.strip()
        if not req_clean:
            return {
                "success": False,
                "error": "Task request cannot be empty.",
            }

        if self.companion_ai is None:
            return {
                "success": False,
                "error": "Companion AI is not configured. Planning unavailable.",
            }

        self.is_planning = True
        # Invalidate any prior plan or approval
        self.invalidate_plan()

        try:
            task = self.companion_ai.plan_task(req_clean)
        except Exception as e:
            self.is_planning = False
            return {
                "success": False,
                "error": f"AI planning error: {str(e)}",
            }
        finally:
            self.is_planning = False

        if task.status == TaskStatus.FAILED or not task.steps:
            self.pending_task = None
            self.is_plan_approved = False
            return {
                "success": False,
                "error": task.error or "Plan rejected: No valid executable plan could be formulated.",
            }

        # Validate task steps against registered tools and safety boundaries
        if self.tool_registry is not None:
            is_valid, val_err = validate_task(task, self.tool_registry)
            if not is_valid:
                self.pending_task = None
                self.is_plan_approved = False
                return {
                    "success": False,
                    "error": f"Task validation rejected: {val_err}",
                }

        self.pending_task = task
        self.is_plan_approved = False

        steps_data = []
        for s in task.steps:
            steps_data.append({
                "step_id": s.step_id,
                "tool_name": s.tool_name,
                "args": s.args,
                "description": s.description,
                "depends_on": s.depends_on,
            })

        return {
            "success": True,
            "task_id": task.task_id,
            "user_request": task.user_request,
            "total_steps": len(task.steps),
            "steps": steps_data,
            "plan_status": "validated_awaiting_approval",
            "is_approved": False,
            "message": f"Formulated and validated {len(task.steps)} step(s) successfully.",
            "execution_enabled": False,  # Execution is disabled until explicit user approval
            "execution_notice": (
                "[PHASE 8A SAFETY CONTROL / PHASE 8B CONTROLLED EXECUTION]\n"
                "Plan validated successfully. Desktop execution is held until you explicitly click 'Approve Plan'.\n"
                "Individual desktop actions will enforce runtime permission policies."
            ),
        }

    def approve_plan(self) -> Dict[str, Any]:
        """Explicitly approve the currently pending plan for execution."""
        if self.is_executing:
            return {
                "success": False,
                "error": "Cannot approve plan while execution is in progress.",
            }

        if self.pending_task is None:
            return {
                "success": False,
                "error": "No valid plan available to approve. Please formulate a plan first.",
            }

        self.is_plan_approved = True
        return {
            "success": True,
            "task_id": self.pending_task.task_id,
            "message": "Plan explicitly approved by user. Ready for controlled execution.",
        }

    def invalidate_plan(self) -> None:
        """Invalidate the pending plan and any user approval."""
        self.pending_task = None
        self.is_plan_approved = False

    def execute_approved_task(self) -> Dict[str, Any]:
        """
        Execute the approved task plan using TaskExecutor.
        Enforces:
        - Only approved plans can run.
        - Single execution lock to prevent duplicate submissions or concurrent runs.
        - Sequential step execution through ToolRegistry with PermissionPolicy enforcement.
        - Verification and bounded recovery.
        """
        if self.is_executing:
            return {
                "success": False,
                "error": "Task execution already in progress. Duplicate execution prevented.",
            }

        if self.pending_task is None:
            return {
                "success": False,
                "error": "No plan available for execution. Please formulate a plan first.",
            }

        if not self.is_plan_approved:
            return {
                "success": False,
                "error": "Plan has not been approved. Explicit user approval is required before execution.",
            }

        if self.task_executor is None:
            return {
                "success": False,
                "error": "Task executor is not configured. Execution unavailable.",
            }

        self.is_executing = True
        task = self.pending_task

        try:
            summary = self.task_executor.execute(task)
            self.last_execution_summary = summary
        finally:
            # Plan is consumed; require new formulation and approval for any subsequent execution
            self.pending_task = None
            self.is_plan_approved = False
            self.is_executing = False

        # Gather detailed per-step execution, verification, and recovery outcomes
        steps_info = []
        for s in task.steps:
            steps_info.append({
                "step_id": s.step_id,
                "tool_name": s.tool_name,
                "args": s.args,
                "description": s.description,
                "depends_on": s.depends_on,
                "status": s.status.value,
                "attempts": s.attempts,
                "verification": s.verification.to_dict() if s.verification else None,
                "recovery": s.recovery.to_dict() if s.recovery else None,
                "result": s.result.to_dict() if s.result else None,
                "error": s.error,
            })

        return {
            "success": summary.status == TaskStatus.COMPLETED,
            "task_id": summary.task_id,
            "status": summary.status.value,
            "user_request": summary.user_request,
            "total_steps": summary.total_steps,
            "executed_steps": summary.executed_steps,
            "successful_steps": summary.successful_steps,
            "failed_step_id": summary.failed_step_id,
            "failure_type": summary.failure_type,
            "retry_attempts": summary.retry_attempts,
            "skipped_steps": summary.skipped_steps,
            "skipped_step_ids": summary.skipped_step_ids,
            "error": summary.error,
            "duration_seconds": summary.duration_seconds,
            "steps": steps_info,
        }


class BlindspotMainWindow:
    """
    Main desktop window for BLINDSPOT.
    Provides running status, active context display, and controlled task execution area.
    """

    BG_DARK = "#1a1b26"
    CARD_BG = "#24283b"
    ACCENT_BLUE = "#7aa2f7"
    COLOR_RUNNING = "#9ece6a"   # Green
    COLOR_STOPPED = "#f7768e"   # Red
    COLOR_WARNING = "#e0af68"   # Yellow
    FG_MAIN = "#c0caf5"
    FG_MUTED = "#7982a9"
    BORDER_COLOR = "#414868"

    def __init__(
        self,
        root: tk.Tk,
        bridge: UIStateBridge,
        poll_interval_ms: int = 300,
        action_approver: Optional[Callable[[ActionRequest], bool]] = None,
    ):
        self.root = root
        self.bridge = bridge
        self.poll_interval_ms = poll_interval_ms
        self._poll_job: Optional[str] = None
        self._is_destroyed = False
        self._active_worker: Optional[threading.Thread] = None
        self._active_approval_event: Optional[threading.Event] = None
        self._approval_lock = threading.Lock()
        self._on_approval_waiting: Optional[Callable[[], None]] = None
        self._last_plan_res: Optional[Dict[str, Any]] = None
        self._is_chat_busy: bool = False
        self._active_chat_worker: Optional[threading.Thread] = None
        self._ui_queue: queue.Queue = queue.Queue()

        self.desktop_io = getattr(bridge, "desktop_io", None)
        if self.desktop_io is None and bridge.state:
            self.desktop_io = bridge.state.get("desktop_io")
        from desktop_io import get_backend_label
        self.backend_label = get_backend_label(self.desktop_io)

        self._init_window()
        self._setup_styles()
        self._build_widgets()

        # Connect UI action approver if policy is default auto-deny and no custom approver passed
        if action_approver is not None:
            self.bridge.set_action_approver(action_approver)
        elif self.bridge.tool_registry and hasattr(self.bridge.tool_registry, "_policy"):
            current_approver = self.bridge.tool_registry._policy._approvers.get(ActionRisk.LOW_RISK_ACTION)
            if current_approver is _auto_deny:
                self.bridge.set_action_approver(self._ui_action_approver)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Trigger initial view update and start safe polling
        self._refresh_view()
        self._schedule_poll()

    def _init_window(self) -> None:
        self.root.title("BLINDSPOT — Desktop Assistant")
        self.root.geometry("780x700")
        self.root.minsize(680, 560)
        self.root.configure(bg=self.BG_DARK)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            if style.theme_use() != "clam":
                style.theme_use("clam")
        except Exception:
            pass

        style.configure("Root.TFrame", background=self.BG_DARK)
        style.configure("Card.TFrame", background=self.CARD_BG, relief="solid", borderwidth=1)

        style.configure(
            "Title.TLabel",
            background=self.BG_DARK,
            foreground=self.ACCENT_BLUE,
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG_DARK,
            foreground=self.FG_MUTED,
            font=("Segoe UI", 9),
        )

        style.configure(
            "CardHeader.TLabel",
            background=self.CARD_BG,
            foreground=self.ACCENT_BLUE,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=self.CARD_BG,
            foreground=self.FG_MUTED,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "FieldValue.TLabel",
            background=self.CARD_BG,
            foreground=self.FG_MAIN,
            font=("Segoe UI", 9),
        )

        style.configure(
            "StatusRunning.TLabel",
            background=self.BG_DARK,
            foreground=self.COLOR_RUNNING,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "StatusStopped.TLabel",
            background=self.BG_DARK,
            foreground=self.COLOR_STOPPED,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "BackendBadge.TLabel",
            background=self.BG_DARK,
            foreground="#89b4fa" if self.backend_label == "REAL WINDOWS I/O" else "#f9e2af",
            font=("Segoe UI", 9, "bold"),
        )

        style.configure(
            "Footer.TLabel",
            background=self.BG_DARK,
            foreground=self.FG_MUTED,
            font=("Segoe UI", 8),
        )
        style.configure(
            "Error.TLabel",
            background=self.BG_DARK,
            foreground=self.COLOR_STOPPED,
            font=("Segoe UI", 9, "bold"),
        )

        # Tabbed notebook styling (Phase 9A)
        style.configure(
            "TNotebook",
            background=self.BG_DARK,
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background=self.CARD_BG,
            foreground=self.FG_MUTED,
            padding=[14, 6],
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.CARD_BG), ("active", "#2d334a")],
            foreground=[("selected", self.ACCENT_BLUE), ("active", self.FG_MAIN)],
        )

    def _build_widgets(self) -> None:
        main_container = ttk.Frame(self.root, style="Root.TFrame", padding=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        # ---------------- HEADER SECTION ----------------
        header_frame = ttk.Frame(main_container, style="Root.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_box = ttk.Frame(header_frame, style="Root.TFrame")
        title_box.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(title_box, text="BLINDSPOT", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            title_box,
            text="Context-Aware Desktop AI Assistant (Controlled Execution)",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W)

        ctrl_box = ttk.Frame(header_frame, style="Root.TFrame")
        ctrl_box.pack(side=tk.RIGHT, fill=tk.Y)

        self.backend_badge = ttk.Label(
            ctrl_box,
            text=f"[{self.backend_label}]",
            style="BackendBadge.TLabel",
        )
        self.backend_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.status_badge = ttk.Label(ctrl_box, text="● RUNNING", style="StatusRunning.TLabel")
        self.status_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.toggle_btn = ttk.Button(
            ctrl_box,
            text="Pause Perception",
            command=self._on_toggle_clicked,
        )
        self.toggle_btn.pack(side=tk.LEFT)

        # ---------------- CONTEXT CARD ----------------
        context_card = ttk.Frame(main_container, style="Card.TFrame", padding=12)
        context_card.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(context_card, text="ACTIVE DESKTOP CONTEXT", style="CardHeader.TLabel").pack(
            anchor=tk.W, pady=(0, 8)
        )

        grid_frame = ttk.Frame(context_card, style="Card.TFrame")
        grid_frame.pack(fill=tk.X)

        # Row 0: Active Window
        ttk.Label(grid_frame, text="Active Window:", style="FieldLabel.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2
        )
        self.val_window = ttk.Label(grid_frame, text="Detecting...", style="FieldValue.TLabel")
        self.val_window.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Row 1: Current Activity
        ttk.Label(grid_frame, text="Current Activity:", style="FieldLabel.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=2
        )
        self.val_activity = ttk.Label(grid_frame, text="Detecting...", style="FieldValue.TLabel")
        self.val_activity.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Row 2: Confidence
        ttk.Label(grid_frame, text="Confidence:", style="FieldLabel.TLabel").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=2
        )
        self.val_confidence = ttk.Label(grid_frame, text="--", style="FieldValue.TLabel")
        self.val_confidence.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Row 3: Duration
        ttk.Label(grid_frame, text="Context Duration:", style="FieldLabel.TLabel").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 10), pady=2
        )
        self.val_duration = ttk.Label(grid_frame, text="0s", style="FieldValue.TLabel")
        self.val_duration.grid(row=3, column=1, sticky=tk.W, pady=2)

        # ---------------- TABBED INTERACTIVE INTERFACE (PHASE 9A) ----------------
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Tab 1: Task Planning & Execution
        task_tab = ttk.Frame(self.notebook, style="Root.TFrame")
        self.notebook.add(task_tab, text="Task Planning & Execution")

        task_card = ttk.Frame(task_tab, style="Card.TFrame", padding=12)
        task_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(task_card, text="TASK PLANNING & CONTROLLED EXECUTION", style="CardHeader.TLabel").pack(
            anchor=tk.W, pady=(0, 2)
        )
        ttk.Label(
            task_card,
            text="Formulate, inspect, explicitly approve, and execute tasks with desktop verification.",
            style="FieldValue.TLabel",
        ).pack(anchor=tk.W, pady=(0, 8))

        # Input row
        input_row = ttk.Frame(task_card, style="Card.TFrame")
        input_row.pack(fill=tk.X, pady=(0, 8))

        self.task_entry = ttk.Entry(input_row)
        self.task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.task_entry.bind("<Return>", lambda e: self._on_plan_clicked())
        self.task_entry.bind("<KeyRelease>", self._on_input_changed)

        self.plan_btn = ttk.Button(input_row, text="Plan Task", command=self._on_plan_clicked)
        self.plan_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.approve_btn = ttk.Button(
            input_row,
            text="Approve Plan",
            command=self._on_approve_clicked,
            state="disabled",
        )
        self.approve_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.execute_btn = ttk.Button(
            input_row,
            text="Execute Task",
            command=self._on_execute_clicked,
            state="disabled",
        )
        self.execute_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.clear_btn = ttk.Button(input_row, text="Clear", command=self._on_clear_clicked)
        self.clear_btn.pack(side=tk.LEFT)

        # Plan output text area
        self.plan_output = scrolledtext.ScrolledText(
            task_card,
            height=12,
            bg="#181825",
            fg="#c0caf5",
            insertbackground="#c0caf5",
            font=("Consolas", 9),
            relief="flat",
            borderwidth=1,
        )
        self.plan_output.pack(fill=tk.BOTH, expand=True)
        self.plan_output.insert(
            tk.END,
            "[PLAN DRAFT]\n"
            "Enter a task request above (e.g. 'Open notepad and focus it') and click 'Plan Task'.\n"
            "Execution requires two-stage explicit confirmation: Plan Task -> Approve Plan -> Execute Task.\n"
        )
        self.plan_output.configure(state="disabled")

        # Tab 2: Companion Chat & Desktop Q&A (Phase 9A)
        chat_tab = ttk.Frame(self.notebook, style="Root.TFrame")
        self.notebook.add(chat_tab, text="Companion Chat & Q&A")

        chat_card = ttk.Frame(chat_tab, style="Card.TFrame", padding=12)
        chat_card.pack(fill=tk.BOTH, expand=True)

        chat_header = ttk.Frame(chat_card, style="Card.TFrame")
        chat_header.pack(fill=tk.X, pady=(0, 2))

        ttk.Label(chat_header, text="COMPANION CHAT & DESKTOP Q&A", style="CardHeader.TLabel").pack(
            side=tk.LEFT, anchor=tk.W
        )
        self.clear_chat_btn = ttk.Button(
            chat_header,
            text="Clear Chat",
            command=self._on_clear_chat_clicked,
        )
        self.clear_chat_btn.pack(side=tk.RIGHT)

        ttk.Label(
            chat_card,
            text="Ask natural-language questions about your current desktop context, active applications, or recent activity.",
            style="FieldValue.TLabel",
        ).pack(anchor=tk.W, pady=(0, 8))

        # Chat output text area
        self.chat_output = scrolledtext.ScrolledText(
            chat_card,
            height=12,
            bg="#181825",
            fg="#c0caf5",
            insertbackground="#c0caf5",
            font=("Consolas", 9),
            relief="flat",
            borderwidth=1,
            wrap=tk.WORD,
        )
        self.chat_output.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Text tags for distinct role styling
        self.chat_output.tag_config("user", foreground="#7aa2f7", font=("Consolas", 9, "bold"))
        self.chat_output.tag_config("assistant", foreground="#9ece6a", font=("Consolas", 9))
        self.chat_output.tag_config("system", foreground="#bb9af7", font=("Consolas", 9, "italic"))
        self.chat_output.tag_config("error", foreground="#f7768e", font=("Consolas", 9, "bold"))
        self.chat_output.tag_config("time", foreground="#7982a9", font=("Consolas", 8))

        # Initial informational prompt
        self._append_chat_message(
            "SYSTEM",
            "Welcome to BLINDSPOT Companion Chat!\nAsk questions about your current desktop context (e.g. 'What am I working on?', 'What was I doing recently?').",
            tag="system",
        )

        # Chat input row
        chat_input_row = ttk.Frame(chat_card, style="Card.TFrame")
        chat_input_row.pack(fill=tk.X)

        self.chat_entry = ttk.Entry(chat_input_row)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.chat_entry.bind("<Return>", self._on_chat_return)

        self.send_chat_btn = ttk.Button(
            chat_input_row,
            text="Send",
            command=self._on_send_chat_clicked,
        )
        self.send_chat_btn.pack(side=tk.LEFT)

        # ---------------- FOOTER / STATUS BAR ----------------
        footer_frame = ttk.Frame(main_container, style="Root.TFrame")
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.footer_label = ttk.Label(
            footer_frame,
            text="BLINDSPOT UI online. Background perception active.",
            style="Footer.TLabel",
        )
        self.footer_label.pack(side=tk.LEFT)

        self.footer_backend_label = ttk.Label(
            footer_frame,
            text=f"Backend: {self.backend_label}",
            style="Footer.TLabel",
        )
        self.footer_backend_label.pack(side=tk.RIGHT)

    def _ui_action_approver(self, request: ActionRequest) -> bool:
        """
        Thread-safe callback to request user approval for LOW_RISK_ACTION.
        Called by ToolRegistry on a background execution worker thread.
        Prompts user on the main UI thread via Tk messagebox.
        """
        with self._approval_lock:
            if self._is_destroyed:
                return False
            approved = [False]
            evt = threading.Event()
            self._active_approval_event = evt

        def _prompt():
            with self._approval_lock:
                if self._is_destroyed:
                    approved[0] = False
                    evt.set()
                    return
            try:
                if not self.root or not self.root.winfo_exists():
                    approved[0] = False
                    evt.set()
                    return
                args_str = ", ".join(f"{k}={v!r}" for k, v in request.args.items()) if request.args else "none"
                msg = (
                    f"BLINDSPOT Action Permission Request\n\n"
                    f"Tool:        {request.tool_name}\n"
                    f"Risk level:  {request.risk.value}\n"
                    f"Description: {request.description}\n"
                    f"Arguments:   {args_str}\n\n"
                    f"Do you grant permission to execute this desktop action?"
                )
                res = messagebox.askyesno(
                    title="BLINDSPOT — Action Permission Request",
                    message=msg,
                    parent=self.root,
                )
                approved[0] = bool(res)
            except Exception:
                approved[0] = False
            finally:
                evt.set()

        try:
            with self._approval_lock:
                if self._is_destroyed:
                    self._active_approval_event = None
                    return False
            self.root.after(0, _prompt)
        except Exception:
            with self._approval_lock:
                if self._active_approval_event is evt:
                    self._active_approval_event = None
            return False

        # Signal that the approver is actively waiting on the approval event
        if self._on_approval_waiting is not None:
            try:
                self._on_approval_waiting()
            except Exception:
                pass

        while not evt.is_set():
            with self._approval_lock:
                if self._is_destroyed:
                    approved[0] = False
                    break
            evt.wait(timeout=0.05)

        with self._approval_lock:
            if self._active_approval_event is evt:
                self._active_approval_event = None

        return approved[0]

    def _refresh_view(self) -> None:
        """Fetch latest snapshot from bridge and update UI elements."""
        try:
            snapshot = self.bridge.get_status_snapshot()
            self.update_view(snapshot)
        except Exception as e:
            self.set_status_error(f"Error updating perception view: {str(e)}")

    def update_view(self, snapshot: Dict[str, Any]) -> None:
        """Update widget labels with values from a status snapshot."""
        if self._is_destroyed:
            return

        is_running = snapshot.get("is_running", True)
        if is_running:
            self.status_badge.configure(text="● RUNNING", style="StatusRunning.TLabel")
            self.toggle_btn.configure(text="Pause Perception")
        else:
            self.status_badge.configure(text="○ STOPPED", style="StatusStopped.TLabel")
            self.toggle_btn.configure(text="Resume Perception")

        self.val_window.configure(text=str(snapshot.get("active_window", "None")))
        self.val_activity.configure(text=str(snapshot.get("active_context", "None")))
        conf = snapshot.get("confidence", 0)
        self.val_confidence.configure(text=f"{conf}/3" if conf else "--")
        self.val_duration.configure(text=str(snapshot.get("duration_formatted", "0s")))

    def set_status_message(self, message: str) -> None:
        if not self._is_destroyed:
            self.footer_label.configure(text=message, style="Footer.TLabel")

    def set_status_error(self, error_message: str) -> None:
        if not self._is_destroyed:
            self.footer_label.configure(text=f"Error: {error_message}", style="Error.TLabel")

    def _safe_ui_dispatch(self, callback: Callable[[], None]) -> None:
        """Safely schedule a callback on the UI thread."""
        with self._approval_lock:
            if self._is_destroyed:
                return
            self._ui_queue.put(callback)
            if self.root and threading.current_thread() == threading.main_thread():
                try:
                    self.root.after(0, self._process_ui_queue)
                except Exception:
                    pass

    def _process_ui_queue(self) -> None:
        """Process any pending UI callbacks on the Tkinter main thread."""
        if self._is_destroyed:
            return
        while not self._ui_queue.empty():
            try:
                cb = self._ui_queue.get_nowait()
                cb()
            except queue.Empty:
                break
            except Exception:
                pass

    def _schedule_poll(self) -> None:
        """Schedule next background poll."""
        if not self._is_destroyed:
            self._poll_job = self.root.after(self.poll_interval_ms, self._on_poll)

    def _on_poll(self) -> None:
        if self._is_destroyed:
            return
        self._process_ui_queue()
        self._refresh_view()
        self._schedule_poll()

    def _on_toggle_clicked(self) -> None:
        """Handle start/stop button click."""
        try:
            new_state = self.bridge.toggle_running()
            if new_state:
                self.set_status_message("Perception monitoring resumed.")
            else:
                self.set_status_message("Perception monitoring paused.")
            self._refresh_view()
        except Exception as e:
            self.set_status_error(f"Failed to toggle monitoring: {str(e)}")

    def _on_input_changed(self, event=None) -> None:
        """Invalidate plan and approval whenever input is changed."""
        if self.bridge.pending_task is not None or self.bridge.is_plan_approved:
            current_req = self.bridge.pending_task.user_request if self.bridge.pending_task else ""
            input_now = self.task_entry.get().strip()
            if input_now != current_req:
                self.bridge.invalidate_plan()
                self.approve_btn.configure(state="disabled")
                self.execute_btn.configure(state="disabled")
                self._last_plan_res = None
                self.set_status_message("Task input modified. Previous plan approval invalidated.")

    def _on_clear_clicked(self) -> None:
        self.task_entry.delete(0, tk.END)
        self.bridge.invalidate_plan()
        self._last_plan_res = None
        self.approve_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self.plan_btn.configure(state="normal")
        self._set_output_text(
            "Type a task request above (e.g. 'Open notepad and focus it') and click 'Plan Task'.\n"
            "Execution requires two-stage explicit confirmation: Plan Task -> Approve Plan -> Execute Task.\n"
        )
        self.set_status_message("Input cleared.")

    def _on_plan_clicked(self) -> None:
        """Handle Plan Task button click safely on a worker thread."""
        query = self.task_entry.get().strip()
        if not query:
            self.set_status_error("Please enter a task request.")
            self._set_output_text(
                "[TASK REQUEST EMPTY]\n"
                "Please enter a task request in the field above (e.g. 'open notepad').\n"
            )
            return

        self.plan_btn.configure(state="disabled")
        self.approve_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self.set_status_message(f"Formulating plan for: '{query}'...")
        self._set_output_text(
            f"[PLANNING TASK]\n"
            f"Request: {query}\n\n"
            "Analyzing available tools and formulating plan...\n"
        )

        def _worker():
            try:
                res = self.bridge.plan_task(query)
            except Exception as e:
                res = {"success": False, "error": f"Unexpected planning error: {str(e)}"}

            self._safe_ui_dispatch(lambda: self._handle_plan_result(res))

        self._active_worker = threading.Thread(target=_worker, daemon=True)
        self._active_worker.start()

    def _handle_plan_result(self, res: Dict[str, Any]) -> None:
        if self._is_destroyed:
            return

        self.plan_btn.configure(state="normal")
        if not res.get("success"):
            err = res.get("error", "Unknown planning error.")
            status_line = err if len(err) <= 80 else err[:77] + "..."
            self.set_status_error(status_line)
            self.approve_btn.configure(state="disabled")
            self.execute_btn.configure(state="disabled")
            self._last_plan_res = None

            # Detailed, helpful rejection display in plan preview area
            req_text = self.task_entry.get().strip()
            lines = [
                "[PLAN REJECTED]",
                "=" * 60,
                f"Task Request: {req_text or 'None'}",
                f"Reason:       {err}",
                "=" * 60,
                "\nGuidance:",
                "- Verify that the task uses registered BLINDSPOT tools.",
                "- For opening applications, only allowlisted applications are supported:",
                "  notepad, calculator, paint, wordpad, explorer.",
                "- Dangerous system commands (cmd, powershell, bash, regedit) are blocked.",
            ]
            self._set_output_text("\n".join(lines))
            return

        self._last_plan_res = res
        self.approve_btn.configure(state="normal")
        self.execute_btn.configure(state="disabled")

        # Format plan review display
        lines = [
            "[VALIDATED PLAN — AWAITING USER APPROVAL]",
            "=" * 60,
            f"Task ID:     {res.get('task_id')}",
            f"Request:     {res.get('user_request')}",
            f"Total Steps: {res.get('total_steps')}",
            f"Status:      VALIDATED (Awaiting Explicit Plan Approval)",
            "=" * 60,
            "",
        ]
        for s in res.get("steps", []):
            deps = f" [requires: {s['depends_on']}]" if s.get("depends_on") else ""
            args_str = ", ".join(f"{k}={v!r}" for k, v in s.get("args", {}).items()) if s.get("args") else "none"
            desc = f" -- {s['description']}" if s.get("description") else ""
            lines.append(f"  Step {s['step_id']}: {s['tool_name']}({args_str}){desc}{deps}")

        lines.append("")
        lines.append("=" * 60)
        if res.get("execution_notice"):
            lines.append(res.get("execution_notice"))
        else:
            lines.append("SAFETY NOTICE: Explicit approval required before desktop execution.")
            lines.append("Click 'Approve Plan' to authorize execution.")
            lines.append("Modifying the request above will immediately invalidate this approval.")
        lines.append("=" * 60)

        self._set_output_text("\n".join(lines))
        self.set_status_message(f"Plan formulated successfully ({res.get('total_steps')} steps). Awaiting explicit approval.")

    def _on_approve_clicked(self) -> None:
        """Handle user clicking 'Approve Plan'."""
        res = self.bridge.approve_plan()
        if not res.get("success"):
            self.set_status_error(res.get("error", "Could not approve plan."))
            return

        self.approve_btn.configure(state="disabled")
        self.execute_btn.configure(state="normal")

        # Re-render output area with approved status
        if self._last_plan_res:
            lines = [
                "[PLAN APPROVED — READY FOR EXECUTION]",
                "=" * 60,
                f"Task ID:     {self._last_plan_res.get('task_id')}",
                f"Request:     {self._last_plan_res.get('user_request')}",
                f"Total Steps: {self._last_plan_res.get('total_steps')}",
                f"Status:      APPROVED BY USER (Ready for Execution)",
                "=" * 60,
                "",
            ]
            for s in self._last_plan_res.get("steps", []):
                deps = f" [requires: {s['depends_on']}]" if s.get("depends_on") else ""
                args_str = ", ".join(f"{k}={v!r}" for k, v in s.get("args", {}).items()) if s.get("args") else "none"
                desc = f" -- {s['description']}" if s.get("description") else ""
                lines.append(f"  Step {s['step_id']}: {s['tool_name']}({args_str}){desc}{deps}")

            lines.append("")
            lines.append("=" * 60)
            lines.append("CONTROLLED EXECUTION READY:")
            lines.append("Click 'Execute Task' to begin execution via TaskExecutor.")
            lines.append("Low-risk actions will request runtime confirmation where required.")
            lines.append("=" * 60)

            self._set_output_text("\n".join(lines))

        self.set_status_message("Plan approved by user. Ready for controlled execution.")

    def _on_execute_clicked(self) -> None:
        """Handle user clicking 'Execute Task'."""
        if not self.bridge.is_plan_approved:
            self.set_status_error("Cannot execute: plan has not been approved.")
            return

        self.plan_btn.configure(state="disabled")
        self.approve_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.task_entry.configure(state="disabled")

        self.set_status_message("Executing approved task through TaskExecutor...")
        self._set_output_text(
            "[EXECUTING TASK]\n"
            "=" * 60 + "\n"
            "Task execution initiated in background worker.\n"
            "Executing ordered steps through TaskExecutor...\n"
            "Monitoring tool execution, permissions, retries, and desktop verification.\n"
            "=" * 60 + "\n"
        )

        def _worker():
            try:
                res = self.bridge.execute_approved_task()
            except Exception as e:
                res = {"success": False, "error": f"Unexpected execution error: {str(e)}"}

            self._safe_ui_dispatch(lambda: self._handle_execution_result(res))

        self._active_worker = threading.Thread(target=_worker, daemon=True)
        self._active_worker.start()

    def _handle_execution_result(self, res: Dict[str, Any]) -> None:
        """Render comprehensive execution outcomes, verification results, and recovery records."""
        if self._is_destroyed:
            return

        self.task_entry.configure(state="normal")
        self.plan_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.approve_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self._last_plan_res = None

        status = res.get("status", "unknown").upper()
        is_success = res.get("success", False)

        # Distinguish verified success from completed unverified or failures
        has_verified_steps = any(
            (s.get("verification") or {}).get("status") == "verified"
            for s in res.get("steps", [])
        )
        has_failed_verification = any(
            (s.get("verification") or {}).get("status") == "failed"
            for s in res.get("steps", [])
        )

        if is_success:
            if has_verified_steps:
                banner = "[TASK EXECUTION COMPLETED — ALL STEPS VERIFIED]"
            else:
                banner = "[TASK EXECUTION COMPLETED — STEPS EXECUTED (UNVERIFIED)]"
        elif status == "BLOCKED":
            banner = "[TASK EXECUTION BLOCKED — PERMISSION DENIED OR SAFETY BOUNDARY]"
        elif has_failed_verification or res.get("failure_type") == "verification_failure":
            banner = "[TASK EXECUTION FAILED — POST-EXECUTION VERIFICATION FAILED]"
        else:
            banner = f"[TASK EXECUTION FAILED — {res.get('failure_type', 'EXECUTION ERROR').upper()}]"

        duration_sec = res.get("duration_seconds", 0.0)
        lines = [
            banner,
            "=" * 70,
            f"Task ID:          {res.get('task_id')}",
            f"Request:          {res.get('user_request')}",
            f"Terminal Status:  {status}",
            f"Progress:         {res.get('successful_steps')}/{res.get('total_steps')} steps successful "
            f"({res.get('executed_steps')} executed, {res.get('retry_attempts', 0)} retried, {res.get('skipped_steps', 0)} skipped)",
            f"Duration:         {duration_sec:.2f}s",
        ]
        if res.get("error"):
            lines.append(f"Error / Halt:     {res.get('error')}")
        lines.append("=" * 70)
        lines.append("\nSTEP-BY-STEP EXECUTION AND VERIFICATION LOG:")

        for s in res.get("steps", []):
            step_id = s.get("step_id")
            tool_name = s.get("tool_name")
            step_status = (s.get("status") or "unknown").upper()
            attempts = s.get("attempts", 0)
            args = s.get("args", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else "none"
            deps = f" [requires: {s['depends_on']}]" if s.get("depends_on") else ""

            lines.append(f"\n  • Step {step_id}: {tool_name}({args_str}){deps}")
            lines.append(f"    Status:       {step_status} (Attempts: {attempts})")

            # Verification detail
            v_info = s.get("verification")
            if v_info:
                v_stat = (v_info.get("status") or "").upper()
                v_msg = v_info.get("message") or "No verification message."
                lines.append(f"    Verification: [{v_stat}] {v_msg}")
            else:
                lines.append("    Verification: [NOT VERIFIED]")

            # Recovery / Retry detail
            r_info = s.get("recovery")
            if r_info and r_info.get("failure_type") != "none":
                r_dec = (r_info.get("decision") or "").upper()
                r_msg = r_info.get("message") or ""
                lines.append(f"    Recovery:     [{r_dec}] {r_msg}")

            # Error details
            if s.get("error"):
                lines.append(f"    Error:        {s.get('error')}")

        lines.append("\n" + "=" * 70)
        lines.append("TASK EXECUTION FINISHED.")
        lines.append("To run another task, enter a request above and click 'Plan Task'.")
        lines.append("=" * 70)

        self._set_output_text("\n".join(lines))
        if is_success:
            self.set_status_message(f"Task completed successfully in {duration_sec:.2f}s.")
        elif status == "BLOCKED":
            self.set_status_error("Task stopped: action permission denied.")
        else:
            self.set_status_error(f"Task execution failed: {res.get('error', 'Execution error.')}")

    def _set_output_text(self, text: str) -> None:
        if self._is_destroyed:
            return
        self.plan_output.configure(state="normal")
        self.plan_output.delete("1.0", tk.END)
        self.plan_output.insert(tk.END, text)
        self.plan_output.configure(state="disabled")

    # ---------------- COMPANION CHAT METHODS (PHASE 9A) ----------------

    def _append_chat_message(self, sender: str, text: str, tag: str = "assistant") -> None:
        """Safely append a message to the conversation display."""
        if self._is_destroyed:
            return
        self.chat_output.configure(state="normal")
        timestamp_str = time.strftime("[%H:%M:%S] ")
        self.chat_output.insert(tk.END, timestamp_str, "time")
        sender_tag = tag if tag in ("user", "assistant", "system", "error") else ("user" if sender.lower() == "user" else "assistant")
        self.chat_output.insert(tk.END, f"{sender}: ", sender_tag)
        self.chat_output.insert(tk.END, f"{text}\n\n", tag)
        self.chat_output.see(tk.END)
        self.chat_output.configure(state="disabled")

    def _on_send_chat_clicked(self) -> None:
        """Handle sending chat query to Companion AI."""
        if self._is_chat_busy:
            return

        query = self.chat_entry.get().strip()
        if not query:
            return

        self._is_chat_busy = True
        self.chat_entry.delete(0, tk.END)
        self.chat_entry.configure(state="disabled")
        self.send_chat_btn.configure(state="disabled")
        self.clear_chat_btn.configure(state="disabled")

        self._append_chat_message("User", query, tag="user")
        self.set_status_message("BLINDSPOT is thinking...")

        def _worker():
            import gc
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
            try:
                res = self.bridge.ask(query)
            except Exception as e:
                res = {"success": False, "error": f"Unexpected chat error: {str(e)}"}
            finally:
                if gc_was_enabled:
                    gc.enable()

            self._safe_ui_dispatch(lambda: self._handle_chat_response(res))

        self._active_chat_worker = threading.Thread(target=_worker, daemon=True)
        self._active_chat_worker.start()

    def _on_chat_return(self, event=None) -> None:
        """Handle Return key press in chat entry."""
        self._on_send_chat_clicked()

    def _handle_chat_response(self, res: Dict[str, Any]) -> None:
        """Process chat response on UI thread and restore controls."""
        if self._is_destroyed:
            return

        try:
            if res.get("success"):
                answer = res.get("answer", "")
                if "[Local Fallback" in answer or "[Gemini Quota Exceeded]" in answer:
                    self._append_chat_message("BLINDSPOT (Local Fallback)", answer, tag="assistant")
                    self.set_status_message("Notice: Gemini quota exceeded. Local fallback active.")
                else:
                    self._append_chat_message("BLINDSPOT", answer, tag="assistant")
                    self.set_status_message("BLINDSPOT replied.")
            else:
                err = res.get("error", "Unknown error")
                err_lower = err.lower()
                if "429" in err_lower or "quota" in err_lower or "resource_exhausted" in err_lower:
                    clean_err = "Gemini API daily quota or rate limit exceeded (HTTP 429)."
                else:
                    clean_err = err
                self._append_chat_message("ERROR", clean_err, tag="error")
                self.set_status_error(f"Chat error: {clean_err}")
        finally:
            self._is_chat_busy = False
            try:
                self.chat_entry.configure(state="normal")
                self.send_chat_btn.configure(state="normal")
                self.clear_chat_btn.configure(state="normal")
                self.chat_entry.focus()
            except Exception:
                pass

    def _on_clear_chat_clicked(self) -> None:
        """Clear conversation session and chat display."""
        if self._is_chat_busy:
            return
        self.bridge.clear_conversation()
        if not self._is_destroyed:
            self.chat_output.configure(state="normal")
            self.chat_output.delete("1.0", tk.END)
            self.chat_output.configure(state="disabled")
            self._append_chat_message(
                "SYSTEM",
                "Conversation history cleared.",
                tag="system",
            )
            self.set_status_message("Conversation history cleared.")

    def on_close(self) -> None:
        """Clean shutdown handler."""
        with self._approval_lock:
            if self._is_destroyed:
                return
            self._is_destroyed = True
            if self._active_approval_event is not None:
                self._active_approval_event.set()
                self._active_approval_event = None

        self._is_chat_busy = False

        if self._poll_job is not None and self.root:
            try:
                self.root.after_cancel(self._poll_job)
            except Exception:
                pass
            self._poll_job = None

        # Join any active background worker threads before tearing down Tkinter
        if self._active_chat_worker is not None and self._active_chat_worker.is_alive():
            try:
                self._active_chat_worker.join(timeout=3.0)
            except Exception:
                pass

        if self._active_worker is not None and self._active_worker.is_alive():
            try:
                self._active_worker.join(timeout=4.0)
            except Exception:
                pass

        try:
            self.root.update_idletasks()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass


def run_ui(
    state: Dict[str, Any],
    state_lock: threading.Lock,
    companion_ai: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
    task_executor: Optional[Any] = None,
    action_approver: Optional[Callable[[ActionRequest], bool]] = None,
    desktop_io: Optional[Any] = None,
    analyzer: Optional[Any] = None,
) -> None:
    """Launch the BLINDSPOT desktop window."""
    bridge = UIStateBridge(
        state=state,
        state_lock=state_lock,
        companion_ai=companion_ai,
        tool_registry=tool_registry,
        task_executor=task_executor,
        action_approver=action_approver,
        desktop_io=desktop_io,
        analyzer=analyzer,
    )
    root = tk.Tk()
    BlindspotMainWindow(root=root, bridge=bridge, action_approver=action_approver)
    root.mainloop()


def main():
    """Standalone entry point for launching BLINDSPOT with UI."""
    from main import initialize_system, monitor

    use_mock_io = "--mock-io" in sys.argv
    components = initialize_system(mock_io=use_mock_io)
    state = components["state"]
    state_lock = components["state_lock"]
    stop_event = components["stop_event"]
    desktop_io = components.get("desktop_io")

    # Start background perception monitor thread
    monitor_thread = threading.Thread(
        target=monitor,
        args=(state, state_lock, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    try:
        run_ui(
            state=state,
            state_lock=state_lock,
            companion_ai=components["companion_ai"],
            tool_registry=components["tool_registry"],
            task_executor=components["task_executor"],
            desktop_io=desktop_io,
            analyzer=components.get("analyzer"),
        )
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
