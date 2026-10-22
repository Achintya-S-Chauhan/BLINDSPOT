"""
BLINDSPOT - Phase 8A: Desktop UI Foundation.

Provides:
- UIStateBridge: Thread-safe data and control bridge connecting UI to backend state,
  perception pipeline, and task planning.
- BlindspotMainWindow: Native Windows desktop UI using standard library Tkinter/ttk.
  Displays running/stopped state, active application, detected context, and a safe
  task planning area (with real desktop execution strictly disabled for Phase 8A).
"""

import sys
import time
import threading
from typing import Dict, Any, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext

from task import validate_task, TaskStatus
from main import format_time


class UIStateBridge:
    """
    Thread-safe bridge between BLINDSPOT backend state and the UI.
    Decoupled from Tk widgets to allow full headless testing.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        state_lock: threading.Lock,
        companion_ai: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        task_executor: Optional[Any] = None,
    ):
        self.state = state
        self.state_lock = state_lock
        self.companion_ai = companion_ai
        self.tool_registry = tool_registry
        self.task_executor = task_executor

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
        Phase 8A Safety Guarantee: Task is planned and validated, but NEVER executed.
        """
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

        try:
            task = self.companion_ai.plan_task(req_clean)
        except Exception as e:
            return {
                "success": False,
                "error": f"AI planning error: {str(e)}",
            }

        if task.status == TaskStatus.FAILED or not task.steps:
            return {
                "success": False,
                "error": task.error or "Plan rejected: No valid executable plan could be formulated.",
            }

        # Validate task steps against registered tools and safety boundaries
        if self.tool_registry is not None:
            is_valid, val_err = validate_task(task, self.tool_registry)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Task validation rejected: {val_err}",
                }

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
            "message": f"Formulated {len(task.steps)} step(s) successfully.",
            "execution_enabled": False,
            "execution_notice": (
                "[PHASE 8A SAFETY CONTROL]\n"
                "Desktop action execution from the UI is disabled in Phase 8A.\n"
                "The task above has been safely planned and validated without touching hardware."
            ),
        }


class BlindspotMainWindow:
    """
    Main desktop window for BLINDSPOT.
    Provides running status, active context display, and task input area.
    """

    BG_DARK = "#1a1b26"
    CARD_BG = "#24283b"
    ACCENT_BLUE = "#7aa2f7"
    COLOR_RUNNING = "#9ece6a"  # Green
    COLOR_STOPPED = "#f7768e"  # Red
    FG_MAIN = "#c0caf5"
    FG_MUTED = "#7982a9"
    BORDER_COLOR = "#414868"

    def __init__(
        self,
        root: tk.Tk,
        bridge: UIStateBridge,
        poll_interval_ms: int = 300,
    ):
        self.root = root
        self.bridge = bridge
        self.poll_interval_ms = poll_interval_ms
        self._poll_job: Optional[str] = None
        self._is_destroyed = False

        self._init_window()
        self._setup_styles()
        self._build_widgets()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Trigger initial view update and start safe polling
        self._refresh_view()
        self._schedule_poll()

    def _init_window(self) -> None:
        self.root.title("BLINDSPOT — Desktop Assistant")
        self.root.geometry("740x620")
        self.root.minsize(640, 500)
        self.root.configure(bg=self.BG_DARK)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
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
            text="Context-Aware Desktop AI Assistant (Phase 8A UI Foundation)",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W)

        ctrl_box = ttk.Frame(header_frame, style="Root.TFrame")
        ctrl_box.pack(side=tk.RIGHT, fill=tk.Y)

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

        # ---------------- TASK PLANNING CARD (PHASE 8A) ----------------
        task_card = ttk.Frame(main_container, style="Card.TFrame", padding=12)
        task_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(task_card, text="TASK PLANNING (SAFE PREVIEW)", style="CardHeader.TLabel").pack(
            anchor=tk.W, pady=(0, 2)
        )
        ttk.Label(
            task_card,
            text="Formulate and inspect multi-step plans. Real execution is disabled for safety.",
            style="FieldValue.TLabel",
        ).pack(anchor=tk.W, pady=(0, 8))

        # Input row
        input_row = ttk.Frame(task_card, style="Card.TFrame")
        input_row.pack(fill=tk.X, pady=(0, 8))

        self.task_entry = ttk.Entry(input_row)
        self.task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.task_entry.bind("<Return>", lambda e: self._on_plan_clicked())

        self.plan_btn = ttk.Button(input_row, text="Plan Task", command=self._on_plan_clicked)
        self.plan_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.clear_btn = ttk.Button(input_row, text="Clear", command=self._on_clear_clicked)
        self.clear_btn.pack(side=tk.LEFT)

        # Plan output text area
        self.plan_output = scrolledtext.ScrolledText(
            task_card,
            height=10,
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
            "Type a task request above (e.g. 'Open notepad and focus it') and click 'Plan Task'.\n"
            "Plans will be validated with dependency ordering, but desktop execution is held in Phase 8A.\n"
        )
        self.plan_output.configure(state="disabled")

        # ---------------- FOOTER / STATUS BAR ----------------
        footer_frame = ttk.Frame(main_container, style="Root.TFrame")
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.footer_label = ttk.Label(
            footer_frame,
            text="BLINDSPOT UI online. Background perception active.",
            style="Footer.TLabel",
        )
        self.footer_label.pack(side=tk.LEFT)

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

    def _schedule_poll(self) -> None:
        """Schedule next background poll."""
        if not self._is_destroyed:
            self._poll_job = self.root.after(self.poll_interval_ms, self._on_poll)

    def _on_poll(self) -> None:
        if self._is_destroyed:
            return
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

    def _on_clear_clicked(self) -> None:
        self.task_entry.delete(0, tk.END)
        self._set_output_text(
            "Type a task request above (e.g. 'Open notepad and focus it') and click 'Plan Task'.\n"
            "Plans will be validated with dependency ordering, but desktop execution is held in Phase 8A.\n"
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

            # Safely schedule update on UI thread using after(0, ...)
            if not self._is_destroyed:
                try:
                    self.root.after(0, lambda: self._handle_plan_result(res))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def _handle_plan_result(self, res: Dict[str, Any]) -> None:
        if self._is_destroyed:
            return

        self.plan_btn.configure(state="normal")
        if not res.get("success"):
            err = res.get("error", "Unknown planning error.")
            status_line = err if len(err) <= 80 else err[:77] + "..."
            self.set_status_error(status_line)

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

        # Format plan output
        lines = []
        lines.append(f"Task ID: {res.get('task_id')} | Request: {res.get('user_request')}")
        lines.append(f"Total Steps: {res.get('total_steps')}\n")
        for s in res.get("steps", []):
            deps = f" [requires: {s['depends_on']}]" if s.get("depends_on") else ""
            args_str = ", ".join(f"{k}={v!r}" for k, v in s.get("args", {}).items()) if s.get("args") else "none"
            desc = f" -- {s['description']}" if s.get("description") else ""
            lines.append(f"  Step {s['step_id']}: {s['tool_name']}({args_str}){desc}{deps}")

        lines.append("\n" + "=" * 55)
        lines.append(res.get("execution_notice", ""))
        lines.append("=" * 55)

        self._set_output_text("\n".join(lines))
        self.set_status_message(f"Plan formulated successfully ({res.get('total_steps')} steps).")

    def _set_output_text(self, text: str) -> None:
        if self._is_destroyed:
            return
        self.plan_output.configure(state="normal")
        self.plan_output.delete("1.0", tk.END)
        self.plan_output.insert(tk.END, text)
        self.plan_output.configure(state="disabled")

    def on_close(self) -> None:
        """Clean shutdown handler."""
        self._is_destroyed = True
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except Exception:
                pass
            self._poll_job = None
        try:
            self.root.destroy()
        except Exception:
            pass


def run_ui(
    state: Dict[str, Any],
    state_lock: threading.Lock,
    companion_ai: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
    task_executor: Optional[Any] = None,
) -> None:
    """Launch the BLINDSPOT desktop window."""
    bridge = UIStateBridge(
        state=state,
        state_lock=state_lock,
        companion_ai=companion_ai,
        tool_registry=tool_registry,
        task_executor=task_executor,
    )
    root = tk.Tk()
    BlindspotMainWindow(root=root, bridge=bridge)
    root.mainloop()


def main():
    """Standalone entry point for launching BLINDSPOT with UI."""
    from main import initialize_system, monitor

    components = initialize_system()
    state = components["state"]
    state_lock = components["state_lock"]
    stop_event = components["stop_event"]

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
        )
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
