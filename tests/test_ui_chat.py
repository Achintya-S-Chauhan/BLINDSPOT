"""
BLINDSPOT - Phase 9A: Integrated UI Companion Chat & Desktop Q&A Tests.

Tests:
- UIStateBridge conversational query dispatch, context construction, and error handling
- Prevention of duplicate/concurrent submissions in UIStateBridge and GUI
- Conversation turn recording into ConversationSession and Clear Chat semantics
- BlindspotMainWindow tabbed layout preserving Task Planning & Execution widgets
- Chat UI components: chat_output, chat_entry, send_chat_btn, clear_chat_btn, <Return> binding
- Background worker execution with safe root.after() thread dispatch
- Headless and mocked UI interaction tests without live Gemini API calls
"""

import unittest
import threading
import time
import tkinter as tk
from unittest.mock import MagicMock, patch

from ui import UIStateBridge, BlindspotMainWindow
from context import DesktopContext, RawObservation, InterpretedContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer, WorkflowUnderstanding
from conversation import ConversationSession, ConversationTurn


class MockCompanionAI:
    """Mock CompanionAI for testing conversational Q&A without live API calls."""

    def __init__(self, answer_return: str = "Mock answer about your active window.", should_raise: bool = False):
        self.answer_return = answer_return
        self.should_raise = should_raise
        self.conversation = ConversationSession()
        self.last_ask_call = None

    def ask(
        self,
        current_context=None,
        context_start_time=None,
        history=None,
        understanding=None,
        user_query=None,
        conversation=None,
        now=None,
    ) -> str:
        self.last_ask_call = {
            "current_context": current_context,
            "context_start_time": context_start_time,
            "history": history,
            "understanding": understanding,
            "user_query": user_query,
            "conversation": conversation,
            "now": now,
        }
        if self.should_raise:
            raise RuntimeError("Simulated AI provider failure")

        active_session = conversation if conversation is not None else self.conversation
        active_session.add_user_message(user_query)
        active_session.add_assistant_message(self.answer_return)
        return self.answer_return

    def clear_conversation(self) -> None:
        self.conversation.clear()


class TestUIStateBridgeChat(unittest.TestCase):
    """Unit tests for UIStateBridge conversational query methods."""

    def setUp(self):
        self.history = ContextHistory()
        self.ctx = DesktopContext(
            observation=RawObservation(window_title="code - ui.py", ocr_text="def ask"),
            interpretation=InterpretedContext(activity="coding", confidence=3),
        )
        self.start_time = time.time() - 30.0
        self.state = {
            "current_context": self.ctx,
            "context_start_time": self.start_time,
            "history": self.history,
            "last_window": "code - ui.py",
            "last_context": "coding",
            "is_running": True,
        }
        self.state_lock = threading.Lock()
        self.mock_ai = MockCompanionAI()
        self.analyzer = ContextualUnderstandingAnalyzer()
        self.bridge = UIStateBridge(
            state=self.state,
            state_lock=self.state_lock,
            companion_ai=self.mock_ai,
            analyzer=self.analyzer,
        )

    def test_01_ask_success(self):
        """Bridge.ask() returns success, answer, and records turns into session."""
        res = self.bridge.ask("What application am I using?")
        self.assertTrue(res["success"])
        self.assertEqual(res["query"], "What application am I using?")
        self.assertEqual(res["answer"], "Mock answer about your active window.")
        self.assertFalse(self.bridge.is_asking)

        # Check turn recording in conversation session
        turns = self.bridge.get_conversation_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[0]["content"], "What application am I using?")
        self.assertEqual(turns[1]["role"], "assistant")
        self.assertEqual(turns[1]["content"], "Mock answer about your active window.")

    def test_02_ask_empty_query_rejected(self):
        """Empty or whitespace queries must be rejected without calling AI."""
        res1 = self.bridge.ask("")
        self.assertFalse(res1["success"])
        self.assertIn("empty", res1["error"].lower())

        res2 = self.bridge.ask("   \t\n  ")
        self.assertFalse(res2["success"])
        self.assertIn("empty", res2["error"].lower())
        self.assertIsNone(self.mock_ai.last_ask_call)

    def test_03_ask_no_companion_ai_rejected(self):
        """When companion_ai is None, ask() returns structured error."""
        bridge_no_ai = UIStateBridge(
            state=self.state,
            state_lock=self.state_lock,
            companion_ai=None,
        )
        res = bridge_no_ai.ask("Hello?")
        self.assertFalse(res["success"])
        self.assertIn("not configured", res["error"].lower())

    def test_04_ask_concurrent_guard(self):
        """If is_asking is True, concurrent submissions are rejected."""
        self.bridge.is_asking = True
        res = self.bridge.ask("Another query while busy")
        self.assertFalse(res["success"])
        self.assertIn("already in progress", res["error"].lower())
        self.bridge.is_asking = False

    def test_05_ask_handles_companion_ai_exception(self):
        """Exceptions in CompanionAI.ask are caught safely and reset is_asking."""
        failing_ai = MockCompanionAI(should_raise=True)
        bridge = UIStateBridge(
            state=self.state,
            state_lock=self.state_lock,
            companion_ai=failing_ai,
        )
        res = bridge.ask("Trigger exception")
        self.assertFalse(res["success"])
        self.assertIn("AI Q&A error", res["error"])
        self.assertIn("Simulated AI provider failure", res["error"])
        self.assertFalse(bridge.is_asking)

    def test_06_clear_conversation(self):
        """clear_conversation() clears conversation turns from session."""
        self.bridge.ask("Query 1")
        self.assertEqual(len(self.bridge.get_conversation_turns()), 2)

        self.bridge.clear_conversation()
        self.assertEqual(len(self.bridge.get_conversation_turns()), 0)

    def test_07_get_conversation_turns_empty_or_none(self):
        """get_conversation_turns() safely returns empty list if no conversation exists."""
        bridge_no_ai = UIStateBridge(state=self.state, state_lock=self.state_lock, companion_ai=None)
        self.assertEqual(bridge_no_ai.get_conversation_turns(), [])

    def test_08_ask_context_wiring_matches_cli(self):
        """Context assembly in ask() extracts current_context, history, and analyzer understanding."""
        self.bridge.ask("Verify context wiring")
        call = self.mock_ai.last_ask_call
        self.assertIsNotNone(call)
        self.assertIs(call["current_context"], self.ctx)
        self.assertEqual(call["context_start_time"], self.start_time)
        self.assertIs(call["history"], self.history)
        self.assertIsInstance(call["understanding"], WorkflowUnderstanding)
        self.assertEqual(call["user_query"], "Verify context wiring")


class TestBlindspotMainWindowChat(unittest.TestCase):
    """UI interaction and widget tests for Companion Chat in BlindspotMainWindow."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.state = {
            "current_context": DesktopContext(
                observation=RawObservation(window_title="notepad.exe", ocr_text="Meeting notes"),
                interpretation=InterpretedContext(activity="documenting", confidence=3),
            ),
            "context_start_time": time.time() - 10.0,
            "history": ContextHistory(),
            "last_window": "notepad.exe",
            "last_context": "documenting",
            "is_running": True,
        }
        self.lock = threading.Lock()
        self.mock_ai = MockCompanionAI(answer_return="You are editing notes in Notepad.")
        self.bridge = UIStateBridge(
            state=self.state,
            state_lock=self.lock,
            companion_ai=self.mock_ai,
        )
        self.app = BlindspotMainWindow(root=self.root, bridge=self.bridge, poll_interval_ms=1000)

    def tearDown(self):
        try:
            self.app.on_close()
        except Exception:
            pass

    def test_09_widgets_created_and_task_widgets_preserved(self):
        """Confirms tabbed notebook with preserved Task Planning widgets and new Chat widgets."""
        # Notebook exists
        self.assertTrue(hasattr(self.app, "notebook"))
        tabs = [self.app.notebook.tab(i, "text") for i in range(self.app.notebook.index("end"))]
        self.assertIn("Task Planning & Execution", tabs)
        self.assertIn("Companion Chat & Q&A", tabs)

        # Task planning widgets preserved
        self.assertTrue(hasattr(self.app, "task_entry"))
        self.assertTrue(hasattr(self.app, "plan_btn"))
        self.assertTrue(hasattr(self.app, "approve_btn"))
        self.assertTrue(hasattr(self.app, "execute_btn"))
        self.assertTrue(hasattr(self.app, "clear_btn"))
        self.assertTrue(hasattr(self.app, "plan_output"))

        # Chat widgets exist
        self.assertTrue(hasattr(self.app, "chat_output"))
        self.assertTrue(hasattr(self.app, "chat_entry"))
        self.assertTrue(hasattr(self.app, "send_chat_btn"))
        self.assertTrue(hasattr(self.app, "clear_chat_btn"))

        # Initial chat output contains welcome prompt
        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("Welcome to BLINDSPOT Companion Chat", out_text)

    def test_10_send_chat_empty_query_ignored(self):
        """Sending empty query does nothing and does not mark chat as busy."""
        self.app.chat_entry.delete(0, tk.END)
        self.app._on_send_chat_clicked()
        self.assertFalse(self.app._is_chat_busy)
        self.assertIsNone(self.mock_ai.last_ask_call)

    def test_11_send_chat_worker_and_dispatch(self):
        """Submitting a query runs worker, updates chat_output, and restores controls."""
        self.app.chat_entry.delete(0, tk.END)
        self.app.chat_entry.insert(0, "What am I working on?")

        self.app._on_send_chat_clicked()

        # Chat worker spawned
        self.assertTrue(self.app._is_chat_busy)
        self.assertIsNotNone(self.app._active_chat_worker)

        # Wait briefly for background thread to execute
        self.app._active_chat_worker.join(timeout=2.0)
        # Drain safe UI queue and update widgets
        self.app._process_ui_queue()
        self.root.update()

        # Chat should no longer be busy
        self.assertFalse(self.app._is_chat_busy)

        # Output text contains both user message and assistant reply
        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("User: What am I working on?", out_text)
        self.assertIn("BLINDSPOT: You are editing notes in Notepad.", out_text)

        # Controls restored
        self.assertEqual(str(self.app.chat_entry.cget("state")), "normal")
        self.assertEqual(str(self.app.send_chat_btn.cget("state")), "normal")
        self.assertEqual(str(self.app.clear_chat_btn.cget("state")), "normal")
        self.assertEqual(self.app.chat_entry.get(), "")

    def test_12_send_chat_error_response_handling(self):
        """When AI raises an error, chat displays error tag and restores controls."""
        failing_ai = MockCompanionAI(should_raise=True)
        self.bridge.companion_ai = failing_ai

        self.app.chat_entry.insert(0, "Failing question")
        self.app._on_send_chat_clicked()

        self.app._active_chat_worker.join(timeout=2.0)
        self.app._process_ui_queue()
        self.root.update()

        self.assertFalse(self.app._is_chat_busy)
        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("ERROR: AI Q&A error", out_text)
        self.assertEqual(str(self.app.chat_entry.cget("state")), "normal")
        self.assertEqual(str(self.app.send_chat_btn.cget("state")), "normal")

    def test_13_duplicate_send_prevented_while_busy(self):
        """Clicking send while _is_chat_busy is True is ignored."""
        self.app._is_chat_busy = True
        self.app.chat_entry.insert(0, "Query")
        self.app._on_send_chat_clicked()
        # Entry was not cleared because send was ignored
        self.assertEqual(self.app.chat_entry.get(), "Query")
        self.app._is_chat_busy = False

    def test_14_clear_chat_button(self):
        """Clicking clear_chat_btn clears conversation session and chat output."""
        # Insert a turn
        self.bridge.ask("Test question")
        self.assertEqual(len(self.bridge.get_conversation_turns()), 2)

        self.app._on_clear_chat_clicked()

        # Conversation session is cleared
        self.assertEqual(len(self.bridge.get_conversation_turns()), 0)

        # Chat output shows cleared system notice
        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("Conversation history cleared", out_text)

    def test_15_enter_key_triggers_send(self):
        """Pressing Return key in chat_entry triggers send."""
        with patch.object(self.app, "_on_send_chat_clicked") as mock_send:
            self.app._on_chat_return()
            mock_send.assert_called_once()

    def test_16_destroyed_window_safety(self):
        """Dispatching response to destroyed window does not raise."""
        self.app.on_close()
        self.assertTrue(self.app._is_destroyed)
        # Calling response handler directly when destroyed should return cleanly
        self.app._handle_chat_response({"success": True, "answer": "Nothing"})
        self.app._append_chat_message("BLINDSPOT", "Nothing")

    def test_17_chat_gemini_quota_exceeded_local_fallback_ui(self):
        """When Gemini quota is exceeded, UI displays BLINDSPOT (Local Fallback) and status notice."""
        fallback_msg = (
            "[Local Fallback — Gemini Quota Exceeded]\n"
            "You are currently working on documenting in 'notepad.exe'."
        )
        self.mock_ai.answer_return = fallback_msg

        self.app.chat_entry.insert(0, "what am i working on?")
        self.app._on_send_chat_clicked()

        self.app._active_chat_worker.join(timeout=2.0)
        self.app._process_ui_queue()
        self.root.update()

        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("BLINDSPOT (Local Fallback)", out_text)
        self.assertIn("[Local Fallback — Gemini Quota Exceeded]", out_text)
        self.assertIn("documenting in 'notepad.exe'", out_text)
        self.assertIn("Gemini quota exceeded", self.app.footer_label.cget("text"))
        self.assertEqual(str(self.app.chat_entry.cget("state")), "normal")

    def test_18_chat_gemini_quota_exceeded_friendly_error_ui(self):
        """When quota is exceeded on non-context query, UI displays friendly message without raw JSON."""
        quota_err = (
            "[Gemini Quota Exceeded] Gemini API rate limit or quota exceeded (HTTP 429). "
            "The online AI model is currently unavailable, and this question "
            "cannot be answered using local desktop perception."
        )
        self.mock_ai.answer_return = quota_err

        self.app.chat_entry.insert(0, "explain general relativity")
        self.app._on_send_chat_clicked()

        self.app._active_chat_worker.join(timeout=2.0)
        self.app._process_ui_queue()
        self.root.update()

        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("HTTP 429", out_text)
        self.assertNotIn('{"error"', out_text)
        self.assertNotIn("RESOURCE_EXHAUSTED", out_text)
        self.assertEqual(str(self.app.chat_entry.cget("state")), "normal")

    def test_19_chat_gemini_quota_exceeded_missing_context_ui(self):
        """When quota is exceeded and desktop context is unavailable, UI displays honest notice."""
        missing_ctx_msg = (
            "[Local Fallback — Gemini Quota Exceeded]\n"
            "No active desktop activity or context is currently detected."
        )
        self.mock_ai.answer_return = missing_ctx_msg

        self.app.chat_entry.insert(0, "what is my context?")
        self.app._on_send_chat_clicked()

        self.app._active_chat_worker.join(timeout=2.0)
        self.app._process_ui_queue()
        self.root.update()

        out_text = self.app.chat_output.get("1.0", tk.END)
        self.assertIn("No active desktop activity or context is currently detected", out_text)
        self.assertEqual(str(self.app.chat_entry.cget("state")), "normal")


if __name__ == "__main__":
    unittest.main()
