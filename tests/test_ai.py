import unittest
from unittest.mock import patch, MagicMock
import io
import json
from typing import Optional, List, Dict, Any
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer
from conversation import ConversationSession, ConversationTurn
from ai import (
    LLMContextPayload,
    format_context_payload,
    LLMProvider,
    GeminiProvider,
    CompanionAI,
    MissingAPIKeyError,
    LLMProviderError,
    QuotaExceededError,
    deterministic_local_answer,
    ToolCall,
    parse_tool_calls,
    extract_text_response,
    format_tool_results_for_interactions,
)
from tools import (
    ToolRegistry,
    ToolResult,
    BaseTool,
    GetCurrentContextTool,
    GetRecentActivityTool,
    GetConversationHistoryTool,
)


class MockLLMProvider(LLMProvider):
    def __init__(self, response_text: str = "Mocked AI answer", should_raise: Exception = None):
        self.response_text = response_text
        self.should_raise = should_raise
        self.last_system_instruction = None
        self.last_user_prompt = None
        self.last_tools = None
        self.last_tool_registry = None

    def generate_response(
        self,
        system_instruction: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> str:
        self.last_system_instruction = system_instruction
        self.last_user_prompt = user_prompt
        self.last_tools = tools
        self.last_tool_registry = tool_registry
        if self.should_raise:
            raise self.should_raise
        return self.response_text


class MockMutatingTool(BaseTool):
    """Mutating action tool used to verify safety boundary in tool calling."""
    def __init__(self):
        super().__init__(
            name="mock_mutating_action",
            description="Simulates an unsafe mutating action.",
            is_read_only=False,
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output="Mutating action executed!")


class TestAICompanionCore(unittest.TestCase):

    def setUp(self):
        self.analyzer = ContextualUnderstandingAnalyzer()

    def _create_context(self, activity: str = "coding", window_title: str = "VSCode", ocr_text: str = "def foo(): pass") -> DesktopContext:
        obs = RawObservation(window_title=window_title, ocr_text=ocr_text)
        interp = InterpretedContext(activity=activity, confidence=3)
        return DesktopContext(observation=obs, interpretation=interp)

    def test_1_context_payload_construction(self):
        """1. Test that format_context_payload builds a complete LLMContextPayload."""
        ctx = self._create_context("coding", "VSCode - main.py", "import time")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1045.0)

        payload = format_context_payload(
            current_context=ctx,
            context_start_time=1000.0,
            history=history,
            understanding=understanding,
            user_query="What am I doing?",
            now=1045.0,
        )

        self.assertIsInstance(payload, LLMContextPayload)
        d = payload.to_dict()
        self.assertEqual(d["current_context"]["activity"], "coding")
        self.assertEqual(d["current_context"]["window_title"], "VSCode - main.py")
        self.assertEqual(d["current_context"]["duration_seconds"], 45.0)
        self.assertEqual(d["user_query"], "What am I doing?")

    def test_2_current_context_inclusion(self):
        """2. Test that current context is accurately reflected in prompt text."""
        ctx = self._create_context("coding", "VSCode - app.py", "const x = 42;")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1060.0)

        payload = format_context_payload(ctx, 1000.0, history, understanding, "status check", now=1060.0)
        prompt = payload.to_prompt_text()

        self.assertIn("Active Activity: coding", prompt)
        self.assertIn("Active Window:   VSCode - app.py", prompt)
        self.assertIn("Active Duration: 1m", prompt)
        self.assertIn("const x = 42;", prompt)

    def test_3_recent_history_inclusion(self):
        """3. Test that recent context history transitions are included in prompt text."""
        history = ContextHistory(maxlen=10)
        ctx1 = self._create_context("coding", "VSCode")
        ctx2 = self._create_context("reading in a browser", "Docs - Chrome")

        history.record_transition(ctx1, start_time=1000.0, end_time=1100.0)
        history.record_transition(ctx2, start_time=1100.0, end_time=1180.0)

        ctx_active = self._create_context("coding", "VSCode")
        understanding = self.analyzer.analyze(ctx_active, 1180.0, history, now=1200.0)

        payload = format_context_payload(ctx_active, 1180.0, history, understanding, "Summarize recent workflow", now=1200.0)
        prompt = payload.to_prompt_text()

        self.assertIn("reading in a browser", prompt)
        self.assertIn("Docs - Chrome", prompt)
        self.assertIn("Flow Sequence: coding -> reading in a browser -> coding", prompt)

    def test_4_workflow_understanding_inclusion(self):
        """4. Test that workflow pattern type and interpretation are present."""
        history = ContextHistory(maxlen=10)
        ctx1 = self._create_context("coding", "VSCode")
        ctx2 = self._create_context("reading in a browser", "Chrome")

        history.record_transition(ctx1, start_time=1000.0, end_time=1180.0)
        history.record_transition(ctx2, start_time=1180.0, end_time=1265.0)

        ctx_active = self._create_context("coding", "VSCode")
        understanding = self.analyzer.analyze(ctx_active, 1265.0, history, now=1310.0)

        payload = format_context_payload(ctx_active, 1265.0, history, understanding, "Why did I switch?", now=1310.0)
        prompt = payload.to_prompt_text()

        self.assertIn("Pattern Type:   resumed_task", prompt)
        self.assertIn(understanding.interpretation, prompt)
        self.assertIn(f"Confidence:     {understanding.confidence}/3", prompt)

    def test_5_user_question_inclusion(self):
        """5. Test that user's question is explicitly included at the prompt end."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        query = "Can you explain what I was working on 10 minutes ago?"
        payload = format_context_payload(ctx, 1000.0, history, understanding, query, now=1010.0)
        prompt = payload.to_prompt_text()

        self.assertIn("=== USER QUESTION ===", prompt)
        self.assertIn(query, prompt)

    def test_6_token_conscious_truncation(self):
        """6. Test that long OCR text is truncated to max_ocr_chars."""
        huge_ocr = "A" * 1000
        ctx = self._create_context("reading in a browser", "Chrome", huge_ocr)
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1020.0)

        payload = format_context_payload(ctx, 1000.0, history, understanding, "query", max_ocr_chars=100, now=1020.0)

        self.assertTrue(len(payload.current_ocr_snippet) <= 105)  # 100 chars + "..."
        self.assertTrue(payload.current_ocr_snippet.endswith("..."))

    def test_7_provider_error_handling(self):
        """7. Test that LLM provider errors are caught gracefully without crashing."""
        provider = MockLLMProvider(should_raise=LLMProviderError("Connection timed out"))
        ai = CompanionAI(provider=provider)

        ctx = self._create_context("coding")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        response = ai.ask(ctx, 1000.0, history, understanding, "hello")
        self.assertIn("[AI Provider Error]", response)
        self.assertIn("Connection timed out", response)

    def test_8_missing_api_key_handling(self):
        """8. Test that MissingAPIKeyError provides a clear configuration explanation."""
        # Initialize GeminiProvider with no API key and no env var
        with patch.dict("os.environ", {}, clear=True):
            provider = GeminiProvider(api_key=None)
            ai = CompanionAI(provider=provider)

            ctx = self._create_context("coding")
            history = ContextHistory(maxlen=10)
            understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

            response = ai.ask(ctx, 1000.0, history, understanding, "hello")
            self.assertIn("BLINDSPOT_GEMINI_API_KEY", response)
            self.assertIn("not configured", response)

    def test_9_successful_mocked_assistant_response(self):
        """9. Test successful end-to-end question answering with a mocked provider."""
        expected_reply = "You were coding in VS Code for 3 minutes, researched in Chrome for 1 minute, and are now back to coding."
        mock_provider = MockLLMProvider(response_text=expected_reply)
        ai = CompanionAI(provider=mock_provider)

        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1045.0)

        response = ai.ask(
            current_context=ctx,
            context_start_time=1000.0,
            history=history,
            understanding=understanding,
            user_query="Summarize what I did.",
            now=1045.0,
        )

        self.assertEqual(response, expected_reply)
        self.assertIn("=== OBSERVED DESKTOP CONTEXT ===", mock_provider.last_user_prompt)
        self.assertIn("Summarize what I did.", mock_provider.last_user_prompt)
        self.assertIn("BLINDSPOT", mock_provider.last_system_instruction)

    def test_10_conversation_payload_inclusion(self):
        """10. Test that conversation session turns appear in prompt text."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        session = ConversationSession(maxlen=10)
        session.add_user_message("What am I working on?")
        session.add_assistant_message("You are editing main.py in VS Code.")

        payload = format_context_payload(
            ctx, 1000.0, history, understanding, "explain that",
            conversation=session, now=1010.0,
        )
        prompt = payload.to_prompt_text()

        self.assertIn("=== RECENT CONVERSATION ===", prompt)
        self.assertIn("User: What am I working on?", prompt)
        self.assertIn("BLINDSPOT: You are editing main.py in VS Code.", prompt)
        self.assertIn("=== USER QUESTION ===\nexplain that", prompt)

    def test_11_companion_ai_session_accumulation(self):
        """11. Test that CompanionAI records turns into its ConversationSession."""
        mock_provider = MockLLMProvider(response_text="You're writing python tests.")
        ai = CompanionAI(provider=mock_provider)

        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        self.assertEqual(len(ai.conversation), 0)

        reply = ai.ask(ctx, 1000.0, history, understanding, "what am I doing?")
        self.assertEqual(reply, "You're writing python tests.")
        self.assertEqual(len(ai.conversation), 2)

        turns = ai.conversation.get_turns()
        self.assertEqual(turns[0].role, "user")
        self.assertEqual(turns[0].content, "what am I doing?")
        self.assertEqual(turns[1].role, "assistant")
        self.assertEqual(turns[1].content, "You're writing python tests.")

    def test_12_followup_question_includes_prior_turns_in_llm_payload(self):
        """12. Test that follow-up questions include previous conversation in provider prompt."""
        mock_provider = MockLLMProvider(response_text="First answer")
        ai = CompanionAI(provider=mock_provider)

        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        # Turn 1
        ai.ask(ctx, 1000.0, history, understanding, "what am I doing?")
        self.assertNotIn("=== RECENT CONVERSATION ===", mock_provider.last_user_prompt)

        # Turn 2: Follow-up
        mock_provider.response_text = "Second answer"
        ai.ask(ctx, 1000.0, history, understanding, "what was I doing before this?")
        self.assertIn("=== RECENT CONVERSATION ===", mock_provider.last_user_prompt)
        self.assertIn("User: what am I doing?", mock_provider.last_user_prompt)
        self.assertIn("BLINDSPOT: First answer", mock_provider.last_user_prompt)
        self.assertIn("=== USER QUESTION ===\nwhat was I doing before this?", mock_provider.last_user_prompt)

    def test_13_bounded_conversation_in_payload(self):
        """13. Test that format_context_payload limits conversation turns to max_conversation_turns."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        session = ConversationSession(maxlen=50)
        for i in range(20):
            session.add_user_message(f"Question {i}")
            session.add_assistant_message(f"Answer {i}")

        payload = format_context_payload(
            ctx, 1000.0, history, understanding, "Latest question",
            conversation=session, max_conversation_turns=4, now=1010.0,
        )

        # Should only have last 4 turns (2 user, 2 assistant)
        self.assertEqual(len(payload.conversation_history), 4)
        prompt = payload.to_prompt_text()
        self.assertIn("Question 19", prompt)
        self.assertIn("Answer 19", prompt)
        self.assertNotIn("Question 0", prompt)

    def test_14_clear_conversation(self):
        """14. Test that clear_conversation resets the session and clears conversation in subsequent prompts."""
        mock_provider = MockLLMProvider(response_text="Sure, here is the answer.")
        ai = CompanionAI(provider=mock_provider)

        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        ai.ask(ctx, 1000.0, history, understanding, "Initial question")
        self.assertEqual(len(ai.conversation), 2)

        ai.clear_conversation()
        self.assertEqual(len(ai.conversation), 0)
        self.assertTrue(ai.conversation.is_empty)

        # Asking after clear should not have RECENT CONVERSATION section
        ai.ask(ctx, 1000.0, history, understanding, "Fresh question")
        self.assertNotIn("=== RECENT CONVERSATION ===", mock_provider.last_user_prompt)
        self.assertIn("Fresh question", mock_provider.last_user_prompt)
        self.assertEqual(len(ai.conversation), 2)

    def test_15_provider_error_does_not_record_conversation_turn(self):
        """15. Test that failed queries do not pollute conversation history."""
        mock_provider = MockLLMProvider(should_raise=LLMProviderError("Timeout"))
        ai = CompanionAI(provider=mock_provider)

        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        response = ai.ask(ctx, 1000.0, history, understanding, "Failing question")
        self.assertIn("[AI Provider Error]", response)
        self.assertEqual(len(ai.conversation), 0)

    def test_16_parse_tool_calls_interactions_steps(self):
        """16. Test parsing tool_call from Interactions API steps structure."""
        resp_data = {
            "id": "interactions/123",
            "steps": [
                {
                    "type": "tool_call",
                    "tool_call": {
                        "name": "get_current_context",
                        "arguments": {"include_ocr": True},
                        "id": "call_1",
                    },
                }
            ],
        }
        calls = parse_tool_calls(resp_data)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_current_context")
        self.assertEqual(calls[0].args, {"include_ocr": True})
        self.assertEqual(calls[0].call_id, "call_1")

    def test_17_parse_tool_calls_candidates_and_output(self):
        """17. Test parsing functionCall from standard candidates and output shapes with stringified args."""
        # Candidates shape with stringified JSON arguments
        resp_cand = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_recent_activity",
                                    "args": '{"limit": 3}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        calls_cand = parse_tool_calls(resp_cand)
        self.assertEqual(len(calls_cand), 1)
        self.assertEqual(calls_cand[0].name, "get_recent_activity")
        self.assertEqual(calls_cand[0].args, {"limit": 3})

        # Output shape
        resp_out = {
            "output": [
                {
                    "function_call": {
                        "name": "get_conversation_history",
                        "arguments": {"limit": 5},
                    }
                }
            ]
        }
        calls_out = parse_tool_calls(resp_out)
        self.assertEqual(len(calls_out), 1)
        self.assertEqual(calls_out[0].name, "get_conversation_history")
        self.assertEqual(calls_out[0].args, {"limit": 5})

    def test_18_parse_multiple_tool_calls(self):
        """18. Test parsing multiple tool calls in a single turn."""
        resp_data = {
            "steps": [
                {
                    "type": "tool_call",
                    "tool_call": {"name": "get_current_context", "args": {}},
                },
                {
                    "type": "tool_call",
                    "tool_call": {"name": "get_recent_activity", "args": {"limit": 5}},
                },
            ]
        }
        calls = parse_tool_calls(resp_data)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "get_current_context")
        self.assertEqual(calls[1].name, "get_recent_activity")
        self.assertEqual(calls[1].args, {"limit": 5})

    def test_19_gemini_provider_single_tool_call_loop(self):
        """19. Test successful Gemini tool execution loop returning result back to Gemini."""
        provider = GeminiProvider(api_key="fake-test-key")
        ctx = self._create_context("coding", "VSCode - main.py")
        registry = ToolRegistry()
        registry.register(GetCurrentContextTool(state_getter=lambda: {"current_context": ctx}))

        mock_turn1 = {
            "id": "interaction_step_1",
            "steps": [
                {
                    "type": "tool_call",
                    "tool_call": {
                        "name": "get_current_context",
                        "args": {"include_ocr": False},
                    },
                }
            ],
        }
        mock_turn2 = {
            "id": "interaction_step_2",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"text": "Based on the tool result, you are coding in VS Code."}],
                }
            ],
        }

        with patch.object(provider, "_send_http_request", side_effect=[mock_turn1, mock_turn2]) as mock_send:
            response = provider.generate_response(
                system_instruction="You are BLINDSPOT.",
                user_prompt="What am I doing?",
                tools=registry.get_tool_schemas(),
                tool_registry=registry,
            )

            self.assertEqual(response, "Based on the tool result, you are coding in VS Code.")
            self.assertEqual(mock_send.call_count, 2)

            # Verify call 1 had tool declarations
            first_body = mock_send.call_args_list[0][0][0]
            self.assertIn("tools", first_body)
            self.assertEqual(first_body["tools"][0]["name"], "get_current_context")
            self.assertEqual(first_body["tools"][0]["type"], "function")

            # Verify call 2 passed previous_interaction_id and function_result
            second_body = mock_send.call_args_list[1][0][0]
            self.assertEqual(second_body["previous_interaction_id"], "interaction_step_1")
            self.assertEqual(second_body["input"][0]["type"], "function_result")
            self.assertEqual(second_body["input"][0]["name"], "get_current_context")
            self.assertTrue(second_body["input"][0]["result"]["has_active_context"])

    def test_20_gemini_provider_multiple_tool_calls_single_turn(self):
        """20. Test handling multiple tool calls issued simultaneously by Gemini."""
        provider = GeminiProvider(api_key="fake-test-key")
        ctx = self._create_context("coding", "VSCode")
        hist = ContextHistory()
        hist.record_transition(ctx, 1000.0, 1050.0)

        registry = ToolRegistry()
        registry.register(GetCurrentContextTool(state_getter=lambda: {"current_context": ctx}))
        registry.register(GetRecentActivityTool(history=hist))

        mock_turn1 = {
            "id": "interaction_multi_1",
            "steps": [
                {"type": "tool_call", "tool_call": {"name": "get_current_context", "args": {}}},
                {"type": "tool_call", "tool_call": {"name": "get_recent_activity", "args": {"limit": 2}}},
            ],
        }
        mock_turn2 = {
            "id": "interaction_multi_2",
            "steps": [
                {"type": "model_output", "content": [{"text": "You are currently coding, and before that you were coding."}]},
            ],
        }

        with patch.object(provider, "_send_http_request", side_effect=[mock_turn1, mock_turn2]) as mock_send:
            response = provider.generate_response(
                system_instruction="sys",
                user_prompt="prompt",
                tools=registry.get_tool_schemas(),
                tool_registry=registry,
            )

            self.assertIn("You are currently coding", response)
            self.assertEqual(mock_send.call_count, 2)
            second_body = mock_send.call_args_list[1][0][0]
            # Input should be a list of 2 tool results
            self.assertIsInstance(second_body["input"], list)
            self.assertEqual(len(second_body["input"]), 2)
            self.assertEqual(second_body["input"][0]["type"], "function_result")
            self.assertEqual(second_body["input"][0]["name"], "get_current_context")
            self.assertEqual(second_body["input"][1]["type"], "function_result")
            self.assertEqual(second_body["input"][1]["name"], "get_recent_activity")

    def test_21_gemini_provider_sequential_tool_calls_multi_turn(self):
        """21. Test multi-turn sequential tool calling (Tool A -> Result A -> Tool B -> Result B -> Final)."""
        provider = GeminiProvider(api_key="fake-test-key")
        ctx = self._create_context("coding")
        hist = ContextHistory()

        registry = ToolRegistry()
        registry.register(GetCurrentContextTool(state_getter=lambda: {"current_context": ctx}))
        registry.register(GetRecentActivityTool(history=hist))

        # Turn 1 requests Tool 1
        turn1 = {"id": "int_1", "steps": [{"type": "tool_call", "tool_call": {"name": "get_current_context", "args": {}}}]}
        # Turn 2 requests Tool 2
        turn2 = {"id": "int_2", "steps": [{"type": "tool_call", "tool_call": {"name": "get_recent_activity", "args": {}}}]}
        # Turn 3 returns final text
        turn3 = {"id": "int_3", "steps": [{"type": "model_output", "content": [{"text": "Sequential tool loop completed."}]}]}

        with patch.object(provider, "_send_http_request", side_effect=[turn1, turn2, turn3]) as mock_send:
            response = provider.generate_response(
                system_instruction="sys",
                user_prompt="prompt",
                tools=registry.get_tool_schemas(),
                tool_registry=registry,
            )

            self.assertEqual(response, "Sequential tool loop completed.")
            self.assertEqual(mock_send.call_count, 3)

    def test_22_gemini_provider_unknown_tool_request(self):
        """22. Test that requesting an unknown tool returns a structured error to Gemini safely."""
        provider = GeminiProvider(api_key="fake-test-key")
        registry = ToolRegistry()

        turn1 = {"id": "int_err", "steps": [{"type": "tool_call", "tool_call": {"name": "nonexistent_scanner", "args": {}}}]}
        turn2 = {"id": "int_err_done", "steps": [{"type": "model_output", "content": [{"text": "I could not scan because tool is unavailable."}]}]}

        with patch.object(provider, "_send_http_request", side_effect=[turn1, turn2]) as mock_send:
            response = provider.generate_response(
                system_instruction="sys",
                user_prompt="prompt",
                tools=registry.get_tool_schemas(),
                tool_registry=registry,
            )

            self.assertEqual(response, "I could not scan because tool is unavailable.")
            second_body = mock_send.call_args_list[1][0][0]
            self.assertEqual(second_body["input"][0]["type"], "function_result")
            self.assertIn("error", second_body["input"][0]["result"])
            self.assertIn("not registered", second_body["input"][0]["result"]["error"])

    def test_23_gemini_provider_safety_boundary_blocks_action_tool(self):
        """23. Test that mutating action tools requested by Gemini are rejected by the safety boundary."""
        provider = GeminiProvider(api_key="fake-test-key")
        registry = ToolRegistry(allow_actions=False)
        registry.register(MockMutatingTool())

        turn1 = {"id": "int_unsafe", "steps": [{"type": "tool_call", "tool_call": {"name": "mock_mutating_action", "args": {}}}]}
        turn2 = {"id": "int_unsafe_done", "steps": [{"type": "model_output", "content": [{"text": "I was blocked by the safety boundary."}]}]}

        with patch.object(provider, "_send_http_request", side_effect=[turn1, turn2]) as mock_send:
            response = provider.generate_response(
                system_instruction="sys",
                user_prompt="prompt",
                tools=registry.get_tool_schemas(),
                tool_registry=registry,
            )

            self.assertEqual(response, "I was blocked by the safety boundary.")
            second_body = mock_send.call_args_list[1][0][0]
            self.assertEqual(second_body["input"][0]["type"], "function_result")
            err_output = second_body["input"][0]["result"]["error"]
            self.assertIn("safety boundary", err_output)

    def test_24_companion_ai_with_gemini_tool_loop(self):
        """24. Test full CompanionAI.ask() flow with Gemini tool calling and conversation update."""
        provider = GeminiProvider(api_key="fake-test-key")
        ctx = self._create_context("coding", "VSCode")
        registry = ToolRegistry()
        registry.register(GetCurrentContextTool(state_getter=lambda: {"current_context": ctx}))

        ai = CompanionAI(provider=provider, tool_registry=registry)
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        turn1 = {"id": "ai_tool_1", "steps": [{"type": "tool_call", "tool_call": {"name": "get_current_context", "args": {}}}]}
        turn2 = {"id": "ai_tool_2", "steps": [{"type": "model_output", "content": [{"text": "Tool confirmed you are coding."}]}]}

        with patch.object(provider, "_send_http_request", side_effect=[turn1, turn2]):
            reply = ai.ask(ctx, 1000.0, history, understanding, "Check my active app")

            self.assertEqual(reply, "Tool confirmed you are coding.")
            self.assertEqual(len(ai.conversation), 2)
            turns = ai.conversation.get_turns()
            self.assertEqual(turns[0].content, "Check my active app")
            self.assertEqual(turns[1].content, "Tool confirmed you are coding.")

    def test_25_failed_tool_calls_do_not_corrupt_conversation_history(self):
        """25. Test that network / provider errors during tool calling do not pollute conversation history."""
        provider = GeminiProvider(api_key="fake-test-key")
        ctx = self._create_context("coding")
        registry = ToolRegistry()
        registry.register(GetCurrentContextTool(state_getter=lambda: {"current_context": ctx}))

        ai = CompanionAI(provider=provider, tool_registry=registry)
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        # First call succeeds in requesting tool, but second call fails with HTTP error
        turn1 = {"id": "fail_tool_1", "steps": [{"type": "tool_call", "tool_call": {"name": "get_current_context", "args": {}}}]}

        with patch.object(provider, "_send_http_request", side_effect=[turn1, LLMProviderError("Connection timeout during tool loop")]):
            reply = ai.ask(ctx, 1000.0, history, understanding, "Check my active app")

            self.assertIn("[AI Provider Error]", reply)
            self.assertEqual(len(ai.conversation), 0)

    def test_26_gemini_provider_http_429_raises_quota_exceeded_error(self):
        """26. Test that HTTP 429 parses error cleanly, redacts API key, and raises QuotaExceededError."""
        from urllib.error import HTTPError
        import urllib.request

        secret_key = "super_secret_gemini_api_key_12345"
        provider = GeminiProvider(api_key=secret_key)

        raw_json = json.dumps({
            "error": {
                "code": 429,
                "message": f"Rate limit exceeded for model gemini-3.6-flash (key: {secret_key})",
                "status": "RESOURCE_EXHAUSTED",
            }
        }).encode("utf-8")

        mock_http_err = HTTPError(
            url=f"https://generativelanguage.googleapis.com/v1beta/interactions?key={secret_key}",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(raw_json),
        )

        with patch("urllib.request.urlopen", side_effect=mock_http_err):
            with self.assertRaises(QuotaExceededError) as cm:
                provider._send_http_request({"model": "gemini-3.6-flash"}, secret_key)

            err_msg = str(cm.exception)
            self.assertIn("429", err_msg)
            self.assertIn("quota", err_msg.lower())
            # API secret must never be exposed
            self.assertNotIn(secret_key, err_msg)
            # Raw provider JSON wrapper must not be dumped directly
            self.assertNotIn('"status": "RESOURCE_EXHAUSTED"', err_msg)

    def test_27_deterministic_local_answer_active_window(self):
        """27. Test deterministic local fallback answering active window queries."""
        ctx = self._create_context("coding", "VSCode - main.py")
        ans = deterministic_local_answer(
            user_query="What window is active?",
            current_context=ctx,
            context_start_time=1000.0,
            history=None,
            understanding=None,
            now=1045.0,
        )
        self.assertIsNotNone(ans)
        self.assertIn("VSCode - main.py", ans)
        self.assertIn("active for 45s", ans)

    def test_28_deterministic_local_answer_working_on(self):
        """28. Test deterministic local fallback answering 'what am I working on' queries."""
        ctx = self._create_context("coding", "VSCode - main.py")
        history = ContextHistory()
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1030.0)

        ans = deterministic_local_answer(
            user_query="What am I working on?",
            current_context=ctx,
            context_start_time=1000.0,
            history=history,
            understanding=understanding,
            now=1030.0,
        )
        self.assertIsNotNone(ans)
        self.assertIn("coding", ans)
        self.assertIn("VSCode - main.py", ans)

    def test_29_deterministic_local_answer_recent_activity(self):
        """29. Test deterministic local fallback answering recent activity history queries."""
        history = ContextHistory()
        c1 = self._create_context("researching in browser", "Chrome")
        c2 = self._create_context("coding", "VSCode")
        history.record_transition(c1, 1000.0, 1060.0)
        history.record_transition(c2, 1060.0, 1120.0)

        understanding = self.analyzer.analyze(c2, 1120.0, history, now=1130.0)

        ans = deterministic_local_answer(
            user_query="What was I doing recently?",
            current_context=c2,
            context_start_time=1120.0,
            history=history,
            understanding=understanding,
            now=1130.0,
        )
        self.assertIsNotNone(ans)
        self.assertIn("Recent activity history", ans)
        self.assertIn("researching in browser", ans)
        self.assertIn("coding", ans)

    def test_30_deterministic_local_answer_missing_context_honesty(self):
        """30. Test that missing context honestly reports unavailable rather than fabricating."""
        # 1. Missing window
        ans_win = deterministic_local_answer(
            user_query="What window is active?",
            current_context=None,
            context_start_time=None,
            history=None,
            understanding=None,
        )
        self.assertEqual(ans_win, "No active window is currently detected.")

        # 2. Missing activity
        ans_act = deterministic_local_answer(
            user_query="What am I working on?",
            current_context=None,
            context_start_time=None,
            history=None,
            understanding=None,
        )
        self.assertEqual(ans_act, "No active desktop activity or context is currently detected.")

        # 3. Empty history
        empty_hist = ContextHistory()
        ans_hist = deterministic_local_answer(
            user_query="What was I doing recently?",
            current_context=None,
            context_start_time=None,
            history=empty_hist,
            understanding=None,
        )
        self.assertEqual(ans_hist, "No recent activity history has been recorded yet in this session.")

    def test_31_deterministic_local_answer_unrelated_query_returns_none(self):
        """31. Test that non-context questions return None so they are not fabricated locally."""
        ans = deterministic_local_answer(
            user_query="Write a poem about rust and python",
            current_context=self._create_context("coding"),
            context_start_time=1000.0,
            history=None,
            understanding=None,
        )
        self.assertIsNone(ans)

    def test_32_companion_ai_ask_quota_exceeded_local_fallback(self):
        """32. Test that QuotaExceededError in CompanionAI.ask triggers local fallback and records turn."""
        provider = MockLLMProvider(should_raise=QuotaExceededError("Rate limit exceeded (HTTP 429)"))
        ai = CompanionAI(provider=provider)

        ctx = self._create_context("coding", "VSCode - task.py")
        history = ContextHistory()
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1020.0)

        reply = ai.ask(
            current_context=ctx,
            context_start_time=1000.0,
            history=history,
            understanding=understanding,
            user_query="What am I working on?",
            now=1020.0,
        )

        self.assertIn("[Local Fallback — Gemini Quota Exceeded]", reply)
        self.assertIn("coding", reply)
        self.assertIn("VSCode - task.py", reply)

        # Conversation session records the turn
        turns = ai.conversation.get_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].content, "What am I working on?")
        self.assertIn("[Local Fallback — Gemini Quota Exceeded]", turns[1].content)

    def test_33_companion_ai_ask_quota_exceeded_unrelated_query(self):
        """33. Test QuotaExceededError on non-context query returns concise friendly message without raw JSON."""
        provider = MockLLMProvider(should_raise=QuotaExceededError("Rate limit exceeded (HTTP 429)"))
        ai = CompanionAI(provider=provider)

        ctx = self._create_context("coding")
        history = ContextHistory()
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1010.0)

        reply = ai.ask(
            current_context=ctx,
            context_start_time=1000.0,
            history=history,
            understanding=understanding,
            user_query="How do I configure nginx reverse proxy?",
            now=1010.0,
        )

        self.assertIn("[Gemini Quota Exceeded]", reply)
        self.assertIn("HTTP 429", reply)
        self.assertNotIn("{", reply)  # No raw JSON
        self.assertNotIn("RESOURCE_EXHAUSTED", reply)

        # Conversation turn recorded
        turns = ai.conversation.get_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].content, "How do I configure nginx reverse proxy?")

    def test_34_deterministic_local_answer_summary_short_and_natural_language(self):
        """34. Test summary requests with short commands and natural-language phrasings."""
        history = ContextHistory()
        c1 = self._create_context("researching in browser", "Chrome")
        c2 = self._create_context("coding", "VSCode - main.py")
        history.record_transition(c1, 1000.0, 1060.0)
        history.record_transition(c2, 1060.0, 1120.0)
        understanding = self.analyzer.analyze(c2, 1120.0, history, now=1180.0)

        test_queries = [
            "summary",
            "summarize my activity",
            "give me a desktop summary",
            "summarize what I've been doing",
            "can you summarize what I've been doing?",
            "desktop summary",
            "activity summary",
            "give me a summary",
        ]

        for q in test_queries:
            ans = deterministic_local_answer(
                user_query=q,
                current_context=c2,
                context_start_time=1120.0,
                history=history,
                understanding=understanding,
                now=1180.0,
            )
            self.assertIsNotNone(ans, f"Expected non-None response for query '{q}'")
            self.assertIn("Desktop Activity Summary:", ans)
            self.assertIn("coding", ans)
            self.assertIn("VSCode - main.py", ans)
            self.assertIn("active for 1m", ans)
            self.assertIn("Recent Activity (2 recorded episodes):", ans)
            self.assertIn("researching in browser on 'Chrome'", ans)
            self.assertIn("Flow: researching in browser -> coding", ans)

    def test_35_deterministic_local_answer_summary_empty_history(self):
        """35. Test summary when current context exists but no prior history has been recorded."""
        ctx = self._create_context("debugging", "PyCharm - test_core.py")
        empty_hist = ContextHistory()

        ans = deterministic_local_answer(
            user_query="summary",
            current_context=ctx,
            context_start_time=1000.0,
            history=empty_hist,
            understanding=None,
            now=1045.0,
        )
        self.assertIsNotNone(ans)
        self.assertIn("Desktop Activity Summary:", ans)
        self.assertIn("Current Focus: debugging in 'PyCharm - test_core.py' (active for 45s).", ans)
        self.assertIn("Recent History: No prior activity transitions recorded yet in this session.", ans)

    def test_36_deterministic_local_answer_summary_missing_current_context(self):
        """36. Test summary when no current window is active but prior history exists."""
        history = ContextHistory()
        c1 = self._create_context("writing documentation", "Notion")
        c2 = self._create_context("terminal ops", "PowerShell")
        history.record_transition(c1, 1000.0, 1100.0)
        history.record_transition(c2, 1100.0, 1200.0)
        understanding = self.analyzer.analyze(c2, 1200.0, history, now=1210.0)

        ans = deterministic_local_answer(
            user_query="give me a desktop summary",
            current_context=None,
            context_start_time=None,
            history=history,
            understanding=understanding,
            now=1210.0,
        )
        self.assertIsNotNone(ans)
        self.assertIn("Desktop Activity Summary:", ans)
        self.assertIn("Current Focus: No active window or activity currently detected.", ans)
        self.assertIn("Recent Activity (2 recorded episodes):", ans)
        self.assertIn("writing documentation on 'Notion'", ans)
        self.assertIn("terminal ops on 'PowerShell'", ans)

    def test_37_deterministic_local_answer_summary_no_activity_honesty(self):
        """37. Test that summary honestly reports not enough recorded activity when context and history are absent."""
        empty_hist = ContextHistory()
        ans = deterministic_local_answer(
            user_query="summarize what I've been doing",
            current_context=None,
            context_start_time=None,
            history=empty_hist,
            understanding=None,
        )
        self.assertEqual(ans, "There is not enough recorded desktop activity or history to summarize.")

    def test_38_deterministic_local_answer_summary_unrelated_query_returns_none(self):
        """38. Test that unrelated queries with 'summary' words return None and do not trigger desktop summary."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory()
        unrelated_queries = [
            "Write a summary of Hamlet",
            "Summarize the French Revolution",
            "Please summarize this PDF article about quantum computing",
            "Summary of War and Peace",
        ]
        for q in unrelated_queries:
            ans = deterministic_local_answer(
                user_query=q,
                current_context=ctx,
                context_start_time=1000.0,
                history=history,
                understanding=None,
            )
            self.assertIsNone(ans, f"Expected None for unrelated query '{q}' but got: {ans}")

    def test_39_companion_ai_ask_quota_exceeded_summary_fallback(self):
        """39. Test that summary query on QuotaExceededError in CompanionAI returns local summary fallback."""
        provider = MockLLMProvider(should_raise=QuotaExceededError("Rate limit exceeded (HTTP 429)"))
        ai = CompanionAI(provider=provider)

        ctx = self._create_context("coding", "VSCode - verification.py")
        history = ContextHistory()
        c1 = self._create_context("terminal testing", "Command Prompt")
        history.record_transition(c1, 1000.0, 1050.0)
        history.record_transition(ctx, 1050.0, 1100.0)

        understanding = self.analyzer.analyze(ctx, 1100.0, history, now=1130.0)

        reply = ai.ask(
            current_context=ctx,
            context_start_time=1100.0,
            history=history,
            understanding=understanding,
            user_query="summary",
            now=1130.0,
        )

        self.assertIn("[Local Fallback — Gemini Quota Exceeded]", reply)
        self.assertIn("Desktop Activity Summary:", reply)
        self.assertIn("coding in 'VSCode - verification.py'", reply)
        self.assertIn("Recent Activity", reply)
        self.assertIn("terminal testing on 'Command Prompt'", reply)

        # Check conversation history recorded correctly
        turns = ai.conversation.get_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].content, "summary")
        self.assertIn("Desktop Activity Summary:", turns[1].content)


if __name__ == "__main__":
    unittest.main()
