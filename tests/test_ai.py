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
            self.assertEqual(first_body["tools"][0]["function_declarations"][0]["name"], "get_current_context")

            # Verify call 2 passed previous_interaction_id and tool_result
            second_body = mock_send.call_args_list[1][0][0]
            self.assertEqual(second_body["previous_interaction_id"], "interaction_step_1")
            self.assertEqual(second_body["input"]["type"], "tool_result")
            self.assertEqual(second_body["input"]["tool_result"]["name"], "get_current_context")
            self.assertTrue(second_body["input"]["tool_result"]["output"]["has_active_context"])

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
            self.assertEqual(second_body["input"][0]["tool_result"]["name"], "get_current_context")
            self.assertEqual(second_body["input"][1]["tool_result"]["name"], "get_recent_activity")

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
            self.assertIn("error", second_body["input"]["tool_result"]["output"])
            self.assertIn("not registered", second_body["input"]["tool_result"]["output"]["error"])

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
            err_output = second_body["input"]["tool_result"]["output"]["error"]
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


if __name__ == "__main__":
    unittest.main()
