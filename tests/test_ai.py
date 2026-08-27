import unittest
from unittest.mock import patch, MagicMock
import io
import json
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer
from ai import (
    LLMContextPayload,
    format_context_payload,
    LLMProvider,
    GeminiProvider,
    CompanionAI,
    MissingAPIKeyError,
    LLMProviderError,
)


class MockLLMProvider(LLMProvider):
    def __init__(self, response_text: str = "Mocked AI answer", should_raise: Exception = None):
        self.response_text = response_text
        self.should_raise = should_raise
        self.last_system_instruction = None
        self.last_user_prompt = None

    def generate_response(self, system_instruction: str, user_prompt: str) -> str:
        self.last_system_instruction = system_instruction
        self.last_user_prompt = user_prompt
        if self.should_raise:
            raise self.should_raise
        return self.response_text


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


if __name__ == "__main__":
    unittest.main()
