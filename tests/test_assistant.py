import unittest
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer
from assistant import CompanionAssistant, CompanionResponse


class TestCompanionAssistant(unittest.TestCase):

    def setUp(self):
        self.analyzer = ContextualUnderstandingAnalyzer()
        self.assistant = CompanionAssistant()

    def _create_context(self, activity: str, window_title: str = "TestApp", confidence: int = 3) -> DesktopContext:
        obs = RawObservation(window_title=window_title, ocr_text="")
        interp = InterpretedContext(activity=activity, confidence=confidence)
        return DesktopContext(observation=obs, interpretation=interp)

    def test_focused_continuous_activity(self):
        """1. Focused continuous activity produces focused companion response."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1045.0)

        response = self.assistant.respond(understanding)

        self.assertEqual(response.category, "focus")
        self.assertEqual(response.confidence, 3)
        self.assertIn("focused on the same coding activity", response.message)
        self.assertEqual(response.supporting_context["current_activity"], "coding")

    def test_task_resumption_a_b_a(self):
        """2. A -> B -> A produces resumption companion response."""
        history = ContextHistory(maxlen=10)
        ctx_code = self._create_context("coding", "VSCode")
        ctx_research = self._create_context("reading in a browser", "Chrome")

        history.record_transition(ctx_code, start_time=1000.0, end_time=1180.0)
        history.record_transition(ctx_research, start_time=1180.0, end_time=1265.0)

        understanding = self.analyzer.analyze(ctx_code, 1265.0, history, now=1310.0)
        response = self.assistant.respond(understanding)

        self.assertEqual(response.category, "resumption")
        self.assertEqual(response.confidence, 3)
        self.assertIn("back to your coding task after a short research session", response.message)
        self.assertEqual(response.supporting_context["pattern_type"], "resumed_task")

    def test_workflow_shift_a_b_c(self):
        """3. A -> B -> C produces workflow shift companion response."""
        history = ContextHistory(maxlen=10)
        ctx_code = self._create_context("coding", "VSCode")
        ctx_research = self._create_context("reading in a browser", "Chrome")

        history.record_transition(ctx_code, start_time=1000.0, end_time=1100.0)

        understanding = self.analyzer.analyze(ctx_research, 1100.0, history, now=1140.0)
        response = self.assistant.respond(understanding)

        self.assertEqual(response.category, "transition")
        self.assertIn("shifted from coding into research", response.message)

    def test_repeated_same_activity(self):
        """4. Repeated same activity without false transitions preserves focus."""
        history = ContextHistory(maxlen=10)
        ctx = self._create_context("coding", "VSCode")

        history.record_transition(ctx, start_time=1000.0, end_time=1100.0)
        history.record_transition(ctx, start_time=1100.0, end_time=1200.0)

        understanding = self.analyzer.analyze(ctx, 1200.0, history, now=1230.0)
        response = self.assistant.respond(understanding)

        self.assertEqual(response.category, "focus")
        self.assertIn("focused", response.message)

    def test_generic_activity_names(self):
        """5. Non-application-specific activity names are handled naturally."""
        history = ContextHistory(maxlen=10)
        ctx_cad = self._create_context("3D modeling", "Blender")
        ctx_music = self._create_context("audio mixing", "Audacity")

        history.record_transition(ctx_cad, start_time=1000.0, end_time=1200.0)
        history.record_transition(ctx_music, start_time=1200.0, end_time=1260.0)

        understanding = self.analyzer.analyze(ctx_cad, 1260.0, history, now=1300.0)
        response = self.assistant.respond(understanding)

        self.assertEqual(response.category, "resumption")
        self.assertIn("back to your 3D modeling task after a short audio mixing session", response.message)

    def test_structured_serialization(self):
        """6. to_dict produces complete structured dictionary for future AI layers."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1045.0)

        response = self.assistant.respond(understanding)
        d = response.to_dict()

        self.assertIn("message", d)
        self.assertIn("category", d)
        self.assertIn("confidence", d)
        self.assertIn("supporting_context", d)
        self.assertIn("timestamp", d)
        self.assertEqual(d["category"], "focus")


if __name__ == "__main__":
    unittest.main()
