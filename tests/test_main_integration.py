import unittest
import time
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer
from assistant import CompanionAssistant
from main import get_summary_times, close_active_context, format_time, format_timestamp


class TestMainIntegration(unittest.TestCase):

    def setUp(self):
        self.history = ContextHistory(maxlen=10)
        self.analyzer = ContextualUnderstandingAnalyzer()
        self.assistant = CompanionAssistant()
        self.state = {
            "last_window": None,
            "last_context": None,
            "last_confidence": 0,
            "context_start_time": None,
            "window_start_time": None,
            "last_ocr_time": None,
            "activity_time": {},
            "candidate_context": None,
            "candidate_start_time": None,
            "current_context": None,
            "previous_context_obj": None,
            "history": self.history,
        }

    def test_get_summary_times_ongoing(self):
        t0 = 1000.0
        self.state["last_context"] = "coding"
        self.state["context_start_time"] = t0
        self.state["activity_time"]["coding"] = 20.0

        # Current time t0 + 10s
        times = get_summary_times(self.state, t0 + 10.0)
        self.assertEqual(times["coding"], 30.0)
        # Verify state is not mutated
        self.assertEqual(self.state["context_start_time"], t0)
        self.assertEqual(self.state["activity_time"]["coding"], 20.0)

    def test_close_active_context(self):
        t0 = 1000.0
        t1 = 1050.0
        obs = RawObservation(window_title="VSCode", ocr_text="code")
        interp = InterpretedContext(activity="coding", confidence=3)
        ctx = DesktopContext(observation=obs, interpretation=interp)

        self.state["last_context"] = "coding"
        self.state["context_start_time"] = t0
        self.state["current_context"] = ctx
        self.state["previous_context_obj"] = ctx

        close_active_context(self.state, t1)

        self.assertEqual(self.state["activity_time"]["coding"], 50.0)
        self.assertEqual(len(self.history), 1)
        record = self.history.get_recent()[0]
        self.assertEqual(record.activity, "coding")
        self.assertEqual(record.duration, 50.0)
        self.assertEqual(record.start_time, t0)
        self.assertEqual(record.end_time, t1)

    def test_main_context_analyzer_integration(self):
        t0 = 1000.0
        obs = RawObservation(window_title="VSCode", ocr_text="code")
        interp = InterpretedContext(activity="coding", confidence=3)
        ctx = DesktopContext(observation=obs, interpretation=interp)

        self.state["current_context"] = ctx
        self.state["context_start_time"] = t0

        understanding = self.analyzer.analyze(
            current_context=self.state["current_context"],
            context_start_time=self.state["context_start_time"],
            history=self.state["history"],
            now=t0 + 60.0,
        )

        self.assertEqual(understanding.current_activity, "coding")
        self.assertEqual(understanding.pattern_type, "focused_session")
        self.assertEqual(understanding.current_duration, 60.0)

    def test_main_assistant_integration(self):
        t0 = 1000.0
        obs = RawObservation(window_title="VSCode", ocr_text="code")
        interp = InterpretedContext(activity="coding", confidence=3)
        ctx = DesktopContext(observation=obs, interpretation=interp)

        self.state["current_context"] = ctx
        self.state["context_start_time"] = t0

        understanding = self.analyzer.analyze(
            current_context=self.state["current_context"],
            context_start_time=self.state["context_start_time"],
            history=self.state["history"],
            now=t0 + 60.0,
        )
        response = self.assistant.respond(understanding)

        self.assertEqual(response.category, "focus")
        self.assertIn("focused on the same coding activity", response.message)

    def test_formatting_helpers(self):
        self.assertEqual(format_time(45), "45s")
        self.assertEqual(format_time(125), "2m")
        self.assertEqual(format_time(3665), "1h 1m")
        self.assertTrue(len(format_timestamp(time.time())) > 0)


if __name__ == "__main__":
    unittest.main()
