import unittest
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer, WorkflowUnderstanding


class TestContextualUnderstanding(unittest.TestCase):

    def setUp(self):
        self.analyzer = ContextualUnderstandingAnalyzer()

    def _create_context(self, activity: str, window_title: str = "TestApp", confidence: int = 3) -> DesktopContext:
        obs = RawObservation(window_title=window_title, ocr_text="")
        interp = InterpretedContext(activity=activity, confidence=confidence)
        return DesktopContext(observation=obs, interpretation=interp)

    def test_sequence_1_coding_only(self):
        """Sequence 1: Coding only (no prior history)."""
        ctx = self._create_context("coding", "Visual Studio Code - main.py")
        history = ContextHistory(maxlen=10)

        t_start = 1000.0
        now = 1045.0  # 45 seconds later

        understanding = self.analyzer.analyze(ctx, t_start, history, now=now)

        self.assertEqual(understanding.current_activity, "coding")
        self.assertEqual(understanding.current_window, "Visual Studio Code - main.py")
        self.assertEqual(understanding.current_duration, 45.0)
        self.assertEqual(understanding.pattern_type, "focused_session")
        self.assertEqual(understanding.confidence, 3)
        self.assertIn("coding", understanding.interpretation)
        self.assertIn("45s", understanding.interpretation)

    def test_sequence_2_coding_research_coding(self):
        """Sequence 2: Coding -> Research -> Coding (resumed task)."""
        history = ContextHistory(maxlen=10)
        ctx_code = self._create_context("coding", "Visual Studio Code - main.py")
        ctx_research = self._create_context("reading in a browser", "Stack Overflow - Google Chrome")

        # Episode 1: Coding from t=1000 to t=1180 (3m)
        history.record_transition(ctx_code, start_time=1000.0, end_time=1180.0)
        # Episode 2: Research from t=1180 to t=1265 (1m 25s)
        history.record_transition(ctx_research, start_time=1180.0, end_time=1265.0)

        # Current episode: Returned to Coding at t=1265, now t=1310 (45s active)
        understanding = self.analyzer.analyze(ctx_code, 1265.0, history, now=1310.0)

        self.assertEqual(understanding.current_activity, "coding")
        self.assertEqual(understanding.pattern_type, "resumed_task")
        self.assertEqual(understanding.confidence, 3)
        self.assertIn("Returned to coding", understanding.interpretation)
        self.assertIn("research session", understanding.interpretation)
        self.assertEqual(
            understanding.recent_activities,
            ["coding", "reading in a browser", "coding"],
        )

    def test_sequence_3_coding_research_another_activity(self):
        """Sequence 3: Coding -> Research -> another activity (workflow shift)."""
        history = ContextHistory(maxlen=10)
        ctx_code = self._create_context("coding", "Visual Studio Code - main.py")
        ctx_research = self._create_context("reading in a browser", "Google Chrome")
        ctx_terminal = self._create_context("working in a terminal", "PowerShell")

        # Episode 1: Coding
        history.record_transition(ctx_code, start_time=1000.0, end_time=1200.0)
        # Episode 2: Research
        history.record_transition(ctx_research, start_time=1200.0, end_time=1300.0)

        # Current episode: Terminal
        understanding = self.analyzer.analyze(ctx_terminal, 1300.0, history, now=1330.0)

        self.assertEqual(understanding.current_activity, "working in a terminal")
        self.assertEqual(understanding.pattern_type, "workflow_shift")
        self.assertIn("coding", understanding.interpretation)
        self.assertIn("reading in a browser", understanding.interpretation)
        self.assertIn("working in a terminal", understanding.interpretation)
        self.assertEqual(
            understanding.recent_activities,
            ["coding", "reading in a browser", "working in a terminal"],
        )

    def test_sequence_4_repeated_same_activity(self):
        """Sequence 4: Repeated same activity without unnecessary transitions."""
        history = ContextHistory(maxlen=10)
        ctx1 = self._create_context("coding", "VSCode - window1")
        ctx2 = self._create_context("coding", "VSCode - window2")

        history.record_transition(ctx1, start_time=1000.0, end_time=1100.0)
        history.record_transition(ctx2, start_time=1100.0, end_time=1200.0)

        # Active is also coding
        understanding = self.analyzer.analyze(ctx1, 1200.0, history, now=1250.0)

        self.assertEqual(understanding.current_activity, "coding")
        self.assertEqual(understanding.pattern_type, "focused_session")
        self.assertIn("Continuously engaged in coding", understanding.interpretation)
        self.assertEqual(understanding.recent_activities, ["coding"])

    def test_generic_activity_resumption(self):
        """Verify understanding works generically with arbitrary activity names, not just hardcoded strings."""
        history = ContextHistory(maxlen=10)
        ctx_doc = self._create_context("writing documentation", "Obsidian")
        ctx_chat = self._create_context("chatting", "Slack")

        history.record_transition(ctx_doc, start_time=1000.0, end_time=1200.0)
        history.record_transition(ctx_chat, start_time=1200.0, end_time=1260.0)

        understanding = self.analyzer.analyze(ctx_doc, 1260.0, history, now=1300.0)

        self.assertEqual(understanding.pattern_type, "resumed_task")
        self.assertIn("Returned to writing documentation", understanding.interpretation)
        self.assertIn("chatting session", understanding.interpretation)

    def test_serialization_structure(self):
        """Verify to_dict produces structured output suitable for AI consumption."""
        ctx = self._create_context("coding", "VSCode")
        history = ContextHistory(maxlen=10)
        understanding = self.analyzer.analyze(ctx, 1000.0, history, now=1060.0)

        d = understanding.to_dict()
        self.assertIn("current", d)
        self.assertIn("recent", d)
        self.assertIn("understanding", d)
        self.assertEqual(d["current"]["activity"], "coding")
        self.assertEqual(d["understanding"]["pattern_type"], "focused_session")
        self.assertEqual(d["understanding"]["confidence"], 3)


if __name__ == "__main__":
    unittest.main()
