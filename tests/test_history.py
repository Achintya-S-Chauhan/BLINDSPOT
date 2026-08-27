import unittest
import time
import threading
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextRecord, ContextHistory


class TestContextHistory(unittest.TestCase):

    def _create_mock_context(self, activity="coding", title="test_app", ocr="print('hello')"):
        obs = RawObservation(window_title=title, ocr_text=ocr)
        interp = InterpretedContext(activity=activity, confidence=3)
        return DesktopContext(observation=obs, interpretation=interp)

    def test_record_transition_and_get_recent(self):
        history = ContextHistory(maxlen=10)
        ctx1 = self._create_mock_context("coding", "VSCode")
        ctx2 = self._create_mock_context("browsing", "Chrome")

        t0 = time.time()
        t1 = t0 + 10.0
        t2 = t1 + 5.0

        rec1 = history.record_transition(ctx1, start_time=t0, end_time=t1)
        rec2 = history.record_transition(ctx2, start_time=t1, end_time=t2)

        self.assertEqual(len(history), 2)
        self.assertEqual(rec1.activity, "coding")
        self.assertEqual(rec1.duration, 10.0)
        self.assertEqual(rec2.activity, "browsing")
        self.assertEqual(rec2.duration, 5.0)

        recent = history.get_recent(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].activity, "browsing")

    def test_bounded_capacity_eviction(self):
        history = ContextHistory(maxlen=3)
        for i in range(5):
            ctx = self._create_mock_context(activity=f"task_{i}")
            history.record_transition(ctx, start_time=float(i), end_time=float(i + 1))

        self.assertEqual(len(history), 3)
        recent = history.get_recent()
        self.assertEqual([r.activity for r in recent], ["task_2", "task_3", "task_4"])

    def test_timeline_serialization(self):
        history = ContextHistory(maxlen=5)
        ctx = self._create_mock_context("coding", "VSCode", "def foo(): pass")
        history.record_transition(ctx, start_time=100.0, end_time=150.0)

        timeline = history.get_timeline()
        self.assertEqual(len(timeline), 1)
        item = timeline[0]
        self.assertEqual(item["activity"], "coding")
        self.assertEqual(item["window_title"], "VSCode")
        self.assertEqual(item["duration"], 50.0)
        self.assertIn("observation", item["context"])
        self.assertEqual(item["context"]["observation"]["ocr_text"], "def foo(): pass")

    def test_thread_safety(self):
        history = ContextHistory(maxlen=50)
        threads = []

        def worker(idx):
            for j in range(20):
                ctx = self._create_mock_context(f"thread_{idx}_task_{j}")
                history.record_transition(ctx, start_time=0.0, end_time=1.0)
                _ = history.get_recent()

        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(history), 50)


if __name__ == "__main__":
    unittest.main()
