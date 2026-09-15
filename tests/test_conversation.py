import unittest
import threading
from conversation import ConversationTurn, ConversationSession


class TestConversationSession(unittest.TestCase):

    def test_1_turn_creation_and_serialization(self):
        """Test ConversationTurn fields and dictionary serialization."""
        turn = ConversationTurn(role="user", content="Hello BLINDSPOT", timestamp=1000.0)
        self.assertEqual(turn.role, "user")
        self.assertEqual(turn.content, "Hello BLINDSPOT")
        self.assertEqual(turn.timestamp, 1000.0)

        d = turn.to_dict()
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "Hello BLINDSPOT")
        self.assertEqual(d["timestamp"], 1000.0)

    def test_2_session_add_messages(self):
        """Test recording user and assistant messages in session."""
        session = ConversationSession(maxlen=10)
        self.assertTrue(session.is_empty)
        self.assertEqual(len(session), 0)

        t1 = session.add_user_message("What am I doing?")
        t2 = session.add_assistant_message("You are coding in VS Code.")

        self.assertFalse(session.is_empty)
        self.assertEqual(len(session), 2)
        self.assertEqual(t1.role, "user")
        self.assertEqual(t1.content, "What am I doing?")
        self.assertEqual(t2.role, "assistant")
        self.assertEqual(t2.content, "You are coding in VS Code.")

    def test_3_session_chronological_order(self):
        """Test that get_turns returns items chronologically (oldest to newest)."""
        session = ConversationSession(maxlen=10)
        session.add_user_message("Q1", timestamp=100.0)
        session.add_assistant_message("A1", timestamp=101.0)
        session.add_user_message("Q2", timestamp=102.0)
        session.add_assistant_message("A2", timestamp=103.0)

        turns = session.get_turns()
        self.assertEqual(len(turns), 4)
        self.assertEqual([t.content for t in turns], ["Q1", "A1", "Q2", "A2"])

    def test_4_session_limit(self):
        """Test that get_turns and get_history_dicts respect limit parameter."""
        session = ConversationSession(maxlen=10)
        for i in range(6):
            session.add_user_message(f"Msg {i}")

        recent = session.get_turns(limit=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual([t.content for t in recent], ["Msg 3", "Msg 4", "Msg 5"])

        dicts = session.get_history_dicts(limit=2)
        self.assertEqual(len(dicts), 2)
        self.assertEqual([d["content"] for d in dicts], ["Msg 4", "Msg 5"])

    def test_5_bounded_history(self):
        """Test that conversation session enforces maxlen bounds automatically."""
        max_size = 4
        session = ConversationSession(maxlen=max_size)

        for i in range(10):
            session.add_user_message(f"Message {i}")

        self.assertEqual(len(session), max_size)
        turns = session.get_turns()
        self.assertEqual([t.content for t in turns], ["Message 6", "Message 7", "Message 8", "Message 9"])

    def test_6_session_clear(self):
        """Test clearing conversation history."""
        session = ConversationSession(maxlen=10)
        session.add_user_message("What's up?")
        session.add_assistant_message("All systems normal.")
        self.assertEqual(len(session), 2)

        session.clear()
        self.assertEqual(len(session), 0)
        self.assertTrue(session.is_empty)
        self.assertEqual(session.get_turns(), [])
        self.assertEqual(session.get_history_dicts(), [])

    def test_7_thread_safety(self):
        """Test thread-safety under concurrent message additions."""
        session = ConversationSession(maxlen=100)
        num_threads = 8
        msgs_per_thread = 10

        def worker(thread_id):
            for j in range(msgs_per_thread):
                session.add_user_message(f"T{thread_id}-M{j}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(session), num_threads * msgs_per_thread)


if __name__ == "__main__":
    unittest.main()
