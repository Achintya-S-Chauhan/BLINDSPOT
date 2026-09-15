import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ConversationTurn:
    """Represents a single conversational turn (user or assistant)."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized dictionary representation of the turn."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class ConversationSession:
    """
    Thread-safe, bounded in-memory session storing conversation turns.
    Differentiates conversational dialogue from desktop context history.
    """

    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self._turns: deque[ConversationTurn] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add_turn(self, role: str, content: str, timestamp: Optional[float] = None) -> ConversationTurn:
        """Add a conversation turn with role 'user' or 'assistant'."""
        ts = time.time() if timestamp is None else timestamp
        turn = ConversationTurn(role=role, content=content, timestamp=ts)
        with self._lock:
            self._turns.append(turn)
        return turn

    def add_user_message(self, content: str, timestamp: Optional[float] = None) -> ConversationTurn:
        """Convenience method to record a user message."""
        return self.add_turn("user", content, timestamp=timestamp)

    def add_assistant_message(self, content: str, timestamp: Optional[float] = None) -> ConversationTurn:
        """Convenience method to record an assistant message."""
        return self.add_turn("assistant", content, timestamp=timestamp)

    def get_turns(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """Return chronological list of recent turns (oldest to newest within limit)."""
        with self._lock:
            items = list(self._turns)
            if limit is not None and limit > 0:
                return items[-limit:]
            return items

    def get_history_dicts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return serialized list of turn dicts for LLM payload or API formatting."""
        return [t.to_dict() for t in self.get_turns(limit=limit)]

    def clear(self) -> None:
        """Clear all conversation history in this session."""
        with self._lock:
            self._turns.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._turns) == 0
