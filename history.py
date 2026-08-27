from dataclasses import dataclass, field, asdict
from collections import deque
import threading
import time
from typing import Optional, List, Dict, Any
from context import DesktopContext


@dataclass
class ContextRecord:
    """Represents a discrete context episode over a time interval."""
    context: DesktopContext
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def close(self, end_time: float):
        """Close this episode at end_time and compute duration."""
        self.end_time = end_time
        self.duration = max(0.0, end_time - self.start_time)

    @property
    def activity(self) -> str:
        return self.context.activity

    @property
    def window_title(self) -> str:
        return self.context.window_title

    @property
    def confidence(self) -> int:
        return self.context.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "activity": self.activity,
            "window_title": self.window_title,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "context": self.context.to_dict(),
            "metadata": self.metadata,
        }


class ContextHistory:
    """
    Thread-safe, bounded in-memory history of desktop context transitions.
    Keeps only significant context episodes rather than every raw OCR frame.
    """

    def __init__(self, maxlen: int = 50):
        self.maxlen = maxlen
        self._records: deque[ContextRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add_record(self, record: ContextRecord) -> None:
        """Add a completed or updated context record."""
        with self._lock:
            self._records.append(record)

    def record_transition(
        self,
        context: DesktopContext,
        start_time: float,
        end_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextRecord:
        """Create, close, and store a completed context episode."""
        record = ContextRecord(
            context=context,
            start_time=start_time,
            end_time=end_time,
            duration=max(0.0, end_time - start_time),
            metadata=metadata or {},
        )
        with self._lock:
            self._records.append(record)
        return record

    def get_recent(self, limit: Optional[int] = None) -> List[ContextRecord]:
        """Return a list of recent records (oldest to newest within limit)."""
        with self._lock:
            items = list(self._records)
            if limit is not None and limit > 0:
                return items[-limit:]
            return items

    def get_timeline(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return serializable summary records suitable for future AI context."""
        records = self.get_recent(limit=limit)
        return [r.to_dict() for r in records]

    def clear(self) -> None:
        """Clear all stored history records."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
