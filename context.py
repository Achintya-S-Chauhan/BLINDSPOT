from dataclasses import dataclass, field, asdict
import time
from typing import Optional, Dict, Any


@dataclass
class RawObservation:
    """Raw perceived data from the desktop environment."""
    window_title: str
    ocr_text: str
    window_rect: Optional[Dict[str, int]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpretedContext:
    """Heuristic or model-derived activity and confidence."""
    activity: str
    confidence: int  # 1 (low) to 3 (high)


@dataclass
class DesktopContext:
    """Complete snapshot separating raw observations from interpreted context."""
    observation: RawObservation
    interpretation: InterpretedContext
    timestamp: float = field(default_factory=time.time)

    @property
    def activity(self) -> str:
        return self.interpretation.activity

    @property
    def confidence(self) -> int:
        return self.interpretation.confidence

    @property
    def window_title(self) -> str:
        return self.observation.window_title

    @property
    def ocr_text(self) -> str:
        return self.observation.ocr_text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to a dictionary."""
        return asdict(self)
