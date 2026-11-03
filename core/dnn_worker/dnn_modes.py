"""ResiMem dnn modes module."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
import time


class DNNMode(Enum):
    """DNNMode implementation."""

    FULL = "full"
    TINY = "tiny"
    MINIMAL = "minimal"


@dataclass
class DNNModelConfig:
    """DNNModelConfig implementation."""

    mode: DNNMode
    memory_gb: float
    models: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    @staticmethod
    def create_full_config() -> "DNNModelConfig":
        """Create full config."""
        return DNNModelConfig(
            mode=DNNMode.FULL,
            memory_gb=10.0,
            models={
                "vision": "yolov8x",
                "vision_track": "deepsort",
                "text": "bert-base-uncased",
                "speech": "whisper-medium",
                "ocr": "paddleocr-full",
            },
            description="create full config:,,",
        )

    @staticmethod
    def create_tiny_config() -> "DNNModelConfig":
        """Create tiny config."""
        return DNNModelConfig(
            mode=DNNMode.TINY,
            memory_gb=0.2,
            models={
                "vision": "yolov8n",
                "text": "distilbert-base-uncased",
            },
            description="create tiny config:,",
        )

    @staticmethod
    def create_minimal_config() -> "DNNModelConfig":
        """Create minimal config."""
        return DNNModelConfig(
            mode=DNNMode.MINIMAL,
            memory_gb=0.05,
            models={
                "motion": "opencv_bgsubtract",
            },
            description="create minimal config:",
        )


@dataclass
class DNNWorkerStatus:
    """DNNWorkerStatus implementation."""

    worker_id: str
    current_mode: DNNMode
    memory_used_gb: float
    is_degraded: bool
    last_mode_change: float = field(default_factory=time.time)
    degradation_count: int = 0
    restoration_count: int = 0

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "current_mode": self.current_mode.value,
            "memory_used_gb": self.memory_used_gb,
            "is_degraded": self.is_degraded,
            "last_mode_change": self.last_mode_change,
            "degradation_count": self.degradation_count,
            "restoration_count": self.restoration_count,
        }


@dataclass
class DegradeRequest:
    """DegradeRequest implementation."""

    worker_id: str
    target_mode: DNNMode
    reason: str
    priority: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class DegradeResponse:
    """DegradeResponse implementation."""

    worker_id: str
    success: bool
    freed_memory_gb: float
    old_mode: DNNMode
    new_mode: DNNMode
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "success": self.success,
            "freed_memory_gb": self.freed_memory_gb,
            "old_mode": self.old_mode.value,
            "new_mode": self.new_mode.value,
            "message": self.message,
            "timestamp": self.timestamp,
        }


FULL_CONFIG = DNNModelConfig.create_full_config()
TINY_CONFIG = DNNModelConfig.create_tiny_config()
MINIMAL_CONFIG = DNNModelConfig.create_minimal_config()


MODE_MEMORY_DELTA = {
    (DNNMode.FULL, DNNMode.TINY): 9.8,
    (DNNMode.FULL, DNNMode.MINIMAL): 9.95,
    (DNNMode.TINY, DNNMode.MINIMAL): 0.15,
    (DNNMode.TINY, DNNMode.FULL): -9.8,
    (DNNMode.MINIMAL, DNNMode.TINY): -0.15,
    (DNNMode.MINIMAL, DNNMode.FULL): -9.95,
}


def get_memory_delta(from_mode: DNNMode, to_mode: DNNMode) -> float:
    """Return memory delta."""
    return MODE_MEMORY_DELTA.get((from_mode, to_mode), 0.0)
