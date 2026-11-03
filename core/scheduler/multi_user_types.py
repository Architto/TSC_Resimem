"""ResiMem multi user types module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Set
from datetime import datetime


class UserState(Enum):
    """UserState implementation."""

    ACTIVE = "active"
    WARM = "warm"
    COLD = "cold"


@dataclass
class UserInfo:
    """UserInfo implementation."""

    user_id: str = "default_user"
    session_id: Optional[str] = None

    first_request_time: datetime = field(default_factory=datetime.now)
    last_activity_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0

    def update_activity(self):
        """Update activity."""
        self.last_activity_time = datetime.now()
        self.total_requests += 1


@dataclass
class UserMetrics:
    """UserMetrics implementation."""

    user_id: str

    num_sequences: int = 0
    num_blocks_gpu: int = 0
    num_blocks_cpu: int = 0
    total_tokens: int = 0

    swap_out_count: int = 0
    swap_in_count: int = 0
    total_swap_out_latency_ms: float = 0.0
    total_swap_in_latency_ms: float = 0.0

    def get_avg_swap_out_latency(self) -> float:
        """Return avg swap out latency."""
        if self.swap_out_count == 0:
            return 0.0
        return self.total_swap_out_latency_ms / self.swap_out_count

    def get_avg_swap_in_latency(self) -> float:
        """Return avg swap in latency."""
        if self.swap_in_count == 0:
            return 0.0
        return self.total_swap_in_latency_ms / self.swap_in_count


@dataclass
class SwapMetadata:
    """SwapMetadata implementation."""

    user_id: str

    sequence_ids: List[str]
    block_mapping: Dict[int, int]

    swap_out_time: datetime = field(default_factory=datetime.now)
    original_state: UserState = UserState.ACTIVE

    num_blocks_swapped: int = 0
    swap_out_latency_ms: float = 0.0


@dataclass
class UserStateInfo:
    """UserStateInfo implementation."""

    user_id: str

    state: UserState = UserState.COLD
    last_activity_time: datetime = field(default_factory=datetime.now)
    last_state_change_time: datetime = field(default_factory=datetime.now)

    sequence_ids: Set[str] = field(default_factory=set)
    metrics: UserMetrics = field(default_factory=lambda: None)

    swap_metadata: Optional[SwapMetadata] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = UserMetrics(user_id=self.user_id)

    def update_activity(self):
        """Update activity."""
        self.last_activity_time = datetime.now()

    def change_state(self, new_state: UserState):
        """Change state."""
        self.state = new_state
        self.last_state_change_time = datetime.now()

    def get_idle_time_seconds(self) -> float:
        """Return idle time seconds."""
        return (datetime.now() - self.last_activity_time).total_seconds()


@dataclass
class FocusManagerConfig:
    """FocusManagerConfig implementation."""

    cooling_off_period_s: float = 7.0
    check_interval_s: float = 1.0

    max_users_in_gpu: int = 1
    swap_batch_size_gb: float = 2.0

    target_swap_latency_ms: float = 100.0

    enable_swap: bool = True
    enable_logging: bool = True


@dataclass
class ExtendedChatRequest:
    """ExtendedChatRequest implementation."""

    messages: List[Dict]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    user: str = "default_user"
    session_id: Optional[str] = None

    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


def extract_user_id(request_dict: Dict) -> str:
    """Extract user id."""
    return request_dict.get("user", "default_user")


def extract_session_id(request_dict: Dict) -> Optional[str]:
    """Extract session id."""

    metadata = request_dict.get("metadata", {})
    return metadata.get("session_id", None)


__all__ = [
    "UserState",
    "UserInfo",
    "UserMetrics",
    "SwapMetadata",
    "UserStateInfo",
    "ExtendedChatRequest",
    "FocusManagerConfig",
    "extract_user_id",
    "extract_session_id",
]
