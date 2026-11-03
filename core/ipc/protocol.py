"""ResiMem protocol module."""

from dataclasses import dataclass, fields
from enum import Enum
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """MessageType implementation."""

    # vLLM → GRM
    REGISTER_VLLM = 1
    REQUEST_EXPANSION = 2
    HEARTBEAT_VLLM = 3
    UNREGISTER_VLLM = 4

    # DNN → GRM
    REGISTER_DNN = 5
    MEMORY_RELEASED = 6
    HEARTBEAT_DNN = 7
    UNREGISTER_DNN = 8

    # GRM → vLLM
    EXPANSION_APPROVED = 9
    EXPANSION_DENIED = 10

    # GRM → DNN
    REQUEST_RELEASE = 11
    CANCEL_REQUEST = 12

    DNN_EVICT_REQUEST = 15
    DNN_EVICT_RESPONSE = 16
    DNN_RESTORE_REQUEST = 17
    DNN_RESTORE_RESPONSE = 18

    ACK = 13
    ERROR = 14


class WorkerStatus(Enum):
    """WorkerStatus implementation."""

    # vLLM Worker
    VLLM_IDLE = 0
    VLLM_REQUESTING = 1
    VLLM_EXPANDING = 2
    VLLM_RUNNING = 3

    # DNN Worker
    DNN_RUNNING = 10
    DNN_RELEASING = 11
    DNN_PAUSED = 12

    CRASHED = 99
    OFFLINE = 100


@dataclass
class Message:
    """Message implementation."""

    msg_type: MessageType
    sender_id: str
    timestamp: float
    payload: Dict[str, Any]

    @classmethod
    def create(
        cls, msg_type: MessageType, sender_id: str, payload: Dict[str, Any] = None
    ):
        """Create."""
        return cls(
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=time.time(),
            payload=payload or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """From dict."""
        raw_type = data.get("msg_type")
        if raw_type is None:
            raise ValueError(
                "Message.from_dict: from dict: missing required value msg_type "
            )
        try:
            msg_type = MessageType(raw_type)
        except (ValueError, KeyError):
            raise ValueError(f"Message.from_dict: from dict msg_type: {raw_type}")

        sender_id = data.get("sender_id", "unknown")
        timestamp = data.get("timestamp", 0.0)
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        return cls(
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=timestamp,
            payload=payload,
        )


@dataclass
class ExpansionRequest:
    """ExpansionRequest implementation."""

    num_blocks: int
    size_bytes: int
    priority: int = 5
    timeout_ms: int = 50
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = f"exp_{int(time.time() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "num_blocks": self.num_blocks,
            "size_bytes": self.size_bytes,
            "priority": self.priority,
            "timeout_ms": self.timeout_ms,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpansionRequest":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ReleaseRequest:
    """ReleaseRequest implementation."""

    size_bytes: int
    timeout_ms: int = 100
    request_id: Optional[str] = None
    priority: int = 5

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = f"rel_{int(time.time() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "size_bytes": self.size_bytes,
            "timeout_ms": self.timeout_ms,
            "request_id": self.request_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReleaseRequest":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class WorkerInfo:
    """WorkerInfo implementation."""

    worker_id: str
    worker_type: str  # "vllm" or "dnn"
    status: WorkerStatus
    memory_used_gb: float = 0.0
    memory_capacity_gb: float = 0.0
    last_heartbeat: float = 0.0

    num_blocks: Optional[int] = None
    kv_cache_size_gb: Optional[float] = None

    memory_releasable_gb: Optional[float] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        self.last_heartbeat = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "status": self.status.value,
            "memory_used_gb": self.memory_used_gb,
            "memory_capacity_gb": self.memory_capacity_gb,
            "last_heartbeat": self.last_heartbeat,
            "num_blocks": self.num_blocks,
            "kv_cache_size_gb": self.kv_cache_size_gb,
            "memory_releasable_gb": self.memory_releasable_gb,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerInfo":
        """From dict."""
        data = data.copy()

        raw_status = data.get("status")
        if raw_status is None:
            data["status"] = WorkerStatus.VLLM_IDLE
        elif isinstance(raw_status, WorkerStatus):
            pass
        else:
            try:
                data["status"] = WorkerStatus(raw_status)
            except (ValueError, KeyError):
                logger.warning(f"from dict WorkerStatus : {raw_status},")
                data["status"] = WorkerStatus.VLLM_IDLE

        import inspect

        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def update_heartbeat(self):
        """Update heartbeat."""
        self.last_heartbeat = time.time()

    def is_alive(self, timeout_seconds: float = 5.0) -> bool:
        """Return whether alive."""
        return (time.time() - self.last_heartbeat) < timeout_seconds


def create_register_message(
    worker_id: str, worker_type: str, worker_info: WorkerInfo
) -> Message:
    """Create register message."""
    msg_type = (
        MessageType.REGISTER_VLLM if worker_type == "vllm" else MessageType.REGISTER_DNN
    )
    return Message.create(
        msg_type=msg_type, sender_id=worker_id, payload=worker_info.to_dict()
    )


def create_heartbeat_message(
    worker_id: str, worker_type: str, memory_used_gb: float
) -> Message:
    """Create heartbeat message."""
    msg_type = (
        MessageType.HEARTBEAT_VLLM
        if worker_type == "vllm"
        else MessageType.HEARTBEAT_DNN
    )
    return Message.create(
        msg_type=msg_type,
        sender_id=worker_id,
        payload={"memory_used_gb": memory_used_gb},
    )


def create_expansion_request_message(
    worker_id: str, request: ExpansionRequest
) -> Message:
    """Create expansion request message."""
    return Message.create(
        msg_type=MessageType.REQUEST_EXPANSION,
        sender_id=worker_id,
        payload=request.to_dict(),
    )


def create_release_request_message(request: ReleaseRequest) -> Message:
    """Create release request message."""
    return Message.create(
        msg_type=MessageType.REQUEST_RELEASE, sender_id="grm", payload=request.to_dict()
    )


def create_approval_message(num_blocks: int) -> Message:
    """Create approval message."""
    return Message.create(
        msg_type=MessageType.EXPANSION_APPROVED,
        sender_id="grm",
        payload={"num_blocks": num_blocks},
    )


def create_denial_message(reason: str) -> Message:
    """Create denial message."""
    return Message.create(
        msg_type=MessageType.EXPANSION_DENIED,
        sender_id="grm",
        payload={"reason": reason},
    )


def create_memory_released_message(
    worker_id: str, size_bytes: int, request_id: str
) -> Message:
    """Create memory released message."""
    return Message.create(
        msg_type=MessageType.MEMORY_RELEASED,
        sender_id=worker_id,
        payload={"size_bytes": size_bytes, "request_id": request_id},
    )


@dataclass
class DNNWorkerConfig:
    """DNNWorkerConfig implementation."""

    worker_id: str
    worker_type: str = "dnn_worker"

    priority: int = 50
    evictable: bool = True
    residency_class: str = "elastic_serving"
    reload_cost_band: str = "medium"
    user_visibility_band: str = "medium"
    request_pressure_band: str = "runtime_pending"

    expected_memory_gb: float = 1.0
    current_memory_gb: float = 0.0

    task_type: str = "unknown"  # "vision", "audio", "text", "multimodal"
    task_name: str = "unknown"

    status: str = "active"  # "active", "evicted", "loading"
    last_used_time: float = 0.0

    def __post_init__(self):
        if self.last_used_time == 0.0:
            self.last_used_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "priority": self.priority,
            "evictable": self.evictable,
            "residency_class": self.residency_class,
            "reload_cost_band": self.reload_cost_band,
            "user_visibility_band": self.user_visibility_band,
            "request_pressure_band": self.request_pressure_band,
            "expected_memory_gb": self.expected_memory_gb,
            "current_memory_gb": self.current_memory_gb,
            "task_type": self.task_type,
            "task_name": self.task_name,
            "status": self.status,
            "last_used_time": self.last_used_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNWorkerConfig":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DNNEvictionPolicy:
    """DNNEvictionPolicy implementation."""

    min_priority_to_evict: int = 0
    prefer_least_recently_used: bool = True

    safety_buffer_gb: float = 0.5
    min_dnn_to_keep: int = 0

    auto_restore_on_idle: bool = True
    idle_threshold_s: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "min_priority_to_evict": self.min_priority_to_evict,
            "prefer_least_recently_used": self.prefer_least_recently_used,
            "safety_buffer_gb": self.safety_buffer_gb,
            "min_dnn_to_keep": self.min_dnn_to_keep,
            "auto_restore_on_idle": self.auto_restore_on_idle,
            "idle_threshold_s": self.idle_threshold_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNEvictionPolicy":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DNNEvictRequest:
    """DNNEvictRequest implementation."""

    worker_id: str
    request_id: str
    reason: str = "memory_pressure"  # "memory_pressure", "manual", "policy"
    required_memory_gb: float = 0.0
    timeout_s: float = 30.0

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"evict_{int(time.time() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "request_id": self.request_id,
            "reason": self.reason,
            "required_memory_gb": self.required_memory_gb,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNEvictRequest":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DNNEvictResponse:
    """DNNEvictResponse implementation."""

    worker_id: str
    request_id: str
    success: bool
    freed_memory_gb: float = 0.0
    eviction_time_s: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "request_id": self.request_id,
            "success": self.success,
            "freed_memory_gb": self.freed_memory_gb,
            "eviction_time_s": self.eviction_time_s,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNEvictResponse":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DNNRestoreRequest:
    """DNNRestoreRequest implementation."""

    worker_id: str
    request_id: str
    timeout_s: float = 60.0

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"restore_{int(time.time() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "request_id": self.request_id,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNRestoreRequest":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DNNRestoreResponse:
    """DNNRestoreResponse implementation."""

    worker_id: str
    request_id: str
    success: bool
    restore_time_s: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "worker_id": self.worker_id,
            "request_id": self.request_id,
            "success": self.success,
            "restore_time_s": self.restore_time_s,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNRestoreResponse":
        """From dict."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def create_dnn_evict_request_message(request: DNNEvictRequest) -> Message:
    """Create dnn evict request message."""
    return Message.create(
        msg_type=MessageType.DNN_EVICT_REQUEST,
        sender_id="grm",
        payload=request.to_dict(),
    )


def create_dnn_evict_response_message(response: DNNEvictResponse) -> Message:
    """Create dnn evict response message."""
    return Message.create(
        msg_type=MessageType.DNN_EVICT_RESPONSE,
        sender_id=response.worker_id,
        payload=response.to_dict(),
    )


def create_dnn_restore_request_message(request: DNNRestoreRequest) -> Message:
    """Create dnn restore request message."""
    return Message.create(
        msg_type=MessageType.DNN_RESTORE_REQUEST,
        sender_id="grm",
        payload=request.to_dict(),
    )


def create_dnn_restore_response_message(response: DNNRestoreResponse) -> Message:
    """Create dnn restore response message."""
    return Message.create(
        msg_type=MessageType.DNN_RESTORE_RESPONSE,
        sender_id=response.worker_id,
        payload=response.to_dict(),
    )
