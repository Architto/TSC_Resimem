"""ResiMem grm client module."""

import time
import logging
import threading
from typing import Optional

from ..ipc import (
    GRMSharedMemory,
    Message,
    MessageType,
    WorkerInfo,
    WorkerStatus,
    ExpansionRequest,
    create_register_message,
    create_heartbeat_message,
    create_expansion_request_message,
)


class GRMClient:
    """GRMClient implementation."""

    def __init__(
        self,
        worker_id: str,
        worker_type: str = "vllm",
        heartbeat_interval_s: float = 1.0,
    ):

        self.worker_id = worker_id
        self.worker_type = worker_type
        self.heartbeat_interval_s = heartbeat_interval_s

        self.logger = logging.getLogger(f"GRMClient_{worker_id}")

        try:
            self.shm = GRMSharedMemory(create=False)
            self.logger.info(f"Connected to GRM shared memory")
        except Exception as e:
            self.logger.error(f"Failed to connect to GRM: {e}")
            raise

        self.worker_info = WorkerInfo(
            worker_id=worker_id,
            worker_type=worker_type,
            status=(
                WorkerStatus.VLLM_IDLE
                if worker_type == "vllm"
                else WorkerStatus.DNN_RUNNING
            ),
        )

        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        self.stats = {
            "expansion_requests": 0,
            "approved": 0,
            "denied": 0,
            "timeouts": 0,
        }

    def register(self) -> bool:
        """Register."""
        try:
            msg = create_register_message(
                self.worker_id, self.worker_type, self.worker_info
            )
            success = self.shm.push_request(msg)

            if success:
                self.logger.info(f"✅ Registered to GRM as {self.worker_type} worker")

                self._start_heartbeat()
            else:
                self.logger.error("❌ Failed to register: message queue full")

            return success
        except Exception as e:
            self.logger.error(f"❌ Failed to register: {e}")
            return False

    def unregister(self) -> bool:
        """Unregister."""
        try:

            self._stop_heartbeat()

            msg_type = (
                MessageType.UNREGISTER_VLLM
                if self.worker_type == "vllm"
                else MessageType.UNREGISTER_DNN
            )
            msg = Message.create(
                msg_type=msg_type, sender_id=self.worker_id, payload={}
            )
            self.shm.push_request(msg)

            self.logger.info(f"✅ Unregistered from GRM")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to unregister: {e}")
            return False

    def request_expansion(self, request: ExpansionRequest) -> bool:
        """Request expansion."""
        self.stats["expansion_requests"] += 1

        try:

            self.worker_info.status = WorkerStatus.VLLM_REQUESTING

            msg = create_expansion_request_message(self.worker_id, request)
            success = self.shm.push_request(msg)

            if not success:
                self.logger.error("❌ Failed to send expansion request: queue full")
                self.worker_info.status = WorkerStatus.VLLM_IDLE
                return False

            self.logger.info(
                f"📤 Sent expansion request: {request.num_blocks} blocks, "
                f"{request.size_bytes / 1024**3:.2f} GB"
            )

            result = self._wait_for_response(request.timeout_ms)

            if result:
                self.worker_info.status = WorkerStatus.VLLM_EXPANDING
                self.stats["approved"] += 1
                self.logger.info(f"✅ Expansion approved by GRM")
            else:
                self.worker_info.status = WorkerStatus.VLLM_IDLE
                self.logger.warning(f"❌ Expansion denied or timeout")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error during expansion request: {e}")
            self.worker_info.status = WorkerStatus.VLLM_IDLE
            return False

    def _wait_for_response(self, timeout_ms: int) -> bool:

        start_time = time.time()

        while (time.time() - start_time) * 1000 < timeout_ms:

            msg = self.shm.pop_response_for_worker(self.worker_id)

            if msg is not None:
                if msg.msg_type == MessageType.EXPANSION_APPROVED:
                    return True
                elif msg.msg_type == MessageType.EXPANSION_DENIED:
                    reason = msg.payload.get("reason", "unknown")
                    self.logger.warning(f"Expansion denied: {reason}")
                    self.stats["denied"] += 1
                    return False

            time.sleep(0.001)

        self.logger.warning(f"⏰ Expansion request timeout after {timeout_ms}ms")
        self.stats["timeouts"] += 1
        return False

    def update_memory_usage(self, memory_used_gb: float):
        """Update memory usage."""
        self.worker_info.memory_used_gb = memory_used_gb

    def update_kv_cache_info(self, num_blocks: int, kv_cache_size_gb: float):
        """Update kv cache info."""
        if self.worker_type == "vllm":
            self.worker_info.num_blocks = num_blocks
            self.worker_info.kv_cache_size_gb = kv_cache_size_gb

    def _start_heartbeat(self):

        if self.heartbeat_running:
            return

        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        self.logger.debug(
            f"Started heartbeat thread (interval={self.heartbeat_interval_s}s)"
        )

    def _stop_heartbeat(self):

        self.heartbeat_running = False
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=2.0)
            self.logger.debug("Stopped heartbeat thread")

    def _heartbeat_loop(self):

        while self.heartbeat_running:
            try:

                msg = create_heartbeat_message(
                    self.worker_id, self.worker_type, self.worker_info.memory_used_gb
                )
                self.shm.push_request(msg)

                self.worker_info.update_heartbeat()

            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

            time.sleep(self.heartbeat_interval_s)

    def get_stats(self) -> dict:
        """Return stats."""
        return {
            **self.stats,
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "memory_used_gb": self.worker_info.memory_used_gb,
        }

    def is_connected(self) -> bool:
        """Return whether connected."""
        try:

            magic = self.shm._read_uint64(0)
            return magic == self.shm.MAGIC_NUMBER
        except Exception:
            return False

    def close(self):
        """Close."""
        self._stop_heartbeat()
        self.shm.close()
        self.logger.info("GRM Client closed")

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unregister()
        self.close()
