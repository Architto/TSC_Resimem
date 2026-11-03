"""ResiMem dnn worker module."""

import time
import torch
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from ..ipc.protocol import (
    DNNWorkerConfig,
    DNNEvictRequest,
    DNNEvictResponse,
    DNNRestoreRequest,
    DNNRestoreResponse,
    WorkerInfo,
    WorkerStatus,
)
from ..grm.grm_client import GRMClient
from ..monitoring import mark_global

logger = logging.getLogger(__name__)


class DNNWorker(ABC):
    """DNNWorker implementation."""

    def __init__(
        self,
        worker_id: str,
        config: DNNWorkerConfig,
        grm_client: Optional[GRMClient] = None,
        device: str = "cuda",
    ):

        self.worker_id = worker_id
        self.config = config
        self.grm = grm_client
        self.device = device

        self.model = None
        self.is_loaded = False
        self.is_running = False

        logger.info(f"init DNN Worker: {worker_id} ({config.task_name})")
        logger.info(f"  init: {config.priority}")
        logger.info(f"  init: evicted: {config.evictable}")
        logger.info(f"  init: {config.residency_class}")
        logger.info(f"  init: restored: {config.reload_cost_band}")
        logger.info(f"  init: {config.user_visibility_band}")
        logger.info(f"  init: {config.expected_memory_gb} GB")

    @abstractmethod
    def load_model(self):
        """Load model."""
        raise NotImplementedError

    @abstractmethod
    def unload_model(self):
        """Unload model."""
        raise NotImplementedError

    def start(self, run_message_loop=True):
        """Start the service."""
        logger.info(f"start: started DNN Worker: {self.worker_id}")

        self._safe_load_model()

        if self.grm:
            self.register_to_grm()

        self.is_running = True

        if run_message_loop and self.grm:
            import threading

            self._message_thread = threading.Thread(
                target=self._message_loop,
                daemon=True,
                name=f"DNNWorker-{self.worker_id}-MessageLoop",
            )
            self._message_thread.start()
            logger.info(f"DNN Worker {self.worker_id} start: started")

        logger.info(f"DNN Worker {self.worker_id} start: completed successfully")

    def stop(self):
        """Stop the service."""
        logger.info(f"stop: stopped DNN Worker: {self.worker_id}")

        self.is_running = False
        if hasattr(self, "_message_thread") and self._message_thread:
            self._message_thread.join(timeout=1.0)
            logger.info(f"DNN Worker {self.worker_id} stop: stopped")

        if self.is_loaded:
            self._safe_unload_model()

        if self.grm:
            self.grm.unregister()

        logger.info(f"DNN Worker {self.worker_id} stop: stopped")

    def _message_loop(self):

        logger.info(f"[{self.worker_id}] message loop: started")

        from ..ipc import MessageType, DNNEvictRequest, DNNRestoreRequest

        while self.is_running:
            try:

                msg = self.grm.shm.pop_response_for_worker(self.worker_id)

                if msg is None:
                    time.sleep(0.01)
                    continue

                logger.debug(
                    f"[{self.worker_id}] message loop: type={msg.msg_type}, sender={msg.sender_id}"
                )

                if msg.msg_type == MessageType.DNN_EVICT_REQUEST:
                    payload = msg.payload
                    if payload.get("worker_id") == self.worker_id:
                        logger.info(
                            f"[{self.worker_id}] message loop: evicted: {payload.get('request_id')}"
                        )

                        request = DNNEvictRequest(
                            worker_id=payload["worker_id"],
                            request_id=payload["request_id"],
                            reason=payload.get("reason", "unknown"),
                            required_memory_gb=payload.get("required_memory_gb", 0.0),
                            timeout_s=payload.get("timeout_s", 30.0),
                        )

                        response = self.on_evict_request(request)

                        from ..ipc import create_dnn_evict_response_message

                        response_msg = create_dnn_evict_response_message(response)
                        self.grm.shm.push_request(response_msg)

                        logger.info(
                            f"[{self.worker_id}] message loop: evicted: success={response.success}"
                        )

                elif msg.msg_type == MessageType.DNN_RESTORE_REQUEST:
                    payload = msg.payload
                    if payload.get("worker_id") == self.worker_id:
                        logger.info(
                            f"[{self.worker_id}] message loop: restored: {payload.get('request_id')}"
                        )

                        request = DNNRestoreRequest(
                            worker_id=payload["worker_id"],
                            request_id=payload["request_id"],
                            timeout_s=payload.get("timeout_s", 30.0),
                        )

                        response = self.on_restore_request(request)

                        from ..ipc import create_dnn_restore_response_message

                        response_msg = create_dnn_restore_response_message(response)
                        self.grm.shm.push_request(response_msg)

                        logger.info(
                            f"[{self.worker_id}] message loop: restored: success={response.success}"
                        )

            except Exception as e:
                logger.error(
                    f"[{self.worker_id}] message loop: error: {e}", exc_info=True
                )
                time.sleep(0.1)

        logger.info(f"[{self.worker_id}] message loop: finished")

    def register_to_grm(self):
        """Register to grm."""
        if not self.grm:
            logger.warning(
                f"{self.worker_id}: GRM register to grm: not initialized,skipped"
            )
            return

        self.grm.worker_info.memory_used_gb = self.config.current_memory_gb
        self.grm.worker_info.memory_capacity_gb = self.config.expected_memory_gb
        self.grm.worker_info.memory_releasable_gb = (
            self.config.expected_memory_gb if self.config.evictable else 0.0
        )
        self.grm.worker_info.model_name = getattr(self.config, "task_name", "")

        try:
            from ..ipc import Message, MessageType

            payload = self.grm.worker_info.to_dict()
            payload["dnn_config"] = {
                "worker_id": self.config.worker_id,
                "task_name": self.config.task_name,
                "priority": self.config.priority,
                "evictable": self.config.evictable,
                "expected_memory_gb": self.config.expected_memory_gb,
                "current_memory_gb": self.config.current_memory_gb,
                "status": self.config.status,
            }

            msg = Message.create(
                msg_type=MessageType.REGISTER_DNN,
                sender_id=self.worker_id,
                payload=payload,
            )
            success = self.grm.shm.push_request(msg)
            if success:

                if not self.grm.heartbeat_running:
                    self.grm._start_heartbeat()
                logger.info(
                    f"{self.worker_id} register to grm: completed successfully GRM "
                    f"(priority={self.config.priority}, "
                    f"evictable={self.config.evictable})"
                )
            else:
                logger.warning(
                    f"{self.worker_id} register to grm: registered GRM failed"
                )
        except Exception as e:
            logger.warning(f"{self.worker_id} register to grm: registered GRM : {e}")

    def on_evict_request(self, request: DNNEvictRequest) -> DNNEvictResponse:
        """On evict request."""
        logger.info(f"on evict request: evicted: {request.request_id}")
        logger.info(f"  on evict request: {request.reason}")
        logger.info(f"  on evict request: released: {request.required_memory_gb} GB")

        if not self.is_loaded:
            return DNNEvictResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=False,
                freed_memory_gb=0.0,
                eviction_time_s=0.0,
                message="on evict request",
            )

        if not self.config.evictable:
            return DNNEvictResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=False,
                freed_memory_gb=0.0,
                eviction_time_s=0.0,
                message="on evict request DNN evicted(evictable=False)",
            )

        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() / (1024**3)

        try:
            logger.info(f"on evict request: evicted: {self.config.task_name}")

            self.unload_model()
            self.model = None
            self.is_loaded = False

            torch.cuda.empty_cache()

            memory_after = torch.cuda.memory_allocated() / (1024**3)
            freed_memory = memory_before - memory_after
            eviction_time = time.time() - start_time

            self.config.status = "evicted"
            self.config.current_memory_gb = 0.0

            logger.info(f"on evict request: completed successfully:")
            logger.info(f"  on evict request: released: {freed_memory:.3f} GB")
            logger.info(f"  on evict request: {eviction_time:.3f} ")

            return DNNEvictResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=True,
                freed_memory_gb=freed_memory,
                eviction_time_s=eviction_time,
                message=f"on evict request: completed successfully {self.config.task_name}",
            )

        except Exception as e:
            logger.error(f"on evict request: failed: {e}", exc_info=True)
            return DNNEvictResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=False,
                freed_memory_gb=0.0,
                eviction_time_s=time.time() - start_time,
                message=f"on evict request: failed: {str(e)}",
            )

    def on_restore_request(self, request: DNNRestoreRequest) -> DNNRestoreResponse:
        """On restore request."""
        logger.info(f"on restore request: restored: {request.request_id}")

        if self.is_loaded:
            return DNNRestoreResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=True,
                restore_time_s=0.0,
                message="on restore request",
            )

        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() / (1024**3)

        try:
            logger.info(f"on restore request: restored: {self.config.task_name}")

            self.load_model()
            self.is_loaded = True

            memory_after = torch.cuda.memory_allocated() / (1024**3)
            memory_used = memory_after - memory_before
            restore_time = time.time() - start_time

            self.config.status = "active"
            self.config.current_memory_gb = memory_used
            self.config.last_used_time = time.time()

            logger.info(f"on restore request: completed successfully:")
            logger.info(f"  on restore request: {memory_used:.3f} GB")
            logger.info(f"  on restore request: {restore_time:.3f} ")

            return DNNRestoreResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=True,
                restore_time_s=restore_time,
                message=f"on restore request: completed successfully {self.config.task_name}",
            )

        except Exception as e:
            logger.error(f"on restore request: failed: {e}", exc_info=True)
            return DNNRestoreResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                success=False,
                restore_time_s=time.time() - start_time,
                message=f"on restore request: failed: {str(e)}",
            )

    def _safe_load_model(self):

        try:
            mark_global(
                "dnn_load_start",
                {"worker_id": self.worker_id, "task": self.config.task_name},
            )
            memory_before = torch.cuda.memory_allocated() / (1024**3)

            self.load_model()
            self.is_loaded = True

            memory_after = torch.cuda.memory_allocated() / (1024**3)
            memory_used = memory_after - memory_before

            self.config.status = "active"
            self.config.current_memory_gb = memory_used
            self.config.last_used_time = time.time()

            logger.info(
                f"safe load model: completed successfully: {self.config.task_name}"
            )
            logger.info(f"  safe load model: {memory_used:.3f} GB")
            mark_global(
                "dnn_load_done",
                {
                    "worker_id": self.worker_id,
                    "task": self.config.task_name,
                    "memory_used_gb": round(memory_used, 3),
                },
            )

        except Exception as e:
            logger.error(f"safe load model: failed: {e}", exc_info=True)
            self.is_loaded = False
            mark_global(
                "dnn_load_failed",
                {
                    "worker_id": self.worker_id,
                    "task": self.config.task_name,
                    "error": str(e),
                },
            )
            raise

    def _safe_unload_model(self):

        try:
            mark_global(
                "dnn_unload_start",
                {"worker_id": self.worker_id, "task": self.config.task_name},
            )
            self.unload_model()
            self.model = None
            self.is_loaded = False

            torch.cuda.empty_cache()

            self.config.status = "evicted"
            self.config.current_memory_gb = 0.0

            logger.info(f"safe unload model: evicted: {self.config.task_name}")
            mark_global(
                "dnn_unload_done",
                {"worker_id": self.worker_id, "task": self.config.task_name},
            )

        except Exception as e:
            logger.error(f"safe unload model: failed: {e}", exc_info=True)
            mark_global(
                "dnn_unload_failed",
                {
                    "worker_id": self.worker_id,
                    "task": self.config.task_name,
                    "error": str(e),
                },
            )
            raise

    def get_memory_info(self) -> Dict[str, Any]:
        """Return memory info."""
        return {
            "worker_id": self.worker_id,
            "is_loaded": self.is_loaded,
            "status": self.config.status,
            "current_memory_gb": self.config.current_memory_gb,
            "expected_memory_gb": self.config.expected_memory_gb,
            "evictable": self.config.evictable,
            "priority": self.config.priority,
        }

    def update_last_used_time(self):
        """Update last used time."""
        self.config.last_used_time = time.time()
