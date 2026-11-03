"""ResiMem grm server module."""

import time
import logging
import threading
from typing import Dict, Optional
from dataclasses import dataclass

from ..ipc import (
    GRMSharedMemory,
    Message,
    MessageType,
    WorkerInfo,
    WorkerStatus,
    ExpansionRequest,
    ReleaseRequest,
    create_release_request_message,
    create_approval_message,
    create_denial_message,
    DNNWorkerConfig,
    DNNEvictionPolicy,
    DNNEvictRequest,
    DNNEvictResponse,
    DNNRestoreRequest,
    DNNRestoreResponse,
    create_dnn_evict_request_message,
    create_dnn_evict_response_message,
    create_dnn_restore_request_message,
    create_dnn_restore_response_message,
)


@dataclass
class GRMConfig:
    """GRMConfig implementation."""

    total_gpu_memory_gb: float = 24.0
    heartbeat_timeout_s: float = 5.0
    poll_interval_ms: float = 1.0
    expansion_default_timeout_ms: int = 50
    release_default_timeout_ms: int = 100
    min_free_memory_gb: float = 1.0


class GRMServer:
    """GRMServer implementation."""

    def __init__(self, config: Optional[GRMConfig] = None):

        self.config = config or GRMConfig()
        self.logger = logging.getLogger("GRMServer")

        try:
            self.shm = GRMSharedMemory(create=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize shared memory: {e}")
            raise

        self.vllm_workers: Dict[str, WorkerInfo] = {}
        self.dnn_workers: Dict[str, WorkerInfo] = {}

        self.dnn_configs: Dict[str, DNNWorkerConfig] = {}
        self.eviction_policy = DNNEvictionPolicy()
        self.pending_evict_requests: Dict[str, DNNEvictRequest] = (
            {}
        )  # request_id -> request
        self.pending_restore_requests: Dict[str, DNNRestoreRequest] = (
            {}
        )  # request_id -> request

        self.running = False
        self.main_thread: Optional[threading.Thread] = None

        self.last_expansion_request_time_s: float = 0.0
        self.last_memory_pressure_time_s: float = 0.0
        self.last_auto_restore_scan_time_s: float = 0.0

        self.stats = {
            "total_expansion_requests": 0,
            "approved_expansions": 0,
            "denied_expansions": 0,
            "dnn_releases": 0,
            "dnn_release_timeouts": 0,
            "dnn_evictions": 0,
            "dnn_restores": 0,
            "dnn_eviction_failures": 0,
            "dnn_restore_failures": 0,
            "response_send_failures": 0,
        }

        self.logger.info(
            f"GRM Server initialized (Total GPU Memory: {self.config.total_gpu_memory_gb:.2f} GB)"
        )

    def start(self):
        """Start the service."""
        self.running = True
        self.logger.info("GRM Server started")

        try:
            while self.running:

                self._check_heartbeats()

                self._process_messages()

                self._maybe_auto_restore_evicted_dnns()

                now = time.time()
                if now - getattr(self, "_last_stats_log_time", 0) >= 10.0:
                    self._last_stats_log_time = now
                    self._log_stats()

                time.sleep(self.config.poll_interval_ms / 1000.0)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping...")
        except Exception as e:
            self.logger.error(f"GRM Server crashed: {e}", exc_info=True)
        finally:
            self.stop()

    def start_in_background(self):
        """Start in background."""
        if self.running:
            self.logger.warning("GRM Server is already running")
            return

        self.main_thread = threading.Thread(target=self.start, daemon=True)
        self.main_thread.start()
        self.logger.info("GRM Server started in background")

    def stop(self):
        """Stop the service."""
        self.running = False

        if hasattr(self, "main_thread") and self.main_thread is not None:
            current_thread = threading.current_thread()
            if self.main_thread is not current_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=5.0)
                if self.main_thread.is_alive():
                    self.logger.warning("GRM stop 5s ")

        try:
            self.shm.close()
            self.shm.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to cleanup shared memory: {e}")

        self.logger.info("GRM Server stopped")

    def is_running(self) -> bool:
        """Return whether running."""
        return self.running

    def __enter__(self):

        self.start_in_background()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.stop()
        return False

    def _process_messages(self):

        max_iterations = 100

        for _ in range(max_iterations):

            msg = self.shm.pop_request()
            if msg is None:
                break

            try:
                self._handle_message(msg)
            except Exception as e:
                self.logger.error(
                    f"Error handling message {msg.msg_type}: {e}", exc_info=True
                )

    def _handle_message(self, msg: Message):

        msg_type = msg.msg_type

        if msg_type == MessageType.REGISTER_VLLM:
            self._handle_register_vllm(msg)
        elif msg_type == MessageType.REGISTER_DNN:
            self._handle_register_dnn(msg)
        elif msg_type == MessageType.REQUEST_EXPANSION:
            self._handle_expansion_request(msg)
        elif msg_type == MessageType.MEMORY_RELEASED:
            self._handle_memory_released(msg)
        elif msg_type == MessageType.HEARTBEAT_VLLM:
            self._handle_heartbeat_vllm(msg)
        elif msg_type == MessageType.HEARTBEAT_DNN:
            self._handle_heartbeat_dnn(msg)
        elif msg_type == MessageType.UNREGISTER_VLLM:
            self._handle_unregister_vllm(msg)
        elif msg_type == MessageType.UNREGISTER_DNN:
            self._handle_unregister_dnn(msg)

        elif msg_type == MessageType.DNN_EVICT_RESPONSE:
            self._handle_dnn_evict_response(msg)
        elif msg_type == MessageType.DNN_RESTORE_RESPONSE:
            self._handle_dnn_restore_response(msg)
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")

    def _handle_register_vllm(self, msg: Message):

        worker_info = WorkerInfo.from_dict(msg.payload)
        self.vllm_workers[worker_info.worker_id] = worker_info
        self.shm.register_vllm_worker(worker_info)
        self.logger.info(f"✅ Registered vLLM worker: {worker_info.worker_id}")

    def _handle_register_dnn(self, msg: Message):

        worker_info = WorkerInfo.from_dict(msg.payload)
        self.dnn_workers[worker_info.worker_id] = worker_info
        self.shm.register_dnn_worker(worker_info)

        if "dnn_config" in msg.payload:
            dnn_config = DNNWorkerConfig.from_dict(msg.payload["dnn_config"])
            self.dnn_configs[worker_info.worker_id] = dnn_config
            self.logger.info(
                f"✅ Registered DNN worker: {worker_info.worker_id} "
                f"(priority={dnn_config.priority}, "
                f"residency_class={dnn_config.residency_class}, "
                f"evictable={dnn_config.evictable}, "
                f"memory={dnn_config.expected_memory_gb}GB)"
            )
        else:

            default_config = DNNWorkerConfig(
                worker_id=worker_info.worker_id,
                expected_memory_gb=worker_info.memory_capacity_gb,
                current_memory_gb=worker_info.memory_used_gb,
            )
            self.dnn_configs[worker_info.worker_id] = default_config
            self.logger.info(
                f"✅ Registered DNN worker: {worker_info.worker_id} (legacy mode, default config)"
            )

    def _handle_heartbeat_vllm(self, msg: Message):

        worker_id = msg.sender_id
        if worker_id in self.vllm_workers:
            self.vllm_workers[worker_id].update_heartbeat()
            if "memory_used_gb" in msg.payload:
                self.vllm_workers[worker_id].memory_used_gb = msg.payload[
                    "memory_used_gb"
                ]

    def _handle_heartbeat_dnn(self, msg: Message):

        worker_id = msg.sender_id
        if worker_id in self.dnn_workers:
            self.dnn_workers[worker_id].update_heartbeat()
            if "memory_used_gb" in msg.payload:
                self.dnn_workers[worker_id].memory_used_gb = msg.payload[
                    "memory_used_gb"
                ]

    def _handle_unregister_vllm(self, msg: Message):

        worker_id = msg.sender_id
        if worker_id in self.vllm_workers:
            del self.vllm_workers[worker_id]
            self.logger.info(f"❌ Unregistered vLLM worker: {worker_id}")

    def _handle_unregister_dnn(self, msg: Message):

        worker_id = msg.sender_id
        if worker_id in self.dnn_workers:
            del self.dnn_workers[worker_id]

        if worker_id in self.dnn_configs:
            del self.dnn_configs[worker_id]
        self.logger.info(f"❌ Unregistered DNN worker: {worker_id}")

    def _handle_expansion_request(self, msg: Message):

        request = ExpansionRequest.from_dict(msg.payload)
        sender_id = msg.sender_id

        self.last_expansion_request_time_s = time.time()

        self.stats["total_expansion_requests"] += 1

        self.logger.info(
            f"📥 Expansion request from {sender_id}: "
            f"{request.num_blocks} blocks, "
            f"{request.size_bytes / 1024**3:.2f} GB, "
            f"priority={request.priority}"
        )

        vllm_used = sum(w.memory_used_gb for w in self.vllm_workers.values())
        dnn_used = sum(w.memory_used_gb for w in self.dnn_workers.values())
        total_used = vllm_used + dnn_used
        available = self.config.total_gpu_memory_gb - total_used

        needed_gb = request.size_bytes / 1024**3

        self.logger.debug(
            f"Memory: vLLM={vllm_used:.2f}GB, DNN={dnn_used:.2f}GB, "
            f"Available={available:.2f}GB, Needed={needed_gb:.2f}GB"
        )

        if available >= needed_gb + self.config.min_free_memory_gb:
            self.logger.info(
                f"✅ Approved: sufficient free memory ({available:.2f} GB)"
            )
            self._send_approval(sender_id, request.num_blocks)
            self.stats["approved_expansions"] += 1
            return

        deficit_gb = needed_gb - available + self.config.min_free_memory_gb
        self.logger.info(
            f"⚠️ Insufficient memory, entering L2 pressure handler: {deficit_gb:.2f} GB"
        )

        release_success = self.handle_memory_pressure(
            required_memory_gb=deficit_gb,
            reason=f"expansion:{sender_id}:{request.request_id}",
            timeout_s=max(0.001, request.timeout_ms / 1000.0),
        )

        if release_success:
            self.logger.info(f"✅ DNN released memory, approving expansion")
            self._send_approval(sender_id, request.num_blocks)
            self.stats["approved_expansions"] += 1
            self.stats["dnn_releases"] += 1
        else:
            self.logger.warning(
                "❌ L2 memory pressure handling failed, denying expansion"
            )
            self._send_denial(sender_id, "L2 memory pressure unresolved")
            self.stats["denied_expansions"] += 1
            self.stats["dnn_release_timeouts"] += 1

    def _handle_memory_released(self, msg: Message):

        worker_id = msg.sender_id
        size_bytes = msg.payload.get("size_bytes", 0)
        request_id = msg.payload.get("request_id", "unknown")

        self.logger.info(
            f"✅ DNN {worker_id} released {size_bytes / 1024**3:.2f} GB "
            f"(request_id: {request_id})"
        )

        if worker_id in self.dnn_workers:
            self.dnn_workers[worker_id].status = WorkerStatus.DNN_RUNNING
            released_gb = size_bytes / 1024**3
            self.dnn_workers[worker_id].memory_used_gb = max(
                0.0,
                self.dnn_workers[worker_id].memory_used_gb - released_gb,
            )

    def request_dnn_degrade(
        self, required_memory_gb: float, timeout_s: float = 30.0
    ) -> bool:
        """Request dnn degrade."""
        if required_memory_gb <= 0:
            self.logger.info(
                "request_dnn_degrade: required_memory_gb<=0,request dnn degrade: skipped DNN processed"
            )
            return True

        dnn_candidate = self.select_dnn_to_evict(required_memory_gb)
        if dnn_candidate is not None:
            self.logger.info(
                f"L2 degrade path A: request dnn degrade: evicted DNN {dnn_candidate},released {required_memory_gb:.2f}GB"
            )
            if self.evict_dnn(dnn_candidate, required_memory_gb):
                return True
            self.logger.warning(
                f"L2 degrade path A failed: DNN {dnn_candidate} request dnn degrade: failed, legacy release "
            )

        fallback_candidate = self._find_releasable_dnn(required_memory_gb)
        if fallback_candidate is None:
            self.logger.warning(
                "L2 degrade failed: request dnn degrade: evicted/released DNN"
            )
            return False

        timeout_ms = max(1, int(timeout_s * 1000))
        self.logger.info(
            f"L2 degrade path B: legacy release -> {fallback_candidate}, "
            f"required={required_memory_gb:.2f}GB, timeout={timeout_ms}ms"
        )
        return self._request_dnn_release(
            fallback_candidate,
            required_memory_gb,
            timeout_ms=timeout_ms,
        )

    def handle_memory_pressure(
        self,
        required_memory_gb: float,
        reason: str = "unknown",
        timeout_s: float = 30.0,
    ) -> bool:
        """Handle memory pressure."""
        if required_memory_gb <= 0:
            self.logger.info(
                f"handle_memory_pressure: reason={reason}, required<=0,handle memory pressure: completed successfully"
            )
            return True

        self.logger.info(
            f"🧠 Handle memory pressure: reason={reason}, required={required_memory_gb:.2f}GB"
        )

        self.last_memory_pressure_time_s = time.time()

        success = self.request_dnn_degrade(required_memory_gb, timeout_s=timeout_s)

        if success:
            self.logger.info("✅ L2 memory pressure handled by DNN coordination")
        else:
            self.logger.warning("❌ L2 memory pressure handling failed")

        return success

    def _calculate_available_memory_gb(self) -> float:

        vllm_used = sum(worker.memory_used_gb for worker in self.vllm_workers.values())
        dnn_used = sum(worker.memory_used_gb for worker in self.dnn_workers.values())
        return self.config.total_gpu_memory_gb - (vllm_used + dnn_used)

    def _maybe_auto_restore_evicted_dnns(self, now_s: Optional[float] = None) -> int:

        if not self.eviction_policy.auto_restore_on_idle:
            return 0

        now_s = now_s or time.time()

        if now_s - self.last_auto_restore_scan_time_s < 1.0:
            return 0
        self.last_auto_restore_scan_time_s = now_s

        last_activity = max(
            self.last_expansion_request_time_s, self.last_memory_pressure_time_s
        )
        if last_activity > 0:
            idle_duration_s = now_s - last_activity
            if idle_duration_s < self.eviction_policy.idle_threshold_s:
                return 0

        evicted_candidates = []
        for worker_id, config in self.dnn_configs.items():
            if config.status != "evicted":
                continue
            if worker_id not in self.dnn_workers:
                continue
            evicted_candidates.append((worker_id, config))

        if not evicted_candidates:
            return 0

        evicted_candidates.sort(
            key=lambda item: (-item[1].priority, -item[1].last_used_time)
        )

        restored_count = 0
        for worker_id, config in evicted_candidates:
            required_memory_gb = (
                config.expected_memory_gb + self.eviction_policy.safety_buffer_gb
            )
            available_gb = self._calculate_available_memory_gb()

            if available_gb < required_memory_gb:
                self.logger.info(
                    f"maybe auto restore evicted dnns: skipped {worker_id}: insufficient capacity "
                    f"({available_gb:.2f}GB < {required_memory_gb:.2f}GB)"
                )
                continue

            self.logger.info(
                f"🔄 Auto restore DNN: {worker_id} "
                f"(priority={config.priority}, expected={config.expected_memory_gb:.2f}GB)"
            )
            if self.restore_dnn(worker_id):
                restored_count += 1

        return restored_count

    def _find_releasable_dnn(self, needed_gb: float) -> Optional[str]:

        for worker_id, worker in self.dnn_workers.items():
            if (
                worker.status == WorkerStatus.DNN_RUNNING
                and worker.memory_releasable_gb is not None
                and worker.memory_releasable_gb >= needed_gb
            ):
                return worker_id
        return None

    def _request_dnn_release(
        self, dnn_id: str, size_gb: float, timeout_ms: int
    ) -> bool:

        release_request = ReleaseRequest(
            size_bytes=int(size_gb * 1024**3), timeout_ms=timeout_ms, priority=5
        )

        msg = create_release_request_message(release_request)
        msg.payload["target_worker"] = dnn_id
        success = self.shm.push_response(msg)

        if not success:
            self.logger.error(f"Failed to send release request to {dnn_id}")
            return False

        self.logger.info(
            f"📤 Sent release request to {dnn_id} "
            f"(size={size_gb:.2f}GB, timeout={timeout_ms}ms)"
        )

        if dnn_id in self.dnn_workers:
            self.dnn_workers[dnn_id].status = WorkerStatus.DNN_RELEASING

        start_time = time.time()
        while (time.time() - start_time) * 1000 < timeout_ms:

            self._process_messages()

            if dnn_id in self.dnn_workers:
                if self.dnn_workers[dnn_id].status == WorkerStatus.DNN_RUNNING:
                    return True

            time.sleep(0.001)

        self.logger.warning(f"DNN {dnn_id} release timeout after {timeout_ms}ms")
        if dnn_id in self.dnn_workers:
            self.dnn_workers[dnn_id].status = WorkerStatus.DNN_RUNNING
        return False

    def _send_approval(self, vllm_id: str, num_blocks: int):

        msg = create_approval_message(num_blocks)
        msg.payload["target_worker"] = vllm_id
        success = self.shm.push_response(msg)
        if not success:
            self.stats["response_send_failures"] += 1
            self.logger.error(f"❌ Failed to send approval to {vllm_id}")
            return
        self.logger.debug(f"📤 Sent approval to {vllm_id}")

    def _send_denial(self, vllm_id: str, reason: str):

        msg = create_denial_message(reason)
        msg.payload["target_worker"] = vllm_id
        success = self.shm.push_response(msg)
        if not success:
            self.stats["response_send_failures"] += 1
            self.logger.error(f"❌ Failed to send denial to {vllm_id}: {reason}")
            return
        self.logger.debug(f"📤 Sent denial to {vllm_id}: {reason}")

    def _check_heartbeats(self):

        current_time = time.time()

        dead_vllm = []
        for worker_id, worker in self.vllm_workers.items():
            if not worker.is_alive(self.config.heartbeat_timeout_s):
                self.logger.warning(
                    f"⚠️ vLLM worker {worker_id} timeout "
                    f"(last heartbeat: {current_time - worker.last_heartbeat:.1f}s ago)"
                )
                dead_vllm.append(worker_id)

        for worker_id in dead_vllm:
            del self.vllm_workers[worker_id]

        dead_dnn = []
        for worker_id, worker in self.dnn_workers.items():
            if not worker.is_alive(self.config.heartbeat_timeout_s):
                self.logger.warning(
                    f"⚠️ DNN worker {worker_id} timeout "
                    f"(last heartbeat: {current_time - worker.last_heartbeat:.1f}s ago)"
                )
                dead_dnn.append(worker_id)

        for worker_id in dead_dnn:
            del self.dnn_workers[worker_id]

            self.dnn_configs.pop(worker_id, None)

    def _log_stats(self):

        self.logger.info("=" * 50)
        self.logger.info("GRM Statistics:")
        self.logger.info(f"  vLLM Workers: {len(self.vllm_workers)}")
        self.logger.info(f"  DNN Workers: {len(self.dnn_workers)}")
        self.logger.info(
            f"  Total Expansion Requests: {self.stats['total_expansion_requests']}"
        )
        self.logger.info(f"  Approved: {self.stats['approved_expansions']}")
        self.logger.info(f"  Denied: {self.stats['denied_expansions']}")
        self.logger.info(f"  DNN Releases: {self.stats['dnn_releases']}")
        self.logger.info(
            f"  DNN Release Timeouts: {self.stats['dnn_release_timeouts']}"
        )
        self.logger.info(f"  DNN Evictions: {self.stats['dnn_evictions']}")
        self.logger.info(
            f"  DNN Eviction Failures: {self.stats['dnn_eviction_failures']}"
        )
        self.logger.info(f"  DNN Restores: {self.stats['dnn_restores']}")
        self.logger.info(
            f"  DNN Restore Failures: {self.stats['dnn_restore_failures']}"
        )
        self.logger.info(
            f"  Response Send Failures: {self.stats['response_send_failures']}"
        )

        if self.stats["total_expansion_requests"] > 0:
            approval_rate = (
                self.stats["approved_expansions"]
                / self.stats["total_expansion_requests"]
                * 100
            )
            self.logger.info(f"  Approval Rate: {approval_rate:.1f}%")

        self.logger.info("=" * 50)

    def get_stats(self) -> Dict:
        """Return stats."""
        return {
            **self.stats,
            "num_vllm_workers": len(self.vllm_workers),
            "num_dnn_workers": len(self.dnn_workers),
        }

    def select_dnn_to_evict(self, required_memory_gb: float) -> Optional[str]:
        """Select dnn to evict."""
        evictable_dnns = []
        active_dnn_count = sum(
            1 for cfg in self.dnn_configs.values() if cfg.status == "active"
        )
        min_dnn_to_keep = max(0, int(self.eviction_policy.min_dnn_to_keep))

        if active_dnn_count <= min_dnn_to_keep:
            self.logger.warning(
                f"select dnn to evict min_dnn_to_keep : active={active_dnn_count}, keep={min_dnn_to_keep}"
            )
            return None

        max_priority_allowed = int(self.eviction_policy.min_priority_to_evict)

        for worker_id, config in self.dnn_configs.items():

            if not config.evictable:
                continue
            if config.status != "active":
                continue
            if worker_id not in self.dnn_workers:
                continue
            if max_priority_allowed > 0 and config.priority > max_priority_allowed:
                continue

            evictable_dnns.append((worker_id, config))

        if not evictable_dnns:
            self.logger.warning("select dnn to evict: evicted DNN Worker")
            return None

        if self.eviction_policy.prefer_least_recently_used:
            evictable_dnns.sort(key=lambda x: (x[1].priority, x[1].last_used_time))
        else:
            evictable_dnns.sort(key=lambda x: (x[1].priority, x[0]))

        self.logger.info(
            f"select dnn to evict: evicted DNN  ( {len(evictable_dnns)} ):"
        )
        for wid, cfg in evictable_dnns:
            self.logger.info(
                f"  {wid}: priority={cfg.priority}, "
                f"memory={cfg.expected_memory_gb}GB, "
                f"last_used={time.time()-cfg.last_used_time:.1f}s ago"
            )

        for worker_id, config in evictable_dnns:
            if config.expected_memory_gb >= required_memory_gb:
                self.logger.info(
                    f"select dnn to evict: evicted DNN: {worker_id} ({config.task_name}, {config.expected_memory_gb}GB)"
                )
                return worker_id

        worker_id, config = evictable_dnns[0]
        self.logger.info(
            f"select dnn to evict DNN insufficient capacity,evicted: {worker_id} "
            f"({config.expected_memory_gb}GB < {required_memory_gb}GB)"
        )
        return worker_id

    def evict_dnn(self, worker_id: str, required_memory_gb: float = 0.0) -> bool:
        """Evict dnn."""
        if worker_id not in self.dnn_configs:
            self.logger.error(f"DNN Worker {worker_id} evict dnn: not found")
            return False

        config = self.dnn_configs[worker_id]

        request = DNNEvictRequest(
            worker_id=worker_id,
            request_id=f"evict_{int(time.time() * 1000000)}",
            reason="memory_pressure",
            required_memory_gb=required_memory_gb,
            timeout_s=30.0,
        )

        self.pending_evict_requests[request.request_id] = request

        msg = create_dnn_evict_request_message(request)
        sent = self.shm.push_response(msg)
        if not sent:
            self.pending_evict_requests.pop(request.request_id, None)
            self.stats["dnn_eviction_failures"] += 1
            self.stats["response_send_failures"] += 1
            self.logger.error(
                f"evict dnn: failed: {worker_id} (request_id={request.request_id})"
            )
            return False

        self.logger.info(
            f"evict dnn: evicted: {worker_id} (request_id={request.request_id})"
        )

        start_time = time.time()
        while time.time() - start_time < request.timeout_s:

            self._process_messages()

            if request.request_id not in self.pending_evict_requests:

                self.logger.info(f"DNN {worker_id} evict dnn: completed successfully")
                return True

            time.sleep(0.01)

        self.logger.warning(f"DNN {worker_id} evict dnn: timed out")
        self.pending_evict_requests.pop(request.request_id, None)
        self.stats["dnn_eviction_failures"] += 1
        return False

    def restore_dnn(self, worker_id: str) -> bool:
        """Restore dnn."""
        if worker_id not in self.dnn_configs:
            self.logger.error(f"DNN Worker {worker_id} restore dnn: not found")
            return False

        config = self.dnn_configs[worker_id]

        if config.status != "evicted":
            self.logger.warning(
                f"DNN Worker {worker_id} restore dnn: evicted (status={config.status})"
            )
            return False

        request = DNNRestoreRequest(
            worker_id=worker_id,
            request_id=f"restore_{int(time.time() * 1000000)}",
            timeout_s=60.0,
        )

        self.pending_restore_requests[request.request_id] = request

        msg = create_dnn_restore_request_message(request)
        sent = self.shm.push_response(msg)
        if not sent:
            self.pending_restore_requests.pop(request.request_id, None)
            self.stats["dnn_restore_failures"] += 1
            self.stats["response_send_failures"] += 1
            self.logger.error(
                f"restore dnn: failed: {worker_id} (request_id={request.request_id})"
            )
            return False

        self.logger.info(
            f"restore dnn: restored: {worker_id} (request_id={request.request_id})"
        )

        start_time = time.time()
        while time.time() - start_time < request.timeout_s:

            self._process_messages()

            if request.request_id not in self.pending_restore_requests:
                self.logger.info(f"DNN {worker_id} restore dnn: completed successfully")
                return True

            time.sleep(0.01)

        self.logger.warning(f"DNN {worker_id} restore dnn: timed out")
        self.pending_restore_requests.pop(request.request_id, None)
        self.stats["dnn_restore_failures"] += 1
        return False

    def _handle_dnn_evict_response(self, msg: Message):

        response = DNNEvictResponse.from_dict(msg.payload)

        if response.request_id not in self.pending_evict_requests:
            self.logger.warning(
                f"handle dnn evict response: evicted: {response.request_id}"
            )
            return

        request = self.pending_evict_requests.pop(response.request_id)

        if response.success:

            if response.worker_id in self.dnn_configs:
                self.dnn_configs[response.worker_id].status = "evicted"
                self.dnn_configs[response.worker_id].current_memory_gb = 0.0

            self.stats["dnn_evictions"] += 1

            self.logger.info(
                f"✅ DNN handle dnn evict response: completed successfully: {response.worker_id}, "
                f"released {response.freed_memory_gb:.3f} GB, "
                f" {response.eviction_time_s:.3f}s"
            )
        else:
            self.stats["dnn_eviction_failures"] += 1
            self.logger.error(
                f"❌ DNN handle dnn evict response: failed: {response.worker_id}, "
                f": {response.message}"
            )

    def _handle_dnn_restore_response(self, msg: Message):

        response = DNNRestoreResponse.from_dict(msg.payload)

        if response.request_id not in self.pending_restore_requests:
            self.logger.warning(
                f"handle dnn restore response: restored: {response.request_id}"
            )
            return

        request = self.pending_restore_requests.pop(response.request_id)

        if response.success:

            if response.worker_id in self.dnn_configs:
                self.dnn_configs[response.worker_id].status = "active"
                self.dnn_configs[response.worker_id].last_used_time = time.time()

            self.stats["dnn_restores"] += 1

            self.logger.info(
                f"✅ DNN handle dnn restore response: completed successfully: {response.worker_id}, "
                f" {response.restore_time_s:.3f}s"
            )
        else:
            self.stats["dnn_restore_failures"] += 1
            self.logger.error(
                f"❌ DNN handle dnn restore response: failed: {response.worker_id}, "
                f": {response.message}"
            )
