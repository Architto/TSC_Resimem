"""ResiMem orchestrator module."""

import threading
import logging
from typing import Optional, Dict, Callable
from dataclasses import dataclass

import torch

from .memory.giant_tensor import GiantTensor
from .memory.memory_pool import MemoryPool
from .memory.kv_cache_adapter import KVCacheAdapter, KVCacheConfig
from .scheduler.focus_manager import FocusManager, FocusManagerConfig
from .scheduler.group_swap import GroupSwapManager, GroupSwapConfig
from .scheduler.multi_user_types import UserState
from .scheduler.api_interceptor import setup_request_interception
from .monitoring import GpuMemoryMonitor, set_global_monitor, mark_global


logger = logging.getLogger(__name__)


@dataclass
class ResiMemConfig:
    """ResiMemConfig implementation."""

    device: str = "cuda:0"
    total_gpu_memory_gb: float = 24.0

    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 16
    initial_blocks: int = 1024
    max_blocks: int = 4096

    cooling_off_period_s: float = 7.0
    enable_swap: bool = True

    enable_logging: bool = True

    enable_memory_monitor: bool = True
    monitor_output_dir: str = "logs/mem"
    monitor_sample_interval_s: float = 0.1
    monitor_smi_interval_s: float = 0.5
    monitor_max_samples: int = 6000
    monitor_flush_interval_s: float = 5.0


class ResiMemOrchestrator:
    """ResiMemOrchestrator implementation."""

    def __init__(self, config: ResiMemConfig):

        self.config = config
        self._lock = threading.RLock()
        self._initialized = False
        self._running = False

        self.giant_tensor: Optional[GiantTensor] = None
        self.memory_pool: Optional[MemoryPool] = None
        self.kv_cache_adapter: Optional[KVCacheAdapter] = None
        self.focus_manager: Optional[FocusManager] = None
        self.group_swap_manager: Optional[GroupSwapManager] = None
        self.memory_monitor: Optional[GpuMemoryMonitor] = None

        self.user_blocks: Dict[str, set] = {}

        self.stats = {
            "total_allocations": 0,
            "total_swap_outs": 0,
            "total_swap_ins": 0,
            "total_expansions": 0,
            "total_shrinks": 0,
        }

        logger.info(f"ResiMemOrchestrator init: created,: {config}")

    def initialize(self) -> bool:
        """Initialize."""
        if self._initialized:
            logger.warning("ResiMemOrchestrator initialize: already active")
            return True

        logger.info("=" * 60)
        logger.info("ResiMemOrchestrator initialize: started")
        logger.info("=" * 60)

        try:

            logger.info("initialize 1/5: created Giant Tensor...")
            self.giant_tensor = GiantTensor(
                total_size_gb=self.config.total_gpu_memory_gb,
                device=self.config.device,
            )

            logger.info("initialize 2/5: created Memory Pool...")
            self.memory_pool = MemoryPool(self.giant_tensor)

            logger.info("initialize 3/5: created KV Cache ...")
            self.kv_cache_adapter = KVCacheAdapter(
                giant_tensor_gpu=self.giant_tensor,
                config=KVCacheConfig(
                    num_layers=self.config.num_layers,
                    num_kv_heads=self.config.num_kv_heads,
                    head_dim=self.config.head_dim,
                    block_size=self.config.block_size,
                ),
                initial_blocks=self.config.initial_blocks,
                max_blocks=self.config.max_blocks,
            )

            logger.info("initialize 4/5: created GroupSwapManager...")
            swap_config = GroupSwapConfig(
                target_latency_ms=100.0,
                pin_memory=True,
            )
            self.group_swap_manager = GroupSwapManager(
                config=swap_config,
                kv_cache_gpu=self.kv_cache_adapter.kv_cache_gpu,
                kv_cache_cpu=self.kv_cache_adapter.kv_cache_cpu,
                block_size=self.config.block_size,
            )

            logger.info("initialize 5/5: created FocusManager...")
            focus_config = FocusManagerConfig(
                cooling_off_period_s=self.config.cooling_off_period_s,
                enable_swap=self.config.enable_swap,
                enable_logging=self.config.enable_logging,
            )
            self.focus_manager = FocusManager(
                config=focus_config,
                swap_out_callback=self._handle_swap_out,
                swap_in_callback=self._handle_swap_in,
            )

            setup_request_interception(self.focus_manager)

            self._initialized = True

            logger.info("=" * 60)
            logger.info("✅ ResiMemOrchestrator initialize: completed successfully")
            logger.info("=" * 60)
            self._print_status()

            return True

        except Exception as e:
            logger.error(
                f"❌ ResiMemOrchestrator initialize: failed: {e}", exc_info=True
            )
            self._cleanup()
            return False

    def start(self) -> bool:
        """Start the service."""
        if not self._initialized:
            logger.error("ResiMemOrchestrator start: not initialized, initialize()")
            return False

        if self._running:
            logger.warning("ResiMemOrchestrator start: already active")
            return True

        logger.info("ResiMemOrchestrator start: started...")

        if self.config.enable_memory_monitor and self.memory_monitor is None:
            self.memory_monitor = GpuMemoryMonitor(
                sample_interval_s=self.config.monitor_sample_interval_s,
                smi_interval_s=self.config.monitor_smi_interval_s,
                output_dir=self.config.monitor_output_dir,
                max_samples=self.config.monitor_max_samples,
                flush_interval_s=self.config.monitor_flush_interval_s,
            )
            self.memory_monitor.start()
            set_global_monitor(self.memory_monitor)
            self.memory_monitor.mark("orchestrator_start")

        self.focus_manager.start()

        self._running = True
        logger.info("✅ ResiMemOrchestrator start: started")

        return True

    def stop(self):
        """Stop the service."""
        if not self._running:
            return

        logger.info("ResiMemOrchestrator stop: stopped...")

        if self.focus_manager:
            self.focus_manager.stop()

        if self.memory_monitor is not None:
            self.memory_monitor.mark("orchestrator_stop")
            self.memory_monitor.stop()
            output_paths = self.memory_monitor.save()
            logger.info(f"stop: {output_paths}")
            set_global_monitor(None)
            self.memory_monitor = None

        self._running = False
        logger.info("✅ ResiMemOrchestrator stop: stopped")

    def _cleanup(self):

        self.stop()

        self.focus_manager = None
        self.group_swap_manager = None
        self.kv_cache_adapter = None
        self.memory_pool = None

        if self.giant_tensor:
            del self.giant_tensor
            self.giant_tensor = None
            torch.cuda.empty_cache()

        self._initialized = False

    def on_user_request(self, user_id: str, sequence_id: Optional[str] = None):
        """On user request."""
        with self._lock:

            if self.focus_manager:
                self.focus_manager.notify_activity(user_id)

            if user_id not in self.user_blocks:
                self.user_blocks[user_id] = set()

            if self.config.enable_logging:
                logger.debug(f"on user request: request: user_id={user_id}")

    def register_block_for_user(self, user_id: str, block_id: int):
        """Register block for user."""
        with self._lock:
            if user_id not in self.user_blocks:
                self.user_blocks[user_id] = set()

            self.user_blocks[user_id].add(block_id)

            if self.group_swap_manager:
                self.group_swap_manager.register_block_for_user(
                    user_id, f"seq_{block_id}", block_id
                )

            self.stats["total_allocations"] += 1

    def unregister_block(self, block_id: int):
        """Unregister block."""
        with self._lock:

            for user_id, blocks in self.user_blocks.items():
                if block_id in blocks:
                    blocks.remove(block_id)
                    break

            if self.group_swap_manager:
                self.group_swap_manager.unregister_block(block_id)

    def get_user_blocks(self, user_id: str) -> set:
        """Return user blocks."""
        with self._lock:
            return self.user_blocks.get(user_id, set()).copy()

    def _handle_swap_out(self, user_id: str) -> bool:

        logger.info(f"handle swap out: processed Swap Out:  {user_id}")
        mark_global("swap_out_start", {"user_id": user_id})

        with self._lock:
            if self.group_swap_manager:
                metadata = self.group_swap_manager.swap_out_user(user_id)

                if metadata:
                    self.stats["total_swap_outs"] += 1
                    logger.info(
                        f"Swap Out handle swap out: completed successfully:  {user_id}, "
                        f"{metadata.num_blocks_swapped} , "
                        f" {metadata.swap_out_latency_ms:.2f}ms"
                    )
                    mark_global("swap_out_done", {"user_id": user_id})
                    return True

            return False

    def _handle_swap_in(self, user_id: str) -> bool:

        logger.info(f"handle swap in: processed Swap In:  {user_id}")
        mark_global("swap_in_start", {"user_id": user_id})

        with self._lock:
            if self.group_swap_manager:
                success = self.group_swap_manager.swap_in_user(user_id)

                if success:
                    self.stats["total_swap_ins"] += 1
                    logger.info(
                        f"Swap In handle swap in: completed successfully:  {user_id}"
                    )
                    mark_global("swap_in_done", {"user_id": user_id})
                    return True

            return False

    def expand_kv_cache(self, additional_blocks: int) -> bool:
        """Expand kv cache."""
        with self._lock:
            if self.kv_cache_adapter:
                success = self.kv_cache_adapter.expand(additional_blocks)
                if success:
                    self.stats["total_expansions"] += 1
                return success
            return False

    def shrink_kv_cache(self, blocks_to_remove: int) -> bool:
        """Shrink kv cache."""
        with self._lock:
            if self.kv_cache_adapter:
                success = self.kv_cache_adapter.shrink(blocks_to_remove)
                if success:
                    self.stats["total_shrinks"] += 1
                return success
            return False

    def handle_memory_pressure(self, needed_blocks: int) -> bool:
        """Handle memory pressure."""
        logger.info(f"handle memory pressure: processed:  {needed_blocks} ")

        with self._lock:

            if self._try_l1_swap_out(needed_blocks):
                return True

            if self._try_l2_sandwich_eviction(needed_blocks):
                return True

            logger.warning(f"handle memory pressure: unavailable")
            return False

    def _try_l1_swap_out(self, needed_blocks: int) -> bool:

        if not self.focus_manager:
            return False

        warm_users = self.focus_manager.get_users_by_state(UserState.WARM)

        freed_blocks = 0
        for user_id in warm_users:
            user_block_count = len(self.get_user_blocks(user_id))

            if user_block_count > 0:
                if self.focus_manager.force_swap_out(user_id):
                    freed_blocks += user_block_count
                    logger.info(
                        f"L1: try l1 swap out: evicted {user_id}, released {user_block_count} "
                    )

                    if freed_blocks >= needed_blocks:
                        return True

        return freed_blocks >= needed_blocks

    def _try_l2_sandwich_eviction(self, needed_blocks: int) -> bool:

        logger.warning(
            "L2 try l2 sandwich eviction: evicted ResiMemScheduler processed,"
            " Orchestrator "
        )
        return False

    def get_active_user(self) -> Optional[str]:
        """Return active user."""
        if self.focus_manager:
            return self.focus_manager.get_active_user()
        return None

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        """Return user state."""
        if self.focus_manager:
            return self.focus_manager.get_user_state(user_id)
        return None

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            result = {
                "orchestrator": self.stats.copy(),
                "initialized": self._initialized,
                "running": self._running,
            }

            if self.kv_cache_adapter:
                result["kv_cache"] = self.kv_cache_adapter.get_stats()

            if self.focus_manager:
                result["focus_manager"] = self.focus_manager.get_statistics()

            if self.group_swap_manager:
                result["group_swap"] = self.group_swap_manager.get_statistics()

            return result

    def _print_status(self):

        logger.info("print status: state:")
        logger.info(f"  Giant Tensor: {self.giant_tensor}")
        logger.info(f"  KV Cache: {self.kv_cache_adapter}")
        logger.info(f"  FocusManager: {self.focus_manager}")
        logger.info(f"  GroupSwapManager: {self.group_swap_manager}")

    def __repr__(self) -> str:
        return (
            f"ResiMemOrchestrator(initialized={self._initialized}, "
            f"running={self._running}, "
            f"active_user={self.get_active_user()})"
        )


def create_resimem_orchestrator(
    device: str = "cuda:0",
    total_gpu_memory_gb: float = 24.0,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    initial_blocks: int = 1024,
    max_blocks: int = 4096,
    cooling_off_period_s: float = 7.0,
    auto_start: bool = True,
) -> ResiMemOrchestrator:
    """Create resimem orchestrator."""
    config = ResiMemConfig(
        device=device,
        total_gpu_memory_gb=total_gpu_memory_gb,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        initial_blocks=initial_blocks,
        max_blocks=max_blocks,
        cooling_off_period_s=cooling_off_period_s,
    )

    orchestrator = ResiMemOrchestrator(config)
    orchestrator.initialize()

    if auto_start:
        orchestrator.start()

    return orchestrator
