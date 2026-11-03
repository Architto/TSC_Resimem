"""ResiMem group swap module."""

import time
import logging
import threading
from typing import Dict, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch

from .multi_user_types import SwapMetadata, UserState


logger = logging.getLogger(__name__)


@dataclass
class GroupSwapConfig:
    """GroupSwapConfig implementation."""

    target_latency_ms: float = 100.0
    max_blocks_per_batch: int = 512

    cpu_swap_space_gb: float = 32.0
    pin_memory: bool = True

    verify_after_swap: bool = False
    enable_logging: bool = True


@dataclass
class BlockTableEntry:
    """BlockTableEntry implementation."""

    block_id: int
    sequence_id: str
    user_id: str
    num_tokens: int = 0

    in_gpu: bool = True
    gpu_block_id: Optional[int] = None
    cpu_block_id: Optional[int] = None


class UserBlockTracker:
    """UserBlockTracker implementation."""

    def __init__(self):
        # user_id → set of block_ids
        self._user_blocks: Dict[str, Set[int]] = {}

        # block_id → BlockTableEntry
        self._block_info: Dict[int, BlockTableEntry] = {}

        self._lock = threading.RLock()

    def register_block(
        self, user_id: str, sequence_id: str, block_id: int, num_tokens: int = 0
    ):
        """Register block."""
        with self._lock:

            if user_id not in self._user_blocks:
                self._user_blocks[user_id] = set()
            self._user_blocks[user_id].add(block_id)

            entry = BlockTableEntry(
                block_id=block_id,
                sequence_id=sequence_id,
                user_id=user_id,
                num_tokens=num_tokens,
                in_gpu=True,
                gpu_block_id=block_id,
            )
            self._block_info[block_id] = entry

    def unregister_block(self, block_id: int):
        """Unregister block."""
        with self._lock:
            if block_id in self._block_info:
                entry = self._block_info[block_id]
                user_id = entry.user_id

                if user_id in self._user_blocks:
                    self._user_blocks[user_id].discard(block_id)

                del self._block_info[block_id]

    def get_user_blocks(self, user_id: str) -> Set[int]:
        """Return user blocks."""
        with self._lock:
            return self._user_blocks.get(user_id, set()).copy()

    def get_block_info(self, block_id: int) -> Optional[BlockTableEntry]:
        """Return block info."""
        with self._lock:
            return self._block_info.get(block_id)

    def update_block_location(
        self,
        block_id: int,
        in_gpu: bool,
        gpu_id: Optional[int] = None,
        cpu_id: Optional[int] = None,
    ):
        """Update block location."""
        with self._lock:
            if block_id in self._block_info:
                entry = self._block_info[block_id]
                entry.in_gpu = in_gpu
                entry.gpu_block_id = gpu_id
                entry.cpu_block_id = cpu_id

    def get_user_gpu_blocks(self, user_id: str) -> List[int]:
        """Return user gpu blocks."""
        with self._lock:
            result = []
            for block_id in self._user_blocks.get(user_id, set()):
                entry = self._block_info.get(block_id)
                if entry and entry.in_gpu:
                    result.append(block_id)
            return result

    def get_all_gpu_blocks(self) -> Set[int]:
        """Return all gpu blocks."""
        with self._lock:
            result: Set[int] = set()
            for block_id, entry in self._block_info.items():
                if entry.in_gpu:
                    result.add(block_id)
            return result


class GroupSwapManager:
    """GroupSwapManager implementation."""

    def __init__(
        self,
        config: GroupSwapConfig,
        kv_cache_gpu: Optional[torch.Tensor] = None,
        kv_cache_cpu: Optional[torch.Tensor] = None,
        block_size: int = 16,
    ):

        self.config = config
        self.kv_cache_gpu = kv_cache_gpu
        self.kv_cache_cpu = kv_cache_cpu
        self.block_size = block_size

        self.block_tracker = UserBlockTracker()

        self.swap_metadata: Dict[str, SwapMetadata] = {}

        self._cpu_free_blocks: List[int] = []
        self._cpu_total_blocks: int = 0
        self._init_cpu_blocks()

        self._lock = threading.RLock()

        self.stats = {
            "total_swap_outs": 0,
            "total_swap_ins": 0,
            "total_blocks_swapped_out": 0,
            "total_blocks_swapped_in": 0,
            "total_swap_out_time_ms": 0.0,
            "total_swap_in_time_ms": 0.0,
        }

        logger.info(f"GroupSwapManager init: completed successfully,: {config}")

    def _init_cpu_blocks(self):

        if self.kv_cache_cpu is not None:

            if len(self.kv_cache_cpu.shape) == 6:
                self._cpu_total_blocks = self.kv_cache_cpu.shape[2]
            elif len(self.kv_cache_cpu.shape) > 0:
                self._cpu_total_blocks = self.kv_cache_cpu.shape[0]
            else:
                self._cpu_total_blocks = 0
            self._cpu_free_blocks = list(range(self._cpu_total_blocks))
        else:

            self._cpu_total_blocks = 0
            self._cpu_free_blocks = []
            logger.warning(
                "GroupSwapManager: kv_cache_cpu init cpu blocks,CPU  0,Swap Out "
            )

    def _allocate_cpu_block(self) -> Optional[int]:

        if self._cpu_free_blocks:
            return self._cpu_free_blocks.pop()
        return None

    def _free_cpu_block(self, block_id: int):

        self._cpu_free_blocks.append(block_id)

    def swap_out_user(self, user_id: str) -> Optional[SwapMetadata]:
        """Swap out user."""
        start_time = time.time()

        with self._lock:

            gpu_blocks = self.block_tracker.get_user_gpu_blocks(user_id)

            if not gpu_blocks:
                logger.warning(f"swap out user {user_id}  GPU ,skipped Swap Out")
                return None

            logger.info(
                f"swap out user: started Swap Out  {user_id}, {len(gpu_blocks)} "
            )

            block_mapping: Dict[int, int] = {}  # GPU block ID → CPU block ID
            sequence_ids: Set[str] = set()

            for gpu_block_id in gpu_blocks:

                cpu_block_id = self._allocate_cpu_block()
                if cpu_block_id is None:
                    logger.error(f"CPU swap out user: failed,Swap Out ")

                    for allocated_cpu_id in block_mapping.values():
                        self._free_cpu_block(allocated_cpu_id)
                    return None

                block_mapping[gpu_block_id] = cpu_block_id

                block_info = self.block_tracker.get_block_info(gpu_block_id)
                if block_info:
                    sequence_ids.add(block_info.sequence_id)

            if not self._copy_blocks_gpu_to_cpu(block_mapping):
                logger.error("Swap Out swap out user: failed,evicted")
                for allocated_cpu_id in block_mapping.values():
                    self._free_cpu_block(allocated_cpu_id)
                return None

            for gpu_block_id, cpu_block_id in block_mapping.items():
                self.block_tracker.update_block_location(
                    gpu_block_id, in_gpu=False, gpu_id=None, cpu_id=cpu_block_id
                )

            elapsed_ms = (time.time() - start_time) * 1000
            metadata = SwapMetadata(
                user_id=user_id,
                sequence_ids=list(sequence_ids),
                block_mapping=block_mapping,
                num_blocks_swapped=len(block_mapping),
                swap_out_latency_ms=elapsed_ms,
            )

            self.swap_metadata[user_id] = metadata

            self.stats["total_swap_outs"] += 1
            self.stats["total_blocks_swapped_out"] += len(block_mapping)
            self.stats["total_swap_out_time_ms"] += elapsed_ms

            logger.info(
                f"Swap Out swap out user: completed successfully: {user_id},"
                f"{len(block_mapping)} ,"
                f" {elapsed_ms:.2f}ms"
            )

            return metadata

    def _copy_blocks_gpu_to_cpu(self, block_mapping: Dict[int, int]) -> bool:

        if self.kv_cache_gpu is None or self.kv_cache_cpu is None:
            logger.error(
                "KV Cache copy blocks gpu to cpu: not initialized,unavailable Swap Out "
            )
            return False

        is_6d = len(self.kv_cache_gpu.shape) == 6

        success = True
        for gpu_block_id, cpu_block_id in block_mapping.items():
            try:
                if is_6d:

                    self.kv_cache_cpu[:, :, cpu_block_id, :, :, :].copy_(
                        self.kv_cache_gpu[:, :, gpu_block_id, :, :, :],
                        non_blocking=True,
                    )
                else:

                    self.kv_cache_cpu[cpu_block_id].copy_(
                        self.kv_cache_gpu[gpu_block_id], non_blocking=True
                    )
            except Exception as e:
                logger.error(
                    f"copy blocks gpu to cpu {gpu_block_id} → {cpu_block_id} failed: {e}"
                )
                success = False

        if self.kv_cache_gpu.is_cuda:
            torch.cuda.synchronize()

        return success

    def swap_in_user(self, user_id: str) -> bool:
        """Swap in user."""
        start_time = time.time()

        with self._lock:

            metadata = self.swap_metadata.get(user_id)
            if metadata is None:
                logger.warning(f"swap in user {user_id}  Swap ,skipped Swap In")
                return False

            logger.info(
                f"swap in user: started Swap In  {user_id},"
                f" {metadata.num_blocks_swapped} "
            )

            if not self._copy_blocks_cpu_to_gpu(metadata.block_mapping):
                logger.error("Swap In swap in user: failed,")
                return False

            for gpu_block_id, cpu_block_id in metadata.block_mapping.items():
                self.block_tracker.update_block_location(
                    gpu_block_id, in_gpu=True, gpu_id=gpu_block_id, cpu_id=None
                )

                self._free_cpu_block(cpu_block_id)

            del self.swap_metadata[user_id]

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_swap_ins"] += 1
            self.stats["total_blocks_swapped_in"] += metadata.num_blocks_swapped
            self.stats["total_swap_in_time_ms"] += elapsed_ms

            logger.info(
                f"Swap In swap in user: completed successfully: {user_id},"
                f"{metadata.num_blocks_swapped} ,"
                f" {elapsed_ms:.2f}ms"
            )

            return True

    def _copy_blocks_cpu_to_gpu(self, block_mapping: Dict[int, int]) -> bool:

        if self.kv_cache_gpu is None or self.kv_cache_cpu is None:
            logger.error(
                "KV Cache copy blocks cpu to gpu: not initialized,unavailable Swap In "
            )
            return False

        is_6d = len(self.kv_cache_gpu.shape) == 6

        success = True
        for gpu_block_id, cpu_block_id in block_mapping.items():
            try:
                if is_6d:

                    self.kv_cache_gpu[:, :, gpu_block_id, :, :, :].copy_(
                        self.kv_cache_cpu[:, :, cpu_block_id, :, :, :],
                        non_blocking=True,
                    )
                else:

                    self.kv_cache_gpu[gpu_block_id].copy_(
                        self.kv_cache_cpu[cpu_block_id], non_blocking=True
                    )
            except Exception as e:
                logger.error(
                    f"copy blocks cpu to gpu {cpu_block_id} → {gpu_block_id} failed: {e}"
                )
                success = False

        if self.kv_cache_gpu.is_cuda:
            torch.cuda.synchronize()

        return success

    def get_user_swap_metadata(self, user_id: str) -> Optional[SwapMetadata]:
        """Return user swap metadata."""
        with self._lock:
            return self.swap_metadata.get(user_id)

    def is_user_swapped_out(self, user_id: str) -> bool:
        """Return whether user swapped out."""
        with self._lock:
            return user_id in self.swap_metadata

    def get_swapped_out_users(self) -> Set[str]:
        """Return swapped out users."""
        with self._lock:
            return set(self.swap_metadata.keys())

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            avg_swap_out = (
                self.stats["total_swap_out_time_ms"] / self.stats["total_swap_outs"]
                if self.stats["total_swap_outs"] > 0
                else 0
            )
            avg_swap_in = (
                self.stats["total_swap_in_time_ms"] / self.stats["total_swap_ins"]
                if self.stats["total_swap_ins"] > 0
                else 0
            )

            return {
                **self.stats,
                "avg_swap_out_latency_ms": avg_swap_out,
                "avg_swap_in_latency_ms": avg_swap_in,
                "swapped_out_users": len(self.swap_metadata),
                "cpu_free_blocks": len(self._cpu_free_blocks),
                "cpu_total_blocks": self._cpu_total_blocks,
            }

    def register_block_for_user(
        self, user_id: str, sequence_id: str, block_id: int, num_tokens: int = 0
    ):
        """Register block for user."""
        self.block_tracker.register_block(user_id, sequence_id, block_id, num_tokens)

    def unregister_block(self, block_id: int):
        """Unregister block."""
        self.block_tracker.unregister_block(block_id)

    def get_user_block_count(self, user_id: str) -> int:
        """Return user block count."""
        return len(self.block_tracker.get_user_blocks(user_id))

    def get_active_gpu_block_ids(self) -> Set[int]:
        """Return active gpu block ids."""
        return self.block_tracker.get_all_gpu_blocks()

    def reset(self):
        """Reset."""
        with self._lock:

            with self.block_tracker._lock:
                self.block_tracker._user_blocks.clear()
                self.block_tracker._block_info.clear()

            self.swap_metadata.clear()
            self._init_cpu_blocks()
            self.stats = {
                "total_swap_outs": 0,
                "total_swap_ins": 0,
                "total_blocks_swapped_out": 0,
                "total_blocks_swapped_in": 0,
                "total_swap_out_time_ms": 0.0,
                "total_swap_in_time_ms": 0.0,
            }
            logger.info("GroupSwapManager reset")

    def __repr__(self) -> str:
        return (
            f"GroupSwapManager("
            f"swapped_users={len(self.swap_metadata)}, "
            f"cpu_free={len(self._cpu_free_blocks)}/{self._cpu_total_blocks})"
        )


def create_group_swap_manager(
    kv_cache_gpu: Optional[torch.Tensor] = None,
    kv_cache_cpu: Optional[torch.Tensor] = None,
    block_size: int = 16,
    target_latency_ms: float = 100.0,
) -> GroupSwapManager:
    """Create group swap manager."""
    config = GroupSwapConfig(
        target_latency_ms=target_latency_ms,
    )

    return GroupSwapManager(
        config=config,
        kv_cache_gpu=kv_cache_gpu,
        kv_cache_cpu=kv_cache_cpu,
        block_size=block_size,
    )
