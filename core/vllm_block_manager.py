"""ResiMem vllm block manager module."""

import logging
import threading
import time
from typing import Dict, Optional, List, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


@dataclass
class UserBlockInfo:
    """UserBlockInfo implementation."""

    user_id: str
    block_ids: Set[int] = field(default_factory=set)
    sequence_ids: Set[str] = field(default_factory=set)

    kv_cache_backup: Optional[torch.Tensor] = None
    backup_block_mapping: Dict[int, int] = field(
        default_factory=dict
    )  # gpu_block → backup_index

    is_swapped_out: bool = False
    swap_out_time: Optional[float] = None
    swap_in_time: Optional[float] = None


@dataclass
class SwapStats:
    """SwapStats implementation."""

    total_swap_outs: int = 0
    total_swap_ins: int = 0
    total_blocks_swapped_out: int = 0
    total_blocks_swapped_in: int = 0
    total_swap_out_time_ms: float = 0.0
    total_swap_in_time_ms: float = 0.0
    blocks_freed_to_vllm: int = 0


class VLLMBlockManager:
    """VLLMBlockManager implementation."""

    def __init__(
        self,
        kv_cache_gpu: Optional[torch.Tensor] = None,
        cpu_swap_space_gb: float = 32.0,
        block_pool: Optional[Any] = None,  # vLLM BlockPool
    ):

        self.kv_cache_gpu = kv_cache_gpu
        self.block_pool = block_pool

        self.users: Dict[str, UserBlockInfo] = {}

        self.block_to_user: Dict[int, str] = {}

        self._init_cpu_swap_space(cpu_swap_space_gb)

        self.stats = SwapStats()

        self._lock = threading.RLock()

        logger.info(f"VLLMBlockManager init: completed successfully")

    def _init_cpu_swap_space(self, size_gb: float):

        if self.kv_cache_gpu is not None:

            shape = self.kv_cache_gpu.shape
            dtype = self.kv_cache_gpu.dtype

            if len(shape) == 6:
                block_bytes = (
                    shape[0]
                    * shape[1]
                    * shape[3]
                    * shape[4]
                    * shape[5]
                    * dtype.itemsize
                )
            else:
                block_bytes = self.kv_cache_gpu[0].numel() * dtype.itemsize

            max_blocks = int(size_gb * 1024**3 / block_bytes)

            if len(shape) == 6:
                cpu_shape = (
                    shape[0],
                    shape[1],
                    max_blocks,
                    shape[3],
                    shape[4],
                    shape[5],
                )
            else:
                cpu_shape = (max_blocks,) + shape[1:]

            self.kv_cache_cpu = torch.zeros(
                cpu_shape, dtype=dtype, device="cpu", pin_memory=True
            )
            self.cpu_max_blocks = max_blocks
            self.cpu_free_indices = list(range(max_blocks))

            logger.info(f"CPU init cpu swap space: {max_blocks} , {size_gb:.1f} GB")
        else:
            self.kv_cache_cpu = None
            self.cpu_max_blocks = 0
            self.cpu_free_indices = []

    def register_block(self, user_id: str, sequence_id: str, block_id: int):
        """Register block."""
        with self._lock:

            if user_id not in self.users:
                self.users[user_id] = UserBlockInfo(user_id=user_id)

            user_info = self.users[user_id]
            user_info.block_ids.add(block_id)
            user_info.sequence_ids.add(sequence_id)

            self.block_to_user[block_id] = user_id

            logger.debug(
                f"register block: registered: user={user_id}, block={block_id}"
            )

    def unregister_block(self, block_id: int):
        """Unregister block."""
        with self._lock:
            if block_id in self.block_to_user:
                user_id = self.block_to_user[block_id]

                if user_id in self.users:
                    self.users[user_id].block_ids.discard(block_id)

                del self.block_to_user[block_id]

                logger.debug(f"unregister block: unregistered: block={block_id}")

    # Swap Out
    def swap_out_user(self, user_id: str) -> bool:
        """Swap out user."""
        start_time = time.time()

        with self._lock:
            if user_id not in self.users:
                logger.warning(f"swap out user {user_id} not found")
                return False

            user_info = self.users[user_id]

            if user_info.is_swapped_out:
                logger.warning(f"swap out user {user_id} already active")
                return False

            if not user_info.block_ids:
                logger.warning(f"swap out user {user_id} ")
                return False

            block_ids = list(user_info.block_ids)
            num_blocks = len(block_ids)

            logger.info(f"swap out user: started Swap Out  {user_id}, {num_blocks} ")

            if len(self.cpu_free_indices) < num_blocks:
                logger.error(
                    f"CPU swap out user: insufficient capacity:  {num_blocks}, {len(self.cpu_free_indices)}"
                )
                return False

            cpu_indices = [self.cpu_free_indices.pop() for _ in range(num_blocks)]

            try:
                if self.kv_cache_gpu is not None and self.kv_cache_cpu is not None:
                    self._copy_blocks_to_cpu(block_ids, cpu_indices)
            except Exception as e:

                self.cpu_free_indices.extend(cpu_indices)
                logger.error(f"Swap Out swap out user {user_id}  CPU failed, CPU : {e}")
                return False

            user_info.backup_block_mapping = {
                block_id: cpu_idx for block_id, cpu_idx in zip(block_ids, cpu_indices)
            }

            if self.block_pool is not None:
                actually_freed = self._free_blocks_to_vllm(block_ids)
                self.stats.blocks_freed_to_vllm += actually_freed

            user_info.is_swapped_out = True
            user_info.swap_out_time = time.time()

            for block_id in block_ids:
                if block_id in self.block_to_user:
                    del self.block_to_user[block_id]
            user_info.block_ids.clear()

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.total_swap_outs += 1
            self.stats.total_blocks_swapped_out += num_blocks
            self.stats.total_swap_out_time_ms += elapsed_ms

            logger.info(
                f"Swap Out swap out user: completed successfully:  {user_id}, "
                f"{num_blocks} , {elapsed_ms:.2f}ms"
            )

            return True

    def _copy_blocks_to_cpu(self, block_ids: List[int], cpu_indices: List[int]):

        is_6d = len(self.kv_cache_gpu.shape) == 6

        for block_id, cpu_idx in zip(block_ids, cpu_indices):
            try:
                if is_6d:
                    self.kv_cache_cpu[:, :, cpu_idx, :, :, :].copy_(
                        self.kv_cache_gpu[:, :, block_id, :, :, :], non_blocking=True
                    )
                else:
                    self.kv_cache_cpu[cpu_idx].copy_(
                        self.kv_cache_gpu[block_id], non_blocking=True
                    )
            except Exception as e:
                logger.error(
                    f"copy blocks to cpu {block_id} → CPU[{cpu_idx}] failed: {e}"
                )
                raise

        if self.kv_cache_gpu.is_cuda:
            torch.cuda.synchronize()

    def _free_blocks_to_vllm(self, block_ids: List[int]) -> int:

        try:
            blocks = []
            for bid in block_ids:
                if bid < 0 or bid >= len(self.block_pool.blocks):
                    logger.warning(
                        f"block_id {bid} free blocks to vllm BlockPool configured range [0, {len(self.block_pool.blocks)}), skipped"
                    )
                    continue
                blocks.append(self.block_pool.blocks[bid])

            if blocks:
                self.block_pool.free_blocks(blocks)
                logger.info(f"free blocks to vllm: released {len(blocks)}  vLLM")
                return len(blocks)
            else:
                logger.warning("free blocks to vllm: released vLLM")
                return 0
        except Exception as e:
            logger.error(f"free blocks to vllm: released vLLM failed: {e}")
            return 0

    # Swap In
    def swap_in_user(self, user_id: str) -> bool:
        """Swap in user."""
        start_time = time.time()

        with self._lock:
            if user_id not in self.users:
                logger.warning(f"swap in user {user_id} not found")
                return False

            user_info = self.users[user_id]

            if not user_info.is_swapped_out:
                logger.warning(f"swap in user {user_id} evicted")
                return False

            if not user_info.backup_block_mapping:
                logger.warning(f"swap in user {user_id} ")
                return False

            num_blocks = len(user_info.backup_block_mapping)

            logger.info(f"swap in user: started Swap In  {user_id}, {num_blocks} ")

            new_block_ids = self._allocate_blocks_from_vllm(num_blocks)

            if new_block_ids is None:
                logger.error("swap in user vLLM failed")
                return False

            old_block_ids = list(user_info.backup_block_mapping.keys())
            cpu_indices = list(user_info.backup_block_mapping.values())

            try:
                if self.kv_cache_gpu is not None and self.kv_cache_cpu is not None:
                    self._copy_blocks_from_cpu(new_block_ids, cpu_indices)
            except Exception as e:

                if self.block_pool is not None:
                    try:
                        blocks = [
                            self.block_pool.blocks[bid]
                            for bid in new_block_ids
                            if 0 <= bid < len(self.block_pool.blocks)
                        ]
                        if blocks:
                            self.block_pool.free_blocks(blocks)
                    except Exception as free_err:
                        logger.error(f"swap in user: released GPU failed: {free_err}")
                logger.error(
                    f"Swap In swap in user {user_id}  CPU→GPU failed, GPU : {e}"
                )
                return False

            user_info.block_ids = set(new_block_ids)
            for block_id in new_block_ids:
                self.block_to_user[block_id] = user_id

            for cpu_idx in cpu_indices:
                self.cpu_free_indices.append(cpu_idx)
            user_info.backup_block_mapping.clear()

            user_info.is_swapped_out = False
            user_info.swap_in_time = time.time()

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.total_swap_ins += 1
            self.stats.total_blocks_swapped_in += num_blocks
            self.stats.total_swap_in_time_ms += elapsed_ms

            logger.info(
                f"Swap In swap in user: completed successfully:  {user_id}, "
                f"{num_blocks} , {elapsed_ms:.2f}ms"
            )

            return True

    def _allocate_blocks_from_vllm(self, num_blocks: int) -> Optional[List[int]]:

        if self.block_pool is None:

            logger.error(
                "BlockPool allocate blocks from vllm,unavailable vLLM allocation updated."
                " on_vllm_initialized()  BlockPool."
            )
            return None

        try:

            free_blocks = self.block_pool.get_num_free_blocks()
            if free_blocks < num_blocks:
                logger.error(
                    f"vLLM allocate blocks from vllm: insufficient capacity:  {num_blocks}, {free_blocks}"
                )
                return None

            blocks = self.block_pool.get_new_blocks(num_blocks)
            block_ids = [block.block_id for block in blocks]

            logger.info(
                f"allocate blocks from vllm vLLM allocation updated {len(block_ids)} "
            )
            return block_ids

        except Exception as e:
            logger.error(f"allocate blocks from vllm vLLM failed: {e}")
            return None

    def _copy_blocks_from_cpu(self, block_ids: List[int], cpu_indices: List[int]):

        is_6d = len(self.kv_cache_gpu.shape) == 6

        for block_id, cpu_idx in zip(block_ids, cpu_indices):
            try:
                if is_6d:
                    self.kv_cache_gpu[:, :, block_id, :, :, :].copy_(
                        self.kv_cache_cpu[:, :, cpu_idx, :, :, :], non_blocking=True
                    )
                else:
                    self.kv_cache_gpu[block_id].copy_(
                        self.kv_cache_cpu[cpu_idx], non_blocking=True
                    )
            except Exception as e:
                logger.error(
                    f"copy blocks from cpu CPU[{cpu_idx}] →  {block_id} failed: {e}"
                )
                raise

        if self.kv_cache_gpu.is_cuda:
            torch.cuda.synchronize()

    def get_user_block_count(self, user_id: str) -> int:
        """Return user block count."""
        with self._lock:
            if user_id in self.users:
                return len(self.users[user_id].block_ids)
            return 0

    def is_user_swapped_out(self, user_id: str) -> bool:
        """Return whether user swapped out."""
        with self._lock:
            if user_id in self.users:
                return self.users[user_id].is_swapped_out
            return False

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            avg_swap_out = (
                self.stats.total_swap_out_time_ms / self.stats.total_swap_outs
                if self.stats.total_swap_outs > 0
                else 0
            )
            avg_swap_in = (
                self.stats.total_swap_in_time_ms / self.stats.total_swap_ins
                if self.stats.total_swap_ins > 0
                else 0
            )

            return {
                "total_swap_outs": self.stats.total_swap_outs,
                "total_swap_ins": self.stats.total_swap_ins,
                "total_blocks_swapped_out": self.stats.total_blocks_swapped_out,
                "total_blocks_swapped_in": self.stats.total_blocks_swapped_in,
                "blocks_freed_to_vllm": self.stats.blocks_freed_to_vllm,
                "avg_swap_out_ms": avg_swap_out,
                "avg_swap_in_ms": avg_swap_in,
                "total_users": len(self.users),
                "swapped_out_users": sum(
                    1 for u in self.users.values() if u.is_swapped_out
                ),
                "cpu_free_blocks": len(self.cpu_free_indices),
                "cpu_total_blocks": self.cpu_max_blocks,
            }

    def get_active_gpu_block_ids(self) -> Set[int]:
        """Return active gpu block ids."""
        with self._lock:
            active: Set[int] = set()
            for user_info in self.users.values():
                if not user_info.is_swapped_out:
                    active.update(user_info.block_ids)
            return active

    def __repr__(self) -> str:
        return (
            f"VLLMBlockManager("
            f"users={len(self.users)}, "
            f"swapped_out={sum(1 for u in self.users.values() if u.is_swapped_out)})"
        )


def create_vllm_block_manager(
    kv_cache_gpu: Optional[torch.Tensor] = None,
    cpu_swap_space_gb: float = 32.0,
    block_pool: Optional[Any] = None,
) -> VLLMBlockManager:
    """Create vllm block manager."""
    return VLLMBlockManager(
        kv_cache_gpu=kv_cache_gpu,
        cpu_swap_space_gb=cpu_swap_space_gb,
        block_pool=block_pool,
    )
