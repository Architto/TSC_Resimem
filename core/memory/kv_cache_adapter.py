"""ResiMem kv cache adapter module."""

import threading

import torch
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import logging

from .giant_tensor import GiantTensor
from .memory_pool import MemoryPool

logger = logging.getLogger(__name__)


@dataclass
class KVCacheConfig:
    """KVCacheConfig implementation."""

    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 16
    dtype: torch.dtype = torch.float16

    @property
    def bytes_per_token(self) -> int:
        """Bytes per token."""

        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size

    @property
    def bytes_per_block(self) -> int:
        """Bytes per block."""
        return self.bytes_per_token * self.block_size

    def __repr__(self) -> str:
        return (
            f"KVCacheConfig(layers={self.num_layers}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"block_size={self.block_size}, dtype={self.dtype}, "
            f"bytes_per_block={self.bytes_per_block / 1024:.1f}KB)"
        )


class KVCacheAdapter:
    """KVCacheAdapter implementation."""

    def __init__(
        self,
        giant_tensor_gpu: GiantTensor,
        config: KVCacheConfig,
        initial_blocks: int = 1024,
        max_blocks: int = 4096,
        cpu_swap_space_gb: float = 32.0,
    ):

        self.giant_tensor_gpu = giant_tensor_gpu
        self.config = config
        self.initial_blocks = initial_blocks
        self.max_blocks = max_blocks

        self.bytes_per_block = config.bytes_per_block
        initial_bytes = initial_blocks * self.bytes_per_block
        max_bytes = max_blocks * self.bytes_per_block

        logger.info(f"KVCacheAdapter init:")
        logger.info(f"  init: {config}")
        logger.info(f"  init: {initial_blocks}")
        logger.info(f"  init: {max_blocks}")
        logger.info(f"  init: {self.bytes_per_block / 1024:.2f} KB")
        logger.info(f"  init: {initial_bytes / 1024**3:.2f} GB")
        logger.info(f"  init: {max_bytes / 1024**3:.2f} GB")

        self._raw_gpu_memory = giant_tensor_gpu.get_slice(0, max_bytes)

        # [num_layers, 2, max_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache_gpu = self._reshape_to_kv_cache(
            self._raw_gpu_memory, max_blocks, config
        )

        self._lock = threading.RLock()

        self.current_blocks = initial_blocks

        cpu_bytes = int(cpu_swap_space_gb * 1024**3)
        cpu_blocks = min(cpu_bytes // self.bytes_per_block, max_blocks)

        self.kv_cache_cpu = torch.zeros(
            config.num_layers,
            2,
            cpu_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
            dtype=config.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.cpu_blocks = cpu_blocks

        logger.info(
            f"  CPU init: {cpu_blocks}  ({cpu_blocks * self.bytes_per_block / 1024**3:.2f} GB)"
        )
        logger.info("✅ KVCacheAdapter init: completed successfully")

    def _reshape_to_kv_cache(
        self,
        raw_memory: torch.Tensor,
        num_blocks: int,
        config: KVCacheConfig,
    ) -> torch.Tensor:

        # [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
        target_shape = (
            config.num_layers,
            2,  # K and V
            num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        )

        total_elements = (
            config.num_layers
            * 2
            * num_blocks
            * config.block_size
            * config.num_kv_heads
            * config.head_dim
        )

        dtype_size = torch.tensor([], dtype=config.dtype).element_size()
        needed_bytes = total_elements * dtype_size

        assert (
            raw_memory.numel() >= needed_bytes
        ), f"reshape to kv cache: insufficient capacity:  {needed_bytes} ,  {raw_memory.numel()} "

        raw_bytes = raw_memory[:needed_bytes]

        kv_cache = raw_bytes.view(config.dtype).reshape(target_shape)

        logger.debug(f"KV Cache reshape to kv cache: {kv_cache.shape}")
        logger.debug(f"KV Cache reshape to kv cache: {kv_cache.device}")
        logger.debug(f"KV Cache reshape to kv cache: {kv_cache.dtype}")

        return kv_cache

    def get_block(self, block_id: int, device: str = "gpu") -> torch.Tensor:
        """Return block."""
        if device == "gpu":
            if block_id >= self.current_blocks:
                raise IndexError(
                    f"get block ID {block_id} configured range [0, {self.current_blocks})"
                )
            return self.kv_cache_gpu[:, :, block_id, :, :, :]
        else:
            if block_id >= self.cpu_blocks:
                raise IndexError(
                    f"get block ID {block_id}  CPU configured range [0, {self.cpu_blocks})"
                )
            return self.kv_cache_cpu[:, :, block_id, :, :, :]

    def copy_block_gpu_to_cpu(self, gpu_block_id: int, cpu_block_id: int):
        """Copy block gpu to cpu."""
        self.kv_cache_cpu[:, :, cpu_block_id, :, :, :].copy_(
            self.kv_cache_gpu[:, :, gpu_block_id, :, :, :], non_blocking=True
        )

    def copy_block_cpu_to_gpu(self, cpu_block_id: int, gpu_block_id: int):
        """Copy block cpu to gpu."""
        self.kv_cache_gpu[:, :, gpu_block_id, :, :, :].copy_(
            self.kv_cache_cpu[:, :, cpu_block_id, :, :, :], non_blocking=True
        )

    def batch_copy_gpu_to_cpu(self, block_mapping: Dict[int, int]):
        """Batch copy gpu to cpu."""
        for gpu_id, cpu_id in block_mapping.items():
            self.copy_block_gpu_to_cpu(gpu_id, cpu_id)

        if self.kv_cache_gpu.is_cuda:
            torch.cuda.synchronize()

    def batch_copy_cpu_to_gpu(self, block_mapping: Dict[int, int]):
        """Batch copy cpu to gpu."""
        for gpu_id, cpu_id in block_mapping.items():
            self.copy_block_cpu_to_gpu(cpu_id, gpu_id)

        if self.kv_cache_gpu.is_cuda:
            torch.cuda.synchronize()

    def expand(self, additional_blocks: int) -> bool:
        """Expand."""
        with self._lock:
            new_total = self.current_blocks + additional_blocks

            if new_total > self.max_blocks:
                logger.error(
                    f"expand: unavailable: request {new_total} ,  {self.max_blocks} "
                )
                return False

            old_blocks = self.current_blocks
            self.current_blocks = new_total

            logger.info(
                f"KV Cache expand: expanded: {old_blocks} → {new_total}  "
                f"(+{additional_blocks}, {additional_blocks * self.bytes_per_block / 1024**2:.1f} MB)"
            )

            return True

    def shrink(self, blocks_to_remove: int) -> bool:
        """Shrink."""
        with self._lock:
            if blocks_to_remove >= self.current_blocks:
                logger.error(
                    f"shrink: unavailable: request {blocks_to_remove}, {self.current_blocks}"
                )
                return False

            old_blocks = self.current_blocks
            self.current_blocks -= blocks_to_remove

            logger.info(
                f"KV Cache shrink: shrunk: {old_blocks} → {self.current_blocks}  "
                f"(-{blocks_to_remove}, -{blocks_to_remove * self.bytes_per_block / 1024**2:.1f} MB)"
            )

            return True

    def get_stats(self) -> Dict:
        """Return stats."""
        used_bytes = self.current_blocks * self.bytes_per_block
        max_bytes = self.max_blocks * self.bytes_per_block

        return {
            "current_blocks": self.current_blocks,
            "max_blocks": self.max_blocks,
            "cpu_blocks": self.cpu_blocks,
            "bytes_per_block": self.bytes_per_block,
            "used_bytes_gpu": used_bytes,
            "used_gb_gpu": used_bytes / 1024**3,
            "max_bytes_gpu": max_bytes,
            "max_gb_gpu": max_bytes / 1024**3,
            "utilization": self.current_blocks / self.max_blocks,
            "config": str(self.config),
        }

    def __repr__(self) -> str:
        return (
            f"KVCacheAdapter(blocks={self.current_blocks}/{self.max_blocks}, "
            f"used={self.current_blocks * self.bytes_per_block / 1024**3:.2f}GB)"
        )


def create_kv_cache_adapter(
    giant_tensor: GiantTensor,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    initial_blocks: int = 1024,
    max_blocks: int = 4096,
) -> KVCacheAdapter:
    """Create kv cache adapter."""
    config = KVCacheConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )

    return KVCacheAdapter(
        giant_tensor_gpu=giant_tensor,
        config=config,
        initial_blocks=initial_blocks,
        max_blocks=max_blocks,
    )
