"""ResiMem resimem block pool module."""

from typing import Optional
import logging

try:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
except ImportError as exc:
    raise ImportError("ResiMemBlockPool resimem operation vLLM V1 BlockPool .") from exc

from ..memory import MemoryPool

logger = logging.getLogger(__name__)


class ResiMemBlockPool(BlockPool):
    """ResiMemBlockPool implementation."""

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        memory_pool: Optional[MemoryPool] = None,
        enable_kv_cache_events: bool = False,
        metrics_collector=None,
    ):

        super().__init__(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        self.memory_pool = memory_pool
        self.expansion_count = 0
        self.total_expanded_blocks = 0

        logger.info(f"✅ ResiMemBlockPool initialized with {num_gpu_blocks} blocks")
        if memory_pool:
            logger.info(f"   MemoryPool available: {memory_pool}")
        else:
            logger.warning("   ⚠️  No MemoryPool provided, expansion will fail")

    def expand_capacity(self, additional_blocks: int) -> bool:
        """Expand capacity."""
        if additional_blocks <= 0:
            logger.warning(f"Invalid additional_blocks: {additional_blocks}")
            return False

        logger.info(f"🔄 Expanding BlockPool by {additional_blocks} blocks...")

        try:

            if self.memory_pool is not None:

                pass

            start_block_id = len(self.blocks)
            new_blocks: list[KVCacheBlock] = []

            for i in range(additional_blocks):
                block_id = start_block_id + i

                block = KVCacheBlock(block_id=block_id)
                new_blocks.append(block)

            logger.debug(f"   Created {len(new_blocks)} new KVCacheBlock objects")
            logger.debug(
                f"   Block IDs: [{start_block_id}, {start_block_id + additional_blocks - 1}]"
            )

            self.blocks.extend(new_blocks)

            self.free_block_queue.append_n(new_blocks)

            self.num_gpu_blocks += additional_blocks
            self.expansion_count += 1
            self.total_expanded_blocks += additional_blocks

            logger.info(
                f"✅ BlockPool expanded successfully!\n"
                f"   Added blocks: {additional_blocks}\n"
                f"   Total blocks: {self.num_gpu_blocks}\n"
                f"   Free blocks: {self.get_num_free_blocks()}\n"
                f"   Expansion count: {self.expansion_count}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to expand BlockPool: {e}", exc_info=True)
            return False

    def get_expansion_stats(self) -> dict:
        """Return expansion stats."""
        return {
            "expansion_count": self.expansion_count,
            "total_expanded_blocks": self.total_expanded_blocks,
            "current_total_blocks": self.num_gpu_blocks,
            "current_free_blocks": self.get_num_free_blocks(),
            "expansion_rate": (
                self.total_expanded_blocks / self.num_gpu_blocks
                if self.num_gpu_blocks > 0
                else 0.0
            ),
        }

    def print_expansion_stats(self):
        """Print expansion stats."""
        stats = self.get_expansion_stats()

        print("\n" + "=" * 70)
        print("ResiMemBlockPool Expansion Statistics")
        print("=" * 70)
        print(f"Total Blocks:         {stats['current_total_blocks']}")
        print(f"Free Blocks:          {stats['current_free_blocks']}")
        print(f"Expansion Count:      {stats['expansion_count']}")
        print(f"Total Expanded:       {stats['total_expanded_blocks']}")
        print(f"Expansion Rate:       {stats['expansion_rate']*100:.1f}%")
        print("=" * 70 + "\n")

    def __repr__(self) -> str:
        return (
            f"ResiMemBlockPool(total={self.num_gpu_blocks}, "
            f"free={self.get_num_free_blocks()}, "
            f"expansions={self.expansion_count})"
        )
