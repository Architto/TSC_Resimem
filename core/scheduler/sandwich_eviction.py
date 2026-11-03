"""ResiMem sandwich eviction module."""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .ghost_block import GhostBlockManager, GhostBlock

logger = logging.getLogger(__name__)
DEFAULT_TOKENS_PER_BLOCK = 128


# float16 × 2(K+V) × num_layers × num_kv_heads × head_dim

ESTIMATED_BYTES_PER_TOKEN = 2 * 1024  # 2 KB/token


@dataclass
class SandwichConfig:
    """SandwichConfig implementation."""

    head_tokens: int = 200
    tail_tokens: int = 2000
    min_body_tokens: int = 1000

    evict_threshold_tokens: int = 10000
    evict_threshold_gb: float = 5.0

    auto_restore_on_access: bool = True
    partial_prefill_batch_size: int = 512

    tokens_per_block: int = 16


@dataclass
class EvictionStats:
    """EvictionStats implementation."""

    sequence_id: str
    evicted_tokens: int = 0
    freed_blocks: int = 0
    freed_memory_gb: float = 0.0
    head_tokens: int = 0
    tail_tokens: int = 0
    eviction_time_s: float = 0.0

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            "sequence_id": self.sequence_id,
            "evicted_tokens": self.evicted_tokens,
            "freed_blocks": self.freed_blocks,
            "freed_memory_gb": self.freed_memory_gb,
            "head_tokens": self.head_tokens,
            "tail_tokens": self.tail_tokens,
            "eviction_time_s": self.eviction_time_s,
        }


class SandwichEviction:
    """SandwichEviction implementation."""

    def __init__(
        self,
        ghost_manager: GhostBlockManager,
        config: Optional[SandwichConfig] = None,
    ):

        self.ghost_manager = ghost_manager
        self.config = config or SandwichConfig()

        self._virtual_block_counter = 0

        self.total_evictions = 0
        self.total_restores = 0
        self.total_freed_memory_gb = 0.0

        logger.info("SandwichEviction init: completed successfully")
        logger.info(f"  HEAD tokens: {self.config.head_tokens}")
        logger.info(f"  TAIL tokens: {self.config.tail_tokens}")
        logger.info(f"  init: threshold: {self.config.evict_threshold_tokens} tokens")

    def identify_eviction_region(
        self,
        total_tokens: int,
    ) -> Tuple[int, int, int]:
        """Identify eviction region."""

        head_end = min(self.config.head_tokens, total_tokens)

        tail_start = max(head_end, total_tokens - self.config.tail_tokens)

        body_start = head_end
        body_end = tail_start

        logger.debug(
            f"identify eviction region: total={total_tokens}, "
            f"HEAD=[0, {head_end}), "
            f"BODY=[{body_start}, {body_end}), "
            f"TAIL=[{body_end}, {total_tokens})"
        )

        return head_end, body_start, body_end

    def should_evict(
        self,
        total_tokens: int,
        current_memory_gb: Optional[float] = None,
    ) -> bool:
        """Should evict."""

        if total_tokens < self.config.evict_threshold_tokens:
            return False

        if current_memory_gb is not None:
            if current_memory_gb < self.config.evict_threshold_gb:
                return False

        head_end, body_start, body_end = self.identify_eviction_region(total_tokens)
        body_tokens = body_end - body_start

        if body_tokens < self.config.min_body_tokens:
            logger.info(f"BODY should evict ({body_tokens} tokens),evicted")
            return False

        return True

    def sandwich_evict(
        self,
        sequence_id: str,
        user_id: str,
        total_tokens: int,
        token_ids: List[int],
        block_ids: Optional[List[int]] = None,
    ) -> EvictionStats:
        """Sandwich evict."""
        start_time = time.time()

        head_end, body_start, body_end = self.identify_eviction_region(total_tokens)

        if body_end <= body_start:
            logger.info(f"sandwich evict {sequence_id} ,evicted")
            return EvictionStats(
                sequence_id=sequence_id,
                eviction_time_s=time.time() - start_time,
            )

        body_token_ids = token_ids[body_start:body_end]

        logger.info(f"sandwich evict: evicted: {sequence_id}")
        logger.info(f"  sandwich evict tokens: {total_tokens}")
        logger.info(f"  HEAD: [0, {head_end}) = {head_end} tokens")
        logger.info(
            f"  BODY: [{body_start}, {body_end}) = {len(body_token_ids)} tokens"
        )
        logger.info(
            f"  TAIL: [{body_end}, {total_tokens}) = {total_tokens - body_end} tokens"
        )

        freed_blocks = 0

        if block_ids is not None:

            tokens_per_block = self.config.tokens_per_block

            body_block_start = body_start // tokens_per_block
            body_block_end = (body_end + tokens_per_block - 1) // tokens_per_block
            body_block_end = min(body_block_end, len(block_ids))
            body_block_start = min(body_block_start, body_block_end)

            body_block_ids = block_ids[body_block_start:body_block_end]

            for i, block_id in enumerate(body_block_ids):

                global_token_start = (body_block_start + i) * tokens_per_block
                global_token_end = min(
                    global_token_start + tokens_per_block, total_tokens
                )

                local_start = max(global_token_start, body_start)
                local_end = min(global_token_end, body_end)

                if local_start >= local_end:
                    continue

                block_tokens = token_ids[local_start:local_end]

                if not block_tokens:
                    continue

                self.ghost_manager.create_ghost(
                    block_id=block_id,
                    token_ids=block_tokens,
                    sequence_id=sequence_id,
                    user_id=user_id,
                    position=local_start,
                )

                freed_blocks += 1
        else:

            batch_size = DEFAULT_TOKENS_PER_BLOCK
            num_blocks = (len(body_token_ids) + batch_size - 1) // batch_size

            for i in range(num_blocks):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(body_token_ids))
                block_tokens = body_token_ids[start_idx:end_idx]

                self._virtual_block_counter += 1
                virtual_block_id = self._virtual_block_counter

                self.ghost_manager.create_ghost(
                    block_id=virtual_block_id,
                    token_ids=block_tokens,
                    sequence_id=sequence_id,
                    user_id=user_id,
                    position=body_start + start_idx,
                )

                freed_blocks += 1

        freed_bytes = len(body_token_ids) * ESTIMATED_BYTES_PER_TOKEN
        freed_memory_gb = freed_bytes / (1024**3)

        self.total_evictions += 1
        self.total_freed_memory_gb += freed_memory_gb

        eviction_time = time.time() - start_time

        stats = EvictionStats(
            sequence_id=sequence_id,
            evicted_tokens=len(body_token_ids),
            freed_blocks=freed_blocks,
            freed_memory_gb=freed_memory_gb,
            head_tokens=head_end,
            tail_tokens=total_tokens - body_end,
            eviction_time_s=eviction_time,
        )

        logger.info(f"sandwich evict: completed successfully: {sequence_id}")
        logger.info(f"  sandwich evict: evicted tokens: {stats.evicted_tokens}")
        logger.info(f"  sandwich evict: released blocks: {stats.freed_blocks}")
        logger.info(f"  sandwich evict: {stats.freed_memory_gb:.2f} GB")
        logger.info(f"  sandwich evict: {stats.eviction_time_s:.3f}s")

        return stats

    def restore(
        self,
        sequence_id: str,
    ) -> bool:
        """Restore."""

        ghosts = self.ghost_manager.get_sequence_ghosts(sequence_id)

        if not ghosts:
            logger.warning(f"restore {sequence_id}  Ghost Blocks")
            return False

        logger.info(f"restore: restored: {sequence_id}")
        logger.info(f"  Ghost Blocks: {len(ghosts)}")
        logger.info(f"  restore tokens: {sum(len(g.token_ids) for g in ghosts)}")

        ghosts.sort(key=lambda g: g.position)

        all_token_ids = []
        for ghost in ghosts:
            all_token_ids.extend(ghost.token_ids)

        logger.info(f"restore: ready Partial Prefill: {len(all_token_ids)} tokens")

        cleared = self.ghost_manager.clear_sequence_ghosts(sequence_id)

        self.total_restores += 1

        logger.info(
            f"restore: completed successfully: {sequence_id},  {cleared}  Ghost Blocks"
        )

        return True

    def get_stats(self) -> Dict:
        """Return stats."""
        ghost_stats = self.ghost_manager.get_stats()

        return {
            "total_evictions": self.total_evictions,
            "total_restores": self.total_restores,
            "total_freed_memory_gb": self.total_freed_memory_gb,
            "ghost_stats": ghost_stats.to_dict(),
            "config": {
                "head_tokens": self.config.head_tokens,
                "tail_tokens": self.config.tail_tokens,
                "evict_threshold_tokens": self.config.evict_threshold_tokens,
            },
        }

    def reset(self):
        """Reset."""
        self.total_evictions = 0
        self.total_restores = 0
        self.total_freed_memory_gb = 0.0
        self._virtual_block_counter = 0
