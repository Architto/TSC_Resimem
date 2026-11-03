"""ResiMem test sandwich eviction module."""

import pytest

from core.scheduler.ghost_block import GhostBlockManager
from core.scheduler.sandwich_eviction import (
    SandwichConfig,
    SandwichEviction,
    EvictionStats,
)


class TestSandwichConfig:
    """TestSandwichConfig implementation."""

    def test_default_config(self):

        config = SandwichConfig()
        assert config.head_tokens == 200
        assert config.tail_tokens == 2000
        assert config.min_body_tokens == 1000
        assert config.evict_threshold_tokens == 10000

    def test_custom_config(self):

        config = SandwichConfig(
            head_tokens=500,
            tail_tokens=3000,
            evict_threshold_tokens=20000,
        )
        assert config.head_tokens == 500
        assert config.tail_tokens == 3000
        assert config.evict_threshold_tokens == 20000


class TestSandwichEviction:
    """TestSandwichEviction implementation."""

    @pytest.fixture
    def ghost_manager(self):
        """Ghost manager."""
        return GhostBlockManager()

    @pytest.fixture
    def sandwich_eviction(self, ghost_manager):
        """Sandwich eviction."""
        config = SandwichConfig(
            head_tokens=200,
            tail_tokens=2000,
            min_body_tokens=1000,
            evict_threshold_tokens=10000,
        )
        return SandwichEviction(ghost_manager, config)

    def test_identify_eviction_region_short_sequence(self, sandwich_eviction):

        total_tokens = 500
        head_end, body_start, body_end = sandwich_eviction.identify_eviction_region(
            total_tokens
        )

        assert head_end == 200
        assert body_start == 200

        assert body_end == 200

    def test_identify_eviction_region_long_sequence(self, sandwich_eviction):

        total_tokens = 20000
        head_end, body_start, body_end = sandwich_eviction.identify_eviction_region(
            total_tokens
        )

        assert head_end == 200  # HEAD: [0, 200)
        assert body_start == 200
        assert body_end == 18000
        assert body_end - body_start == 17800

    def test_should_evict_below_threshold(self, sandwich_eviction):

        total_tokens = 5000
        should_evict = sandwich_eviction.should_evict(total_tokens)
        assert not should_evict

    def test_should_evict_above_threshold(self, sandwich_eviction):

        total_tokens = 15000
        should_evict = sandwich_eviction.should_evict(total_tokens)
        assert should_evict

    def test_should_evict_small_body(self, sandwich_eviction):

        # BODY = [200, 500) = 300 tokens

        total_tokens = 2500
        should_evict = sandwich_eviction.should_evict(total_tokens)
        assert not should_evict

    def test_sandwich_evict_basic(self, sandwich_eviction, ghost_manager):

        sequence_id = "seq_1"
        user_id = "user_1"
        total_tokens = 20000
        token_ids = list(range(total_tokens))

        stats = sandwich_eviction.sandwich_evict(
            sequence_id=sequence_id,
            user_id=user_id,
            total_tokens=total_tokens,
            token_ids=token_ids,
        )

        assert stats.sequence_id == sequence_id
        assert stats.evicted_tokens > 0
        assert stats.freed_blocks > 0
        assert stats.freed_memory_gb > 0
        assert stats.head_tokens == 200
        assert stats.tail_tokens == 2000

        assert ghost_manager.has_ghosts(sequence_id=sequence_id)
        ghosts = ghost_manager.get_sequence_ghosts(sequence_id)
        assert len(ghosts) > 0

    def test_sandwich_evict_with_block_ids(self, sandwich_eviction, ghost_manager):

        sequence_id = "seq_2"
        user_id = "user_1"
        total_tokens = 15000
        token_ids = list(range(total_tokens))
        tokens_per_block = sandwich_eviction.config.tokens_per_block
        total_blocks = (total_tokens + tokens_per_block - 1) // tokens_per_block
        block_ids = [100 + i for i in range(total_blocks)]

        stats = sandwich_eviction.sandwich_evict(
            sequence_id=sequence_id,
            user_id=user_id,
            total_tokens=total_tokens,
            token_ids=token_ids,
            block_ids=block_ids,
        )

        assert stats.evicted_tokens > 0
        _, body_start, body_end = sandwich_eviction.identify_eviction_region(
            total_tokens
        )
        body_block_start = body_start // tokens_per_block
        body_block_end = (body_end + tokens_per_block - 1) // tokens_per_block
        expected_freed_blocks = body_block_end - body_block_start
        assert stats.freed_blocks == expected_freed_blocks

        ghosts = ghost_manager.get_sequence_ghosts(sequence_id)
        assert len(ghosts) == expected_freed_blocks

    def test_sandwich_evict_short_sequence(self, sandwich_eviction, ghost_manager):

        sequence_id = "seq_short"
        user_id = "user_1"
        total_tokens = 500
        token_ids = list(range(total_tokens))

        stats = sandwich_eviction.sandwich_evict(
            sequence_id=sequence_id,
            user_id=user_id,
            total_tokens=total_tokens,
            token_ids=token_ids,
        )

        assert stats.evicted_tokens == 0
        assert stats.freed_blocks == 0
        assert not ghost_manager.has_ghosts(sequence_id=sequence_id)

    def test_restore_sequence(self, sandwich_eviction, ghost_manager):

        sequence_id = "seq_restore"
        user_id = "user_1"
        total_tokens = 20000
        token_ids = list(range(total_tokens))

        sandwich_eviction.sandwich_evict(
            sequence_id=sequence_id,
            user_id=user_id,
            total_tokens=total_tokens,
            token_ids=token_ids,
        )

        assert ghost_manager.has_ghosts(sequence_id=sequence_id)

        success = sandwich_eviction.restore(sequence_id)
        assert success

        assert not ghost_manager.has_ghosts(sequence_id=sequence_id)

    def test_restore_nonexistent_sequence(self, sandwich_eviction):

        success = sandwich_eviction.restore("nonexistent_seq")
        assert not success

    def test_get_stats(self, sandwich_eviction):

        for i in range(3):
            sandwich_eviction.sandwich_evict(
                sequence_id=f"seq_{i}",
                user_id="user_1",
                total_tokens=20000,
                token_ids=list(range(20000)),
            )

        sandwich_eviction.restore("seq_0")

        stats = sandwich_eviction.get_stats()
        assert stats["total_evictions"] == 3
        assert stats["total_restores"] == 1
        assert stats["total_freed_memory_gb"] > 0
        assert "ghost_stats" in stats
        assert "config" in stats

    def test_eviction_memory_estimate(self, sandwich_eviction):

        sequence_id = "seq_mem"
        user_id = "user_1"
        total_tokens = 50000
        token_ids = list(range(total_tokens))

        stats = sandwich_eviction.sandwich_evict(
            sequence_id=sequence_id,
            user_id=user_id,
            total_tokens=total_tokens,
            token_ids=token_ids,
        )

        # 50000 tokens, HEAD=200, TAIL=2000
        # BODY = 50000 - 200 - 2000 = 47800 tokens
        # 47800 * 2KB = 95600 KB ≈ 0.093 GB
        assert stats.evicted_tokens == 47800
        assert 0.08 < stats.freed_memory_gb < 0.12

    def test_multiple_sequences_same_user(self, sandwich_eviction, ghost_manager):

        user_id = "user_multi"

        for seq_idx in range(3):
            sandwich_eviction.sandwich_evict(
                sequence_id=f"seq_{seq_idx}",
                user_id=user_id,
                total_tokens=20000,
                token_ids=list(range(20000)),
            )

        for seq_idx in range(3):
            assert ghost_manager.has_ghosts(sequence_id=f"seq_{seq_idx}")

        assert ghost_manager.has_ghosts(user_id=user_id)
        user_ghosts = ghost_manager.get_user_ghosts(user_id)
        assert len(user_ghosts) == 3
