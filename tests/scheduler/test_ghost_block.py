"""ResiMem test ghost block module."""

import pytest
import time

from core.scheduler.ghost_block import (
    GhostBlock,
    GhostBlockManager,
    GhostBlockStats,
)


class TestGhostBlock:
    """TestGhostBlock implementation."""

    def test_ghost_block_creation(self):

        ghost = GhostBlock(
            block_id=1,
            token_ids=[100, 101, 102],
            position=0,
            sequence_id="seq_1",
            user_id="user_1",
            evicted_time=time.time(),
        )

        assert ghost.block_id == 1
        assert ghost.token_ids == [100, 101, 102]
        assert ghost.sequence_id == "seq_1"
        assert ghost.user_id == "user_1"
        assert ghost.age_seconds() >= 0

    def test_ghost_block_age(self):

        ghost = GhostBlock(
            block_id=1,
            token_ids=[100],
            position=0,
            sequence_id="seq_1",
            user_id="user_1",
            evicted_time=time.time() - 10.0,
        )

        age = ghost.age_seconds()
        assert 9.0 < age < 11.0


class TestGhostBlockManager:
    """TestGhostBlockManager implementation."""

    @pytest.fixture
    def manager(self):
        """Manager."""
        return GhostBlockManager()

    def test_create_ghost(self, manager):

        ghost = manager.create_ghost(
            block_id=1,
            token_ids=[100, 101, 102],
            sequence_id="seq_1",
            user_id="user_1",
            position=0,
        )

        assert ghost.block_id == 1
        assert len(ghost.token_ids) == 3
        assert manager.has_ghosts(sequence_id="seq_1")
        assert manager.has_ghosts(user_id="user_1")

    def test_restore_from_ghost(self, manager):

        manager.create_ghost(
            block_id=1,
            token_ids=[100, 101],
            sequence_id="seq_1",
            user_id="user_1",
            position=0,
        )

        ghost = manager.restore_from_ghost(1)
        assert ghost is not None
        assert ghost.block_id == 1
        assert ghost.token_ids == [100, 101]

    def test_remove_ghost(self, manager):

        manager.create_ghost(
            block_id=1,
            token_ids=[100],
            sequence_id="seq_1",
            user_id="user_1",
            position=0,
        )

        assert manager.has_ghosts(sequence_id="seq_1")

        success = manager.remove_ghost(1)
        assert success
        assert not manager.has_ghosts(sequence_id="seq_1")

    def test_get_sequence_ghosts(self, manager):

        for i in range(3):
            manager.create_ghost(
                block_id=i,
                token_ids=[100 + i],
                sequence_id="seq_1",
                user_id="user_1",
                position=i * 10,
            )

        ghosts = manager.get_sequence_ghosts("seq_1")
        assert len(ghosts) == 3
        assert all(g.sequence_id == "seq_1" for g in ghosts)

    def test_get_user_ghosts(self, manager):

        for seq_id in ["seq_1", "seq_2"]:
            for i in range(2):
                manager.create_ghost(
                    block_id=hash(f"{seq_id}_{i}") % (2**31),
                    token_ids=[100 + i],
                    sequence_id=seq_id,
                    user_id="user_1",
                    position=i * 10,
                )

        user_ghosts = manager.get_user_ghosts("user_1")
        assert len(user_ghosts) == 2
        assert "seq_1" in user_ghosts
        assert "seq_2" in user_ghosts
        assert len(user_ghosts["seq_1"]) == 2
        assert len(user_ghosts["seq_2"]) == 2

    def test_clear_sequence_ghosts(self, manager):

        for i in range(5):
            manager.create_ghost(
                block_id=i,
                token_ids=[100 + i],
                sequence_id="seq_1",
                user_id="user_1",
                position=i * 10,
            )

        assert manager.has_ghosts(sequence_id="seq_1")

        count = manager.clear_sequence_ghosts("seq_1")
        assert count == 5
        assert not manager.has_ghosts(sequence_id="seq_1")

    def test_clear_user_ghosts(self, manager):

        for seq_id in ["seq_1", "seq_2"]:
            for i in range(3):
                manager.create_ghost(
                    block_id=hash(f"{seq_id}_{i}") % (2**31),
                    token_ids=[100 + i],
                    sequence_id=seq_id,
                    user_id="user_1",
                    position=i * 10,
                )

        assert manager.has_ghosts(user_id="user_1")

        count = manager.clear_user_ghosts("user_1")
        assert count == 6
        assert not manager.has_ghosts(user_id="user_1")

    def test_get_stats(self, manager):

        for i in range(10):
            manager.create_ghost(
                block_id=i,
                token_ids=[100 + j for j in range(i + 1)],
                sequence_id=f"seq_{i % 3}",
                user_id=f"user_{i % 2}",
                position=i * 10,
            )

        stats = manager.get_stats()
        assert stats.total_ghosts == 10
        assert stats.total_tokens > 0
        assert stats.total_sequences == 3
        assert stats.total_users == 2
        assert stats.oldest_ghost_age_s >= 0

    def test_get_memory_saved_estimate(self, manager):

        for i in range(10):
            manager.create_ghost(
                block_id=i,
                token_ids=[100 + j for j in range(100)],  # 100 tokens/block
                sequence_id="seq_1",
                user_id="user_1",
                position=i * 100,
            )

        # 1000 * 2KB = 2MB ≈ 0.002 GB
        saved_gb = manager.get_memory_saved_estimate()
        assert saved_gb > 0
        assert saved_gb < 0.1
