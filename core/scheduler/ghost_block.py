"""ResiMem ghost block module."""

import time
import logging
import threading
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GhostBlock:
    """GhostBlock implementation."""

    block_id: int
    token_ids: List[int]
    position: int
    sequence_id: str
    user_id: str
    evicted_time: float
    layer_idx: Optional[int] = None

    def __post_init__(self):
        if self.evicted_time == 0:
            self.evicted_time = time.time()

    def age_seconds(self) -> float:
        """Age seconds."""
        return time.time() - self.evicted_time


@dataclass
class GhostBlockStats:
    """GhostBlockStats implementation."""

    total_ghosts: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    total_users: int = 0
    oldest_ghost_age_s: float = 0
    newest_ghost_age_s: float = 0

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            "total_ghosts": self.total_ghosts,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "total_users": self.total_users,
            "oldest_ghost_age_s": self.oldest_ghost_age_s,
            "newest_ghost_age_s": self.newest_ghost_age_s,
        }


class GhostBlockManager:
    """GhostBlockManager implementation."""

    def __init__(self):

        self._lock = threading.RLock()

        self.ghost_blocks: Dict[int, GhostBlock] = {}  # block_id -> GhostBlock

        self.sequence_ghosts: Dict[str, List[int]] = {}  # sequence_id -> [block_ids]
        self.user_ghosts: Dict[str, Set[str]] = {}  # user_id -> {sequence_ids}

        self.stats = GhostBlockStats()

        logger.info("GhostBlockManager init: completed successfully")

    def create_ghost(
        self,
        block_id: int,
        token_ids: List[int],
        sequence_id: str,
        user_id: str,
        position: int,
        layer_idx: Optional[int] = None,
    ) -> GhostBlock:
        """Create ghost."""
        with self._lock:

            ghost = GhostBlock(
                block_id=block_id,
                token_ids=token_ids.copy(),
                position=position,
                sequence_id=sequence_id,
                user_id=user_id,
                evicted_time=time.time(),
                layer_idx=layer_idx,
            )

            self.ghost_blocks[block_id] = ghost

            if sequence_id not in self.sequence_ghosts:
                self.sequence_ghosts[sequence_id] = []
            self.sequence_ghosts[sequence_id].append(block_id)

            if user_id not in self.user_ghosts:
                self.user_ghosts[user_id] = set()
            self.user_ghosts[user_id].add(sequence_id)

            self._update_stats()

            logger.info(
                f"create ghost: created Ghost Block: block_id={block_id}, "
                f"tokens={len(token_ids)}, sequence={sequence_id}, "
                f"position={position}"
            )

            return ghost

    def restore_from_ghost(
        self,
        block_id: int,
    ) -> Optional[GhostBlock]:
        """Restore from ghost."""
        with self._lock:
            ghost = self.ghost_blocks.get(block_id)

            if ghost is None:
                logger.error(f"Ghost Block {block_id} restore from ghost: not found")
                return None

            logger.info(
                f"restore from ghost: restored Ghost Block: block_id={block_id}, "
                f"tokens={len(ghost.token_ids)}, "
                f"age={ghost.age_seconds():.2f}s"
            )

            return ghost

    def remove_ghost(self, block_id: int) -> bool:
        """Remove ghost."""
        with self._lock:
            if block_id not in self.ghost_blocks:
                logger.warning(f"Ghost Block {block_id} remove ghost: not found")
                return False

            ghost = self.ghost_blocks[block_id]

            if ghost.sequence_id in self.sequence_ghosts:
                self.sequence_ghosts[ghost.sequence_id].remove(block_id)
                if not self.sequence_ghosts[ghost.sequence_id]:
                    del self.sequence_ghosts[ghost.sequence_id]

            if ghost.user_id in self.user_ghosts:
                if ghost.sequence_id in self.user_ghosts[ghost.user_id]:

                    seq_ghosts = self.sequence_ghosts.get(ghost.sequence_id, [])
                    if not seq_ghosts:
                        self.user_ghosts[ghost.user_id].discard(ghost.sequence_id)
                if not self.user_ghosts[ghost.user_id]:
                    del self.user_ghosts[ghost.user_id]

            del self.ghost_blocks[block_id]

            self._update_stats()

            logger.info(f"remove ghost Ghost Block: {block_id}")
            return True

    def get_sequence_ghosts(self, sequence_id: str) -> List[GhostBlock]:
        """Return sequence ghosts."""
        with self._lock:
            block_ids = self.sequence_ghosts.get(sequence_id, [])
            return [
                self.ghost_blocks[bid] for bid in block_ids if bid in self.ghost_blocks
            ]

    def get_user_ghosts(self, user_id: str) -> Dict[str, List[GhostBlock]]:
        """Return user ghosts."""
        with self._lock:
            sequence_ids = self.user_ghosts.get(user_id, set()).copy()
            result = {}

            for seq_id in sequence_ids:
                block_ids = self.sequence_ghosts.get(seq_id, [])
                result[seq_id] = [
                    self.ghost_blocks[bid]
                    for bid in block_ids
                    if bid in self.ghost_blocks
                ]

            return result

    def clear_sequence_ghosts(self, sequence_id: str) -> int:
        """Clear sequence ghosts."""
        with self._lock:
            block_ids = self.sequence_ghosts.pop(sequence_id, [])
            count = 0
            for block_id in block_ids:
                ghost = self.ghost_blocks.pop(block_id, None)
                if ghost is not None:
                    count += 1

            for user_id, seq_set in self.user_ghosts.items():
                seq_set.discard(sequence_id)

            self._update_stats()

        return count

    def clear_user_ghosts(self, user_id: str) -> int:
        """Clear user ghosts."""
        with self._lock:
            sequence_ids = list(self.user_ghosts.pop(user_id, set()))
            count = 0
            for seq_id in sequence_ids:
                block_ids = self.sequence_ghosts.pop(seq_id, [])
                for block_id in block_ids:
                    ghost = self.ghost_blocks.pop(block_id, None)
                    if ghost is not None:
                        count += 1

            self._update_stats()

        return count

    def _update_stats(self):

        self.stats.total_ghosts = len(self.ghost_blocks)
        self.stats.total_tokens = sum(
            len(ghost.token_ids) for ghost in self.ghost_blocks.values()
        )
        self.stats.total_sequences = len(self.sequence_ghosts)
        self.stats.total_users = len(self.user_ghosts)

        if self.ghost_blocks:
            ages = [ghost.age_seconds() for ghost in self.ghost_blocks.values()]
            self.stats.oldest_ghost_age_s = max(ages)
            self.stats.newest_ghost_age_s = min(ages)
        else:
            self.stats.oldest_ghost_age_s = 0
            self.stats.newest_ghost_age_s = 0

    def get_stats(self) -> GhostBlockStats:
        """Return stats."""
        with self._lock:
            self._update_stats()
            return self.stats

    def has_ghosts(
        self, sequence_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> bool:
        """Return whether the object has ghosts."""
        with self._lock:
            if sequence_id is not None:
                return (
                    sequence_id in self.sequence_ghosts
                    and len(self.sequence_ghosts[sequence_id]) > 0
                )

            if user_id is not None:
                return (
                    user_id in self.user_ghosts and len(self.user_ghosts[user_id]) > 0
                )

            return len(self.ghost_blocks) > 0

    def get_memory_saved_estimate(self) -> float:
        """Return memory saved estimate."""
        with self._lock:
            bytes_per_token = 2 * 1024
            total_bytes = self.stats.total_tokens * bytes_per_token
            return total_bytes / (1024**3)

    def reset(self):
        """Reset."""
        with self._lock:
            self.ghost_blocks.clear()
            self.sequence_ghosts.clear()
            self.user_ghosts.clear()
            self._update_stats()
