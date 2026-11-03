"""ResiMem resimem scheduler module."""

import logging
import threading
import time
from typing import Dict, Optional, Set, Callable
from dataclasses import dataclass, field

import torch

from .multi_user_types import (
    UserState,
    FocusManagerConfig,
    UserStateInfo,
)
from .focus_manager import FocusManager, create_focus_manager
from .group_swap import GroupSwapManager, GroupSwapConfig, create_group_swap_manager
from .api_interceptor import setup_request_interception, set_global_scheduler

from .ghost_block import GhostBlockManager
from .sandwich_eviction import SandwichEviction, SandwichConfig


logger = logging.getLogger(__name__)


@dataclass
class ResiMemSchedulerConfig:
    """ResiMemSchedulerConfig implementation."""

    cooling_off_period_s: float = 7.0
    max_users_in_gpu: int = 1

    enable_swap: bool = True
    target_swap_latency_ms: float = 100.0

    total_gpu_memory_gb: float = 24.0
    llm_initial_memory_gb: float = 16.0
    dnn_safety_buffer_gb: float = 0.5

    enable_sandwich_eviction: bool = True
    head_tokens: int = 200
    tail_tokens: int = 2000
    evict_threshold_tokens: int = 10000
    evict_threshold_gb: float = 5.0

    enable_logging: bool = True


class ResiMemScheduler:
    """ResiMemScheduler implementation."""

    def __init__(
        self,
        config: ResiMemSchedulerConfig,
        kv_cache_gpu: Optional[torch.Tensor] = None,
        kv_cache_cpu: Optional[torch.Tensor] = None,
        shrink_callback: Optional[Callable[[int], bool]] = None,
        dnn_degrade_callback: Optional[Callable[[int], bool]] = None,
    ):

        self.config = config
        self.shrink_callback = shrink_callback
        self.dnn_degrade_callback = dnn_degrade_callback

        swap_config = GroupSwapConfig(
            target_latency_ms=config.target_swap_latency_ms,
        )
        self.group_swap_manager = GroupSwapManager(
            config=swap_config,
            kv_cache_gpu=kv_cache_gpu,
            kv_cache_cpu=kv_cache_cpu,
        )

        focus_config = FocusManagerConfig(
            cooling_off_period_s=config.cooling_off_period_s,
            max_users_in_gpu=config.max_users_in_gpu,
            enable_swap=config.enable_swap,
            enable_logging=config.enable_logging,
        )
        self.focus_manager = FocusManager(
            config=focus_config,
            swap_out_callback=self._handle_swap_out,
            swap_in_callback=self._handle_swap_in,
        )

        setup_request_interception(self.focus_manager)
        set_global_scheduler(self)

        self.ghost_manager = GhostBlockManager()
        sandwich_config = SandwichConfig(
            head_tokens=config.head_tokens,
            tail_tokens=config.tail_tokens,
            evict_threshold_tokens=config.evict_threshold_tokens,
            evict_threshold_gb=config.evict_threshold_gb,
        )
        self.sandwich_eviction = SandwichEviction(
            ghost_manager=self.ghost_manager,
            config=sandwich_config,
        )

        self._lock = threading.RLock()

        self._running = False

        self.stats = {
            "l1_swap_outs": 0,
            "l1_swap_ins": 0,
            "l2_degradations": 0,
            "l3_evictions": 0,
        }

        # sequence_id -> {"user_id", "total_tokens", "token_ids", "block_ids", "updated_at"}
        self._sequence_context: Dict[str, Dict] = {}

        logger.info(f"ResiMemScheduler init: completed successfully,: {config}")

    def start(self):
        """Start the service."""
        if self._running:
            logger.warning("ResiMemScheduler start: already active")
            return

        self.focus_manager.start()

        self._running = True
        logger.info("ResiMemScheduler start: started")

    def stop(self):
        """Stop the service."""
        if not self._running:
            return

        self.focus_manager.stop()

        self._running = False
        logger.info("ResiMemScheduler stop: stopped")

    def on_user_input(self, user_id: str, sequence_id: Optional[str] = None):
        """On user input."""
        with self._lock:

            self.focus_manager.notify_activity(user_id)

            if self.config.enable_logging:
                logger.debug(
                    f"on user input: processed: user_id={user_id}, sequence_id={sequence_id}"
                )

    def update_sequence_context(
        self,
        sequence_id: str,
        user_id: str,
        total_tokens: int,
        token_ids: list,
        block_ids: Optional[list] = None,
    ):
        """Update sequence context."""
        with self._lock:
            self._sequence_context[sequence_id] = {
                "user_id": user_id,
                "total_tokens": total_tokens,
                "token_ids": token_ids,
                "block_ids": block_ids,
                "updated_at": time.time(),
            }

    def _handle_swap_out(self, user_id: str) -> bool:

        logger.info(f"handle swap out: processed Swap Out:  {user_id}")

        with self._lock:

            metadata = self.group_swap_manager.swap_out_user(user_id)

            if metadata is not None:
                self.stats["l1_swap_outs"] += 1
                logger.info(
                    f"Swap Out handle swap out: completed successfully:  {user_id}, "
                    f"{metadata.num_blocks_swapped} , "
                    f" {metadata.swap_out_latency_ms:.2f}ms"
                )

                if self.shrink_callback is not None:
                    active_blocks = self.group_swap_manager.get_active_gpu_block_ids()
                    min_blocks = 64
                    if active_blocks:
                        target_blocks = max(max(active_blocks) + 1, min_blocks)
                    else:
                        target_blocks = min_blocks

                    try:
                        if not self.shrink_callback(target_blocks):
                            logger.warning(
                                f"handle swap out: failed: target_blocks={target_blocks}"
                            )
                    except Exception as e:
                        logger.error(
                            f"handle swap out: error: {e}",
                            exc_info=True,
                        )

                return True

            logger.warning(f"Swap Out handle swap out: failed:  {user_id}")
            return False

    def _handle_swap_in(self, user_id: str) -> bool:

        logger.info(f"handle swap in: processed Swap In:  {user_id}")

        with self._lock:

            if self.group_swap_manager.get_user_swap_metadata(user_id) is None:
                logger.info(f"Swap In handle swap in():  {user_id}")
                return True

            success = self.group_swap_manager.swap_in_user(user_id)

            if success:
                self.stats["l1_swap_ins"] += 1
                logger.info(
                    f"Swap In handle swap in: completed successfully:  {user_id}"
                )
                return True

            logger.warning(f"Swap In handle swap in: failed:  {user_id}")
            return False

    def register_block(
        self, user_id: str, sequence_id: str, block_id: int, num_tokens: int = 0
    ):
        """Register block."""
        self.group_swap_manager.register_block_for_user(
            user_id, sequence_id, block_id, num_tokens
        )

    def unregister_block(self, block_id: int):
        """Unregister block."""
        self.group_swap_manager.unregister_block(block_id)

    def handle_memory_pressure(self, needed_blocks: int) -> bool:
        """Handle memory pressure."""
        logger.info(f"handle memory pressure: processed:  {needed_blocks} ")

        with self._lock:

            if self._try_l1_swap_out(needed_blocks):
                logger.info(
                    "L1 handle memory pressure: completed successfully: evicted"
                )
                return True

            if self._try_l2_dnn_degrade(needed_blocks):
                logger.info(
                    "L2 handle memory pressure: completed successfully: DNN released"
                )
                return True

            if self._try_l3_sandwich_eviction(needed_blocks):
                logger.info(
                    "L3 handle memory pressure: completed successfully: evicted"
                )
                return True

            logger.warning(f"handle memory pressure: unavailable:  {needed_blocks} ")
            return False

    def _try_l1_swap_out(self, needed_blocks: int) -> bool:

        warm_users = self.focus_manager.get_users_by_state(UserState.WARM)

        freed_blocks = 0
        for user_id in warm_users:

            user_blocks = self.group_swap_manager.get_user_block_count(user_id)

            if user_blocks > 0:

                if self.focus_manager.force_swap_out(user_id):
                    freed_blocks += user_blocks
                    logger.info(
                        f"L1: try l1 swap out: evicted {user_id}, released {user_blocks} "
                    )

                    if freed_blocks >= needed_blocks:
                        return True

        return freed_blocks >= needed_blocks

    def _try_l2_dnn_degrade(self, needed_blocks: int) -> bool:

        if self.dnn_degrade_callback is None:
            logger.debug("L2 DNN try l2 dnn degrade: skipped: dnn_degrade_callback")
            return False

        try:
            success = bool(self.dnn_degrade_callback(needed_blocks))
        except Exception as exc:
            logger.warning(f"L2 DNN try l2 dnn degrade: error: {exc}")
            return False

        if success:
            self.stats["l2_degradations"] += 1

        return success

    def _try_l3_sandwich_eviction(self, needed_blocks: int) -> bool:

        if not self.config.enable_sandwich_eviction:
            logger.warning("L3 try l3 sandwich eviction: disabled")
            return False

        if not self._sequence_context:
            logger.warning("L3 try l3 sandwich eviction: skipped:")
            return False

        active_user = self.focus_manager.get_active_user()

        candidates = list(self._sequence_context.items())
        if active_user is not None:
            active_first = [
                item for item in candidates if item[1].get("user_id") == active_user
            ]
            other = [
                item for item in candidates if item[1].get("user_id") != active_user
            ]
            candidates = active_first + other

        candidates.sort(key=lambda item: item[1].get("total_tokens", 0), reverse=True)

        freed_blocks = 0
        for sequence_id, ctx in candidates:
            total_tokens = ctx.get("total_tokens", 0)
            token_ids = ctx.get("token_ids", [])
            if total_tokens <= 0 or not token_ids:
                continue

            stats = self.sandwich_evict_sequence(
                sequence_id=sequence_id,
                user_id=ctx.get("user_id", "unknown"),
                total_tokens=total_tokens,
                token_ids=token_ids,
                block_ids=ctx.get("block_ids"),
            )

            if stats is None:
                continue

            freed_blocks += getattr(stats, "freed_blocks", 0)
            if freed_blocks >= needed_blocks:
                return True

        return freed_blocks >= needed_blocks

    def _try_l2_sandwich_eviction(self, needed_blocks: int) -> bool:

        import warnings

        warnings.warn(
            "_try_l2_sandwich_eviction try l2 sandwich eviction, _try_l3_sandwich_eviction",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._try_l3_sandwich_eviction(needed_blocks)

    def get_active_user(self) -> Optional[str]:
        """Return active user."""
        return self.focus_manager.get_active_user()

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        """Return user state."""
        return self.focus_manager.get_user_state(user_id)

    def get_all_users(self) -> Set[str]:
        """Return all users."""
        return self.focus_manager.get_all_users()

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            focus_stats = self.focus_manager.get_statistics()
            swap_stats = self.group_swap_manager.get_statistics()
            sandwich_stats = self.sandwich_eviction.get_stats()
            ghost_stats = self.ghost_manager.get_stats()

            return {
                "scheduler": self.stats,
                "focus_manager": focus_stats,
                "group_swap_manager": swap_stats,
                "sandwich_eviction": sandwich_stats,
                "ghost_manager": ghost_stats.to_dict(),
                "running": self._running,
            }

    def sandwich_evict_sequence(
        self,
        sequence_id: str,
        user_id: str,
        total_tokens: int,
        token_ids: list,
        block_ids: Optional[list] = None,
    ):
        """Sandwich evict sequence."""
        with self._lock:

            if not self.config.enable_sandwich_eviction:
                logger.warning("sandwich evict sequence: disabled")
                return None

            if not self.sandwich_eviction.should_evict(total_tokens):
                logger.info(f"sandwich evict sequence {sequence_id} evicted")
                return None

            stats = self.sandwich_eviction.sandwich_evict(
                sequence_id=sequence_id,
                user_id=user_id,
                total_tokens=total_tokens,
                token_ids=token_ids,
                block_ids=block_ids,
            )

            if stats is None:
                logger.warning(f"sandwich evict sequence: evicted None: {sequence_id}")
                return None

            if stats.evicted_tokens > 0 and stats.freed_blocks > 0:
                self.stats["l3_evictions"] += 1

            logger.info(
                f"sandwich evict sequence: completed successfully: {sequence_id}, "
                f"released {stats.evicted_tokens} tokens, "
                f"{stats.freed_memory_gb:.2f} GB"
            )

            return stats

    def sandwich_restore_sequence(self, sequence_id: str) -> bool:
        """Sandwich restore sequence."""
        with self._lock:
            success = self.sandwich_eviction.restore(sequence_id)

            if success:
                logger.info(
                    f"sandwich restore sequence {sequence_id} completed successfully"
                )
            else:
                logger.warning(f"sandwich restore sequence {sequence_id} failed")

            return success

    def has_ghost_blocks(
        self, sequence_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> bool:
        """Return whether the object has ghost blocks."""
        return self.ghost_manager.has_ghosts(sequence_id=sequence_id, user_id=user_id)

    def reset(self):
        """Reset."""
        with self._lock:
            self.focus_manager.reset()
            self.group_swap_manager.reset()
            self._sequence_context.clear()

            if self.ghost_manager:
                self.ghost_manager.reset()
            if self.sandwich_eviction:
                self.sandwich_eviction.reset()

            self.stats = {
                "l1_swap_outs": 0,
                "l1_swap_ins": 0,
                "l2_degradations": 0,
                "l3_evictions": 0,
            }
            logger.info("ResiMemScheduler reset")

    def __repr__(self) -> str:
        return (
            f"ResiMemScheduler("
            f"running={self._running}, "
            f"active_user={self.get_active_user()}, "
            f"total_users={len(self.get_all_users())})"
        )


def create_resimem_scheduler(
    cooling_off_period_s: float = 7.0,
    enable_swap: bool = True,
    kv_cache_gpu: Optional[torch.Tensor] = None,
    kv_cache_cpu: Optional[torch.Tensor] = None,
    shrink_callback: Optional[Callable[[int], bool]] = None,
    dnn_degrade_callback: Optional[Callable[[int], bool]] = None,
    auto_start: bool = True,
) -> ResiMemScheduler:
    """Create resimem scheduler."""
    config = ResiMemSchedulerConfig(
        cooling_off_period_s=cooling_off_period_s,
        enable_swap=enable_swap,
    )

    scheduler = ResiMemScheduler(
        config=config,
        kv_cache_gpu=kv_cache_gpu,
        kv_cache_cpu=kv_cache_cpu,
        shrink_callback=shrink_callback,
        dnn_degrade_callback=dnn_degrade_callback,
    )

    if auto_start:
        scheduler.start()

    return scheduler


__all__ = [
    "ResiMemSchedulerConfig",
    "ResiMemScheduler",
    "create_resimem_scheduler",
]
