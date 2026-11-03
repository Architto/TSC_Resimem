"""ResiMem kv cache hook module."""

from typing import Optional, Callable
import logging

try:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
except ImportError as exc:
    raise ImportError(
        "KVCacheHookManager resimem operation vLLM V1 KVCacheManager ."
    ) from exc

from .resimem_block_pool import ResiMemBlockPool
from ..grm.grm_client import GRMClient
from ..ipc import ExpansionRequest

logger = logging.getLogger(__name__)


class KVCacheHookManager:
    """KVCacheHookManager implementation."""

    def __init__(
        self,
        expansion_strategy: str = "aggressive",
        expansion_factor: float = 1.5,
        min_expansion_blocks: int = 50,
        grm_client: Optional[GRMClient] = None,
        use_grm: bool = True,
    ):

        self.expansion_strategy = expansion_strategy
        self.expansion_factor = expansion_factor
        self.min_expansion_blocks = min_expansion_blocks

        self.grm_client = grm_client
        self.use_grm = use_grm and (grm_client is not None)

        self._is_enabled = False
        self._original_allocate_slots: Optional[Callable] = None
        self._patched_allocate_slots: Optional[Callable] = None

        self.stats = {
            "total_calls": 0,
            "expansion_triggered": 0,
            "expansion_succeeded": 0,
            "expansion_failed": 0,
            "preemption_triggered": 0,
            "grm_requests": 0,
            "grm_approved": 0,
            "grm_denied": 0,
        }

        grm_status = "ENABLED" if self.use_grm else "DISABLED"
        logger.info(
            f"KVCacheHookManager initialized "
            f"(strategy={expansion_strategy}, GRM={grm_status})"
        )

    def enable(self) -> bool:
        """Enable."""
        if self._is_enabled:
            logger.warning("KVCacheHookManager is already enabled")
            return True

        self._original_allocate_slots = KVCacheManager.allocate_slots

        def _patched_allocate_slots(
            kv_cache_manager_self,
            request,
            num_new_tokens: int,
            num_new_computed_tokens: int = 0,
            new_computed_blocks=None,
            num_lookahead_tokens: int = 0,
            num_external_computed_tokens: int = 0,
            delay_cache_blocks: bool = False,
            num_encoder_tokens: int = 0,
        ):
            return self._hooked_allocate_slots(
                kv_cache_manager=kv_cache_manager_self,
                request=request,
                num_new_tokens=num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
            )

        self._patched_allocate_slots = _patched_allocate_slots
        KVCacheManager.allocate_slots = _patched_allocate_slots

        self._is_enabled = True
        logger.info("✅ KVCacheHookManager enabled")

        return True

    def disable(self) -> bool:
        """Disable."""
        if not self._is_enabled:
            logger.warning("KVCacheHookManager is not enabled")
            return True

        if self._original_allocate_slots is not None:
            KVCacheManager.allocate_slots = self._original_allocate_slots
            self._original_allocate_slots = None
        self._patched_allocate_slots = None

        self._is_enabled = False
        logger.info("✅ KVCacheHookManager disabled")

        return True

    def _hooked_allocate_slots(
        self,
        kv_cache_manager,
        request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks=None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ):

        self.stats["total_calls"] += 1

        try:
            from ..scheduler.api_interceptor import get_global_scheduler

            scheduler = get_global_scheduler()
            if scheduler is not None:
                user_id = "default_user"
                kv_params = getattr(request, "kv_transfer_params", None)
                if isinstance(kv_params, dict):
                    user_id = (
                        kv_params.get("user_id") or kv_params.get("user") or user_id
                    )

                trace_headers = getattr(request, "trace_headers", None)
                if user_id == "default_user" and isinstance(trace_headers, dict):
                    user_id = trace_headers.get("x-resimem-user", user_id)

                scheduler.update_sequence_context(
                    sequence_id=str(request.request_id),
                    user_id=user_id,
                    total_tokens=int(getattr(request, "num_tokens", 0)),
                    token_ids=list(getattr(request, "all_token_ids", [])),
                    block_ids=None,
                )
        except Exception as e:
            logger.debug(f"hooked allocate slots: failed: {e}")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = kv_cache_manager.empty_kv_cache_blocks.blocks

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            kv_cache_manager.max_model_len,
        )
        num_tokens_need_slot = min(
            total_computed_tokens + num_new_tokens + num_lookahead_tokens,
            kv_cache_manager.max_model_len,
        )

        kv_cache_manager.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = (
            kv_cache_manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=num_local_computed_tokens
                + num_external_computed_tokens,
            )
        )

        if num_blocks_to_allocate > kv_cache_manager.block_pool.get_num_free_blocks():
            logger.info(
                f"🔍 Block shortage detected!\n"
                f"   Required: {num_blocks_to_allocate} blocks\n"
                f"   Available: {kv_cache_manager.block_pool.get_num_free_blocks()} blocks\n"
                f"   Shortage: {num_blocks_to_allocate - kv_cache_manager.block_pool.get_num_free_blocks()} blocks"
            )

            if isinstance(kv_cache_manager.block_pool, ResiMemBlockPool):
                self.stats["expansion_triggered"] += 1

                shortage = (
                    num_blocks_to_allocate
                    - kv_cache_manager.block_pool.get_num_free_blocks()
                )

                if self.expansion_strategy == "aggressive":

                    expansion_size = max(
                        int(shortage * self.expansion_factor), self.min_expansion_blocks
                    )
                else:  # conservative

                    expansion_size = max(shortage, self.min_expansion_blocks)

                logger.info(
                    f"🔄 Attempting to expand BlockPool by {expansion_size} blocks..."
                )

                if self.use_grm:
                    self.stats["grm_requests"] += 1

                    bytes_needed = expansion_size * self._get_block_bytes(
                        kv_cache_manager
                    )

                    logger.info(
                        f"📨 Requesting {bytes_needed / (1024**2):.2f} MB "
                        f"({expansion_size} blocks) from GRM..."
                    )

                    request = ExpansionRequest(
                        num_blocks=expansion_size,
                        size_bytes=bytes_needed,
                        priority=5,
                        timeout_ms=100,
                    )
                    approved = self.grm_client.request_expansion(request)

                    if approved:
                        self.stats["grm_approved"] += 1
                        logger.info(
                            f"✅ GRM approved expansion: "
                            f"{bytes_needed / (1024**2):.2f} MB requested"
                        )

                        success = kv_cache_manager.block_pool.expand_capacity(
                            expansion_size
                        )

                        if success:
                            self.stats["expansion_succeeded"] += 1
                            logger.info(
                                f"✅ Expansion successful! Retrying allocation..."
                            )
                        else:
                            self.stats["expansion_failed"] += 1
                            logger.error(
                                f"❌ Expansion failed even after GRM approval! "
                                f"Triggering preemption..."
                            )
                            self.stats["preemption_triggered"] += 1
                            return None
                    else:
                        self.stats["grm_denied"] += 1
                        self.stats["expansion_failed"] += 1
                        logger.warning(
                            f"❌ GRM denied expansion request. "
                            f"Triggering preemption..."
                        )
                        self.stats["preemption_triggered"] += 1
                        return None
                else:

                    success = kv_cache_manager.block_pool.expand_capacity(
                        expansion_size
                    )

                    if success:
                        self.stats["expansion_succeeded"] += 1
                        logger.info(f"✅ Expansion successful! Retrying allocation...")

                    else:
                        self.stats["expansion_failed"] += 1
                        logger.warning(f"❌ Expansion failed! Triggering preemption...")

                        self.stats["preemption_triggered"] += 1
                        return None
            else:

                logger.warning(
                    "BlockPool does not support expansion (not ResiMemBlockPool). "
                    "Triggering preemption..."
                )
                self.stats["preemption_triggered"] += 1
                return None

        if (
            new_computed_block_list is not kv_cache_manager.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            kv_cache_manager.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = kv_cache_manager.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )

        if not delay_cache_blocks:
            kv_cache_manager.coordinator.cache_full_blocks(
                request, num_tokens_need_slot, num_encoder_tokens
            )

        return new_blocks

    def _get_block_bytes(self, kv_cache_manager: Optional[object] = None) -> int:

        target = kv_cache_manager if kv_cache_manager is not None else self
        kv_cache_config = getattr(target, "kv_cache_config", None)
        if kv_cache_config is None:
            return 4096

        kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
        if not kv_cache_groups:
            return 4096

        try:
            return sum(group.kv_cache_spec.page_size_bytes for group in kv_cache_groups)
        except Exception:
            return 4096

    def get_stats(self) -> dict:
        """Return stats."""
        stats = self.stats.copy()

        if stats["total_calls"] > 0:
            stats["expansion_rate"] = (
                stats["expansion_triggered"] / stats["total_calls"]
            )
            stats["success_rate"] = (
                stats["expansion_succeeded"] / stats["expansion_triggered"]
                if stats["expansion_triggered"] > 0
                else 0.0
            )
        else:
            stats["expansion_rate"] = 0.0
            stats["success_rate"] = 0.0

        return stats

    def print_stats(self):
        """Print stats."""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("KVCacheHookManager Statistics")
        print("=" * 70)
        print(f"Total Calls:          {stats['total_calls']}")
        print(
            f"Expansion Triggered:  {stats['expansion_triggered']} ({stats['expansion_rate']*100:.1f}%)"
        )
        print(f"  - Succeeded:        {stats['expansion_succeeded']}")
        print(f"  - Failed:           {stats['expansion_failed']}")

        if self.use_grm:
            print(f"\nGRM Coordination:")
            print(f"  - Requests:         {stats['grm_requests']}")
            print(f"  - Approved:         {stats['grm_approved']}")
            print(f"  - Denied:           {stats['grm_denied']}")
            if stats["grm_requests"] > 0:
                approval_rate = stats["grm_approved"] / stats["grm_requests"] * 100
                print(f"  - Approval Rate:    {approval_rate:.1f}%")

        print(f"\nPreemption Triggered: {stats['preemption_triggered']}")
        print(f"Success Rate:         {stats['success_rate']*100:.1f}%")
        print("=" * 70 + "\n")

    @property
    def is_enabled(self) -> bool:
        """Return whether enabled."""
        return self._is_enabled

    def __enter__(self):

        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.disable()
        return False

    def __repr__(self) -> str:
        status = "ENABLED" if self._is_enabled else "DISABLED"
        return f"KVCacheHookManager({status}, strategy={self.expansion_strategy})"
