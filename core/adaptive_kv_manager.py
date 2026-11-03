"""ResiMem adaptive kv manager module."""

import logging
import math
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)
MIN_EXPAND_BLOCKS = 100
MIN_SHRINK_BLOCKS = 100


SIZE_TOLERANCE_GB = 0.1


ACTIVE_BLOCKS_MARGIN = 1.1


@dataclass
class AdaptiveKVStats:
    """AdaptiveKVStats implementation."""

    total_expansions: int = 0
    total_shrinks: int = 0
    total_freed_gb: float = 0.0
    total_allocated_gb: float = 0.0
    current_size_gb: float = 0.0
    current_usage_rate: float = 0.0
    last_adjustment_time: float = 0.0


class AdaptiveKVCacheManager:
    """AdaptiveKVCacheManager implementation."""

    def __init__(
        self,
        llm_engine: Any,
        block_manager: Any,
        initial_kv_cache_gb: float = 1.5,
        min_kv_cache_gb: float = 1.0,
        max_kv_cache_gb: Optional[float] = None,
        expand_threshold: float = 0.80,
        shrink_threshold: float = 0.30,
        expand_factor: float = 1.5,
        shrink_factor: float = 0.7,
        check_interval_s: float = 5.0,
        local_quantum_contract: Optional[Dict[str, Any]] = None,
    ):

        self.llm_engine = llm_engine
        self.block_manager = block_manager

        self.initial_kv_cache_gb = initial_kv_cache_gb
        self.min_kv_cache_gb = min_kv_cache_gb
        self.max_kv_cache_gb = max_kv_cache_gb
        self.expand_threshold = expand_threshold
        self.shrink_threshold = shrink_threshold
        self.expand_factor = expand_factor
        self.shrink_factor = shrink_factor
        self.check_interval_s = check_interval_s
        self.local_quantum_contract = dict(local_quantum_contract or {})

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self.stats = AdaptiveKVStats()

        self.on_expand: Optional[Callable[[float, float], None]] = None
        self.on_shrink: Optional[Callable[[float, float], None]] = None
        self._local_quantum_snap_count = 0
        self._local_quantum_last_snap: Dict[str, Any] = {}

        logger.info(f"AdaptiveKVCacheManager init: created")
        logger.info(f"  init: initial capacity: {initial_kv_cache_gb:.2f} GB")
        logger.info(
            f"  init: configured range: {min_kv_cache_gb:.2f} - {max_kv_cache_gb or 'unlimited'} GB"
        )
        logger.info(f"  init: expansion threshold: {expand_threshold:.0%}")
        logger.info(f"  init: shrink threshold: {shrink_threshold:.0%}")
        if self.local_quantum_contract:
            logger.info(
                "  Local quantum: family=%s surface=%s status=%s observable_blocks=%s",
                self.local_quantum_contract.get("family_key", "unknown"),
                self.local_quantum_contract.get("surface_key", "unknown"),
                self.local_quantum_contract.get("manifest_status", "unknown"),
                self.local_quantum_contract.get("observable_local_quantum_blocks"),
            )

    def _get_local_quantum_blocks(self) -> Optional[int]:
        if not self.local_quantum_contract:
            return None
        value = self.local_quantum_contract.get("observable_local_quantum_blocks")
        if value is None:
            return None
        try:
            quantum = int(value)
        except Exception:
            return None
        return quantum if quantum > 1 else None

    def _get_local_quantum_snap_mode(self) -> str:
        if not self.local_quantum_contract:
            return "disabled"
        manifest_status = str(
            self.local_quantum_contract.get("manifest_status", "") or ""
        )
        if manifest_status == "verified_manifest_ready":
            return "hard"
        if manifest_status == "numeric_repeatable_contract_pending" and bool(
            self.local_quantum_contract.get("soft_snap_allowed", False)
        ):
            return "soft"
        return "disabled"

    def _snap_target_blocks(
        self,
        *,
        current_blocks: int,
        target_blocks: int,
        mode: str,
    ) -> int:
        quantum = self._get_local_quantum_blocks()
        snap_mode = self._get_local_quantum_snap_mode()
        if quantum is None or snap_mode == "disabled":
            return target_blocks
        if target_blocks <= 0 or current_blocks <= 0:
            return target_blocks

        snapped = target_blocks
        if mode == "expand":
            delta = max(target_blocks - current_blocks, 0)
            if delta <= 0:
                return target_blocks
            snapped = current_blocks + math.ceil(delta / quantum) * quantum
        elif mode == "shrink":
            delta = max(current_blocks - target_blocks, 0)
            if delta <= 0:
                return target_blocks
            aligned_delta = math.floor(delta / quantum) * quantum
            if aligned_delta <= 0:
                return target_blocks
            snapped = current_blocks - aligned_delta

        if snapped != target_blocks:
            self._local_quantum_snap_count += 1
            self._local_quantum_last_snap = {
                "mode": mode,
                "snap_mode": snap_mode,
                "quantum_blocks": quantum,
                "current_blocks": current_blocks,
                "requested_target_blocks": target_blocks,
                "snapped_target_blocks": snapped,
            }
            logger.info(
                "Local quantum snap: mode=%s snap_mode=%s quantum=%s current=%s target=%s snapped=%s",
                mode,
                snap_mode,
                quantum,
                current_blocks,
                target_blocks,
                snapped,
            )
        return snapped

    def start(self):
        """Start the service."""
        if self._running:
            logger.warning("AdaptiveKVCacheManager start: already active")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            name="AdaptiveKVManager",
            daemon=True,
        )
        self._thread.start()
        logger.info("✅ AdaptiveKVCacheManager start: started")

    def stop(self):
        """Stop the service."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=self.check_interval_s + 1.0)
        logger.info("AdaptiveKVCacheManager stop: stopped")

    def _background_loop(self):

        logger.info("AdaptiveKVCacheManager background loop: started")

        while self._running:
            try:
                self._check_and_adjust()
            except Exception as e:
                logger.error(
                    f"AdaptiveKVCacheManager background loop: failed: {e}",
                    exc_info=True,
                )

            time.sleep(self.check_interval_s)

        logger.info("AdaptiveKVCacheManager background loop: finished")

    def _infer_blocks_dim_idx(
        self, shape: list, max_active_block_id: int = -1
    ) -> Optional[int]:

        if not shape:
            return None

        cached_idx = getattr(self, "_blocks_dim_idx", None)
        if cached_idx is not None and 0 <= cached_idx < len(shape):
            if max_active_block_id < 0 or shape[cached_idx] > max_active_block_id:
                return cached_idx

        if len(shape) >= 3 and shape[1] == 2:
            if max_active_block_id < 0 or shape[2] > max_active_block_id:
                return 2

        candidate_indices = []
        for idx, dim in enumerate(shape):
            if dim <= 0:
                continue
            if max_active_block_id >= 0 and dim <= max_active_block_id:
                continue
            candidate_indices.append(idx)

        if candidate_indices:
            if max_active_block_id >= 0:

                return min(candidate_indices, key=lambda i: shape[i])
            return candidate_indices[0]

        return 0 if len(shape) > 0 else None

    def _check_and_adjust(self):

        with self._lock:

            try:
                info = self.llm_engine.resimem_get_kv_cache_info()
            except Exception as e:
                logger.debug(f"check and adjust: unavailable KV Cache : {e}")
                return

            if not info.get("has_kv_cache"):
                logger.debug("KV Cache check and adjust: not initialized,skipped")
                return

            current_bytes = info.get("kv_cache_bytes", 0)
            current_gb = current_bytes / 1024**3

            active_blocks = self.block_manager.get_active_gpu_block_ids()
            num_active = len(active_blocks)

            shape = info.get("kv_cache_tensor_shape", [])
            if len(shape) < 1:
                logger.debug(f"check and adjust: unavailable blocks : shape={shape}")
                return

            max_active_id = max(active_blocks) if active_blocks else -1
            blocks_dim_idx = self._infer_blocks_dim_idx(shape, max_active_id)
            if blocks_dim_idx is None or blocks_dim_idx >= len(shape):
                logger.debug(f"check and adjust: unavailable blocks : shape={shape}")
                return
            self._blocks_dim_idx = blocks_dim_idx
            total_blocks = shape[blocks_dim_idx]

            if total_blocks == 0:
                logger.debug("total_blocks = 0,check and adjust: skipped")
                return

            usage_rate = num_active / total_blocks

            self.stats.current_size_gb = current_gb
            self.stats.current_usage_rate = usage_rate

            logger.debug(
                f"KV Cache check and adjust: state: {current_gb:.2f} GB, "
                f": {usage_rate:.1%} ({num_active}/{total_blocks} blocks)"
            )

            if usage_rate > self.expand_threshold:
                self._try_expand(current_gb, total_blocks, num_active)
            elif usage_rate < self.shrink_threshold:
                self._try_shrink(current_gb, total_blocks, num_active)

    def _try_expand(self, current_gb: float, total_blocks: int, num_active: int):

        target_gb = current_gb * self.expand_factor

        if self.max_kv_cache_gb is not None:
            target_gb = min(target_gb, self.max_kv_cache_gb)

        if target_gb <= current_gb + SIZE_TOLERANCE_GB:
            logger.debug(f"try expand KV Cache  ({current_gb:.2f} GB),skipped")
            return

        bytes_per_block = (
            (current_gb * 1024**3) / total_blocks if total_blocks > 0 else 0
        )
        if bytes_per_block == 0:
            logger.warning("try expand: unavailable bytes_per_block,skipped")
            return

        target_blocks = int((target_gb * 1024**3) / bytes_per_block)
        target_blocks = max(target_blocks, total_blocks + MIN_EXPAND_BLOCKS)
        target_blocks = self._snap_target_blocks(
            current_blocks=total_blocks,
            target_blocks=target_blocks,
            mode="expand",
        )

        logger.info(
            f"🔼 try expand: expanded: {current_gb:.2f} GB → {target_gb:.2f} GB "
            f"({total_blocks} → {target_blocks} blocks)"
        )

        try:
            result = self.llm_engine.resimem_expand_kv_cache(target_blocks)

            if result.get("success"):
                allocated_bytes = result.get("allocated_bytes", 0)
                allocated_gb = allocated_bytes / 1024**3

                self.stats.total_expansions += 1
                self.stats.total_allocated_gb += allocated_gb
                self.stats.last_adjustment_time = time.time()

                logger.info(
                    f"✅ try expand: completed successfully: allocation updated {allocated_gb:.2f} GB, "
                    f" {result.get('new_blocks')} blocks"
                )

                if self.on_expand:
                    self.on_expand(current_gb, target_gb)
            else:
                logger.warning(f"try expand: failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"try expand: error: {e}", exc_info=True)

    def _try_shrink(self, current_gb: float, total_blocks: int, num_active: int):

        target_blocks = int(num_active * ACTIVE_BLOCKS_MARGIN)

        bytes_per_block = (
            (current_gb * 1024**3) / total_blocks if total_blocks > 0 else 0
        )
        if bytes_per_block == 0:
            logger.warning("try shrink: unavailable bytes_per_block,skipped")
            return

        min_blocks = int((self.min_kv_cache_gb * 1024**3) / bytes_per_block)
        target_blocks = max(target_blocks, min_blocks)
        target_blocks = self._snap_target_blocks(
            current_blocks=total_blocks,
            target_blocks=target_blocks,
            mode="shrink",
        )

        if target_blocks >= total_blocks - MIN_SHRINK_BLOCKS:
            logger.debug(f"try shrink ({total_blocks} → {target_blocks}),skipped")
            return

        target_gb = (target_blocks * bytes_per_block) / 1024**3

        logger.info(
            f"🔽 try shrink: {current_gb:.2f} GB → {target_gb:.2f} GB "
            f"({total_blocks} → {target_blocks} blocks)"
        )

        try:
            result = self.llm_engine.resimem_shrink_kv_cache(target_blocks)

            if result.get("success"):
                freed_bytes = result.get("freed_bytes", 0)
                freed_gb = freed_bytes / 1024**3

                self.stats.total_shrinks += 1
                self.stats.total_freed_gb += freed_gb
                self.stats.last_adjustment_time = time.time()

                logger.info(
                    f"✅ try shrink: completed successfully: released {freed_gb:.2f} GB, "
                    f" {result.get('new_blocks')} blocks"
                )

                if self.on_shrink:
                    self.on_shrink(current_gb, target_gb)
            else:
                logger.warning(f"try shrink: failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"try shrink: error: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics."""
        with self._lock:
            return {
                "running": self._running,
                "current_size_gb": self.stats.current_size_gb,
                "current_usage_rate": self.stats.current_usage_rate,
                "total_expansions": self.stats.total_expansions,
                "total_shrinks": self.stats.total_shrinks,
                "total_freed_gb": self.stats.total_freed_gb,
                "total_allocated_gb": self.stats.total_allocated_gb,
                "last_adjustment_time": self.stats.last_adjustment_time,
                "local_quantum_contract": dict(self.local_quantum_contract),
                "local_quantum_snap_count": self._local_quantum_snap_count,
                "local_quantum_last_snap": dict(self._local_quantum_last_snap),
            }

    def force_shrink_to_active(self):
        """Force shrink to active."""
        logger.info("force shrink to active")
        with self._lock:
            try:
                info = self.llm_engine.resimem_get_kv_cache_info()
                if not info.get("has_kv_cache"):
                    logger.warning("KV Cache force shrink to active: not initialized")
                    return

                active_blocks = self.block_manager.get_active_gpu_block_ids()
                num_active = len(active_blocks)
                target_blocks = int(num_active * ACTIVE_BLOCKS_MARGIN)

                shape = info.get("kv_cache_tensor_shape", [])
                max_active_id = max(active_blocks) if active_blocks else -1
                blocks_dim_idx = self._infer_blocks_dim_idx(shape, max_active_id)
                if blocks_dim_idx is None or blocks_dim_idx >= len(shape):
                    logger.warning(
                        f"force shrink to active: unavailable blocks : shape={shape}"
                    )
                    return
                self._blocks_dim_idx = blocks_dim_idx
                total_blocks = shape[blocks_dim_idx]

                if target_blocks >= total_blocks:
                    logger.info("force shrink to active")
                    return
                target_blocks = self._snap_target_blocks(
                    current_blocks=total_blocks,
                    target_blocks=target_blocks,
                    mode="shrink",
                )

                logger.info(
                    f"force shrink to active: {total_blocks} → {target_blocks} blocks"
                )
                result = self.llm_engine.resimem_shrink_kv_cache(target_blocks)

                if result.get("success"):
                    freed_gb = result.get("freed_bytes", 0) / 1024**3
                    logger.info(
                        f"✅ force shrink to active: completed successfully: released {freed_gb:.2f} GB"
                    )
                else:
                    logger.warning(
                        f"force shrink to active: failed: {result.get('message')}"
                    )

            except Exception as e:
                logger.error(f"force shrink to active: error: {e}", exc_info=True)


def create_adaptive_kv_manager(
    llm_engine: Any,
    block_manager: Any,
    config: Any,
) -> AdaptiveKVCacheManager:
    """Create adaptive kv manager."""
    return AdaptiveKVCacheManager(
        llm_engine=llm_engine,
        block_manager=block_manager,
        initial_kv_cache_gb=config.initial_kv_cache_gb,
        min_kv_cache_gb=config.min_kv_cache_gb,
        max_kv_cache_gb=config.max_kv_cache_gb,
        expand_threshold=config.expand_threshold,
        shrink_threshold=config.shrink_threshold,
        expand_factor=config.expand_factor,
        shrink_factor=config.shrink_factor,
        check_interval_s=config.check_interval_s,
        local_quantum_contract=getattr(config, "local_quantum_contract", None),
    )
