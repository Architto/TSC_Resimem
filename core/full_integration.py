"""ResiMem full integration module."""

import logging
import threading
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

import torch

from .vllm_integration import (
    VLLMIntegrationConfig,
    VLLMIntegrationManager,
    VLLMKVCacheInjector,
    create_vllm_integration,
)
from .vllm_block_manager import (
    VLLMBlockManager,
    create_vllm_block_manager,
)
from .scheduler.focus_manager import (
    FocusManager,
    FocusManagerConfig,
)
from .scheduler.multi_user_types import UserState
from .scheduler.api_interceptor import setup_request_interception
from .monitoring import GpuMemoryMonitor, set_global_monitor, mark_global


logger = logging.getLogger(__name__)


@dataclass
class FullIntegrationConfig:
    """FullIntegrationConfig implementation."""

    device: str = "cuda:0"
    total_gpu_memory_gb: float = 20.0
    total_vram_gb: Optional[float] = None
    model_path: Optional[str] = None
    vllm_safe_buffer_gb: float = 1.0

    cpu_swap_space_gb: float = 32.0

    cooling_off_period_s: float = 7.0
    enable_swap: bool = True

    dnn_safety_buffer_gb: float = 0.5
    fallback_on_failure: bool = True

    vllm_integration_config: Optional[Any] = None

    enable_logging: bool = True

    enable_memory_monitor: bool = True
    monitor_output_dir: str = "logs/mem"
    monitor_sample_interval_s: float = 0.1
    monitor_smi_interval_s: float = 0.5
    monitor_max_samples: int = 6000
    monitor_flush_interval_s: float = 5.0


class ResiMemFullIntegration:
    """ResiMemFullIntegration implementation."""

    def __init__(self, config: Optional[FullIntegrationConfig] = None):

        self.config = config or FullIntegrationConfig()

        self.vllm_integration: Optional[VLLMIntegrationManager] = None
        self.block_manager: Optional[VLLMBlockManager] = None
        self.focus_manager: Optional[FocusManager] = None
        self.adaptive_kv_manager: Optional[Any] = None
        self.memory_monitor: Optional[GpuMemoryMonitor] = None

        self._initialized = False
        self._enabled = False
        self._lock = threading.RLock()

        self.stats = {
            "requests_intercepted": 0,
            "focus_switches": 0,
        }

        self.shrink_callback: Optional[Callable[[int], bool]] = None

        self._llm_init_patched = False
        self._original_llm_init = None

        logger.info(f"ResiMemFullIntegration init: created,: {config}")

    def initialize(self) -> bool:
        """Initialize."""
        if self._initialized:
            logger.warning("initialize: already active")
            return True

        logger.info("=" * 60)
        logger.info("ResiMemFullIntegration initialize: started")
        logger.info("=" * 60)

        try:

            logger.info("initialize 1/3: created vLLM ...")

            if self.config.vllm_integration_config is not None:
                vllm_config = self.config.vllm_integration_config
            else:
                vllm_config = VLLMIntegrationConfig(
                    total_gpu_memory_gb=self.config.total_gpu_memory_gb,
                    total_vram_gb=self.config.total_vram_gb,
                    model_path=self.config.model_path,
                    safe_buffer_gb=self.config.vllm_safe_buffer_gb,
                    device=self.config.device,
                    dnn_safety_buffer_gb=self.config.dnn_safety_buffer_gb,
                    fallback_on_failure=self.config.fallback_on_failure,
                )

            self.vllm_integration = VLLMIntegrationManager(vllm_config)
            self.vllm_integration.initialize()

            logger.info("initialize 2/3: created...")

            self.block_manager = create_vllm_block_manager(
                kv_cache_gpu=None,
                cpu_swap_space_gb=self.config.cpu_swap_space_gb,
                block_pool=None,
            )

            logger.info("initialize 3/3: created...")
            focus_config = FocusManagerConfig(
                cooling_off_period_s=self.config.cooling_off_period_s,
                enable_swap=self.config.enable_swap,
                enable_logging=self.config.enable_logging,
            )
            self.focus_manager = FocusManager(
                config=focus_config,
                swap_out_callback=self._on_swap_out,
                swap_in_callback=self._on_swap_in,
            )

            setup_request_interception(self.focus_manager)

            self._initialized = True

            logger.info("=" * 60)
            logger.info("✅ ResiMemFullIntegration initialize: completed successfully")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"❌ initialize: failed: {e}", exc_info=True)
            self._cleanup()
            return False

    def enable(self) -> bool:
        """Enable."""
        if not self._initialized:
            logger.error("enable initialize()")
            return False

        if self._enabled:
            logger.warning("enable: already active")
            return True

        self._install_llm_init_patch()

        if self.config.enable_memory_monitor and self.memory_monitor is None:
            self.memory_monitor = GpuMemoryMonitor(
                sample_interval_s=self.config.monitor_sample_interval_s,
                smi_interval_s=self.config.monitor_smi_interval_s,
                output_dir=self.config.monitor_output_dir,
                max_samples=self.config.monitor_max_samples,
                flush_interval_s=self.config.monitor_flush_interval_s,
            )
            self.memory_monitor.start()
            set_global_monitor(self.memory_monitor)
            self.memory_monitor.mark("service_enable")

        if self.vllm_integration.enable():

            self.focus_manager.start()

            self._enabled = True
            logger.info("✅ ResiMemFullIntegration enable: enabled")
            return True

        logger.error("❌ enable: failed")
        if self.memory_monitor is not None:
            self.memory_monitor.mark("service_enable_failed")
            self.memory_monitor.stop()
            output_paths = self.memory_monitor.save()
            logger.info(f"enable: {output_paths}")
            set_global_monitor(None)
            self.memory_monitor = None
        return False

    def attach_adaptive_kv_manager(self, llm_engine: Any):
        """Attach adaptive kv manager."""
        if self.config.vllm_integration_config is None:
            logger.info(
                "attach adaptive kv manager vllm_integration_config,skipped KV Cache "
            )
            return

        if not getattr(
            self.config.vllm_integration_config, "enable_adaptive_kv", False
        ):
            logger.info("attach adaptive kv manager KV Cache enabled")
            return

        logger.info("=" * 60)
        logger.info("attach adaptive kv manager KV Cache ")
        logger.info("=" * 60)

        try:
            from .adaptive_kv_manager import create_adaptive_kv_manager

            self.adaptive_kv_manager = create_adaptive_kv_manager(
                llm_engine=llm_engine,
                block_manager=self.block_manager,
                config=self.config.vllm_integration_config,
            )

            self.adaptive_kv_manager.start()

            logger.info("✅ attach adaptive kv manager KV Cache started")

        except Exception as e:
            logger.error(
                f"❌ attach adaptive kv manager KV Cache failed: {e}", exc_info=True
            )

    def disable(self) -> bool:
        """Disable."""
        if not self._enabled:
            return True

        if self.adaptive_kv_manager is not None:
            try:
                self.adaptive_kv_manager.stop()
                logger.info("disable KV Cache stopped")
            except Exception as e:
                logger.warning(f"disable: stopped KV Cache failed: {e}")
            self.adaptive_kv_manager = None

        if self.focus_manager:
            self.focus_manager.stop()

        if self.vllm_integration:
            self.vllm_integration.disable()

        self._remove_llm_init_patch()

        if self.memory_monitor is not None:
            try:
                self.memory_monitor.mark("service_disable")
                self.memory_monitor.stop()
                output_paths = self.memory_monitor.save()
                logger.info(f"disable: {output_paths}")
            except Exception as e:
                logger.warning(f"disable: stopped/failed: {e}")
            finally:
                set_global_monitor(None)
                self.memory_monitor = None

        self._enabled = False
        logger.info("✅ ResiMemFullIntegration disable: disabled")

        return True

    def _cleanup(self):

        self.disable()

        if self.vllm_integration:
            self.vllm_integration._cleanup()

        self._initialized = False

    def on_vllm_initialized(self, kv_cache_manager: Any, block_pool: Any):
        """On vllm initialized."""
        with self._lock:
            logger.info(
                "vLLM on vllm initialized: completed successfully, BlockManager..."
            )
            mark_global("vllm_initialized")

            if hasattr(kv_cache_manager, "gpu_cache"):
                kv_cache_gpu = kv_cache_manager.gpu_cache
            else:
                kv_cache_gpu = None

            self.block_manager.kv_cache_gpu = kv_cache_gpu
            self.block_manager.block_pool = block_pool

            if kv_cache_gpu is not None:
                self.block_manager._init_cpu_swap_space(self.config.cpu_swap_space_gb)

            logger.info("✅ BlockManager on vllm initialized")

    def _on_swap_out(self, user_id: str) -> bool:

        logger.info(f"on swap out: processed Swap Out:  {user_id}")
        mark_global("swap_out_start", {"user_id": user_id})

        with self._lock:
            if self.block_manager:
                success = self.block_manager.swap_out_user(user_id)
                if success:
                    logger.info(
                        f"✅ on swap out {user_id} Swap Out completed successfully"
                    )

                    if self.shrink_callback is not None:
                        active_blocks = self.block_manager.get_active_gpu_block_ids()
                        if active_blocks:

                            MIN_BLOCKS = 64
                            target_blocks = max(max(active_blocks) + 1, MIN_BLOCKS)
                        else:
                            target_blocks = 64

                        try:
                            if not self.shrink_callback(target_blocks):
                                logger.warning(
                                    f"on swap out: failed: target_blocks={target_blocks}"
                                )
                        except Exception as e:
                            logger.error(
                                f"on swap out: error: {e}",
                                exc_info=True,
                            )
                    mark_global("swap_out_done", {"user_id": user_id})
                    return True

                logger.warning(f"⚠️ on swap out {user_id} Swap Out failed")
                return False

        return False

    def set_shrink_callback(self, callback: Callable[[int], bool]):
        """Set shrink callback."""
        self.shrink_callback = callback

    def _install_llm_init_patch(self):

        if self._llm_init_patched:
            return

        try:
            from vllm import LLM
        except Exception as e:
            logger.warning(f"vLLM install llm init patch,unavailable LLM Hook: {e}")
            return

        current_init = LLM.__init__

        if hasattr(current_init, "_resimem_patched"):
            logger.warning(
                "LLM.__init__ install llm init patch: already active ResiMem patch,skipped"
            )
            return
        self._original_llm_init = current_init
        integration = self

        def _patched_llm_init(llm_self, *args, **kwargs):
            integration._original_llm_init(llm_self, *args, **kwargs)

            if integration.shrink_callback is None:

                def _default_shrink_cb(target_blocks: int) -> bool:
                    try:
                        result = llm_self.llm_engine.resimem_shrink_kv_cache(
                            target_blocks
                        )
                        return bool(result.get("success", False))
                    except Exception as e:
                        logger.error(
                            f"default shrink cb shrink_callback failed: {e}",
                            exc_info=True,
                        )
                        return False

                integration.shrink_callback = _default_shrink_cb
                logger.info("✅ patched llm init shrink_callback( LLMEngine)")
                mark_global("llm_init_done")

        _patched_llm_init._resimem_patched = True
        LLM.__init__ = _patched_llm_init
        self._llm_init_patched = True
        logger.info("✅ install llm init patch vLLM LLM  Hook")

    def _remove_llm_init_patch(self):

        if not self._llm_init_patched:
            return

        try:
            from vllm import LLM

            if self._original_llm_init is not None:
                LLM.__init__ = self._original_llm_init
        except Exception as e:
            logger.warning(f"remove llm init patch: restored LLM Hook failed: {e}")
        finally:
            self._llm_init_patched = False
            self._original_llm_init = None

    def _on_swap_in(self, user_id: str) -> bool:

        logger.info(f"on swap in: processed Swap In:  {user_id}")
        mark_global("swap_in_start", {"user_id": user_id})

        with self._lock:
            if self.block_manager:
                success = self.block_manager.swap_in_user(user_id)
                if success:
                    logger.info(
                        f"✅ on swap in {user_id} Swap In completed successfully"
                    )
                    mark_global("swap_in_done", {"user_id": user_id})
                    return True

                logger.warning(f"⚠️ on swap in {user_id} Swap In failed")
                return False

        return False

    def on_user_request(self, user_id: str):
        """On user request."""
        with self._lock:
            self.stats["requests_intercepted"] += 1

            if self.focus_manager:
                self.focus_manager.notify_activity(user_id)

    def register_block(self, user_id: str, sequence_id: str, block_id: int):
        """Register block."""
        with self._lock:
            if self.block_manager:
                self.block_manager.register_block(user_id, sequence_id, block_id)

    def unregister_block(self, block_id: int):
        """Unregister block."""
        with self._lock:
            if self.block_manager:
                self.block_manager.unregister_block(block_id)

    def get_active_user(self) -> Optional[str]:
        """Return active user."""
        if self.focus_manager:
            return self.focus_manager.get_active_user()
        return None

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        """Return user state."""
        if self.focus_manager:
            return self.focus_manager.get_user_state(user_id)
        return None

    def get_statistics(self) -> Dict:
        """Return statistics."""
        with self._lock:
            result = {
                "integration": self.stats.copy(),
                "initialized": self._initialized,
                "enabled": self._enabled,
            }

            if self.vllm_integration:
                result["vllm_integration"] = self.vllm_integration.get_status()

            if self.block_manager:
                result["block_manager"] = self.block_manager.get_statistics()

            if self.focus_manager:
                result["focus_manager"] = self.focus_manager.get_statistics()

            return result

    def __enter__(self):

        self.initialize()
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self._cleanup()
        return False

    def __repr__(self) -> str:
        return (
            f"ResiMemFullIntegration("
            f"initialized={self._initialized}, "
            f"enabled={self._enabled}, "
            f"active_user={self.get_active_user()})"
        )


def create_full_integration(
    total_gpu_memory_gb: float = 20.0,
    device: str = "cuda:0",
    cooling_off_period_s: float = 7.0,
    cpu_swap_space_gb: float = 32.0,
    total_vram_gb: Optional[float] = None,
    model_path: Optional[str] = None,
    vllm_safe_buffer_gb: float = 1.0,
    vllm_gpu_utilization: float = 0.45,
    vllm_max_model_len: int = 2048,
    vllm_max_num_seqs: int = 8,
    vllm_enforce_eager: bool = True,
    enable_memory_monitor: bool = True,
    monitor_output_dir: str = "logs/mem",
    monitor_sample_interval_s: float = 0.1,
    monitor_smi_interval_s: float = 0.5,
    monitor_max_samples: int = 6000,
    monitor_flush_interval_s: float = 5.0,
    enable_adaptive_kv: bool = True,
    initial_kv_cache_gb: float = 1.5,
    min_kv_cache_gb: float = 1.0,
    max_kv_cache_gb: Optional[float] = None,
    expand_threshold: float = 0.80,
    shrink_threshold: float = 0.30,
    check_interval_s: float = 5.0,
    auto_enable: bool = True,
) -> ResiMemFullIntegration:
    """Create full integration."""

    _ = (
        vllm_gpu_utilization,
        vllm_max_model_len,
        vllm_max_num_seqs,
        vllm_enforce_eager,
    )

    from .vllm_integration import VLLMIntegrationConfig

    vllm_config = VLLMIntegrationConfig(
        total_gpu_memory_gb=total_gpu_memory_gb,
        device=device,
        total_vram_gb=total_vram_gb,
        model_path=model_path,
        safe_buffer_gb=vllm_safe_buffer_gb,
        enable_adaptive_kv=enable_adaptive_kv,
        initial_kv_cache_gb=initial_kv_cache_gb,
        min_kv_cache_gb=min_kv_cache_gb,
        max_kv_cache_gb=max_kv_cache_gb,
        expand_threshold=expand_threshold,
        shrink_threshold=shrink_threshold,
        check_interval_s=check_interval_s,
    )

    config = FullIntegrationConfig(
        total_gpu_memory_gb=total_gpu_memory_gb,
        device=device,
        cooling_off_period_s=cooling_off_period_s,
        cpu_swap_space_gb=cpu_swap_space_gb,
        total_vram_gb=total_vram_gb,
        model_path=model_path,
        vllm_safe_buffer_gb=vllm_safe_buffer_gb,
        vllm_integration_config=vllm_config,
        enable_memory_monitor=enable_memory_monitor,
        monitor_output_dir=monitor_output_dir,
        monitor_sample_interval_s=monitor_sample_interval_s,
        monitor_smi_interval_s=monitor_smi_interval_s,
        monitor_max_samples=monitor_max_samples,
        monitor_flush_interval_s=monitor_flush_interval_s,
    )

    integration = ResiMemFullIntegration(config)
    integration.initialize()

    if auto_enable:
        integration.enable()

    return integration


__all__ = [
    "FullIntegrationConfig",
    "ResiMemFullIntegration",
    "create_full_integration",
]
