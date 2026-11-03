"""ResiMem memory injector module."""

import torch
from typing import Optional, Dict, Callable
import logging

from ..memory import GiantTensor, MemoryPool

logger = logging.getLogger(__name__)


class MemoryInjector:
    """MemoryInjector implementation."""

    def __init__(
        self,
        memory_pool: MemoryPool,
        fallback_on_failure: bool = True,
    ):

        self.memory_pool = memory_pool
        self.fallback_on_failure = fallback_on_failure

        self._original_allocate_kv_cache: Optional[Callable] = None
        self._original_allocate_kv_cache_tensors_v1: Optional[Callable] = None
        self._original_allocate_kv_cache_tensors_v2: Optional[Callable] = None
        self._is_enabled = False

        self.stats = {
            "total_allocations": 0,
            "injected_allocations": 0,
            "fallback_allocations": 0,
            "total_bytes_allocated": 0,
        }

        if (
            MemoryInjector._instance is not None
            and MemoryInjector._instance._is_enabled
        ):
            logger.warning(
                "⚠️ init: enabled MemoryInjector !" " hook error." " disable() ."
            )

        logger.info("MemoryInjector initialized")
        MemoryInjector._instance = self

    _instance: Optional["MemoryInjector"] = None

    @classmethod
    def get_instance(cls) -> Optional["MemoryInjector"]:
        return cls._instance

    def enable(self) -> bool:
        """Enable."""
        if self._is_enabled:
            logger.warning("MemoryInjector is already enabled")
            return True

        try:

            import vllm.v1.worker.gpu.attn_utils as attn_utils

            try:
                import vllm.v1.worker.gpu_model_runner as gpu_model_runner_v1
            except Exception:
                gpu_model_runner_v1 = None
            try:
                import vllm.v1.worker.gpu.model_runner as gpu_model_runner_v2
            except Exception:
                gpu_model_runner_v2 = None

            self._original_allocate_kv_cache = attn_utils._allocate_kv_cache

            attn_utils._allocate_kv_cache = self._patched_allocate_kv_cache

            if gpu_model_runner_v1 is not None and hasattr(
                gpu_model_runner_v1, "GPUModelRunner"
            ):
                self._original_allocate_kv_cache_tensors_v1 = (
                    gpu_model_runner_v1.GPUModelRunner._allocate_kv_cache_tensors
                )
                gpu_model_runner_v1.GPUModelRunner._allocate_kv_cache_tensors = (
                    MemoryInjector._patched_allocate_kv_cache_tensors  # type: ignore[method-assign]
                )
            if gpu_model_runner_v2 is not None and hasattr(
                gpu_model_runner_v2, "GPUModelRunner"
            ):
                if hasattr(
                    gpu_model_runner_v2.GPUModelRunner, "_allocate_kv_cache_tensors"
                ):
                    self._original_allocate_kv_cache_tensors_v2 = (
                        gpu_model_runner_v2.GPUModelRunner._allocate_kv_cache_tensors
                    )
                    gpu_model_runner_v2.GPUModelRunner._allocate_kv_cache_tensors = (
                        MemoryInjector._patched_allocate_kv_cache_tensors  # type: ignore[method-assign]
                    )

            self._is_enabled = True
            logger.info("✅ MemoryInjector enabled successfully")
            logger.info(f"   Memory Pool: {self.memory_pool}")

            return True

        except ImportError as e:
            logger.error(f"❌ Failed to import vLLM modules: {e}")
            raise RuntimeError(
                "Failed to enable MemoryInjector: vLLM not found. "
                "Please ensure vLLM is installed."
            ) from e

    def disable(self) -> bool:
        """Disable."""
        if not self._is_enabled:
            logger.warning("MemoryInjector is not enabled")
            return True

        errors = []
        try:
            import vllm.v1.worker.gpu.attn_utils as attn_utils

            try:
                import vllm.v1.worker.gpu_model_runner as gpu_model_runner_v1
            except Exception:
                gpu_model_runner_v1 = None
            try:
                import vllm.v1.worker.gpu.model_runner as gpu_model_runner_v2
            except Exception:
                gpu_model_runner_v2 = None

            if self._original_allocate_kv_cache is not None:
                try:
                    attn_utils._allocate_kv_cache = self._original_allocate_kv_cache
                except Exception as e:
                    errors.append(f"attn_utils hook restore: {e}")
                finally:
                    self._original_allocate_kv_cache = None

            if (
                gpu_model_runner_v1 is not None
                and self._original_allocate_kv_cache_tensors_v1 is not None
            ):
                try:
                    gpu_model_runner_v1.GPUModelRunner._allocate_kv_cache_tensors = (
                        self._original_allocate_kv_cache_tensors_v1
                    )
                except Exception as e:
                    errors.append(f"v1 runner hook restore: {e}")
                finally:
                    self._original_allocate_kv_cache_tensors_v1 = None

            if (
                gpu_model_runner_v2 is not None
                and self._original_allocate_kv_cache_tensors_v2 is not None
            ):
                try:
                    if hasattr(
                        gpu_model_runner_v2.GPUModelRunner, "_allocate_kv_cache_tensors"
                    ):
                        gpu_model_runner_v2.GPUModelRunner._allocate_kv_cache_tensors = (
                            self._original_allocate_kv_cache_tensors_v2
                        )
                except Exception as e:
                    errors.append(f"v2 runner hook restore: {e}")
                finally:
                    self._original_allocate_kv_cache_tensors_v2 = None

        except Exception as e:
            errors.append(f"import phase: {e}")
        finally:

            self._is_enabled = False

        if errors:
            logger.error(f"❌ MemoryInjector disable disable: failed: {errors}")
            return False

        logger.info("✅ MemoryInjector disabled, restored original behavior")
        return True

    def _patched_allocate_kv_cache(
        self,
        kv_cache_config,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:

        self.stats["total_allocations"] += 1

        kv_cache_raw_tensors: Dict[str, torch.Tensor] = {}

        try:

            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                size_bytes = kv_cache_tensor.size

                logger.debug(
                    f"Allocating KV Cache: {size_bytes:,} bytes "
                    f"({size_bytes / 1024**2:.1f} MB)"
                )

                tensor_view = self.memory_pool.allocate(size_bytes)

                if tensor_view is None:

                    if self.fallback_on_failure:
                        logger.warning(
                            f"Failed to allocate from Giant Tensor, "
                            f"falling back to torch.zeros()"
                        )
                        return self._fallback_allocate(kv_cache_config, device)
                    else:
                        raise RuntimeError(
                            f"Failed to allocate {size_bytes:,} bytes from Giant Tensor"
                        )

                assert tensor_view.numel() == size_bytes, (
                    f"Allocated tensor size mismatch: "
                    f"expected {size_bytes}, got {tensor_view.numel()}"
                )

                assert tensor_view.dtype == torch.uint8, (
                    f"Allocated tensor dtype mismatch: "
                    f"expected torch.uint8, got {tensor_view.dtype}"
                )

                tensor_int8 = tensor_view.view(torch.int8)

                for layer_name in kv_cache_tensor.shared_by:
                    kv_cache_raw_tensors[layer_name] = tensor_int8

                self.stats["injected_allocations"] += 1
                self.stats["total_bytes_allocated"] += size_bytes

            layer_names = set()
            for group in kv_cache_config.kv_cache_groups:
                for layer_name in group.layer_names:
                    layer_names.add(layer_name)

            assert layer_names == set(
                kv_cache_raw_tensors.keys()
            ), "Some layers are not correctly initialized"

            logger.debug(
                f"✅ Allocated {len(kv_cache_raw_tensors)} KV Cache tensors "
                f"from Giant Tensor"
            )

            return kv_cache_raw_tensors

        except Exception as e:
            logger.error(f"❌ Error in _patched_allocate_kv_cache: {e}")

            if self.fallback_on_failure:
                logger.warning("Falling back to original allocation")
                return self._fallback_allocate(kv_cache_config, device)
            else:
                raise

    @staticmethod
    def _patched_allocate_kv_cache_tensors(self_runner, kv_cache_config):
        device = getattr(self_runner, "device", None)
        if device is None:
            device = torch.device("cuda:0")
        injector = MemoryInjector.get_instance()
        if injector is None:
            raise RuntimeError("MemoryInjector instance not available")
        return injector._patched_allocate_kv_cache(kv_cache_config, device)

    def _fallback_allocate(
        self,
        kv_cache_config,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:

        self.stats["fallback_allocations"] += 1

        if self._original_allocate_kv_cache is None:
            raise RuntimeError(
                "Cannot fallback: original _allocate_kv_cache function not saved"
            )

        logger.debug("Using original vLLM allocation")
        return self._original_allocate_kv_cache(kv_cache_config, device)

    def get_stats(self) -> Dict[str, int]:
        """Return stats."""
        stats = self.stats.copy()
        stats["injection_rate"] = (
            stats["injected_allocations"] / stats["total_allocations"]
            if stats["total_allocations"] > 0
            else 0.0
        )
        return stats

    def print_stats(self):
        """Print stats."""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("MemoryInjector Statistics")
        print("=" * 70)
        print(f"Total Allocations:    {stats['total_allocations']}")
        print(
            f"Injected:             {stats['injected_allocations']} ({stats['injection_rate']*100:.1f}%)"
        )
        print(f"Fallback:             {stats['fallback_allocations']}")
        print(
            f"Total Bytes:          {stats['total_bytes_allocated']:,} ({stats['total_bytes_allocated'] / 1024**3:.2f} GB)"
        )
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
        return f"MemoryInjector({status}, pool={self.memory_pool})"
