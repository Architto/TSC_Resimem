"""ResiMem vllm integration module."""

import logging
import os
import threading
import copy
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

import torch

from .local_quantum import serialize_contract_for_env

logger = logging.getLogger(__name__)


@dataclass
class VLLMIntegrationConfig:
    """VLLMIntegrationConfig implementation."""

    total_gpu_memory_gb: float = 20.0
    device: str = "cuda:0"

    total_vram_gb: Optional[float] = None
    model_weights_gb: Optional[float] = None
    model_path: Optional[str] = None
    safe_buffer_gb: float = 1.0

    dnn_safety_buffer_gb: float = 0.5

    enable_adaptive_kv: bool = True
    initial_kv_cache_gb: float = 1.5
    min_kv_cache_gb: float = 1.0
    max_kv_cache_gb: Optional[float] = None
    expand_threshold: float = 0.80
    shrink_threshold: float = 0.30
    expand_factor: float = 1.5
    shrink_factor: float = 0.7
    check_interval_s: float = 5.0

    enable_logging: bool = True
    fallback_on_failure: bool = True
    lazy_giant_tensor: bool = False
    capture_runtime_overhead_probe_pre_kv: bool = False
    local_quantum_contract: Optional[Dict[str, Any]] = None

    def get_giant_tensor_gb(self) -> float:
        if self.total_vram_gb is None:
            return self.total_gpu_memory_gb

        weights_gb = self.model_weights_gb
        if weights_gb is None and self.model_path:
            weights_gb = estimate_model_weights_gb(self.model_path)
        if weights_gb is None:
            return self.total_gpu_memory_gb

        computed = max(self.total_vram_gb - weights_gb - self.safe_buffer_gb, 0.1)

        if (
            self.enable_adaptive_kv
            and self.max_kv_cache_gb is not None
            and self.max_kv_cache_gb > 0
        ):
            return max(0.1, min(computed, float(self.max_kv_cache_gb)))

        return computed


def estimate_model_weights_gb(model_path: str) -> Optional[float]:
    try:
        from pathlib import Path

        path = Path(model_path)
        if not path.exists():
            return None
        if path.is_file():
            return path.stat().st_size / 1024**3
        patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
        total_bytes = 0
        for pattern in patterns:
            for file in path.rglob(pattern):
                if file.is_file():
                    total_bytes += file.stat().st_size
        if total_bytes <= 0:
            return None
        return total_bytes / 1024**3
    except Exception as e:
        logger.warning(
            f"estimate_model_weights_gb estimate model weights gb: failed (path={model_path}): {e}"
        )
        return None


class VLLMKVCacheInjector:
    """VLLMKVCacheInjector implementation."""

    _instance: Optional["VLLMKVCacheInjector"] = None

    def __init__(
        self,
        giant_tensor: "GiantTensor",
        config: Optional[VLLMIntegrationConfig] = None,
    ):

        self.giant_tensor = giant_tensor
        self.config = config or VLLMIntegrationConfig()

        self._original_allocate_kv_cache: Optional[Callable] = None
        self._original_allocate_kv_cache_tensors_v1: Optional[Callable] = None
        self._original_allocate_kv_cache_tensors_v2: Optional[Callable] = None
        self._is_enabled = False
        self._activation_mode = "inactive"

        self._allocated_offset = 0
        self._allocations: Dict[str, Dict] = {}

        self._lock = threading.RLock()

        self.stats = {
            "total_allocations": 0,
            "total_bytes_allocated": 0,
            "fallback_count": 0,
        }
        self._runtime_overhead_probe_pre_kv: Optional[Dict[str, Any]] = None

        VLLMKVCacheInjector._instance = self

        logger.info(f"VLLMKVCacheInjector init: completed successfully")

    @classmethod
    def get_instance(cls) -> Optional["VLLMKVCacheInjector"]:
        """Return instance."""
        return cls._instance

    def enable(self) -> bool:
        """Enable."""
        if self._is_enabled:
            logger.warning("VLLMKVCacheInjector enable: already active")
            return True

        try:
            kv_cache_bytes = os.environ.get("RESIMEM_KV_CACHE_BYTES")
            if not kv_cache_bytes:
                logger.error(
                    "❌ enable: missing required value RESIMEM_KV_CACHE_BYTES, initialize() completed successfully"
                )
                return False

            #

            self._is_enabled = True
            self._activation_mode = "env_passthrough"
            self._original_allocate_kv_cache = None
            self._original_allocate_kv_cache_tensors_v1 = None
            self._original_allocate_kv_cache_tensors_v2 = None
            logger.info("✅ VLLMKVCacheInjector enable: enabled(env_passthrough)")
            logger.info(f"   RESIMEM_KV_CACHE_BYTES={kv_cache_bytes}")
            logger.info(f"   Giant Tensor: {self.giant_tensor}")

            return True

        except Exception as e:
            logger.error(f"❌ enable: failed: {e}", exc_info=True)
            return False

    def disable(self) -> bool:
        """Disable."""
        if not self._is_enabled:
            return True

        try:
            self._original_allocate_kv_cache = None
            self._original_allocate_kv_cache_tensors_v1 = None
            self._original_allocate_kv_cache_tensors_v2 = None
            self._is_enabled = False
            self._activation_mode = "inactive"
            logger.info("✅ VLLMKVCacheInjector disable: disabled")

            return True

        except Exception as e:
            logger.error(f"❌ disable: failed: {e}", exc_info=True)
            return False

    @staticmethod
    def _patched_allocate_kv_cache_tensors(self_runner, kv_cache_config):
        device = getattr(self_runner, "device", None)
        if device is None:
            device = torch.device("cuda:0")
        injector = VLLMKVCacheInjector.get_instance()
        if injector is None:
            raise RuntimeError("VLLMKVCacheInjector instance not available")
        return injector._hooked_allocate_kv_cache(kv_cache_config, device)

    def _hooked_allocate_kv_cache(
        self,
        kv_cache_config,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:

        with self._lock:
            self.stats["total_allocations"] += 1
            self._capture_runtime_overhead_probe_pre_kv(
                kv_cache_config=kv_cache_config,
                device=device,
            )

            logger.info(f"🔄 hooked allocate kv cache KV Cache allocation updated")
            logger.info(f"   num_blocks: {kv_cache_config.num_blocks}")
            logger.info(
                f"   kv_cache_tensors: {len(kv_cache_config.kv_cache_tensors)} hooked allocate kv cache"
            )

            kv_cache_raw_tensors: Dict[str, torch.Tensor] = {}

            try:

                offset_before = self._allocated_offset
                bytes_allocated_before = self.stats["total_bytes_allocated"]

                for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                    size_bytes = kv_cache_tensor.size

                    logger.info(
                        f"   hooked allocate kv cache: allocation updated: {size_bytes:,}  "
                        f"({size_bytes / 1024**3:.2f} GB)"
                    )

                    tensor_view = self._allocate_from_giant_tensor(size_bytes, device)

                    if tensor_view is None:

                        self._allocated_offset = offset_before
                        self.stats["total_bytes_allocated"] = bytes_allocated_before
                        kv_cache_raw_tensors.clear()
                        self._allocations = {
                            k: v
                            for k, v in self._allocations.items()
                            if v.get("offset", 0) < offset_before
                        }

                        if self.config.fallback_on_failure:
                            logger.warning(
                                "hooked allocate kv cache GiantTensor failed, offset,allocation updated"
                            )
                            return self._fallback_allocate(kv_cache_config, device)
                        else:
                            raise RuntimeError(
                                f"hooked allocate kv cache: unavailable GiantTensor allocation updated {size_bytes} "
                            )

                    for layer_name in kv_cache_tensor.shared_by:
                        kv_cache_raw_tensors[layer_name] = tensor_view

                        self._allocations[layer_name] = {
                            "offset": self._allocated_offset - size_bytes,
                            "size": size_bytes,
                            "tensor": tensor_view,
                        }

                    self.stats["total_bytes_allocated"] += size_bytes

                layer_names = set()
                for group in kv_cache_config.kv_cache_groups:
                    for layer_name in group.layer_names:
                        layer_names.add(layer_name)

                assert layer_names == set(
                    kv_cache_raw_tensors.keys()
                ), "hooked allocate kv cache"

                logger.info(
                    f"✅ KV Cache hooked allocate kv cache: completed successfully: "
                    f"{len(kv_cache_raw_tensors)} , "
                    f"{self.stats['total_bytes_allocated'] / 1024**3:.2f} GB"
                )

                return kv_cache_raw_tensors

            except Exception as e:
                logger.error(
                    f"❌ KV Cache hooked allocate kv cache: failed: {e}", exc_info=True
                )

                if self.config.fallback_on_failure:
                    return self._fallback_allocate(kv_cache_config, device)
                else:
                    raise

    def _capture_runtime_overhead_probe_pre_kv(
        self,
        *,
        kv_cache_config: Any,
        device: torch.device,
    ) -> None:

        if self._runtime_overhead_probe_pre_kv is not None:
            return

        try:
            device_index = (
                device.index if getattr(device, "index", None) is not None else 0
            )
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            gpu_used_gb = float((total_bytes - free_bytes) / 1024**3)
            cuda_free_gb = float(free_bytes / 1024**3)
            model_weights_gb = float(
                getattr(self.config, "model_weights_gb", 0.0) or 0.0
            )
            runtime_base_overhead_gb = 0.0
            status = "invalid"
            if gpu_used_gb >= 0.0 and model_weights_gb > 0.0:
                runtime_base_overhead_gb = max(0.0, gpu_used_gb - model_weights_gb)
                status = "valid"
            self._runtime_overhead_probe_pre_kv = {
                "runtime_probe_stage": "pre_kv_allocation",
                "gpu_used_gb": round(gpu_used_gb, 6),
                "cuda_free_gb": round(cuda_free_gb, 6),
                "model_weights_gb": round(model_weights_gb, 6),
                "kv_allocated_gb": 0.0,
                "loaded_dnn_gb": 0.0,
                "runtime_base_overhead_gb": round(runtime_base_overhead_gb, 6),
                "runtime_base_overhead_status": status,
                "note": "captured_inside_allocate_kv_cache_before_allocation,num_blocks={num_blocks}".format(
                    num_blocks=int(getattr(kv_cache_config, "num_blocks", 0) or 0)
                ),
            }
        except Exception as exc:
            self._runtime_overhead_probe_pre_kv = {
                "runtime_probe_stage": "pre_kv_allocation",
                "gpu_used_gb": -1.0,
                "cuda_free_gb": -1.0,
                "model_weights_gb": float(
                    getattr(self.config, "model_weights_gb", 0.0) or 0.0
                ),
                "kv_allocated_gb": 0.0,
                "loaded_dnn_gb": 0.0,
                "runtime_base_overhead_gb": 0.0,
                "runtime_base_overhead_status": "capture_error",
                "note": f"capture_err={str(exc)[:160]}",
            }

    def get_runtime_overhead_probe_pre_kv(self) -> Optional[Dict[str, Any]]:
        if self._runtime_overhead_probe_pre_kv is None:
            return None
        return copy.deepcopy(self._runtime_overhead_probe_pre_kv)

    def _allocate_from_giant_tensor(
        self,
        size_bytes: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:

        available = self.giant_tensor.total_bytes - self._allocated_offset

        if size_bytes > available:
            logger.error(
                f"GiantTensor allocate from giant tensor: insufficient capacity: "
                f" {size_bytes / 1024**3:.2f} GB, "
                f" {available / 1024**3:.2f} GB"
            )
            return None

        tensor_view = self.giant_tensor.get_slice(self._allocated_offset, size_bytes)

        tensor_int8 = tensor_view.view(torch.int8)

        self._allocated_offset += size_bytes

        logger.debug(
            f"allocate from giant tensor GiantTensor allocation updated: "
            f"offset={self._allocated_offset - size_bytes}, "
            f"size={size_bytes}"
        )

        return tensor_int8

    def _fallback_allocate(
        self,
        kv_cache_config,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:

        self.stats["fallback_count"] += 1

        if self._original_allocate_kv_cache is not None:
            logger.warning("fallback allocate vLLM allocation updated")
            return self._original_allocate_kv_cache(kv_cache_config, device)
        else:
            raise RuntimeError("fallback allocate: unavailable:saved")

    def get_allocation_info(self) -> Dict:
        """Return allocation info."""
        with self._lock:
            return {
                "is_enabled": self._is_enabled,
                "activation_mode": self._activation_mode,
                "allocated_offset": self._allocated_offset,
                "allocated_gb": self._allocated_offset / 1024**3,
                "total_gb": self.giant_tensor.total_bytes / 1024**3,
                "utilization": self._allocated_offset / self.giant_tensor.total_bytes,
                "stats": self.stats.copy(),
                "num_layers": len(self._allocations),
            }

    def __repr__(self) -> str:
        status = "ENABLED" if self._is_enabled else "DISABLED"
        return (
            f"VLLMKVCacheInjector({status}, "
            f"allocated={self._allocated_offset / 1024**3:.2f}GB)"
        )


class VLLMIntegrationManager:
    """VLLMIntegrationManager implementation."""

    def __init__(
        self,
        config: Optional[VLLMIntegrationConfig] = None,
    ):

        self.config = config or VLLMIntegrationConfig()

        self.giant_tensor: Optional["GiantTensor"] = None
        self.kv_cache_injector: Optional[VLLMKVCacheInjector] = None

        self._initialized = False
        self._enabled = False

        logger.info(f"VLLMIntegrationManager init: created,: {config}")

    def initialize(self) -> bool:
        """Initialize."""
        if self._initialized:
            logger.warning("VLLMIntegrationManager initialize: already active")
            return True

        logger.info("=" * 60)
        logger.info("VLLMIntegrationManager initialize: started")
        logger.info("=" * 60)

        try:

            from .memory.giant_tensor import GiantTensor

            giant_gb = self.config.get_giant_tensor_gb()
            logger.info(f"initialize 1/2: created Giant Tensor ({giant_gb:.2f} GB)...")
            self.giant_tensor = GiantTensor(
                total_size_gb=giant_gb,
                device=self.config.device,
                lazy_allocate=self.config.lazy_giant_tensor,
            )

            if self.config.enable_adaptive_kv:
                initial_bytes = int(self.config.initial_kv_cache_gb * 1024**3)
                logger.info(
                    f"🔧 initialize:  KV Cache = {self.config.initial_kv_cache_gb:.2f} GB"
                )
            else:
                initial_bytes = self.giant_tensor.total_bytes
                logger.info(f"🔧 initialize: KV Cache = {giant_gb:.2f} GB")

            os.environ["RESIMEM_KV_CACHE_BYTES"] = str(initial_bytes)
            os.environ["RESIMEM_KV_CACHE_GB"] = f"{initial_bytes / 1024**3:.6f}"
            self._export_local_quantum_contract_env()

            logger.info("initialize 2/2: created KV Cache ...")
            self.kv_cache_injector = VLLMKVCacheInjector(
                giant_tensor=self.giant_tensor,
                config=self.config,
            )

            self._initialized = True

            logger.info("=" * 60)
            logger.info("✅ VLLMIntegrationManager initialize: completed successfully")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"❌ initialize: failed: {e}", exc_info=True)
            self._cleanup()
            return False

    def _export_local_quantum_contract_env(self) -> None:
        contract = self.config.local_quantum_contract
        if not contract:
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_CONTRACT_JSON", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_MANIFEST_STATUS", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_CONTRACT_STATUS", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_RUNTIME_POLICY", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_SEMANTIC_KEY", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_CAPACITY_KEY", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_OBSERVABLE_BLOCKS", None)
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_PROBE_REQUIRED", None)
            return

        contract_json = serialize_contract_for_env(contract)
        os.environ["RESIMEM_LOCAL_QUANTUM_CONTRACT_JSON"] = contract_json
        os.environ["RESIMEM_LOCAL_QUANTUM_MANIFEST_STATUS"] = str(
            contract.get("manifest_status", "")
        )
        os.environ["RESIMEM_LOCAL_QUANTUM_CONTRACT_STATUS"] = str(
            contract.get("contract_status", "")
        )
        os.environ["RESIMEM_LOCAL_QUANTUM_RUNTIME_POLICY"] = str(
            contract.get("runtime_policy", "")
        )
        os.environ["RESIMEM_LOCAL_QUANTUM_SEMANTIC_KEY"] = str(
            contract.get("semantic_mapping_key", "")
        )
        os.environ["RESIMEM_LOCAL_QUANTUM_CAPACITY_KEY"] = str(
            contract.get("capacity_scaling_key", "")
        )
        observable_blocks = contract.get("observable_local_quantum_blocks")
        if observable_blocks is None:
            os.environ.pop("RESIMEM_LOCAL_QUANTUM_OBSERVABLE_BLOCKS", None)
        else:
            os.environ["RESIMEM_LOCAL_QUANTUM_OBSERVABLE_BLOCKS"] = str(
                observable_blocks
            )
        os.environ["RESIMEM_LOCAL_QUANTUM_PROBE_REQUIRED"] = str(
            bool(contract.get("probe_required", False))
        ).lower()
        logger.info(
            "export local quantum contract env Local quantum contract: family=%s surface=%s status=%s policy=%s probe_required=%s",
            contract.get("family_key", "unknown"),
            contract.get("surface_key", "unknown"),
            contract.get("manifest_status", "unknown"),
            contract.get("runtime_policy", "unknown"),
            bool(contract.get("probe_required", False)),
        )

    def enable(self) -> bool:
        """Enable."""
        if not self._initialized:
            logger.error("enable initialize()")
            return False

        if self._enabled:
            logger.warning("enable: already active")
            return True

        if self.kv_cache_injector.enable():
            self._enabled = True
            logger.info("✅ vLLM enable: enabled")
            return True
        else:
            logger.error("❌ enable: failed")
            return False

    def disable(self) -> bool:
        """Disable."""
        if not self._enabled:
            return True

        if self.kv_cache_injector:
            self.kv_cache_injector.disable()

        self._enabled = False
        logger.info("✅ vLLM disable: disabled")

        return True

    def _cleanup(self):

        self.disable()

        if self.giant_tensor is not None:
            try:
                self.giant_tensor.close()
            except Exception as e:
                logger.warning(f"GiantTensor.close() cleanup: failed: {e}")
            self.giant_tensor = None
            torch.cuda.empty_cache()

        self._initialized = False

    def get_status(self) -> Dict:
        """Return status."""
        result = {
            "initialized": self._initialized,
            "enabled": self._enabled,
        }

        if self.giant_tensor:
            result["giant_tensor"] = self.giant_tensor.get_memory_info()

        if self.kv_cache_injector:
            result["kv_cache_injector"] = self.kv_cache_injector.get_allocation_info()
        if self.config.local_quantum_contract:
            result["local_quantum_contract"] = copy.deepcopy(
                self.config.local_quantum_contract
            )
            result["local_quantum_env"] = {
                "manifest_status": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_MANIFEST_STATUS", ""
                ),
                "contract_status": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_CONTRACT_STATUS", ""
                ),
                "runtime_policy": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_RUNTIME_POLICY", ""
                ),
                "semantic_key": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_SEMANTIC_KEY", ""
                ),
                "capacity_key": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_CAPACITY_KEY", ""
                ),
                "observable_blocks": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_OBSERVABLE_BLOCKS", ""
                ),
                "probe_required": os.environ.get(
                    "RESIMEM_LOCAL_QUANTUM_PROBE_REQUIRED", ""
                ),
            }

        return result

    def __enter__(self):

        self.initialize()
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self._cleanup()
        return False

    def get_runtime_overhead_probe_pre_kv(self) -> Optional[Dict[str, Any]]:
        if self.kv_cache_injector is None:
            return None
        getter = getattr(
            self.kv_cache_injector, "get_runtime_overhead_probe_pre_kv", None
        )
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"VLLMIntegrationManager("
            f"initialized={self._initialized}, "
            f"enabled={self._enabled})"
        )


def create_vllm_integration(
    total_gpu_memory_gb: float = 20.0,
    device: str = "cuda:0",
    auto_enable: bool = True,
) -> VLLMIntegrationManager:
    """Create vllm integration."""
    config = VLLMIntegrationConfig(
        total_gpu_memory_gb=total_gpu_memory_gb,
        device=device,
    )

    manager = VLLMIntegrationManager(config)
    manager.initialize()

    if auto_enable:
        manager.enable()

    return manager


__all__ = [
    "VLLMIntegrationConfig",
    "VLLMKVCacheInjector",
    "VLLMIntegrationManager",
    "create_vllm_integration",
]
