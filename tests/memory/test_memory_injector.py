#!/usr/bin/env python3
"""ResiMem test memory injector module."""

import pytest
import torch
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.memory import GiantTensor, MemoryPool
from core.injection import MemoryInjector


class TestMemoryInjectorBasic:
    """TestMemoryInjectorBasic implementation."""

    def test_initialization(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        assert not injector.is_enabled
        assert injector.memory_pool == pool

        print(f"✅ MemoryInjector initialized: {injector}")

    def test_enable_disable(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        success = injector.enable()
        assert success
        assert injector.is_enabled

        print(f"✅ Injector enabled")

        success = injector.disable()
        assert success
        assert not injector.is_enabled

        print(f"✅ Injector disabled")

    def test_context_manager(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        assert not injector.is_enabled

        with injector:
            assert injector.is_enabled
            print(f"✅ Injector enabled in context")

        assert not injector.is_enabled
        print(f"✅ Injector auto-disabled after context")

    def test_double_enable(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        injector.enable()
        injector.enable()

        assert injector.is_enabled

        injector.disable()

        print(f"✅ Double enable handled correctly")

    def test_stats(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        stats = injector.get_stats()

        assert stats["total_allocations"] == 0
        assert stats["injected_allocations"] == 0
        assert stats["fallback_allocations"] == 0

        print(f"✅ Stats initialized correctly")

    def test_hook_verification(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        try:
            import vllm.v1.worker.gpu.attn_utils as attn_utils
        except ImportError:
            pytest.skip("vLLM not found")

        original_func = attn_utils._allocate_kv_cache

        injector.enable()

        assert attn_utils._allocate_kv_cache != original_func
        assert attn_utils._allocate_kv_cache == injector._patched_allocate_kv_cache

        print(f"✅ Hook installed correctly")

        injector.disable()

        assert attn_utils._allocate_kv_cache == original_func

        print(f"✅ Hook removed correctly")


class TestMemoryInjectorAdvanced:
    """TestMemoryInjectorAdvanced implementation."""

    def test_fallback_mechanism(self):

        giant = GiantTensor(total_size_gb=0.1, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool, fallback_on_failure=True)

        injector.enable()

        small_view = pool.allocate(50 * 1024**2)  # 50MB
        assert small_view is not None

        injector.disable()

        print(f"✅ Fallback mechanism test completed")

    def test_stats_tracking(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)
        injector = MemoryInjector(pool)

        injector.enable()

        injector.print_stats()

        injector.disable()

        print(f"✅ Stats tracking verified")


def main():
    """Main."""
    print("\n" + "=" * 70)
    print("Memory Injector")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70 + "\n")

    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
