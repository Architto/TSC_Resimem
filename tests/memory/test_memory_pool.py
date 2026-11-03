#!/usr/bin/env python3
"""ResiMem test memory pool module."""

import pytest
import torch
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.memory import GiantTensor, MemoryPool


class TestMemoryPoolBasic:
    """TestMemoryPoolBasic implementation."""

    def test_initialization(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        assert pool.total_bytes == giant.total_bytes
        assert len(pool.regions) == 1
        assert pool.regions[0].allocated == False
        assert pool.get_free_bytes() == giant.total_bytes
        assert pool.get_allocated_bytes() == 0

        print(f"✅ MemoryPool initialized: {pool}")

    def test_single_allocation(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        size = 512 * 1024**2
        view = pool.allocate(size)

        assert view is not None
        assert view.numel() == size
        assert pool.get_allocated_bytes() == size
        assert pool.get_free_bytes() == giant.total_bytes - size

        print(f"✅ Allocated {size / 1024**2:.1f} MB")

    def test_multiple_allocations(self):

        giant = GiantTensor(total_size_gb=2.0, device="cuda:0")
        pool = MemoryPool(giant)

        views = []
        for i in range(4):
            view = pool.allocate(512 * 1024**2)
            assert view is not None
            views.append(view)

        total_allocated = 4 * 512 * 1024**2
        assert pool.get_allocated_bytes() == total_allocated

        assert len(pool.regions) == 4

        print(f"✅ Allocated 4 blocks, {len(pool.regions)} regions")

    def test_allocation_failure(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        view = pool.allocate(2 * 1024**3)

        assert view is None
        assert pool.get_allocated_bytes() == 0

        print(f"✅ Allocation failure handled correctly")

    def test_free_memory(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        view = pool.allocate(512 * 1024**2)
        assert pool.get_allocated_bytes() == 512 * 1024**2

        success = pool.free(view)
        assert success
        assert pool.get_allocated_bytes() == 0
        assert pool.get_free_bytes() == giant.total_bytes

        print(f"✅ Memory freed successfully")

    def test_region_merging(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        view1 = pool.allocate(256 * 1024**2)
        view2 = pool.allocate(256 * 1024**2)
        view3 = pool.allocate(256 * 1024**2)

        assert len(pool.regions) == 4  # 3 allocated + 1 free

        pool.free(view2)

        assert len(pool.regions) == 4

        pool.free(view1)

        assert len(pool.regions) == 3  # merged_free, allocated, free

        pool.free(view3)

        assert len(pool.regions) == 1
        assert pool.regions[0].allocated == False
        assert pool.get_free_bytes() == giant.total_bytes

        print(f"✅ Region merging works correctly")


class TestMemoryPoolAdvanced:
    """TestMemoryPoolAdvanced implementation."""

    def test_fragmentation(self):

        giant = GiantTensor(total_size_gb=2.0, device="cuda:0")
        pool = MemoryPool(giant)

        views = []
        for i in range(10):
            view = pool.allocate(100 * 1024**2)  # 100MB each
            views.append(view)

        for i in range(1, 10, 2):
            pool.free(views[i])

        frag_ratio = pool.get_fragmentation_ratio()
        print(f"   Fragmentation ratio: {frag_ratio:.2%}")

        assert frag_ratio > 0

        for i in range(0, 10, 2):
            pool.free(views[i])

        assert len(pool.regions) == 1
        assert pool.get_fragmentation_ratio() == 0.0

        print(f"✅ Fragmentation handling works")

    def test_memory_stats(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        view1 = pool.allocate(256 * 1024**2)
        view2 = pool.allocate(256 * 1024**2)

        stats = pool.get_memory_stats()

        assert stats["total_bytes"] == giant.total_bytes
        assert stats["allocated_bytes"] == 512 * 1024**2
        assert stats["free_bytes"] == giant.total_bytes - 512 * 1024**2
        assert 0 < stats["utilization"] < 1
        assert stats["num_regions"] >= 2

        print(f"✅ Memory stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(
                    f"   {key}: {value:.2%}"
                    if key == "utilization" or "ratio" in key
                    else f"   {key}: {value:.2f}"
                )
            else:
                print(f"   {key}: {value:,}")

    def test_print_layout(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        pool.allocate(256 * 1024**2)
        pool.allocate(256 * 1024**2)

        print("\n")
        pool.print_layout()

        print(f"✅ Layout printing works")

    def test_invalid_free(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        invalid_tensor = torch.zeros(1024, dtype=torch.uint8, device="cuda:0")

        success = pool.free(invalid_tensor)
        assert not success

        print(f"✅ Invalid free detected")

    def test_reallocation_after_free(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")
        pool = MemoryPool(giant)

        view1 = pool.allocate(512 * 1024**2)
        pool.free(view1)

        view2 = pool.allocate(512 * 1024**2)
        assert view2 is not None

        assert view2.data_ptr() == view1.data_ptr()

        print(f"✅ Reallocation works correctly")


def main():
    """Main."""
    print("\n" + "=" * 70)
    print("Memory Pool")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70 + "\n")

    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
