#!/usr/bin/env python3
"""ResiMem test giant tensor module."""

import pytest
import torch
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.memory import GiantTensor


class TestGiantTensorBasic:
    """TestGiantTensorBasic implementation."""

    def test_allocation_success(self):

        giant = GiantTensor(total_size_gb=2.0, device="cuda:0")

        assert giant.total_bytes == 2 * 1024**3
        assert giant.tensor.device.type == "cuda"
        assert giant.tensor.dtype == torch.uint8
        assert giant.base_ptr > 0

        print(f"✅ Giant Tensor allocated: {giant}")

    def test_allocation_failure(self):

        with pytest.raises(RuntimeError, match="Failed to allocate"):
            GiantTensor(total_size_gb=1000.0, device="cuda:0")

    def test_zero_copy_slice(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")

        view1 = giant.get_slice(start_bytes=0, size_bytes=512 * 1024**2)

        assert view1.data_ptr() == giant.base_ptr

        view2 = giant.get_slice(start_bytes=512 * 1024**2, size_bytes=512 * 1024**2)

        expected_ptr = giant.base_ptr + 512 * 1024**2
        assert view2.data_ptr() == expected_ptr

        print(f"✅ Zero-copy verified:")
        print(f"   View 1: {view1.data_ptr():#018x}")
        print(f"   View 2: {view2.data_ptr():#018x}")
        print(f"   Offset: {view2.data_ptr() - view1.data_ptr():,} bytes")

    def test_data_sharing(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")

        view = giant.get_slice(start_bytes=0, size_bytes=1024)

        view[100] = 42

        assert giant.tensor[100] == 42

        print(f"✅ Data sharing verified: giant.tensor[100] = {giant.tensor[100]}")

    def test_boundary_checks(self):

        giant = GiantTensor(total_size_gb=1.0, device="cuda:0")

        with pytest.raises(ValueError, match="out of range"):
            giant.get_slice(start_bytes=-1, size_bytes=1024)

        with pytest.raises(ValueError, match="exceeds total size"):
            giant.get_slice(start_bytes=0, size_bytes=giant.total_bytes + 1)

        with pytest.raises(ValueError, match="out of range"):
            giant.get_slice(start_bytes=giant.total_bytes, size_bytes=1)

        print(f"✅ Boundary checks passed")

    def test_physical_continuity(self):

        giant = GiantTensor(total_size_gb=2.0, device="cuda:0")

        is_continuous = giant.verify_continuity()

        assert is_continuous, "Physical address is NOT continuous"

        print(f"✅ Physical address continuity verified")

    def test_memory_info(self):

        giant = GiantTensor(total_size_gb=1.5, device="cuda:0")

        info = giant.get_memory_info()

        assert info["total_bytes"] == int(1.5 * 1024**3)
        assert info["total_gb"] == 1.5
        assert "base_ptr" in info
        assert info["device"] == "cuda:0"
        assert info["dtype"] == "torch.uint8"

        print(f"✅ Memory info:")
        for key, value in info.items():
            print(f"   {key}: {value}")


class TestGiantTensorAdvanced:
    """TestGiantTensorAdvanced implementation."""

    def test_multiple_views(self):

        giant = GiantTensor(total_size_gb=4.0, device="cuda:0")

        views = []
        for i in range(4):
            start = i * 1024**3
            view = giant.get_slice(start_bytes=start, size_bytes=1024**3)
            views.append(view)

        for i in range(len(views) - 1):
            ptr1 = views[i].data_ptr()
            ptr2 = views[i + 1].data_ptr()
            assert ptr2 - ptr1 == 1024**3, f"Views {i} and {i+1} are not continuous"

        for i, view in enumerate(views):
            view[0] = i + 1

        for i, view in enumerate(views):
            assert view[0] == i + 1, f"View {i} data corrupted"

        print(f"✅ Multiple views test passed")
        print(f"   Created {len(views)} views, each 1GB")
        print(f"   All views are continuous and data-isolated")

    def test_large_scale_allocation(self):

        total_memory = torch.cuda.get_device_properties(0).total_memory

        target_size_gb = (total_memory * 0.7) / 1024**3

        print(f"GPU Total Memory: {total_memory / 1024**3:.2f} GB")
        print(f"Target Allocation: {target_size_gb:.2f} GB (70%)")

        try:
            giant = GiantTensor(total_size_gb=target_size_gb, device="cuda:0")

            assert giant.total_bytes == int(target_size_gb * 1024**3)

            assert giant.verify_continuity()

            print(f"✅ Large-scale allocation successful")

        except RuntimeError as e:
            pytest.skip(f"GPU memory insufficient for this test: {e}")


def main():
    """Main."""
    print("\n" + "=" * 70)
    print("Giant Tensor")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70 + "\n")

    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
