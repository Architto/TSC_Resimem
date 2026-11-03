"""ResiMem giant tensor module."""

import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GiantTensor:
    """GiantTensor implementation."""

    def __init__(
        self,
        total_size_gb: float,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.uint8,
        lazy_allocate: bool = False,
    ):

        self.total_size_gb = total_size_gb
        self.device = torch.device(device)
        self.dtype = dtype
        self.lazy_allocate = lazy_allocate

        self.total_bytes = int(total_size_gb * 1024**3)

        self.tensor: Optional[torch.Tensor] = None
        self.base_ptr: Optional[int] = None

        if not self.lazy_allocate:
            self._allocate_tensor()

    def _allocate_tensor(self) -> None:
        if self.tensor is not None:
            return
        logger.info(
            f"Allocating Giant Tensor: {self.total_size_gb:.2f} GB on {self.device}"
        )
        try:

            self.tensor = torch.zeros(
                self.total_bytes, dtype=self.dtype, device=self.device
            )

            self.base_ptr = self.tensor.data_ptr()
            logger.info("✅ Giant Tensor allocated successfully")
            logger.info(f"   Base Address: {self.base_ptr:#018x}")
            logger.info(
                f"   Total Bytes: {self.total_bytes:,} ({self.total_size_gb:.2f} GB)"
            )
        except RuntimeError as e:
            logger.error(f"❌ Failed to allocate Giant Tensor: {e}")
            raise RuntimeError(
                f"Failed to allocate {self.total_size_gb:.2f} GB on {self.device}. "
                f"Please check GPU memory availability."
            ) from e

    def _ensure_allocated(self) -> None:
        if self.tensor is None:
            self._allocate_tensor()

    def get_slice(
        self,
        start_bytes: int,
        size_bytes: int,
    ) -> torch.Tensor:
        """Return slice."""

        if start_bytes < 0 or start_bytes >= self.total_bytes:
            raise ValueError(
                f"start_bytes ({start_bytes}) out of range [0, {self.total_bytes})"
            )

        end_bytes = start_bytes + size_bytes
        if end_bytes > self.total_bytes:
            raise ValueError(
                f"Requested range [{start_bytes}, {end_bytes}) exceeds "
                f"total size {self.total_bytes}"
            )

        self._ensure_allocated()
        assert self.tensor is not None
        view = self.tensor[start_bytes:end_bytes]

        assert self.base_ptr is not None
        expected_ptr = self.base_ptr + start_bytes
        actual_ptr = view.data_ptr()

        assert (
            actual_ptr == expected_ptr
        ), f"Zero-copy failed: expected {expected_ptr:#x}, got {actual_ptr:#x}"

        logger.debug(
            f"Created view: [{start_bytes:,}, {end_bytes:,}) "
            f"({size_bytes / 1024**2:.1f} MB)"
        )

        return view

    def get_memory_info(self) -> dict:
        """Return memory info."""
        return {
            "total_bytes": self.total_bytes,
            "total_gb": self.total_size_gb,
            "base_ptr": (
                f"{self.base_ptr:#018x}" if self.base_ptr is not None else "UNALLOCATED"
            ),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "lazy_allocate": self.lazy_allocate,
            "allocated": self.tensor is not None,
        }

    def verify_continuity(self) -> bool:
        """Verify continuity."""
        if self.total_bytes < 2:
            logger.warning("GiantTensor verify continuity(< 2 bytes),unavailable")
            return False

        size = min(1024**2, self.total_bytes // 2)
        if size == 0:
            size = 1
        view1 = self.get_slice(0, size)
        view2 = self.get_slice(size, size)

        ptr1 = view1.data_ptr()
        ptr2 = view2.data_ptr()

        is_continuous = (ptr2 - ptr1) == size

        if is_continuous:
            logger.info("✅ Physical address continuity verified")
        else:
            logger.error(
                f"❌ Physical address NOT continuous: "
                f"ptr1={ptr1:#x}, ptr2={ptr2:#x}, diff={ptr2-ptr1} (expected {size})"
            )

        return is_continuous

    def __repr__(self) -> str:
        base = f"{self.base_ptr:#x}" if self.base_ptr is not None else "UNALLOCATED"
        return (
            f"GiantTensor(size={self.total_size_gb:.2f}GB, "
            f"device={self.device}, base_ptr={base})"
        )

    def close(self):
        """Close."""
        if self.tensor is not None:
            logger.info(f"Releasing Giant Tensor ({self.total_size_gb:.2f} GB)")
            del self.tensor
            self.tensor = None
            self.base_ptr = None
            torch.cuda.empty_cache()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()
        return False

    def __del__(self):

        try:
            self.close()
        except Exception as e:

            import sys

            print(f"[GiantTensor.__del__] del: failed: {e}", file=sys.stderr)
