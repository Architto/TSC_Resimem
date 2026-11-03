"""ResiMem memory pool module."""

import threading

import torch
from typing import Optional, List, Tuple
import logging

from .giant_tensor import GiantTensor

logger = logging.getLogger(__name__)


class MemoryRegion:
    """MemoryRegion implementation."""

    def __init__(self, start: int, size: int, allocated: bool = False):
        self.start = start
        self.size = size
        self.allocated = allocated

    @property
    def end(self) -> int:
        """End."""
        return self.start + self.size

    def __repr__(self) -> str:
        status = "ALLOCATED" if self.allocated else "FREE"
        return f"MemoryRegion([{self.start:,}, {self.end:,}), {self.size:,} bytes, {status})"


class MemoryPool:
    """MemoryPool implementation."""

    def __init__(self, giant_tensor: GiantTensor):

        self.giant_tensor = giant_tensor
        self.total_bytes = giant_tensor.total_bytes

        self._lock = threading.RLock()

        self.regions: List[MemoryRegion] = [
            MemoryRegion(start=0, size=self.total_bytes, allocated=False)
        ]

        logger.info(f"MemoryPool initialized with {self.total_bytes:,} bytes")

    def allocate(self, size_bytes: int) -> Optional[torch.Tensor]:
        """Allocate."""
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {size_bytes}")

        with self._lock:

            for i, region in enumerate(self.regions):
                if not region.allocated and region.size >= size_bytes:

                    start = region.start

                    if region.size > size_bytes:

                        self.regions[i] = MemoryRegion(
                            start, size_bytes, allocated=True
                        )
                        self.regions.insert(
                            i + 1,
                            MemoryRegion(
                                start + size_bytes,
                                region.size - size_bytes,
                                allocated=False,
                            ),
                        )
                    else:

                        region.allocated = True

                    view = self.giant_tensor.get_slice(start, size_bytes)

                    logger.debug(
                        f"Allocated {size_bytes:,} bytes at offset {start:,} "
                        f"({size_bytes / 1024**2:.1f} MB)"
                    )

                    return view

            free_bytes = self.get_free_bytes()
            logger.warning(
                f"Failed to allocate {size_bytes:,} bytes "
                f"(free: {free_bytes:,} bytes, {free_bytes / 1024**2:.1f} MB)"
            )
            return None

    def free(self, tensor_view: torch.Tensor) -> bool:
        """Free."""
        ptr = tensor_view.data_ptr()
        size = tensor_view.numel() * tensor_view.element_size()

        with self._lock:

            if self.giant_tensor.base_ptr is None:
                logger.error(
                    "free: unavailable: GiantTensor allocation updated (base_ptr is None)"
                )
                return False

            offset = ptr - self.giant_tensor.base_ptr

            if offset < 0 or offset >= self.total_bytes:
                logger.error(
                    f"Invalid tensor view: ptr={ptr:#x} not in Giant Tensor range"
                )
                return False

            for i, region in enumerate(self.regions):
                if region.start == offset and region.size == size and region.allocated:

                    region.allocated = False

                    self._merge_free_regions()

                    logger.debug(f"Freed {size:,} bytes at offset {offset:,}")
                    return True

            logger.error(f"Failed to free tensor view: no matching region found")
            return False

    def _merge_free_regions(self):

        i = 0
        while i < len(self.regions) - 1:
            current = self.regions[i]
            next_region = self.regions[i + 1]

            if not current.allocated and not next_region.allocated:
                merged = MemoryRegion(
                    start=current.start,
                    size=current.size + next_region.size,
                    allocated=False,
                )
                self.regions[i] = merged
                self.regions.pop(i + 1)

            else:
                i += 1

    def get_free_bytes(self) -> int:
        """Return free bytes."""
        with self._lock:
            return sum(r.size for r in self.regions if not r.allocated)

    def get_allocated_bytes(self) -> int:
        """Return allocated bytes."""
        with self._lock:
            return sum(r.size for r in self.regions if r.allocated)

    def get_fragmentation_ratio(self) -> float:
        """Return fragmentation ratio."""
        with self._lock:
            free_regions = [r for r in self.regions if not r.allocated]

            if not free_regions:
                return 0.0

            return (
                (len(free_regions) - 1) / len(free_regions)
                if len(free_regions) > 1
                else 0.0
            )

    def get_memory_stats(self) -> dict:
        """Return memory stats."""
        with self._lock:
            free_bytes = sum(r.size for r in self.regions if not r.allocated)
            allocated_bytes = sum(r.size for r in self.regions if r.allocated)
            num_regions = len(self.regions)
            num_free = sum(1 for r in self.regions if not r.allocated)
            num_alloc = sum(1 for r in self.regions if r.allocated)
            frag = self.get_fragmentation_ratio()

            return {
                "total_bytes": self.total_bytes,
                "allocated_bytes": allocated_bytes,
                "free_bytes": free_bytes,
                "utilization": (
                    allocated_bytes / self.total_bytes if self.total_bytes > 0 else 0.0
                ),
                "fragmentation_ratio": frag,
                "num_regions": num_regions,
                "num_free_regions": num_free,
                "num_allocated_regions": num_alloc,
            }

    def print_layout(self):
        """Print layout."""
        print("\n" + "=" * 70)
        print("Memory Pool Layout")
        print("=" * 70)

        for i, region in enumerate(self.regions):
            status = "█" * 40 if region.allocated else "░" * 40
            size_mb = region.size / 1024**2
            print(f"Region {i:2d}: {status} {size_mb:8.1f} MB  {region}")

        stats = self.get_memory_stats()
        print("\nStatistics:")
        print(f"  Total:        {stats['total_bytes'] / 1024**3:.2f} GB")
        print(
            f"  Allocated:    {stats['allocated_bytes'] / 1024**3:.2f} GB ({stats['utilization']*100:.1f}%)"
        )
        print(f"  Free:         {stats['free_bytes'] / 1024**3:.2f} GB")
        print(f"  Fragmentation: {stats['fragmentation_ratio']*100:.1f}%")
        print(
            f"  Regions:      {stats['num_regions']} (Allocated: {stats['num_allocated_regions']}, Free: {stats['num_free_regions']})"
        )
        print("=" * 70 + "\n")

    def __repr__(self) -> str:
        stats = self.get_memory_stats()
        return (
            f"MemoryPool(total={self.total_bytes / 1024**3:.2f}GB, "
            f"allocated={stats['allocated_bytes'] / 1024**3:.2f}GB, "
            f"free={stats['free_bytes'] / 1024**3:.2f}GB, "
            f"regions={stats['num_regions']})"
        )
