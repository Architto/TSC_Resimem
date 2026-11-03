#!/usr/bin/env python3
"""Demonstrate dynamic BlockPool expansion and hook integration."""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from core.memory import GiantTensor, MemoryPool
from core.vllm_patch import ResiMemBlockPool, KVCacheHookManager


def example_1_basic_expansion():
    """Demonstrate basic BlockPool expansion."""
    print("\n" + "=" * 70)
    print("Example 1: Basic BlockPool Expansion")
    print("=" * 70 + "\n")

    print("📦 Step 1: Creating ResiMemBlockPool...")
    pool = ResiMemBlockPool(
        num_gpu_blocks=100,
        enable_caching=False,
        hash_block_size=16,
    )
    print(f"   {pool}")
    print(f"   Initial blocks: {pool.num_gpu_blocks}")
    print(f"   Free blocks: {pool.get_num_free_blocks()}")

    print("\n🔧 Step 2: Allocating all blocks...")
    free_count = pool.get_num_free_blocks()
    allocated = pool.get_new_blocks(free_count)
    print(f"   Allocated {len(allocated)} blocks")
    print(f"   Free blocks: {pool.get_num_free_blocks()}")

    print("\n📈 Step 3: Dynamically expanding BlockPool...")
    success = pool.expand_capacity(50)
    print(f"   Expansion success: {success}")
    print(f"   {pool}")
    print(f"   Free blocks: {pool.get_num_free_blocks()}")

    print("\n🔧 Step 4: Allocating from expanded pool...")
    new_allocated = pool.get_new_blocks(25)
    print(f"   Allocated {len(new_allocated)} blocks")
    print(f"   Block IDs: {[b.block_id for b in new_allocated[:10]]}...")

    print("\n📊 Step 5: Expansion statistics")
    pool.print_expansion_stats()

    print("✅ Example 1 completed!\n")


def example_2_multiple_expansions():
    """Demonstrate repeated BlockPool expansion."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Expansions")
    print("=" * 70 + "\n")

    pool = ResiMemBlockPool(
        num_gpu_blocks=100,
        enable_caching=False,
        hash_block_size=16,
    )

    print(f"Initial: {pool}\n")

    for round_idx in range(5):
        print(f"Round {round_idx + 1}:")

        free_count = pool.get_num_free_blocks()
        if free_count > 0:
            allocated = pool.get_new_blocks(free_count)
            print(f"  ├─ Allocated {len(allocated)} blocks")

        pool.expand_capacity(50)
        print(f"  ├─ Expanded by 50 blocks")
        print(
            f"  └─ Total blocks: {pool.num_gpu_blocks}, Free: {pool.get_num_free_blocks()}\n"
        )

    print(f"Final: {pool}")
    pool.print_expansion_stats()

    print("✅ Example 2 completed!\n")


def example_3_with_memory_pool():
    """Demonstrate BlockPool integration with MemoryPool."""
    print("\n" + "=" * 70)
    print("Example 3: Integration with MemoryPool")
    print("=" * 70 + "\n")

    print("📦 Step 1: Creating Giant Tensor and MemoryPool...")
    giant = GiantTensor(total_size_gb=4.0, device="cuda:0")
    memory_pool = MemoryPool(giant)
    print(f"   {giant}")
    print(f"   {memory_pool}")

    print("\n📦 Step 2: Creating ResiMemBlockPool with MemoryPool...")
    pool = ResiMemBlockPool(
        num_gpu_blocks=100,
        enable_caching=False,
        hash_block_size=16,
        memory_pool=memory_pool,
    )
    print(f"   {pool}")

    print("\n📈 Step 3: Expanding BlockPool...")
    pool.expand_capacity(100)
    print(f"   {pool}")

    print("\n📊 Step 4: Memory Pool layout")
    memory_pool.print_layout()

    print("✅ Example 3 completed!\n")


def example_4_hook_manager():
    """Demonstrate KVCacheHookManager lifecycle management."""
    print("\n" + "=" * 70)
    print("Example 4: KVCacheHookManager Usage")
    print("=" * 70 + "\n")

    print("🎣 Step 1: Creating KVCacheHookManager...")
    hook_manager = KVCacheHookManager(
        expansion_strategy="aggressive",
        expansion_factor=1.5,
        min_expansion_blocks=50,
    )
    print(f"   {hook_manager}")

    print("\n🔓 Step 2: Enabling Hook...")
    hook_manager.enable()
    print(f"   {hook_manager}")
    print(f"   ✅ vLLM's KVCacheManager.allocate_slots() is now hooked!")

    print("\n🚀 Step 3: Ready for vLLM Integration")
    print("   When vLLM runs and BlockPool becomes full:")
    print("   1. Hook detects block shortage")
    print("   2. Automatically triggers expansion")
    print("   3. vLLM continues without preemption")
    print("   4. Hook records statistics")

    print("\n📊 Step 4: Hook statistics")
    hook_manager.print_stats()

    print("🔒 Step 5: Disabling Hook...")
    hook_manager.disable()
    print(f"   {hook_manager}")

    print("✅ Example 4 completed!\n")


def example_5_context_manager():
    """Demonstrate context-managed hook activation."""
    print("\n" + "=" * 70)
    print("Example 5: Using Context Manager")
    print("=" * 70 + "\n")

    hook_manager = KVCacheHookManager()

    print(f"Before context: {hook_manager}")

    print("\nEntering context...")
    with hook_manager:
        print(f"  Inside context: {hook_manager}")
        print(f"  ✅ Hook is automatically enabled!")

        # llm = LLM(model='...')
        # llm.generate(...)

    print(f"\nAfter context: {hook_manager}")
    print(f"✅ Hook is automatically disabled!")

    print("\n✅ Example 5 completed!\n")


def main():
    """Run all BlockPool examples."""
    print("\n" + "=" * 70)
    print("ResiMem BlockPool Usage Examples")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    example_1_basic_expansion()
    example_2_multiple_expansions()
    example_3_with_memory_pool()
    example_4_hook_manager()
    example_5_context_manager()

    print("\n" + "=" * 70)
    print("🎉 All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
