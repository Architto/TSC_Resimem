#!/usr/bin/env python3
"""Demonstrate the core ResiMem memory-management workflow."""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from core.memory import GiantTensor, MemoryPool
from core.injection import MemoryInjector


def main():
    print("\n" + "=" * 70)
    print("ResiMem Basic Usage Example")
    print("=" * 70)

    print("\n📦 Step 1: Allocating Giant Tensor...")
    giant = GiantTensor(total_size_gb=4.0, device="cuda:0")
    print(f"   {giant}")
    print(f"   Base Address: {giant.base_ptr:#018x}")

    print("\n💾 Step 2: Creating Memory Pool...")
    pool = MemoryPool(giant)
    print(f"   {pool}")

    pool.print_layout()

    print("\n🔧 Step 3: Manual Memory Allocation...")

    llm_view = pool.allocate(1 * 1024**3)
    print(f"   Allocated 1GB for LLM: {llm_view.data_ptr():#018x}")

    dnn_view = pool.allocate(1 * 1024**3)
    print(f"   Allocated 1GB for DNN: {dnn_view.data_ptr():#018x}")

    pool.print_layout()

    print("\n💉 Step 4: Creating Memory Injector...")
    injector = MemoryInjector(pool, fallback_on_failure=True)
    print(f"   {injector}")

    print("\n🎣 Step 5: Enabling Memory Injection...")
    injector.enable()
    print(f"   Status: {injector}")
    print(f"   ✅ vLLM's memory allocation is now redirected to Giant Tensor!")

    print("\n🚀 Step 6: Ready for vLLM Integration")
    print("   Now you can run vLLM, and it will use the Giant Tensor.")
    print("   Example:")
    print("   >>> from vllm import LLM")
    print("   >>> llm = LLM(model='meta-llama/Llama-2-7b-chat-hf')")
    print("   >>> # vLLM will allocate KV Cache from our Giant Tensor!")

    print("\n📊 Step 7: Statistics")
    injector.print_stats()

    print("\n🧹 Step 8: Cleanup...")

    injector.disable()
    print(f"   Injector disabled: {injector}")

    pool.free(llm_view)
    pool.free(dnn_view)
    print(f"   Memory freed")

    pool.print_layout()

    print("\n✅ Example completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
