# ResiMem Architecture

This document describes the core modules, runtime data flow, and external
dependency boundaries of ResiMem.

## Runtime Data Flow

```text
Request and user state
          |
          v
FocusManager / ResiMemScheduler
          |
          +------------------------+
          |                        |
          v                        v
 GroupSwapManager             GRM / IPC
          |                 (optional service)
          +-----------+------------+
                      |
                      v
            MemoryPool / GiantTensor
                      |
                      v
              KV cache / vLLM
                      |
                      v
          GPU runtime and monitoring
```

## Memory and Scheduling

- `core/memory/giant_tensor.py` owns the preallocated contiguous GPU tensor.
- `core/memory/memory_pool.py` provides allocation, release, and range tracking.
- `core/memory/kv_cache_adapter.py` maps KV cache requests into the ResiMem pool.
- `core/scheduler/` tracks user state and implements grouped swapping, Ghost
  Blocks, and segmented eviction.
- `core/injection/memory_injector.py` provides the vLLM allocation-injection
  interface.
- `core/local_quantum.py` resolves runtime KV local-quantum contracts for known
  model and attention surfaces. Unknown surfaces return `probe_required`; the
  module does not generate commands for scripts outside this repository.

## Lifecycle Entry Points

`core/orchestrator.py` manages the lifecycle of GiantTensor, MemoryPool,
KVCacheAdapter, FocusManager, GroupSwapManager, and GPU memory monitoring. It does
not start GRM implicitly.

`core/grm/` and `core/ipc/` implement the standalone global resource-coordination
service. GRM can run in the calling thread or through `start_in_background()`.
Workers that require GRM must explicitly construct and pass a `GRMClient`. This
keeps the unified orchestrator from exposing service switches that have no active
connection.

## vLLM Integration

- `core/vllm_integration.py` manages the fixed KV cache budget and ResiMem
  lifecycle.
- `core/vllm_block_manager.py` tracks request, user, and KV block relationships.
- `core/vllm_kv_cache_hook.py` connects KV cache allocation and resizing events.
- `core/vllm_patch/` extends the vLLM V1 BlockPool and KVCacheManager interfaces.

These modules are integration layers rather than replacements for vLLM source
code. The current fixed-budget interface requires a compatible vLLM fork. The
fork, third-party source, and local source-path fallbacks are intentionally absent
from this repository.

## DNN Runtime and Observability

`core/dnn_worker/` supports `FULL`, `TINY`, and `MINIMAL` GPU modes, together with
GRM-coordinated eviction and restoration. None of these modes represents CPU
fallback.

`core/monitoring/` records GPU memory, requests, DNN lifecycle events, and service
contract metrics. `core/profiling/` provides operation-span profiling that is
disabled by default. The default `core/config/service_contracts.json` applies
`fallback_mode=none` to every ResiMem service, so CPU states are counted as
unavailable. The analyzer can still evaluate separate external baseline contracts
that explicitly permit CPU execution.

## External Dependency Boundary

The repository does not copy vLLM, PyTorch, Transformers, model weights, datasets,
or caches. Operators install those components in a controlled environment and
remain responsible for their licenses and access requirements. Repository
configuration does not contain absolute model paths, device bindings, output
directories, or credentials.

ResiMem-owned code and documentation use the Apache License 2.0. This license does
not alter the terms that apply to external dependencies, models, datasets, or a
compatible vLLM fork.
