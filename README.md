# ResiMem

ResiMem is a GPU memory management and scheduling system for colocated LLM and
DNN services. This repository contains the core runtime, default configuration,
minimal examples, unit tests, and system architecture documentation.

## Repository Layout

- `core/memory/`: GiantTensor, memory pool, and KV cache adapters.
- `core/config/`: default GPU-only service contracts.
- `core/scheduler/`: multi-user state, swapping, and eviction policies.
- `core/grm/`: standalone global resource management service.
- `core/ipc/`: inter-process protocols and shared-memory queues.
- `core/injection/`: vLLM memory-allocation injection.
- `core/dnn_worker/`: GPU DNN worker state and execution interfaces.
- `core/monitoring/`: GPU, request, and service-contract observability.
- `core/profiling/`: optional runtime operation profiling.
- `core/vllm_patch/`: helpers for integrating with vLLM V1.
- `examples/`: minimal API usage examples.
- `tests/`: module-level unit tests.
- `docs/`: system architecture documentation.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module responsibilities and
runtime data flow.

## Requirements

- Linux.
- Python 3.10 or later.
- An NVIDIA GPU, CUDA, and a compatible PyTorch build.
- A compatible vLLM V1 installation when using the vLLM integration modules.

Install the PyTorch build that matches the host CUDA environment first, then
install ResiMem:

```bash
python -m pip install \
  -e .
```

Install development dependencies with:

```bash
python -m pip install \
  -e '.[dev]'
```

## vLLM Compatibility Boundary

`core/vllm_patch/` depends on the vLLM V1 `BlockPool`, `KVCacheBlock`, and
`KVCacheManager` interfaces. `core/injection/` also requires the GPU worker KV
cache allocation interface. The fixed-budget integration reads
`RESIMEM_KV_CACHE_BYTES` and therefore requires a compatible vLLM fork that
provides the corresponding ResiMem KV cache allocation entry point.

This repository does not include that fork and does not load a repository-local
`vllm_src/` tree. Compatibility with an unmodified upstream vLLM release is not
guaranteed. A reproducible deployment should pin and distribute the compatible
fork or patch set separately.

## Validation

```bash
python -m compileall \
  core \
  examples \
  tests

python -m pytest \
  tests \
  -q
```

`examples/basic_usage.py` and `examples/block_pool_usage.py` allocate GPU memory.
Check device capacity before running them.

## Service Contracts

The default
[core/config/service_contracts.json](core/config/service_contracts.json) limits
all ResiMem DNN services to GPU-only execution. The service-contract analyzer can
parse CPU traces produced by external static baselines, but CPU fallback is not a
legal service path under the default ResiMem contract.

## Repository Scope

This repository excludes model weights, datasets, third-party source trees,
runtime logs, evaluation results, paper artifacts, credentials, and machine-local
configuration. Use `.env.example` as the environment-variable template; real
`.env` files are ignored by Git.

## License

ResiMem source code and documentation are released under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution information.
The license includes explicit copyright and patent grants.

vLLM, PyTorch, model weights, and other third-party components are not copied into
this repository. Runtime dependencies and any separately distributed compatible
vLLM fork remain subject to their own licenses.
