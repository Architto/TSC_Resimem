"""ResiMem init module."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "3.0.0"
__author__ = "ResiMem Team"

_EXPORT_MAP = {
    "GiantTensor": (".memory", "GiantTensor"),
    "MemoryPool": (".memory", "MemoryPool"),
    "KVCacheAdapter": (".memory", "KVCacheAdapter"),
    "KVCacheConfig": (".memory", "KVCacheConfig"),
    "MemoryInjector": (".injection", "MemoryInjector"),
    "FocusManager": (".scheduler", "FocusManager"),
    "FocusManagerConfig": (".scheduler", "FocusManagerConfig"),
    "GroupSwapManager": (".scheduler", "GroupSwapManager"),
    "GroupSwapConfig": (".scheduler", "GroupSwapConfig"),
    "UserState": (".scheduler", "UserState"),
    "ResiMemOrchestrator": (".orchestrator", "ResiMemOrchestrator"),
    "ResiMemConfig": (".orchestrator", "ResiMemConfig"),
    "create_resimem_orchestrator": (".orchestrator", "create_resimem_orchestrator"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
