"""Default-off operation span profiler for runtime overhead attribution.

The profiler is intentionally independent from the runtime telemetry monitor so
that enabling operation spans does not create recursive monitor writes.
"""
from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Type


_TRUE_VALUES = {"1", "true", "yes", "on"}
_SCHEMA_VERSION = "resimem.operation_span.v1"

_lock = threading.Lock()
_parent_span_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "resimem_parent_span_id",
    default="",
)

_configured = False
_enabled = False
_jsonl_path: Optional[Path] = None
_run_id = ""
_repeat_id = ""
_cuda_sync = False
_cuda_event = False
_span_seq = 0


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> bool:
        return False

    def add_fields(self, **fields: Any) -> None:
        return None

    def set_status(self, status: str, error_type: str = "") -> None:
        return None


class _SpanContext:
    def __init__(self, operation: str, step: str, attrs: Dict[str, Any]) -> None:
        self.operation = str(operation)
        self.step = str(step)
        self.attrs = _sanitize_mapping(attrs)
        self.status = "ok"
        self.error_type = ""
        self.span_id = _next_span_id()
        self.parent_span_id = _parent_span_id.get("")
        self._token: Optional[contextvars.Token[str]] = None
        self._start_ns = 0
        self._end_ns = 0
        self._sync_start_ns: Optional[int] = None
        self._sync_end_ns: Optional[int] = None
        self._cuda_start_event: Any = None
        self._cuda_end_event: Any = None

    def __enter__(self) -> "_SpanContext":
        self._maybe_cuda_sync()
        if _cuda_sync:
            self._sync_start_ns = time.perf_counter_ns()
        self._maybe_cuda_event_start()
        self._start_ns = time.perf_counter_ns()
        self._token = _parent_span_id.set(self.span_id)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> bool:
        if exc_type is not None:
            self.status = "failed"
            self.error_type = exc_type.__name__
        self._end_ns = time.perf_counter_ns()
        self._maybe_cuda_event_end()
        self._maybe_cuda_sync()
        if _cuda_sync:
            self._sync_end_ns = time.perf_counter_ns()
        if self._token is not None:
            _parent_span_id.reset(self._token)
        _write_record(self._to_record())
        return False

    def add_fields(self, **fields: Any) -> None:
        self.attrs.update(_sanitize_mapping(fields))

    def set_status(self, status: str, error_type: str = "") -> None:
        self.status = str(status or "ok")
        self.error_type = str(error_type or "")

    def _to_record(self) -> Dict[str, Any]:
        wall_async_ms = max(0.0, (self._end_ns - self._start_ns) / 1_000_000.0)
        wall_sync_ms = None
        if self._sync_start_ns is not None and self._sync_end_ns is not None:
            wall_sync_ms = max(
                0.0,
                (self._sync_end_ns - self._sync_start_ns) / 1_000_000.0,
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": _run_id,
            "repeat_id": _repeat_id,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "step": self.step,
            "status": self.status,
            "error_type": self.error_type,
            "ts_start_ns": self._start_ns,
            "ts_end_ns": self._end_ns,
            "wall_async_ms": wall_async_ms,
            "wall_sync_ms": wall_sync_ms,
            "cuda_event_ms": self._cuda_event_elapsed_ms(),
            "attrs": self.attrs,
        }

    @staticmethod
    def _maybe_cuda_sync() -> None:
        if not _cuda_sync:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    def _maybe_cuda_event_start(self) -> None:
        if not _cuda_event:
            return
        try:
            import torch

            if torch.cuda.is_available():
                self._cuda_start_event = torch.cuda.Event(enable_timing=True)
                self._cuda_end_event = torch.cuda.Event(enable_timing=True)
                self._cuda_start_event.record()
        except Exception:
            self._cuda_start_event = None
            self._cuda_end_event = None

    def _maybe_cuda_event_end(self) -> None:
        if self._cuda_start_event is None or self._cuda_end_event is None:
            return
        try:
            self._cuda_end_event.record()
        except Exception:
            self._cuda_start_event = None
            self._cuda_end_event = None

    def _cuda_event_elapsed_ms(self) -> Optional[float]:
        if self._cuda_start_event is None or self._cuda_end_event is None:
            return None
        try:
            self._cuda_end_event.synchronize()
            return float(self._cuda_start_event.elapsed_time(self._cuda_end_event))
        except Exception:
            return None


def configure_from_env(
    *,
    run_id: Optional[str] = None,
    repeat_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    default_filename: str = "operation_spans.jsonl",
    force: bool = False,
) -> bool:
    """Configure the operation profiler from environment variables.

    Returns True when operation profiling is enabled and a JSONL output path is
    available. Calling this function is cheap; pass ``force=True`` when a
    benchmark subprocess starts a new group and needs to refresh run metadata.
    """
    global _configured, _enabled, _jsonl_path, _run_id, _repeat_id
    global _cuda_sync, _cuda_event

    with _lock:
        if _configured and not force:
            return bool(_enabled and _jsonl_path is not None)

        enabled = _env_truthy("RESIMEM_PROFILE_OPS")
        _run_id = str(run_id or os.environ.get("RESIMEM_PROFILE_RUN_ID", "") or "")
        _repeat_id = str(
            repeat_id or os.environ.get("RESIMEM_PROFILE_REPEAT_ID", "") or ""
        )
        _cuda_sync = _env_truthy("RESIMEM_PROFILE_CUDA_SYNC")
        _cuda_event = _env_truthy("RESIMEM_PROFILE_CUDA_EVENT")

        path_text = os.environ.get("RESIMEM_PROFILE_JSONL", "").strip()
        if not path_text and enabled and output_dir:
            path_text = str(Path(output_dir) / default_filename)

        _jsonl_path = Path(path_text).expanduser() if path_text else None
        _enabled = bool(enabled and _jsonl_path is not None)
        _configured = True

        if _enabled and _jsonl_path is not None:
            try:
                _jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                _enabled = False

        return bool(_enabled)


def is_enabled() -> bool:
    _ensure_configured()
    return bool(_enabled and _jsonl_path is not None)


def span(operation: str, step: str, **attrs: Any) -> Any:
    """Return a context manager for one operation span.

    The disabled path returns a no-op context object and does not write files.
    """
    if not is_enabled():
        return _NoopSpan()
    return _SpanContext(operation, step, attrs)


def record_span(
    operation: str,
    step: str,
    *,
    status: str = "ok",
    error_type: str = "",
    **attrs: Any,
) -> None:
    if not is_enabled():
        return
    now_ns = time.perf_counter_ns()
    record = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": _run_id,
        "repeat_id": _repeat_id,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "span_id": _next_span_id(),
        "parent_span_id": _parent_span_id.get(""),
        "operation": str(operation),
        "step": str(step),
        "status": str(status or "ok"),
        "error_type": str(error_type or ""),
        "ts_start_ns": now_ns,
        "ts_end_ns": now_ns,
        "wall_async_ms": 0.0,
        "wall_sync_ms": None,
        "cuda_event_ms": None,
        "attrs": _sanitize_mapping(attrs),
    }
    _write_record(record)


def _ensure_configured() -> None:
    if not _configured:
        configure_from_env()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _next_span_id() -> str:
    global _span_seq
    with _lock:
        _span_seq += 1
        return f"{os.getpid()}-{threading.get_ident()}-{_span_seq}"


def _write_record(record: Dict[str, Any]) -> None:
    if not _enabled or _jsonl_path is None:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _lock:
            with _jsonl_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        return


def _sanitize_mapping(values: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): _sanitize_value(value) for key, value in values.items()}


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item) for item in value]
    return str(value)


def _reset_for_tests() -> None:
    global _configured, _enabled, _jsonl_path, _run_id, _repeat_id
    global _cuda_sync, _cuda_event, _span_seq
    with _lock:
        _configured = False
        _enabled = False
        _jsonl_path = None
        _run_id = ""
        _repeat_id = ""
        _cuda_sync = False
        _cuda_event = False
        _span_seq = 0
