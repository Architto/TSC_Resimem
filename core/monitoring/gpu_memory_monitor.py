#!/usr/bin/env python3
"""
GPU memory monitor with sampling + stage markers.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from core.profiling.resimem_op_profiler import span as profile_span


@dataclass
class MonitorSample:
    ts: float
    gpu_used_mb: Optional[int]
    torch_allocated_mb: Optional[float]
    torch_reserved_mb: Optional[float]


class GpuMemoryMonitor:
    """Lightweight GPU memory monitor with 0.1s sampling by default."""

    def __init__(
        self,
        sample_interval_s: float = 0.1,
        smi_interval_s: float = 0.5,
        device_index: int = 0,
        output_dir: str = "logs/mem",
        run_id: Optional[str] = None,
        enable_nvidia_smi: bool = True,
        max_samples: int = 6000,
        flush_interval_s: float = 5.0,
    ) -> None:
        self.sample_interval_s = sample_interval_s
        self.smi_interval_s = smi_interval_s
        self.device_index = device_index
        self.nvidia_smi_index = self._resolve_nvidia_smi_index(device_index)
        self.output_dir = output_dir
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.enable_nvidia_smi = enable_nvidia_smi
        self.max_samples = max_samples
        self.flush_interval_s = flush_interval_s

        self._samples: Deque[MonitorSample] = deque(maxlen=max_samples)
        self._pending: List[MonitorSample] = []
        self._marks: List[Dict[str, Any]] = []
        self._io_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_smi_ts = 0.0
        self._last_smi_value: Optional[int] = None
        self._last_flush_ts = 0.0
        self._stream_csv_path = os.path.join(
            self.output_dir, f"gpu_mem_{self.run_id}_stream.csv"
        )
        self._stream_jsonl_path = os.path.join(
            self.output_dir, f"gpu_mem_{self.run_id}_stream.jsonl"
        )
        self._events_jsonl_path = os.path.join(
            self.output_dir, f"runtime_events_{self.run_id}.jsonl"
        )
        self._runtime_timeseries_jsonl_path = os.path.join(
            self.output_dir,
            f"runtime_timeseries_{self.run_id}.jsonl",
        )
        self._runtime_timeseries_csv_path = os.path.join(
            self.output_dir,
            f"runtime_timeseries_{self.run_id}.csv",
        )
        self._request_trace_jsonl_path = os.path.join(
            self.output_dir,
            f"llm_request_trace_{self.run_id}.jsonl",
        )
        self._request_trace_csv_path = os.path.join(
            self.output_dir,
            f"llm_request_trace_{self.run_id}.csv",
        )
        self._dnn_state_trace_jsonl_path = os.path.join(
            self.output_dir,
            f"dnn_state_trace_{self.run_id}.jsonl",
        )
        self._dnn_state_trace_csv_path = os.path.join(
            self.output_dir,
            f"dnn_state_trace_{self.run_id}.csv",
        )
        self._dnn_request_trace_jsonl_path = os.path.join(
            self.output_dir,
            f"dnn_request_trace_{self.run_id}.jsonl",
        )
        self._dnn_request_trace_csv_path = os.path.join(
            self.output_dir,
            f"dnn_request_trace_{self.run_id}.csv",
        )
        self._dnn_lifecycle_trace_jsonl_path = os.path.join(
            self.output_dir,
            f"dnn_lifecycle_trace_{self.run_id}.jsonl",
        )
        self._dnn_lifecycle_trace_csv_path = os.path.join(
            self.output_dir,
            f"dnn_lifecycle_trace_{self.run_id}.csv",
        )
        self._runtime_timeseries_count = 0
        self._request_trace_count = 0
        self._dnn_state_trace_count = 0
        self._dnn_request_trace_count = 0
        self._dnn_lifecycle_trace_count = 0
        self._event_count = 0
        self._telemetry_runtime_init_time_s = 0.0
        self._telemetry_runtime_event_write_time_s = 0.0
        self._telemetry_runtime_runtime_trace_write_time_s = 0.0
        self._telemetry_runtime_request_trace_write_time_s = 0.0
        self._telemetry_runtime_dnn_trace_write_time_s = 0.0
        self._telemetry_runtime_service_trace_write_time_s = 0.0
        self._telemetry_runtime_flush_time_s = 0.0
        self._telemetry_runtime_flush_count = 0
        self._peak_gpu_used_mb: Optional[float] = None
        self._peak_torch_allocated_mb: Optional[float] = None
        self._peak_torch_reserved_mb: Optional[float] = None

    @property
    def samples(self) -> List[MonitorSample]:
        return self._samples

    @property
    def marks(self) -> List[Dict[str, Any]]:
        return self._marks

    def start(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_stream_files()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._flush_pending()

    def snapshot_now(self) -> MonitorSample:
        """Snapshot now."""
        sample = self._snapshot()
        self._update_peaks(sample)
        return sample

    def mark(self, label: str, extra: Optional[Dict[str, Any]] = None) -> None:
        snap = self._snapshot()
        start_ts = time.perf_counter()
        with profile_span("telemetry", "event_write", label=label) as op_span:
            record = {
                "ts": snap.ts,
                "label": label,
                "gpu_used_mb": snap.gpu_used_mb,
                "torch_allocated_mb": snap.torch_allocated_mb,
                "torch_reserved_mb": snap.torch_reserved_mb,
            }
            if extra:
                record["extra"] = self._sanitize_payload(extra)
            with self._io_lock:
                self._marks.append(record)
                self._event_count += 1
                self._append_jsonl_locked(self._events_jsonl_path, record)
            op_span.add_fields(event_count=int(self._event_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_event_write_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def record_runtime_stats(
        self,
        payload: Dict[str, Any],
        *,
        sample: Optional[MonitorSample] = None,
    ) -> None:
        snap = sample if sample is not None else self._snapshot()
        start_ts = time.perf_counter()
        with profile_span("telemetry", "runtime_trace_write") as op_span:
            record = {
                "ts": snap.ts,
                "gpu_used_mb": snap.gpu_used_mb,
                "torch_allocated_mb": snap.torch_allocated_mb,
                "torch_reserved_mb": snap.torch_reserved_mb,
            }
            record.update(self._sanitize_payload(payload))
            with self._io_lock:
                self._runtime_timeseries_count += 1
                self._append_jsonl_locked(self._runtime_timeseries_jsonl_path, record)
                self._append_csv_row_locked(
                    self._runtime_timeseries_csv_path,
                    self._runtime_timeseries_csv_fields(),
                    record,
                )
            op_span.add_fields(row_count=int(self._runtime_timeseries_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_runtime_trace_write_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def record_request_trace(self, payload: Dict[str, Any]) -> None:
        start_ts = time.perf_counter()
        with profile_span(
            "telemetry",
            "request_trace_write",
            event=str(payload.get("event", "") or ""),
        ) as op_span:
            record = {"ts": time.time()}
            record.update(self._sanitize_payload(payload))
            with self._io_lock:
                self._request_trace_count += 1
                self._append_jsonl_locked(self._request_trace_jsonl_path, record)
                self._append_csv_row_locked(
                    self._request_trace_csv_path,
                    self._request_trace_csv_fields(),
                    record,
                )
            op_span.add_fields(row_count=int(self._request_trace_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_request_trace_write_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def record_dnn_state(self, payload: Dict[str, Any]) -> None:
        start_ts = time.perf_counter()
        with profile_span("telemetry", "dnn_trace_write") as op_span:
            record = {"ts": time.time()}
            record.update(self._sanitize_payload(payload))
            with self._io_lock:
                self._dnn_state_trace_count += 1
                self._append_jsonl_locked(self._dnn_state_trace_jsonl_path, record)
                self._append_csv_row_locked(
                    self._dnn_state_trace_csv_path,
                    self._dnn_state_trace_csv_fields(),
                    record,
                )
            op_span.add_fields(row_count=int(self._dnn_state_trace_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_dnn_trace_write_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def record_dnn_request_trace(self, payload: Dict[str, Any]) -> None:
        """Record dnn request trace."""
        start_ts = time.perf_counter()
        with profile_span(
            "telemetry",
            "dnn_request_trace_write",
            service_name=str(payload.get("service_name", "") or ""),
        ) as op_span:
            record = {"ts": time.time()}
            record.update(self._sanitize_payload(payload))
            with self._io_lock:
                self._dnn_request_trace_count += 1
                self._append_jsonl_locked(self._dnn_request_trace_jsonl_path, record)
                self._append_csv_row_locked(
                    self._dnn_request_trace_csv_path,
                    self._dnn_request_trace_csv_fields(),
                    record,
                )
            op_span.add_fields(row_count=int(self._dnn_request_trace_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_service_trace_write_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def record_dnn_lifecycle_trace(self, payload: Dict[str, Any]) -> None:
        """Record dnn lifecycle trace."""
        start_ts = time.perf_counter()
        with profile_span(
            "telemetry",
            "dnn_lifecycle_trace_write",
            service_name=str(payload.get("service_name", "") or ""),
            to_state=str(payload.get("to_state", "") or ""),
        ) as op_span:
            record = {"ts": time.time()}
            record.update(self._sanitize_payload(payload))
            record.setdefault("transition_ts_wall", record["ts"])
            with self._io_lock:
                self._dnn_lifecycle_trace_count += 1
                self._append_jsonl_locked(self._dnn_lifecycle_trace_jsonl_path, record)
                self._append_csv_row_locked(
                    self._dnn_lifecycle_trace_csv_path,
                    self._dnn_lifecycle_trace_csv_fields(),
                    record,
                )
            op_span.add_fields(row_count=int(self._dnn_lifecycle_trace_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_service_trace_write_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def save(self) -> Dict[str, str]:
        json_path = os.path.join(self.output_dir, f"gpu_mem_{self.run_id}.json")
        csv_path = os.path.join(self.output_dir, f"gpu_mem_{self.run_id}.csv")
        summary_path = os.path.join(
            self.output_dir, f"gpu_mem_{self.run_id}_summary.json"
        )

        self._save_json(json_path)
        self._save_csv(csv_path)
        self._save_summary(summary_path)
        return {
            "json": json_path,
            "csv": csv_path,
            "summary": summary_path,
            "events_jsonl": self._events_jsonl_path,
            "runtime_timeseries_jsonl": self._runtime_timeseries_jsonl_path,
            "runtime_timeseries_csv": self._runtime_timeseries_csv_path,
            "request_trace_jsonl": self._request_trace_jsonl_path,
            "request_trace_csv": self._request_trace_csv_path,
            "dnn_state_trace_jsonl": self._dnn_state_trace_jsonl_path,
            "dnn_state_trace_csv": self._dnn_state_trace_csv_path,
            "dnn_request_trace_jsonl": self._dnn_request_trace_jsonl_path,
            "dnn_request_trace_csv": self._dnn_request_trace_csv_path,
            "dnn_lifecycle_trace_jsonl": self._dnn_lifecycle_trace_jsonl_path,
            "dnn_lifecycle_trace_csv": self._dnn_lifecycle_trace_csv_path,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._snapshot()
            self._samples.append(sample)
            self._pending.append(sample)
            self._update_peaks(sample)
            now = time.time()
            if (now - self._last_flush_ts) >= self.flush_interval_s:
                self._flush_pending()
            time.sleep(self.sample_interval_s)

    def _snapshot(self) -> MonitorSample:
        ts = time.time()
        gpu_used_mb = None
        if self.enable_nvidia_smi:
            if (ts - self._last_smi_ts) >= self.smi_interval_s:
                self._last_smi_value = self._query_nvidia_smi()
                self._last_smi_ts = ts
            gpu_used_mb = self._last_smi_value

        torch_allocated_mb = None
        torch_reserved_mb = None
        try:
            import torch

            if torch.cuda.is_available():
                torch_allocated_mb = (
                    torch.cuda.memory_allocated(self.device_index) / 1024**2
                )
                torch_reserved_mb = (
                    torch.cuda.memory_reserved(self.device_index) / 1024**2
                )
        except Exception:
            pass

        return MonitorSample(
            ts=ts,
            gpu_used_mb=gpu_used_mb,
            torch_allocated_mb=torch_allocated_mb,
            torch_reserved_mb=torch_reserved_mb,
        )

    def _query_nvidia_smi(self) -> Optional[int]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id",
                    str(self.nvidia_smi_index),
                ],
                encoding="utf-8",
            )
            return int(out.strip())
        except Exception:
            return None

    @staticmethod
    def _resolve_nvidia_smi_index(device_index: int) -> int:
        value = os.environ.get("RESIMEM_NVIDIA_SMI_INDEX", "").strip()
        if not value:
            return int(device_index)
        try:
            return int(value)
        except Exception:
            return int(device_index)

    def _save_json(self, path: str) -> None:
        payload = {
            "run_id": self.run_id,
            "sample_interval_s": self.sample_interval_s,
            "smi_interval_s": self.smi_interval_s,
            "device_index": self.device_index,
            "nvidia_smi_index": self.nvidia_smi_index,
            "max_samples": self.max_samples,
            "flush_interval_s": self.flush_interval_s,
            "samples": [
                {
                    "ts": s.ts,
                    "gpu_used_mb": s.gpu_used_mb,
                    "torch_allocated_mb": s.torch_allocated_mb,
                    "torch_reserved_mb": s.torch_reserved_mb,
                }
                for s in list(self._samples)
            ],
            "marks": self._marks,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["ts", "gpu_used_mb", "torch_allocated_mb", "torch_reserved_mb"]
            )
            for s in list(self._samples):
                writer.writerow(
                    [s.ts, s.gpu_used_mb, s.torch_allocated_mb, s.torch_reserved_mb]
                )

    def _save_summary(self, path: str) -> None:
        summary = {
            "run_id": self.run_id,
            "device_index": self.device_index,
            "nvidia_smi_index": self.nvidia_smi_index,
            "peak_gpu_used_mb": self._peak_gpu_used_mb,
            "peak_torch_allocated_mb": self._peak_torch_allocated_mb,
            "peak_torch_reserved_mb": self._peak_torch_reserved_mb,
            "samples": len(self._samples),
            "marks": len(self._marks),
            "events_jsonl": self._events_jsonl_path,
            "runtime_timeseries_jsonl": self._runtime_timeseries_jsonl_path,
            "runtime_timeseries_csv": self._runtime_timeseries_csv_path,
            "runtime_timeseries_count": self._runtime_timeseries_count,
            "request_trace_jsonl": self._request_trace_jsonl_path,
            "request_trace_csv": self._request_trace_csv_path,
            "request_trace_count": self._request_trace_count,
            "dnn_state_trace_jsonl": self._dnn_state_trace_jsonl_path,
            "dnn_state_trace_csv": self._dnn_state_trace_csv_path,
            "dnn_state_trace_count": self._dnn_state_trace_count,
            "dnn_request_trace_jsonl": self._dnn_request_trace_jsonl_path,
            "dnn_request_trace_csv": self._dnn_request_trace_csv_path,
            "dnn_request_trace_count": self._dnn_request_trace_count,
            "dnn_lifecycle_trace_jsonl": self._dnn_lifecycle_trace_jsonl_path,
            "dnn_lifecycle_trace_csv": self._dnn_lifecycle_trace_csv_path,
            "dnn_lifecycle_trace_count": self._dnn_lifecycle_trace_count,
            "runtime_event_count": self._event_count,
            **self.get_telemetry_runtime_stats(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _init_stream_files(self) -> None:
        start_ts = time.perf_counter()
        if not os.path.exists(self._stream_csv_path):
            with open(self._stream_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["ts", "gpu_used_mb", "torch_allocated_mb", "torch_reserved_mb"]
                )
        if not os.path.exists(self._runtime_timeseries_csv_path):
            with open(
                self._runtime_timeseries_csv_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._runtime_timeseries_csv_fields()
                )
                writer.writeheader()
        if not os.path.exists(self._request_trace_csv_path):
            with open(
                self._request_trace_csv_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=self._request_trace_csv_fields())
                writer.writeheader()
        if not os.path.exists(self._dnn_state_trace_csv_path):
            with open(
                self._dnn_state_trace_csv_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._dnn_state_trace_csv_fields()
                )
                writer.writeheader()
        if not os.path.exists(self._dnn_request_trace_csv_path):
            with open(
                self._dnn_request_trace_csv_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._dnn_request_trace_csv_fields()
                )
                writer.writeheader()
        if not os.path.exists(self._dnn_lifecycle_trace_csv_path):
            with open(
                self._dnn_lifecycle_trace_csv_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._dnn_lifecycle_trace_csv_fields()
                )
                writer.writeheader()
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_init_time_s",
            elapsed_s=time.perf_counter() - start_ts,
        )

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        start_ts = time.perf_counter()
        pending_count = len(self._pending)
        with profile_span(
            "telemetry", "flush", pending_count=int(pending_count)
        ) as op_span:
            with open(self._stream_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for s in self._pending:
                    writer.writerow(
                        [s.ts, s.gpu_used_mb, s.torch_allocated_mb, s.torch_reserved_mb]
                    )
            with open(self._stream_jsonl_path, "a", encoding="utf-8") as f:
                for s in self._pending:
                    f.write(
                        json.dumps(
                            {
                                "ts": s.ts,
                                "gpu_used_mb": s.gpu_used_mb,
                                "torch_allocated_mb": s.torch_allocated_mb,
                                "torch_reserved_mb": s.torch_reserved_mb,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            self._pending.clear()
            self._last_flush_ts = time.time()
            op_span.add_fields(flushed_count=int(pending_count))
        self._add_telemetry_runtime_elapsed(
            field_name="_telemetry_runtime_flush_time_s",
            elapsed_s=time.perf_counter() - start_ts,
            flush_count_delta=1,
        )

    def _update_peaks(self, sample: MonitorSample) -> None:
        self._peak_gpu_used_mb = self._max([self._peak_gpu_used_mb, sample.gpu_used_mb])
        self._peak_torch_allocated_mb = self._max(
            [self._peak_torch_allocated_mb, sample.torch_allocated_mb]
        )
        self._peak_torch_reserved_mb = self._max(
            [self._peak_torch_reserved_mb, sample.torch_reserved_mb]
        )

    @staticmethod
    def _max(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        return max(valid) if valid else None

    def get_telemetry_runtime_stats(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            init_time_s = self._telemetry_runtime_init_time_s
            event_write_time_s = self._telemetry_runtime_event_write_time_s
            runtime_trace_write_time_s = (
                self._telemetry_runtime_runtime_trace_write_time_s
            )
            request_trace_write_time_s = (
                self._telemetry_runtime_request_trace_write_time_s
            )
            dnn_trace_write_time_s = self._telemetry_runtime_dnn_trace_write_time_s
            service_trace_write_time_s = (
                self._telemetry_runtime_service_trace_write_time_s
            )
            flush_time_s = self._telemetry_runtime_flush_time_s
            flush_count = self._telemetry_runtime_flush_count
        total_time_s = (
            init_time_s
            + event_write_time_s
            + runtime_trace_write_time_s
            + request_trace_write_time_s
            + dnn_trace_write_time_s
            + service_trace_write_time_s
            + flush_time_s
        )
        return {
            "telemetry_runtime_init_time_s": init_time_s,
            "telemetry_runtime_event_write_time_s": event_write_time_s,
            "telemetry_runtime_runtime_trace_write_time_s": runtime_trace_write_time_s,
            "telemetry_runtime_request_trace_write_time_s": request_trace_write_time_s,
            "telemetry_runtime_dnn_trace_write_time_s": dnn_trace_write_time_s,
            "telemetry_runtime_service_trace_write_time_s": service_trace_write_time_s,
            "telemetry_runtime_flush_time_s": flush_time_s,
            "telemetry_runtime_flush_count": flush_count,
            "telemetry_runtime_event_count": self._event_count,
            "telemetry_runtime_total_write_ops": (
                self._event_count
                + self._runtime_timeseries_count
                + self._request_trace_count
                + self._dnn_state_trace_count
                + self._dnn_request_trace_count
                + self._dnn_lifecycle_trace_count
            ),
            "telemetry_runtime_total_time_s": total_time_s,
        }

    def _add_telemetry_runtime_elapsed(
        self,
        *,
        field_name: str,
        elapsed_s: float,
        flush_count_delta: int = 0,
    ) -> None:
        elapsed_value = max(0.0, float(elapsed_s))
        with self._telemetry_lock:
            setattr(
                self,
                field_name,
                float(getattr(self, field_name, 0.0)) + elapsed_value,
            )
            if flush_count_delta:
                self._telemetry_runtime_flush_count += int(flush_count_delta)

    def _append_jsonl_locked(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_csv_row_locked(
        self,
        path: str,
        fieldnames: List[str],
        payload: Dict[str, Any],
    ) -> None:
        row = {key: self._stringify_csv_value(payload.get(key)) for key in fieldnames}
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    @staticmethod
    def _sanitize_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        result: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, tuple):
                result[key] = list(value)
            elif isinstance(value, dict):
                result[key] = GpuMemoryMonitor._sanitize_payload(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        GpuMemoryMonitor._sanitize_payload(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                result[key] = value
        return result

    @staticmethod
    def _stringify_csv_value(value: Any) -> Any:
        if isinstance(value, list):
            return "|".join(str(v) for v in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value

    @staticmethod
    def _runtime_timeseries_csv_fields() -> List[str]:
        return [
            "ts",
            "group_name",
            "phase_name",
            "phase_bucket",
            "gpu_used_mb",
            "torch_allocated_mb",
            "torch_reserved_mb",
            "gpu_used_delta_mb",
            "torch_allocated_delta_mb",
            "torch_reserved_delta_mb",
            "cuda_free_delta_gb",
            "reserved_burst_detected",
            "reserved_burst_signature",
            "reserved_burst_event_count",
            "reserved_burst_candidate_req_id",
            "reserved_burst_candidate_status",
            "reserved_burst_candidate_source",
            "reserved_burst_candidate_waiting",
            "reserved_burst_candidate_prefilling",
            "reserved_burst_candidate_prompt_tokens",
            "reserved_burst_candidate_prompt_blocks_estimate",
            "reserved_burst_candidate_blocks_needed",
            "reserved_burst_candidate_free_blocks",
            "reserved_burst_candidate_deficit_blocks",
            "llm_raw_token_tps",
            "llm_strict_token_tps",
            "llm_completed_rps",
            "llm_strict_rps",
            "llm_output_tokens_total_post_warmup",
            "llm_output_tokens_goodput_post_warmup",
            "llm_success_post_warmup",
            "llm_sla_success_post_warmup",
            "inflight_reqs",
            "running_reqs",
            "waiting_reqs",
            "finished_reqs",
            "engine_unfinished_reqs",
            "current_submit_cap",
            "effective_submit_cap",
            "admission_submit_blocked",
            "new_submit_attempts_recent",
            "submit_reject_recent",
            "high_submit_cap_dynamic",
            "recent_prompt_tokens_10s",
            "kv_usage_ratio",
            "cuda_free_gb",
            "prefix_cache_hit_rate",
            "loaded_dnn_count",
            "pending_dnn_count",
            "disabled_dnn_count",
            "policy_state",
            "forecast_guard_level",
            "runtime_action_level",
            "pressure_bucket",
        ]

    @staticmethod
    def _request_trace_csv_fields() -> List[str]:
        return [
            "ts",
            "event",
            "group_name",
            "req_id",
            "engine_req_id",
            "internal_req_id",
            "submit_ts",
            "first_token_ts",
            "finish_ts",
            "fail_ts",
            "cancel_ts",
            "latency_s",
            "ttft_s",
            "decode_duration_s",
            "prompt_tokens",
            "request_max_tokens",
            "output_tokens",
            "total_tokens",
            "success",
            "sla_ok",
            "failure_type",
            "phase_name",
            "phase_bucket",
            "request_priority_policy",
            "request_priority",
            "request_priority_reason",
            "arrival_event_id",
            "arrival_offset_s",
            "arrival_phase_name",
            "arrival_phase_bucket",
            "arrival_plan_mode",
            "arrival_plan_path",
            "arrival_decision",
            "matched_request_key",
            "scheduler_running_hint",
            "scheduler_waiting_hint",
            "scheduler_cycle_id",
            "scheduler_snapshot_ts",
            "scheduler_inflight_rank",
            "scheduler_inflight_size",
            "request_age_s",
            "engine_request_status",
            "engine_num_tokens",
            "engine_num_computed_tokens",
            "engine_num_remaining_tokens",
            "engine_num_output_tokens",
            "engine_num_cached_tokens",
            "engine_num_preemptions",
            "engine_in_running_queue",
            "engine_in_waiting_queue",
            "engine_alloc_decision",
            "engine_alloc_num_new_tokens",
            "engine_alloc_num_new_computed_tokens",
            "engine_alloc_num_external_computed_tokens",
            "engine_alloc_num_lookahead_tokens",
            "engine_alloc_num_encoder_tokens",
            "engine_alloc_total_computed_tokens",
            "engine_alloc_num_tokens_need_slot",
            "engine_alloc_num_blocks_to_allocate",
            "engine_alloc_num_free_blocks",
            "engine_alloc_request_block_count",
            "engine_alloc_block_pool_usage",
            "engine_alloc_total_gpu_blocks",
            "engine_alloc_block_size",
            "engine_lookup_hit",
            "outputproc_lookup_hit",
            "engine_waiting_reason",
            "engine_matched_req_key",
            "engine_scheduler_req_id_sample",
            "engine_outputproc_req_id_sample",
            "outputproc_is_prefilling",
            "outputproc_prompt_len",
            "outputproc_num_cached_tokens",
            "outputproc_num_generation_tokens",
            "outputproc_has_first_token",
            "outputproc_has_last_token",
            "outputproc_scheduled_ts",
            "outputproc_first_token_ts",
            "outputproc_last_token_ts",
            "engine_probe_mode",
            "engine_probe_error",
            "outputproc_request_state_count",
            "scheduler_request_state_count",
            "scheduler_running_reqs",
            "scheduler_waiting_reqs",
            "engine_unfinished_reqs",
            "step_error_index",
            "step_error_consecutive",
            "step_error_message",
            "is_drain",
            "counted_in_post_warmup_non_drain",
            "counted_in_post_warmup_sla",
        ]

    @staticmethod
    def _dnn_state_trace_csv_fields() -> List[str]:
        return [
            "ts",
            "group_name",
            "phase_name",
            "phase_bucket",
            "loaded_dnn_count",
            "pending_dnn_count",
            "disabled_dnn_count",
            "gpu_resident_dnn_count",
            "cpu_degraded_dnn_count",
            "restore_pending_dnn_count",
            "blocked_dnn_count",
            "loaded_dnn",
            "pending_dnn",
            "disabled_dnn",
            "gpu_resident_dnn",
            "cpu_degraded_dnn",
            "restore_pending_dnn",
            "blocked_dnn",
            "service_state_counts",
            "dnn_sla_success_total",
            "dnn_success_total",
            "dnn_total",
            "dnn_unavailable_total",
            "dnn_degrade_switch_count",
            "dnn_restore_from_cpu_count",
            "avg_gpu_to_cpu_degrade_switch_cost_ms",
            "avg_cpu_restore_to_gpu_cost_ms",
        ]

    @staticmethod
    def _dnn_request_trace_csv_fields() -> List[str]:
        return [
            "ts",
            "request_id",
            "group_name",
            "service_name",
            "service_class",
            "service_mode",
            "phase_name",
            "phase_bucket",
            "backend",
            "service_state",
            "submit_ts_wall",
            "start_ts_wall",
            "finish_ts_wall",
            "latency_s",
            "success",
            "service_available",
            "slo_met",
            "unavailable",
            "failure_type",
            "failure_message",
            "latency_sla_s",
            "minimum_availability",
            "outage_budget_s_per_window",
            "max_continuous_outage_s",
            "recovery_deadline_s",
            "fallback_used",
            "fallback_mode",
            "fallback_legal",
            "legal_fallback_backends",
            "counted_post_warmup",
        ]

    @staticmethod
    def _dnn_lifecycle_trace_csv_fields() -> List[str]:
        return [
            "ts",
            "event",
            "transition_id",
            "group_name",
            "service_name",
            "service_class",
            "service_mode",
            "phase_name",
            "phase_bucket",
            "from_state",
            "to_state",
            "reason",
            "transition_ts_perf",
            "transition_ts_wall",
            "state_duration_s",
            "service_available_before",
            "service_available_after",
            "normal_slo_eligible_before",
            "normal_slo_eligible_after",
            "recovery_deadline_s",
            "recovery_deadline_ts_perf",
            "fallback_mode",
            "legal_fallback_backends",
            "availability_debt_s",
        ]


_GLOBAL_MONITOR: Optional[GpuMemoryMonitor] = None


def set_global_monitor(monitor: Optional[GpuMemoryMonitor]) -> None:
    global _GLOBAL_MONITOR
    _GLOBAL_MONITOR = monitor


def get_global_monitor() -> Optional[GpuMemoryMonitor]:
    return _GLOBAL_MONITOR


def mark_global(label: str, extra: Optional[Dict[str, Any]] = None) -> None:
    monitor = _GLOBAL_MONITOR
    if monitor is not None:
        monitor.mark(label, extra)
