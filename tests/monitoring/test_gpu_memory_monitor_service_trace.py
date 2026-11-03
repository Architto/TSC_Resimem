from __future__ import annotations

import csv
import json
from pathlib import Path

from core.monitoring.gpu_memory_monitor import GpuMemoryMonitor


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_service_trace_jsonl_csv_and_summary_are_conserved(tmp_path: Path) -> None:
    monitor = GpuMemoryMonitor(
        sample_interval_s=60.0,
        smi_interval_s=60.0,
        output_dir=str(tmp_path),
        run_id="service_trace_test",
        enable_nvidia_smi=False,
    )
    monitor.start()
    monitor.record_dnn_request_trace(
        {
            "request_id": "dnn-1",
            "service_name": "vision",
            "finish_ts_wall": 123.0,
            "latency_s": 0.1,
            "success": True,
            "service_available": True,
            "slo_met": True,
            "fallback_used": False,
            "fallback_legal": True,
        }
    )
    monitor.record_dnn_lifecycle_trace(
        {
            "event": "transition",
            "transition_id": 1,
            "service_name": "vision",
            "from_state": "restore_pending",
            "to_state": "gpu_resident",
            "transition_ts_wall": 122.0,
            "legal_fallback_backends": ["cpu"],
        }
    )
    monitor.stop()
    paths = monitor.save()

    request_jsonl = Path(paths["dnn_request_trace_jsonl"])
    lifecycle_jsonl = Path(paths["dnn_lifecycle_trace_jsonl"])
    assert len(_read_jsonl(request_jsonl)) == 1
    assert len(_read_jsonl(lifecycle_jsonl)) == 1

    with Path(paths["dnn_request_trace_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with Path(paths["dnn_lifecycle_trace_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 1

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["dnn_request_trace_count"] == 1
    assert summary["dnn_lifecycle_trace_count"] == 1
    assert summary["telemetry_runtime_total_write_ops"] >= 2
    assert summary["telemetry_runtime_service_trace_write_time_s"] > 0.0
