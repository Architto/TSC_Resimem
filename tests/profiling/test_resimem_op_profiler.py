from __future__ import annotations

import json

import pytest

from core.profiling import resimem_op_profiler as profiler


def test_profiler_default_off_does_not_write(tmp_path, monkeypatch):
    out_path = tmp_path / "spans.jsonl"
    monkeypatch.delenv("RESIMEM_PROFILE_OPS", raising=False)
    monkeypatch.setenv("RESIMEM_PROFILE_JSONL", str(out_path))
    profiler._reset_for_tests()

    with profiler.span("kv_shrink", "total", target_blocks=10):
        pass

    assert not out_path.exists()
    assert profiler.is_enabled() is False


def test_profiler_enabled_writes_jsonl(tmp_path, monkeypatch):
    out_path = tmp_path / "spans.jsonl"
    monkeypatch.setenv("RESIMEM_PROFILE_OPS", "1")
    monkeypatch.setenv("RESIMEM_PROFILE_JSONL", str(out_path))
    profiler._reset_for_tests()
    assert profiler.configure_from_env(run_id="run-a", repeat_id="0", force=True)

    with profiler.span("kv_shrink", "total", target_blocks=10) as op_span:
        op_span.add_fields(effective_blocks=8)

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "resimem.operation_span.v1"
    assert record["run_id"] == "run-a"
    assert record["repeat_id"] == "0"
    assert record["operation"] == "kv_shrink"
    assert record["step"] == "total"
    assert record["status"] == "ok"
    assert record["attrs"]["target_blocks"] == 10
    assert record["attrs"]["effective_blocks"] == 8
    assert record["wall_async_ms"] >= 0.0


def test_profiler_records_failed_span_and_reraises(tmp_path, monkeypatch):
    out_path = tmp_path / "spans.jsonl"
    monkeypatch.setenv("RESIMEM_PROFILE_OPS", "1")
    monkeypatch.setenv("RESIMEM_PROFILE_JSONL", str(out_path))
    profiler._reset_for_tests()
    profiler.configure_from_env(run_id="run-b", repeat_id="1", force=True)

    with pytest.raises(ValueError):
        with profiler.span("dnn_backfill", "load_weights", dnn_name="speech"):
            raise ValueError("boom")

    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "failed"
    assert record["error_type"] == "ValueError"
    assert record["attrs"]["dnn_name"] == "speech"


def test_profiler_nested_spans_link_parent(tmp_path, monkeypatch):
    out_path = tmp_path / "nested.jsonl"
    monkeypatch.setenv("RESIMEM_PROFILE_OPS", "1")
    monkeypatch.setenv("RESIMEM_PROFILE_JSONL", str(out_path))
    profiler._reset_for_tests()
    profiler.configure_from_env(run_id="run-c", repeat_id="2", force=True)

    with profiler.span("dnn_backfill", "load_with_guard"):
        with profiler.span("dnn_backfill", "load_weights"):
            pass

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    child, parent = records
    assert child["step"] == "load_weights"
    assert parent["step"] == "load_with_guard"
    assert child["parent_span_id"] == parent["span_id"]
