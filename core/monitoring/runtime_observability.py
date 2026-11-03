"""ResiMem runtime observability module."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _percentile(values: List[float], q: float) -> float:

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * q
    low = int(math.floor(k))
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return float(sorted_values[low])
    w = k - low
    return float(sorted_values[low] * (1.0 - w) + sorted_values[high] * w)


def _init_runtime_observability_summary() -> Dict[str, Any]:

    return {
        "runtime_timeseries_jsonl": "",
        "runtime_timeseries_csv": "",
        "llm_request_trace_jsonl": "",
        "llm_request_trace_csv": "",
        "runtime_events_jsonl": "",
        "dnn_state_trace_jsonl": "",
        "dnn_state_trace_csv": "",
        "runtime_trace_rows": 0,
        "request_trace_rows": 0,
        "dnn_state_trace_rows": 0,
        "dnn_cpu_degraded_peak": 0,
        "dnn_blocked_peak": 0,
        "dnn_restore_pending_peak": 0,
        "dnn_cpu_degraded_seen_rows": 0,
        "dnn_blocked_seen_rows": 0,
        "dnn_latest_cpu_degraded_names": "",
        "dnn_latest_blocked_names": "",
        "dnn_latest_service_state_counts": "",
        "dnn_degrade_switch_count": 0,
        "dnn_restore_from_cpu_count": 0,
        "avg_gpu_to_cpu_degrade_switch_cost_ms": 0.0,
        "avg_cpu_restore_to_gpu_cost_ms": 0.0,
        "llm_raw_token_tps_avg": 0.0,
        "llm_raw_token_tps_p95": 0.0,
        "llm_raw_token_tps_peak": 0.0,
        "llm_completed_rps_avg": 0.0,
        "llm_completed_rps_p95": 0.0,
        "inflight_reqs_avg": 0.0,
        "inflight_reqs_p95": 0.0,
        "inflight_reqs_peak": 0,
        "running_reqs_avg": 0.0,
        "waiting_reqs_p95": 0.0,
        "llm_finish_requests_total": 0,
        "llm_finish_sla_requests_total": 0,
        "llm_output_tokens_total_full": 0,
        "llm_output_tokens_goodput_full": 0,
        "llm_finish_requests_post_warmup_nondrain": 0,
        "llm_finish_sla_requests_post_warmup_nondrain": 0,
        "llm_output_tokens_total_post_warmup_non_drain": 0,
        "llm_output_tokens_goodput_post_warmup_non_drain": 0,
        "llm_finish_raw_req_tps_avg": 0.0,
        "llm_finish_raw_req_tps_p50": 0.0,
        "llm_finish_raw_req_tps_p95": 0.0,
        "llm_finish_raw_decode_tps_avg": 0.0,
        "llm_finish_raw_decode_tps_p50": 0.0,
        "llm_finish_raw_decode_tps_p95": 0.0,
        "request_latest_live_first_token_seen_req_count": 0,
        "request_latest_live_output_tokens_avg": 0.0,
        "request_latest_live_output_tokens_peak": 0,
        "request_latest_live_decode_tps_avg": 0.0,
        "request_latest_live_decode_tps_p50": 0.0,
        "request_latest_live_decode_tps_p95": 0.0,
        "request_latest_live_decode_tps_peak": 0.0,
        "request_pending_count": 0,
        "request_pending_prompt_tokens_total": 0,
        "request_pending_max_prompt_tokens": 0,
        "request_tail_last_finish_age_s": 0.0,
        "request_tail_running_waiting_flip_count": 0,
        "request_tail_active_rows": 0,
        "request_tail_running_rows": 0,
        "request_tail_waiting_rows": 0,
        "request_tail_running_ratio": 0.0,
        "request_tail_waiting_ratio": 0.0,
        "request_tail_latest_scheduler_cycle_id": -1,
        "request_tail_latest_running_req_ids": "",
        "request_tail_latest_waiting_req_ids": "",
        "request_tail_latest_pending_req_ids": "",
        "request_tail_longest_waiting_req_id": "",
        "request_tail_longest_waiting_age_s": 0.0,
        "request_tail_longest_running_req_id": "",
        "request_tail_longest_running_age_s": 0.0,
        "request_tail_most_flipped_req_id": "",
        "request_tail_most_flipped_count": 0,
        "request_tail_latest_waiting_rank": -1,
        "request_tail_latest_max_age_s": 0.0,
        "request_tail_all_waiting_rows": 0,
        "request_tail_all_waiting_max_streak": 0,
        "request_tail_all_waiting_ratio": 0.0,
        "request_tail_zero_tps_rows": 0,
        "request_tail_zero_tps_ratio": 0.0,
        "request_tail_zero_tps_all_waiting_overlap_rows": 0,
        "request_tail_engine_all_waiting_rows": 0,
        "request_tail_engine_all_waiting_max_streak": 0,
        "request_tail_engine_zero_remaining_req_count": 0,
        "request_tail_engine_max_remaining_tokens": 0,
        "request_tail_engine_min_computed_tokens": 0,
        "request_tail_engine_max_computed_tokens": 0,
        "request_tail_engine_latest_statuses": "",
        "request_tail_engine_latest_prefilling_req_ids": "",
        "request_tail_engine_latest_first_token_seen_req_ids": "",
        "request_tail_engine_progress_stalled_req_ids": "",
        "request_tail_engine_progress_stalled_req_count": 0,
        "request_tail_engine_lookup_hit_rows": 0,
        "request_tail_outputproc_lookup_hit_rows": 0,
        "request_tail_engine_waiting_reason_top1": "",
        "request_tail_engine_waiting_reason_counts": "",
        "request_tail_engine_matched_req_keys": "",
        "request_tail_engine_scheduler_req_id_sample": "",
        "request_tail_engine_outputproc_req_id_sample": "",
        "request_tail_engine_alloc_decision_top1": "",
        "request_tail_engine_alloc_decision_counts": "",
        "request_tail_engine_alloc_failed_req_ids": "",
        "request_tail_engine_alloc_failed_req_count": 0,
        "request_tail_engine_alloc_min_free_blocks": 0,
        "request_tail_engine_alloc_max_blocks_to_allocate": 0,
        "request_tail_engine_alloc_block_pool_usage_max": 0.0,
        "request_tail_engine_alloc_latest_snapshots": "",
    }


def _copy_monitor_artifact_paths(
    summary: Dict[str, Any], monitor_paths: Dict[str, str]
) -> None:

    path_keys = [
        "runtime_timeseries_jsonl",
        "runtime_timeseries_csv",
        "request_trace_jsonl",
        "request_trace_csv",
        "events_jsonl",
        "dnn_state_trace_jsonl",
        "dnn_state_trace_csv",
    ]
    alias_map = {
        "request_trace_jsonl": "llm_request_trace_jsonl",
        "request_trace_csv": "llm_request_trace_csv",
        "events_jsonl": "runtime_events_jsonl",
    }
    for key in path_keys:
        if key not in monitor_paths:
            continue
        summary[alias_map.get(key, key)] = monitor_paths.get(key, "") or ""


def _summarize_runtime_trace_csv(summary: Dict[str, Any], runtime_csv: Path) -> None:

    if not runtime_csv.is_file():
        return

    raw_tps_vals: List[float] = []
    completed_rps_vals: List[float] = []
    inflight_vals: List[float] = []
    running_vals: List[float] = []
    waiting_vals: List[float] = []
    with runtime_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary["runtime_trace_rows"] += 1
            raw_tps_vals.append(float(row.get("llm_raw_token_tps", 0) or 0.0))
            completed_rps_vals.append(float(row.get("llm_completed_rps", 0) or 0.0))
            inflight_vals.append(float(row.get("inflight_reqs", 0) or 0.0))
            running_vals.append(float(row.get("running_reqs", 0) or 0.0))
            waiting_vals.append(float(row.get("waiting_reqs", 0) or 0.0))

    if raw_tps_vals:
        summary["llm_raw_token_tps_avg"] = round(float(np.mean(raw_tps_vals)), 6)
        summary["llm_raw_token_tps_p95"] = round(_percentile(raw_tps_vals, 0.95), 6)
        summary["llm_raw_token_tps_peak"] = round(max(raw_tps_vals), 6)
    if completed_rps_vals:
        summary["llm_completed_rps_avg"] = round(float(np.mean(completed_rps_vals)), 6)
        summary["llm_completed_rps_p95"] = round(
            _percentile(completed_rps_vals, 0.95), 6
        )
    if inflight_vals:
        summary["inflight_reqs_avg"] = round(float(np.mean(inflight_vals)), 6)
        summary["inflight_reqs_p95"] = round(_percentile(inflight_vals, 0.95), 6)
        summary["inflight_reqs_peak"] = int(round(max(inflight_vals)))
    if running_vals:
        valid_running = [v for v in running_vals if v >= 0]
        if valid_running:
            summary["running_reqs_avg"] = round(float(np.mean(valid_running)), 6)
    if waiting_vals:
        valid_waiting = [v for v in waiting_vals if v >= 0]
        if valid_waiting:
            summary["waiting_reqs_p95"] = round(_percentile(valid_waiting, 0.95), 6)


def _summarize_request_trace_csv(
    summary: Dict[str, Any],
    request_csv: Path,
    *,
    warmup_deadline_perf: float,
) -> None:

    if not request_csv.is_file():
        return

    rows: List[Dict[str, Any]] = []
    finish_raw_req_tps_vals: List[float] = []
    finish_raw_decode_tps_vals: List[float] = []
    with request_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            summary["request_trace_rows"] += 1
            if str(row.get("event", "")).strip() != "finish":
                continue
            if str(row.get("success", "")).strip().lower() not in {"1", "true", "yes"}:
                continue

            output_tokens = int(float(row.get("output_tokens", 0) or 0))
            sla_ok = str(row.get("sla_ok", "")).strip().lower() in {"1", "true", "yes"}
            summary["llm_finish_requests_total"] += 1
            summary["llm_output_tokens_total_full"] += output_tokens
            if sla_ok:
                summary["llm_finish_sla_requests_total"] += 1
                summary["llm_output_tokens_goodput_full"] += output_tokens

            counted_post_raw = (
                str(row.get("counted_in_post_warmup_non_drain", "")).strip().lower()
            )
            if counted_post_raw:
                counted_post = counted_post_raw in {"1", "true", "yes"}
            else:
                finish_ts = float(row.get("finish_ts", 0) or 0.0)
                is_drain = str(row.get("is_drain", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }
                counted_post = finish_ts >= float(warmup_deadline_perf) and (
                    not is_drain
                )
            if counted_post:
                summary["llm_finish_requests_post_warmup_nondrain"] += 1
                summary[
                    "llm_output_tokens_total_post_warmup_non_drain"
                ] += output_tokens
                latency_s = float(row.get("latency_s", 0) or 0.0)
                decode_duration_s = float(row.get("decode_duration_s", 0) or 0.0)
                if latency_s > 0.0 and output_tokens > 0:
                    finish_raw_req_tps_vals.append(
                        float(output_tokens) / float(latency_s)
                    )
                if decode_duration_s > 0.0 and output_tokens > 0:
                    finish_raw_decode_tps_vals.append(
                        float(output_tokens) / float(decode_duration_s)
                    )

            counted_post_sla_raw = (
                str(row.get("counted_in_post_warmup_sla", "")).strip().lower()
            )
            if counted_post_sla_raw:
                counted_post_sla = counted_post_sla_raw in {"1", "true", "yes"}
            else:
                counted_post_sla = bool(counted_post and sla_ok)
            if counted_post_sla:
                summary["llm_finish_sla_requests_post_warmup_nondrain"] += 1
                summary[
                    "llm_output_tokens_goodput_post_warmup_non_drain"
                ] += output_tokens

    if finish_raw_req_tps_vals:
        summary["llm_finish_raw_req_tps_avg"] = round(
            float(np.mean(finish_raw_req_tps_vals)), 6
        )
        summary["llm_finish_raw_req_tps_p50"] = round(
            _percentile(finish_raw_req_tps_vals, 0.50), 6
        )
        summary["llm_finish_raw_req_tps_p95"] = round(
            _percentile(finish_raw_req_tps_vals, 0.95), 6
        )
    if finish_raw_decode_tps_vals:
        summary["llm_finish_raw_decode_tps_avg"] = round(
            float(np.mean(finish_raw_decode_tps_vals)), 6
        )
        summary["llm_finish_raw_decode_tps_p50"] = round(
            _percentile(finish_raw_decode_tps_vals, 0.50), 6
        )
        summary["llm_finish_raw_decode_tps_p95"] = round(
            _percentile(finish_raw_decode_tps_vals, 0.95), 6
        )

    if not rows:
        return

    submit_rows = [row for row in rows if str(row.get("event", "")).strip() == "submit"]
    finish_rows = [row for row in rows if str(row.get("event", "")).strip() == "finish"]
    finished_req_ids = {
        str(row.get("req_id", "")).strip()
        for row in finish_rows
        if str(row.get("req_id", "")).strip()
    }
    pending_rows = [
        row
        for row in submit_rows
        if str(row.get("req_id", "")).strip() not in finished_req_ids
    ]
    summary["request_pending_count"] = int(len(pending_rows))
    prompt_tokens = [
        int(float(row.get("prompt_tokens", 0) or 0)) for row in pending_rows
    ]
    if prompt_tokens:
        summary["request_pending_prompt_tokens_total"] = int(sum(prompt_tokens))
        summary["request_pending_max_prompt_tokens"] = int(max(prompt_tokens))

    latest_scheduler_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("event", "")).strip() != "scheduler_state":
            continue
        req_id = str(row.get("req_id", "")).strip()
        if not req_id:
            continue
        try:
            cycle_id = int(float(row.get("scheduler_cycle_id", -1) or -1))
        except Exception:
            cycle_id = -1
        try:
            snap_ts = float(row.get("scheduler_snapshot_ts", 0.0) or 0.0)
        except Exception:
            snap_ts = 0.0
        prev = latest_scheduler_rows.get(req_id)
        if prev is None:
            latest_scheduler_rows[req_id] = row
            continue
        try:
            prev_cycle_id = int(float(prev.get("scheduler_cycle_id", -1) or -1))
        except Exception:
            prev_cycle_id = -1
        try:
            prev_snap_ts = float(prev.get("scheduler_snapshot_ts", 0.0) or 0.0)
        except Exception:
            prev_snap_ts = 0.0
        if (cycle_id, snap_ts) >= (prev_cycle_id, prev_snap_ts):
            latest_scheduler_rows[req_id] = row

    latest_live_output_tokens_vals: List[int] = []
    latest_live_decode_tps_vals: List[float] = []
    latest_live_first_token_seen_count = 0
    for row in latest_scheduler_rows.values():
        has_first = str(row.get("outputproc_has_first_token", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not has_first:
            continue
        latest_live_first_token_seen_count += 1
        try:
            output_tokens = int(
                float(row.get("outputproc_num_generation_tokens", 0) or 0)
            )
        except Exception:
            output_tokens = 0
        latest_live_output_tokens_vals.append(max(0, output_tokens))
        try:
            first_token_ts = float(row.get("outputproc_first_token_ts", 0.0) or 0.0)
        except Exception:
            first_token_ts = 0.0
        try:
            last_token_ts = float(row.get("outputproc_last_token_ts", 0.0) or 0.0)
        except Exception:
            last_token_ts = 0.0
        decode_window_s = max(0.0, last_token_ts - first_token_ts)
        if decode_window_s > 0.0 and output_tokens > 0:
            latest_live_decode_tps_vals.append(
                float(output_tokens) / float(decode_window_s)
            )

    summary["request_latest_live_first_token_seen_req_count"] = int(
        latest_live_first_token_seen_count
    )
    if latest_live_output_tokens_vals:
        summary["request_latest_live_output_tokens_avg"] = round(
            float(np.mean(latest_live_output_tokens_vals)), 6
        )
        summary["request_latest_live_output_tokens_peak"] = int(
            max(latest_live_output_tokens_vals)
        )
    if latest_live_decode_tps_vals:
        summary["request_latest_live_decode_tps_avg"] = round(
            float(np.mean(latest_live_decode_tps_vals)), 6
        )
        summary["request_latest_live_decode_tps_p50"] = round(
            _percentile(latest_live_decode_tps_vals, 0.50), 6
        )
        summary["request_latest_live_decode_tps_p95"] = round(
            _percentile(latest_live_decode_tps_vals, 0.95), 6
        )
        summary["request_latest_live_decode_tps_peak"] = round(
            max(latest_live_decode_tps_vals), 6
        )

    submit_ts_vals = [
        float(row.get("submit_ts", 0) or 0.0)
        for row in submit_rows
        if float(row.get("submit_ts", 0) or 0.0) > 0.0
    ]
    if not submit_ts_vals:
        return

    trace_start = min(submit_ts_vals)
    finish_ts_vals = [
        float(row.get("finish_ts", 0) or 0.0)
        for row in finish_rows
        if float(row.get("finish_ts", 0) or 0.0) > 0.0
    ]
    if finish_ts_vals:
        tail_anchor_ts = max(finish_ts_vals)
        summary["request_tail_last_finish_age_s"] = round(
            max(0.0, tail_anchor_ts - trace_start), 6
        )
    else:

        tail_anchor_ts = max(submit_ts_vals)
        summary["request_tail_last_finish_age_s"] = 0.0

    tail_rows = []
    for row in rows:
        snap_ts = float(row.get("scheduler_snapshot_ts", 0) or 0.0)
        if snap_ts > 0.0 and snap_ts >= tail_anchor_ts:
            tail_rows.append(row)
    if not tail_rows:
        return

    summary["request_tail_active_rows"] = int(len(tail_rows))
    running_rows = 0
    waiting_rows = 0
    latest_cycle_id = -1
    latest_running = []
    latest_waiting = []
    latest_pending = []
    latest_per_req: Dict[str, tuple[int, str]] = {}
    cycle_rows: Dict[int, List[Dict[str, Any]]] = {}
    per_req_waiting_age: Dict[str, float] = {}
    per_req_running_age: Dict[str, float] = {}
    per_req_flip_count: Dict[str, int] = {}
    per_req_computed_vals: Dict[str, List[int]] = {}
    per_req_prefilling_latest: Dict[str, bool] = {}
    per_req_first_token_seen_latest: Dict[str, bool] = {}
    waiting_reason_counts: Dict[str, int] = {}
    alloc_decision_counts: Dict[str, int] = {}
    alloc_min_free_blocks: Optional[int] = None
    alloc_max_blocks_to_allocate = 0
    alloc_block_pool_usage_max = 0.0
    engine_lookup_hit_rows = 0
    outputproc_lookup_hit_rows = 0
    latest_scheduler_req_id_sample = ""
    latest_outputproc_req_id_sample = ""
    for row in tail_rows:
        req_id = str(row.get("req_id", "")).strip()
        if not req_id:
            continue
        running_hint = str(row.get("scheduler_running_hint", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        waiting_hint = str(row.get("scheduler_waiting_hint", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if running_hint:
            running_rows += 1
        if waiting_hint:
            waiting_rows += 1
        cycle_id = int(float(row.get("scheduler_cycle_id", -1) or -1))
        latest_per_req[req_id] = (
            cycle_id,
            "running" if running_hint else ("waiting" if waiting_hint else "unknown"),
        )
        age_s = float(row.get("request_age_s", 0) or 0.0)
        if waiting_hint:
            per_req_waiting_age[req_id] = max(
                per_req_waiting_age.get(req_id, 0.0), age_s
            )
        if running_hint:
            per_req_running_age[req_id] = max(
                per_req_running_age.get(req_id, 0.0), age_s
            )
        try:
            per_req_computed_vals.setdefault(req_id, []).append(
                int(float(row.get("engine_num_computed_tokens", 0) or 0))
            )
        except Exception:
            per_req_computed_vals.setdefault(req_id, []).append(0)
        per_req_prefilling_latest[req_id] = str(
            row.get("outputproc_is_prefilling", "")
        ).strip().lower() in {"1", "true", "yes"}
        per_req_first_token_seen_latest[req_id] = str(
            row.get("outputproc_has_first_token", "")
        ).strip().lower() in {"1", "true", "yes"}
        if str(row.get("engine_lookup_hit", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            engine_lookup_hit_rows += 1
        if str(row.get("outputproc_lookup_hit", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            outputproc_lookup_hit_rows += 1
        waiting_reason = str(row.get("engine_waiting_reason", "")).strip()
        if waiting_reason:
            waiting_reason_counts[waiting_reason] = (
                int(waiting_reason_counts.get(waiting_reason, 0)) + 1
            )
        alloc_decision = str(row.get("engine_alloc_decision", "")).strip()
        if alloc_decision:
            alloc_decision_counts[alloc_decision] = (
                int(alloc_decision_counts.get(alloc_decision, 0)) + 1
            )
            try:
                free_blocks = int(
                    float(row.get("engine_alloc_num_free_blocks", 0) or 0)
                )
                alloc_min_free_blocks = (
                    free_blocks
                    if alloc_min_free_blocks is None
                    else min(alloc_min_free_blocks, free_blocks)
                )
            except Exception:
                pass
            try:
                alloc_max_blocks_to_allocate = max(
                    alloc_max_blocks_to_allocate,
                    int(float(row.get("engine_alloc_num_blocks_to_allocate", 0) or 0)),
                )
            except Exception:
                pass
            try:
                alloc_block_pool_usage_max = max(
                    alloc_block_pool_usage_max,
                    float(row.get("engine_alloc_block_pool_usage", 0.0) or 0.0),
                )
            except Exception:
                pass
        latest_scheduler_req_id_sample = str(
            row.get("engine_scheduler_req_id_sample", "")
            or latest_scheduler_req_id_sample
        )
        latest_outputproc_req_id_sample = str(
            row.get("engine_outputproc_req_id_sample", "")
            or latest_outputproc_req_id_sample
        )
        if cycle_id >= 0:
            cycle_rows.setdefault(cycle_id, []).append(row)
            latest_cycle_id = max(latest_cycle_id, cycle_id)

    flip_count = 0
    for req_id in sorted(
        {
            str(row.get("req_id", "")).strip()
            for row in tail_rows
            if str(row.get("req_id", "")).strip()
        }
    ):
        req_states: List[str] = []
        for cycle_id in sorted(cycle_rows.keys()):
            state = None
            for row in cycle_rows[cycle_id]:
                if str(row.get("req_id", "")).strip() != req_id:
                    continue
                running_hint = str(
                    row.get("scheduler_running_hint", "")
                ).strip().lower() in {"1", "true", "yes"}
                waiting_hint = str(
                    row.get("scheduler_waiting_hint", "")
                ).strip().lower() in {"1", "true", "yes"}
                state = (
                    "running"
                    if running_hint
                    else ("waiting" if waiting_hint else "unknown")
                )
                break
            if state is None:
                continue
            if not req_states or req_states[-1] != state:
                req_states.append(state)
        local_flips = 0
        for idx in range(1, len(req_states)):
            if req_states[idx] != req_states[idx - 1]:
                flip_count += 1
                local_flips += 1
        per_req_flip_count[req_id] = int(local_flips)
    summary["request_tail_running_waiting_flip_count"] = int(flip_count)
    if tail_rows:
        summary["request_tail_running_rows"] = int(running_rows)
        summary["request_tail_waiting_rows"] = int(waiting_rows)
        summary["request_tail_running_ratio"] = round(
            float(running_rows) / float(len(tail_rows)), 6
        )
        summary["request_tail_waiting_ratio"] = round(
            float(waiting_rows) / float(len(tail_rows)), 6
        )
    summary["request_tail_latest_scheduler_cycle_id"] = int(latest_cycle_id)
    all_waiting_rows = 0
    all_waiting_max_streak = 0
    zero_tps_rows = 0
    zero_tps_all_waiting_overlap = 0
    cur_all_waiting_streak = 0
    for cycle_id in sorted(cycle_rows.keys()):
        cycle = cycle_rows[cycle_id]
        running_count = 0
        waiting_count = 0
        zero_tps_like = True
        for row in cycle:
            running_hint = str(
                row.get("scheduler_running_hint", "")
            ).strip().lower() in {"1", "true", "yes"}
            waiting_hint = str(
                row.get("scheduler_waiting_hint", "")
            ).strip().lower() in {"1", "true", "yes"}
            running_count += int(running_hint)
            waiting_count += int(waiting_hint)

            if running_hint:
                zero_tps_like = False
        if running_count == 0 and waiting_count > 0:
            all_waiting_rows += 1
            cur_all_waiting_streak += 1
            zero_tps_rows += 1
            zero_tps_all_waiting_overlap += 1
        else:
            all_waiting_max_streak = max(all_waiting_max_streak, cur_all_waiting_streak)
            cur_all_waiting_streak = 0
    all_waiting_max_streak = max(all_waiting_max_streak, cur_all_waiting_streak)
    if tail_rows:
        summary["request_tail_all_waiting_rows"] = int(all_waiting_rows)
        summary["request_tail_all_waiting_max_streak"] = int(all_waiting_max_streak)
        summary["request_tail_all_waiting_ratio"] = (
            round(float(all_waiting_rows) / float(max(len(cycle_rows), 1)), 6)
            if cycle_rows
            else 0.0
        )
        summary["request_tail_zero_tps_rows"] = int(zero_tps_rows)
        summary["request_tail_zero_tps_ratio"] = (
            round(float(zero_tps_rows) / float(max(len(cycle_rows), 1)), 6)
            if cycle_rows
            else 0.0
        )
        summary["request_tail_zero_tps_all_waiting_overlap_rows"] = int(
            zero_tps_all_waiting_overlap
        )

    engine_all_waiting_rows = 0
    engine_all_waiting_max_streak = 0
    cur_engine_all_waiting_streak = 0
    for cycle_id in sorted(cycle_rows.keys()):
        cycle = cycle_rows[cycle_id]
        engine_statuses = []
        remaining_tokens = []
        computed_tokens = []
        for row in cycle:
            engine_statuses.append(str(row.get("engine_request_status", "")).strip())
            try:
                remaining_tokens.append(
                    int(float(row.get("engine_num_remaining_tokens", 0) or 0))
                )
            except Exception:
                pass
            try:
                computed_tokens.append(
                    int(float(row.get("engine_num_computed_tokens", 0) or 0))
                )
            except Exception:
                pass
        if engine_statuses and all(status == "WAITING" for status in engine_statuses):
            engine_all_waiting_rows += 1
            cur_engine_all_waiting_streak += 1
        else:
            engine_all_waiting_max_streak = max(
                engine_all_waiting_max_streak, cur_engine_all_waiting_streak
            )
            cur_engine_all_waiting_streak = 0
    engine_all_waiting_max_streak = max(
        engine_all_waiting_max_streak, cur_engine_all_waiting_streak
    )
    if cycle_rows:
        summary["request_tail_engine_all_waiting_rows"] = int(engine_all_waiting_rows)
        summary["request_tail_engine_all_waiting_max_streak"] = int(
            engine_all_waiting_max_streak
        )

    if latest_cycle_id >= 0:
        latest_rows = cycle_rows.get(latest_cycle_id, [])
        latest_running = sorted(
            {
                str(row.get("req_id", "")).strip()
                for row in latest_rows
                if str(row.get("scheduler_running_hint", "")).strip().lower()
                in {"1", "true", "yes"}
            }
        )
        latest_waiting = sorted(
            {
                str(row.get("req_id", "")).strip()
                for row in latest_rows
                if str(row.get("scheduler_waiting_hint", "")).strip().lower()
                in {"1", "true", "yes"}
            }
        )
        latest_pending = sorted(
            {str(row.get("req_id", "")).strip() for row in latest_rows}
        )
    summary["request_tail_latest_running_req_ids"] = "|".join(latest_running)
    summary["request_tail_latest_waiting_req_ids"] = "|".join(latest_waiting)
    summary["request_tail_latest_pending_req_ids"] = "|".join(latest_pending)
    if per_req_waiting_age:
        rid, age = max(per_req_waiting_age.items(), key=lambda kv: kv[1])
        summary["request_tail_longest_waiting_req_id"] = str(rid)
        summary["request_tail_longest_waiting_age_s"] = round(float(age), 6)
    if per_req_running_age:
        rid, age = max(per_req_running_age.items(), key=lambda kv: kv[1])
        summary["request_tail_longest_running_req_id"] = str(rid)
        summary["request_tail_longest_running_age_s"] = round(float(age), 6)
    if per_req_flip_count:
        rid, cnt = max(per_req_flip_count.items(), key=lambda kv: kv[1])
        summary["request_tail_most_flipped_req_id"] = str(rid)
        summary["request_tail_most_flipped_count"] = int(cnt)
    if latest_cycle_id >= 0:
        waiting_ranks = []
        max_age = 0.0
        for row in cycle_rows.get(latest_cycle_id, []):
            max_age = max(max_age, float(row.get("request_age_s", 0) or 0.0))
            if str(row.get("scheduler_waiting_hint", "")).strip().lower() in {
                "1",
                "true",
                "yes",
            }:
                try:
                    waiting_ranks.append(
                        int(float(row.get("scheduler_inflight_rank", -1) or -1))
                    )
                except Exception:
                    pass
        engine_status_items = []
        matched_req_keys = []
        zero_remaining_count = 0
        remaining_max = 0
        computed_min = None
        computed_max = None
        for row in cycle_rows.get(latest_cycle_id, []):
            req_id = str(row.get("req_id", "")).strip()
            status_name = str(row.get("engine_request_status", "")).strip()
            matched_req_key = str(row.get("engine_matched_req_key", "")).strip()
            if req_id:
                engine_status_items.append(f"{req_id}:{status_name}")
            if matched_req_key:
                matched_req_keys.append(f"{req_id}->{matched_req_key}")
            try:
                remaining = int(float(row.get("engine_num_remaining_tokens", 0) or 0))
            except Exception:
                remaining = 0
            try:
                computed = int(float(row.get("engine_num_computed_tokens", 0) or 0))
            except Exception:
                computed = 0
            if remaining <= 0:
                zero_remaining_count += 1
            remaining_max = max(remaining_max, remaining)
            computed_min = (
                computed if computed_min is None else min(computed_min, computed)
            )
            computed_max = (
                computed if computed_max is None else max(computed_max, computed)
            )
        if waiting_ranks:
            summary["request_tail_latest_waiting_rank"] = int(min(waiting_ranks))
        summary["request_tail_latest_max_age_s"] = round(float(max_age), 6)
        summary["request_tail_engine_zero_remaining_req_count"] = int(
            zero_remaining_count
        )
        summary["request_tail_engine_max_remaining_tokens"] = int(remaining_max)
        summary["request_tail_engine_min_computed_tokens"] = int(computed_min or 0)
        summary["request_tail_engine_max_computed_tokens"] = int(computed_max or 0)
        summary["request_tail_engine_latest_statuses"] = "|".join(engine_status_items)
        latest_prefilling = sorted(
            req_id
            for req_id, is_prefilling in per_req_prefilling_latest.items()
            if is_prefilling
        )
        latest_first_token_seen = sorted(
            req_id for req_id, seen in per_req_first_token_seen_latest.items() if seen
        )
        progress_stalled_req_ids = []
        for req_id, vals in sorted(per_req_computed_vals.items(), key=lambda kv: kv[0]):
            if not vals:
                continue
            if max(vals) == min(vals):
                progress_stalled_req_ids.append(str(req_id))
        summary["request_tail_engine_latest_prefilling_req_ids"] = "|".join(
            latest_prefilling
        )
        summary["request_tail_engine_latest_first_token_seen_req_ids"] = "|".join(
            latest_first_token_seen
        )
        summary["request_tail_engine_progress_stalled_req_ids"] = "|".join(
            progress_stalled_req_ids
        )
        summary["request_tail_engine_progress_stalled_req_count"] = int(
            len(progress_stalled_req_ids)
        )
        summary["request_tail_engine_lookup_hit_rows"] = int(engine_lookup_hit_rows)
        summary["request_tail_outputproc_lookup_hit_rows"] = int(
            outputproc_lookup_hit_rows
        )
        if waiting_reason_counts:
            top_reason = max(waiting_reason_counts.items(), key=lambda kv: kv[1])[0]
            reason_pairs = [
                f"{k}:{waiting_reason_counts[k]}"
                for k in sorted(waiting_reason_counts.keys())
            ]
            summary["request_tail_engine_waiting_reason_top1"] = str(top_reason)
            summary["request_tail_engine_waiting_reason_counts"] = "|".join(
                reason_pairs
            )
        alloc_failed_req_ids = []
        alloc_latest_snapshots = []
        for row in cycle_rows.get(latest_cycle_id, []):
            req_id = str(row.get("req_id", "")).strip()
            alloc_decision = str(row.get("engine_alloc_decision", "")).strip()
            try:
                need_blocks = int(
                    float(row.get("engine_alloc_num_blocks_to_allocate", 0) or 0)
                )
            except Exception:
                need_blocks = 0
            try:
                free_blocks = int(
                    float(row.get("engine_alloc_num_free_blocks", 0) or 0)
                )
            except Exception:
                free_blocks = 0
            try:
                held_blocks = int(
                    float(row.get("engine_alloc_request_block_count", 0) or 0)
                )
            except Exception:
                held_blocks = 0
            if alloc_decision and alloc_decision != "allocated" and req_id:
                alloc_failed_req_ids.append(req_id)
            if req_id and alloc_decision:
                alloc_latest_snapshots.append(
                    f"{req_id}:{alloc_decision},need={need_blocks},free={free_blocks},held={held_blocks}"
                )
        if alloc_decision_counts:
            top_alloc = max(alloc_decision_counts.items(), key=lambda kv: kv[1])[0]
            alloc_pairs = [
                f"{k}:{alloc_decision_counts[k]}"
                for k in sorted(alloc_decision_counts.keys())
            ]
            summary["request_tail_engine_alloc_decision_top1"] = str(top_alloc)
            summary["request_tail_engine_alloc_decision_counts"] = "|".join(alloc_pairs)
        summary["request_tail_engine_alloc_failed_req_ids"] = "|".join(
            sorted(set(alloc_failed_req_ids))
        )
        summary["request_tail_engine_alloc_failed_req_count"] = int(
            len(set(alloc_failed_req_ids))
        )
        summary["request_tail_engine_alloc_min_free_blocks"] = int(
            alloc_min_free_blocks or 0
        )
        summary["request_tail_engine_alloc_max_blocks_to_allocate"] = int(
            alloc_max_blocks_to_allocate
        )
        summary["request_tail_engine_alloc_block_pool_usage_max"] = round(
            float(alloc_block_pool_usage_max), 6
        )
        summary["request_tail_engine_alloc_latest_snapshots"] = "|".join(
            alloc_latest_snapshots
        )
        summary["request_tail_engine_matched_req_keys"] = "|".join(
            sorted(matched_req_keys)
        )
        summary["request_tail_engine_scheduler_req_id_sample"] = str(
            latest_scheduler_req_id_sample
        )
        summary["request_tail_engine_outputproc_req_id_sample"] = str(
            latest_outputproc_req_id_sample
        )


def _summarize_dnn_trace_csv(summary: Dict[str, Any], dnn_csv: Path) -> None:

    if not dnn_csv.is_file():
        return

    with dnn_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary["dnn_state_trace_rows"] += 1
            cpu_degraded_count = int(row.get("cpu_degraded_dnn_count", 0) or 0)
            blocked_count = int(row.get("blocked_dnn_count", 0) or 0)
            restore_pending_count = int(row.get("restore_pending_dnn_count", 0) or 0)
            summary["dnn_cpu_degraded_peak"] = max(
                int(summary.get("dnn_cpu_degraded_peak", 0)),
                cpu_degraded_count,
            )
            summary["dnn_blocked_peak"] = max(
                int(summary.get("dnn_blocked_peak", 0)),
                blocked_count,
            )
            summary["dnn_restore_pending_peak"] = max(
                int(summary.get("dnn_restore_pending_peak", 0)),
                restore_pending_count,
            )
            if cpu_degraded_count > 0:
                summary["dnn_cpu_degraded_seen_rows"] += 1
                summary["dnn_latest_cpu_degraded_names"] = str(
                    row.get("cpu_degraded_dnn", "") or ""
                )
            if blocked_count > 0:
                summary["dnn_blocked_seen_rows"] += 1
                summary["dnn_latest_blocked_names"] = str(
                    row.get("blocked_dnn", "") or ""
                )
            service_state_counts = str(row.get("service_state_counts", "") or "")
            if service_state_counts:
                summary["dnn_latest_service_state_counts"] = service_state_counts
            summary["dnn_degrade_switch_count"] = max(
                int(summary.get("dnn_degrade_switch_count", 0)),
                int(row.get("dnn_degrade_switch_count", 0) or 0),
            )
            summary["dnn_restore_from_cpu_count"] = max(
                int(summary.get("dnn_restore_from_cpu_count", 0)),
                int(row.get("dnn_restore_from_cpu_count", 0) or 0),
            )
            summary["avg_gpu_to_cpu_degrade_switch_cost_ms"] = float(
                row.get("avg_gpu_to_cpu_degrade_switch_cost_ms", 0.0) or 0.0
            )
            summary["avg_cpu_restore_to_gpu_cost_ms"] = float(
                row.get("avg_cpu_restore_to_gpu_cost_ms", 0.0) or 0.0
            )


def summarize_runtime_observability(
    monitor_paths: Dict[str, str],
    *,
    warmup_deadline_perf: float = 0.0,
) -> Dict[str, Any]:
    """Summarize runtime observability."""
    summary = _init_runtime_observability_summary()
    _copy_monitor_artifact_paths(summary, monitor_paths)
    _summarize_runtime_trace_csv(
        summary,
        Path(monitor_paths.get("runtime_timeseries_csv", "") or ""),
    )
    _summarize_request_trace_csv(
        summary,
        Path(monitor_paths.get("request_trace_csv", "") or ""),
        warmup_deadline_perf=float(warmup_deadline_perf),
    )
    _summarize_dnn_trace_csv(
        summary,
        Path(monitor_paths.get("dnn_state_trace_csv", "") or ""),
    )
    return summary
