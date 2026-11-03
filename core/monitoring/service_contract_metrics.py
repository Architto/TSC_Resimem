"""Parse service contracts and aggregate outage, recovery, and joint-SLO metrics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SERVICE_STATE_GPU_RESIDENT = "gpu_resident"
SERVICE_STATE_CPU_DEGRADED = "cpu_degraded"
SERVICE_STATE_RESTORE_PENDING = "restore_pending"
SERVICE_STATE_BLOCKED = "blocked"

FALLBACK_NONE = "none"
FALLBACK_SURVIVAL_ONLY = "survival_only"
FALLBACK_SLO_ELIGIBLE = "slo_eligible"
_VALID_FALLBACK_MODES = {
    FALLBACK_NONE,
    FALLBACK_SURVIVAL_ONLY,
    FALLBACK_SLO_ELIGIBLE,
}

TTFT_SLO_MODE_REPORT_ONLY = "report_only"
TTFT_SLO_MODE_ABSOLUTE = "absolute"
_VALID_TTFT_SLO_MODES = {
    TTFT_SLO_MODE_REPORT_ONLY,
    TTFT_SLO_MODE_ABSOLUTE,
}

DEFAULT_SERVICE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "service_contracts.json"
)

HARD_FAILURE_MARKERS: Tuple[str, ...] = (
    "oom",
    "fatal_async_step",
    "fatal async-step",
    "device assert",
    "device_assert",
    "illegal memory",
    "illegal_memory",
    "runner_fatal",
    "runner fatal",
    "illegal_fallback",
)


@dataclass(frozen=True)
class ServiceContract:
    """Machine-readable contract for one DNN service."""

    service_name: str
    service_class: str
    service_mode: str
    latency_sla_s: float
    minimum_availability: float
    outage_budget_window_s: float
    outage_budget_s_per_window: float
    max_continuous_outage_s: float
    recovery_deadline_s: float
    fallback_mode: str
    legal_fallback_backends: Tuple[str, ...]
    service_priority_weight: float

    @property
    def critical(self) -> bool:
        return self.service_class == "critical"

    def fallback_legal(self, backend: str) -> bool:
        normalized = str(backend or "").strip().lower()
        return (
            self.fallback_mode != FALLBACK_NONE
            and normalized in self.legal_fallback_backends
        )

    def state_available(self, state: str) -> bool:
        normalized = str(state or "").strip().lower()
        if normalized == SERVICE_STATE_GPU_RESIDENT:
            return True
        if normalized == SERVICE_STATE_CPU_DEGRADED:
            return self.fallback_legal("cpu")
        return False

    def state_normal_slo_eligible(self, state: str) -> bool:
        normalized = str(state or "").strip().lower()
        if normalized == SERVICE_STATE_GPU_RESIDENT:
            return True
        return (
            normalized == SERVICE_STATE_CPU_DEGRADED
            and self.fallback_mode == FALLBACK_SLO_ELIGIBLE
            and self.fallback_legal("cpu")
        )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(1.0, max(0.0, float(quantile)))
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def compute_tpot_s(
    *,
    output_tokens: int,
    decode_duration_s: float = 0.0,
    first_token_ts: float = 0.0,
    finish_ts: float = 0.0,
) -> float:
    """Compute per-request TPOT, excluding the first-token interval."""
    token_count = max(0, int(output_tokens))
    if token_count <= 1:
        return 0.0
    duration_s = max(0.0, float(decode_duration_s))
    if duration_s <= 0.0 and finish_ts > first_token_ts > 0.0:
        duration_s = float(finish_ts - first_token_ts)
    return duration_s / float(token_count - 1)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[2] / resolved
    return resolved.resolve()


@lru_cache(maxsize=16)
def load_service_contract_payload(path: str | Path) -> Dict[str, Any]:
    """Load and validate a service-contract file."""
    resolved = _resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version not in {1, 2}:
        raise ValueError(
            f"unsupported service-contract schema version: {payload.get('schema_version')}"
        )
    if not isinstance(payload.get("services"), dict):
        raise ValueError("service contract is missing a services mapping")
    llm_contract = payload.get("llm")
    if not isinstance(llm_contract, dict):
        raise ValueError("service contract is missing an llm mapping")
    if schema_version >= 2:
        ttft_slo_mode = str(llm_contract.get("ttft_slo_mode", "") or "").strip().lower()
        if ttft_slo_mode not in _VALID_TTFT_SLO_MODES:
            raise ValueError(f"invalid ttft_slo_mode: {ttft_slo_mode or '<empty>'}")
        if ttft_slo_mode == TTFT_SLO_MODE_REPORT_ONLY and "ttft_p95_s" in llm_contract:
            raise ValueError("report_only mode must not define an absolute ttft_p95_s")
        if (
            ttft_slo_mode == TTFT_SLO_MODE_ABSOLUTE
            and _to_float(llm_contract.get("ttft_p95_s"), 0.0) <= 0.0
        ):
            raise ValueError("absolute mode requires a positive ttft_p95_s")
    return payload


def resolve_service_contract(
    service_name: str,
    *,
    contract_path: str | Path,
    role_spec: Optional[Mapping[str, Any]] = None,
    default_latency_sla_s: float = 1.0,
) -> ServiceContract:
    """Merge contract defaults, service overrides, and a DNN role specification."""
    payload = load_service_contract_payload(contract_path)
    service_overrides = dict(payload.get("services", {}).get(service_name, {}) or {})
    role = dict(role_spec or {})
    service_class = (
        str(
            service_overrides.get(
                "class",
                (
                    "critical"
                    if str(role.get("service_mode", "realtime")) == "realtime"
                    else "elastic"
                ),
            )
            or "critical"
        )
        .strip()
        .lower()
    )
    if service_class not in {"critical", "elastic"}:
        raise ValueError(
            f"resolve service contract: error service class: {service_name}={service_class}"
        )
    defaults = dict(payload.get("defaults", {}).get(service_class, {}) or {})
    merged = dict(defaults)
    merged.update(service_overrides)
    if "service_mode" in role:
        merged["service_mode"] = role.get("service_mode")
    if "latency_sla_s" in role:
        merged["latency_sla_s"] = role.get("latency_sla_s")
    fallback_mode = (
        str(merged.get("fallback_mode", FALLBACK_NONE) or FALLBACK_NONE).strip().lower()
    )
    if fallback_mode not in _VALID_FALLBACK_MODES:
        raise ValueError(
            f"resolve service contract: error fallback_mode: {service_name}={fallback_mode}"
        )
    window_s = _to_float(
        payload.get("windowing", {}).get("outage_budget_window_s", 600.0),
        600.0,
    )
    return ServiceContract(
        service_name=str(service_name),
        service_class=service_class,
        service_mode=str(
            merged.get("service_mode", role.get("service_mode", "realtime"))
            or "realtime"
        ),
        latency_sla_s=max(
            0.0,
            _to_float(
                merged.get("latency_sla_s", default_latency_sla_s),
                default_latency_sla_s,
            ),
        ),
        minimum_availability=min(
            1.0,
            max(0.0, _to_float(merged.get("minimum_availability", 0.0), 0.0)),
        ),
        outage_budget_window_s=max(1e-6, window_s),
        outage_budget_s_per_window=max(
            0.0,
            _to_float(merged.get("outage_budget_s_per_window", 0.0), 0.0),
        ),
        max_continuous_outage_s=max(
            0.0,
            _to_float(merged.get("max_continuous_outage_s", 0.0), 0.0),
        ),
        recovery_deadline_s=max(
            0.0,
            _to_float(merged.get("recovery_deadline_s", 0.0), 0.0),
        ),
        fallback_mode=fallback_mode,
        legal_fallback_backends=tuple(
            str(item).strip().lower()
            for item in merged.get("legal_fallback_backends", [])
            if str(item).strip()
        ),
        service_priority_weight=max(
            0.0,
            _to_float(merged.get("service_priority_weight", 0.0), 0.0),
        ),
    )


def load_service_contracts(
    contract_path: str | Path,
    *,
    service_names: Optional[Iterable[str]] = None,
) -> Dict[str, ServiceContract]:
    payload = load_service_contract_payload(contract_path)
    names = (
        list(service_names)
        if service_names is not None
        else list(payload.get("services", {}))
    )
    return {
        str(name): resolve_service_contract(str(name), contract_path=contract_path)
        for name in names
    }


def contract_to_dict(contract: ServiceContract) -> Dict[str, Any]:
    payload = asdict(contract)
    payload["critical"] = contract.critical
    payload["legal_fallback_backends"] = list(contract.legal_fallback_backends)
    return payload


def _read_trace(path: str | Path | None) -> List[Dict[str, Any]]:
    if path is None or not str(path).strip():
        return []
    resolved = Path(path)
    if not resolved.is_file():
        return []
    if resolved.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(dict(json.loads(text)))
        return rows
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_ts(row: Mapping[str, Any], *preferred: str) -> float:
    for key in (*preferred, "ts"):
        value = _to_float(row.get(key), 0.0)
        if value > 0.0:
            return value
    return 0.0


def _clip_interval(
    start_ts: float,
    end_ts: float,
    eval_start_ts: float,
    eval_end_ts: float,
) -> Optional[Tuple[float, float]]:
    start = max(float(start_ts), float(eval_start_ts))
    end = min(float(end_ts), float(eval_end_ts))
    if end <= start:
        return None
    return start, end


def _merge_intervals(
    intervals: Iterable[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    ordered = sorted(
        (float(start), float(end)) for start, end in intervals if end > start
    )
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_overlap_s(
    intervals: Sequence[Tuple[float, float]],
    start_ts: float,
    end_ts: float,
) -> float:
    total = 0.0
    for left, right in intervals:
        clipped = _clip_interval(left, right, start_ts, end_ts)
        if clipped is not None:
            total += clipped[1] - clipped[0]
    return total


def _service_state_intervals(
    contract: ServiceContract,
    lifecycle_rows: Sequence[Mapping[str, Any]],
    *,
    eval_start_ts: float,
    eval_end_ts: float,
) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]], List[Dict[str, Any]], str
]:
    transitions = sorted(
        (
            _row_ts(row, "transition_ts_wall"),
            str(row.get("to_state", "") or "").strip(),
            dict(row),
        )
        for row in lifecycle_rows
        if str(row.get("service_name", row.get("dnn_name", "")) or "")
        == contract.service_name
        and str(row.get("to_state", "") or "").strip()
        and _row_ts(row, "transition_ts_wall") > 0.0
    )
    state = SERVICE_STATE_RESTORE_PENDING
    for ts, to_state, _ in transitions:
        if ts <= eval_start_ts:
            state = to_state
        else:
            break

    cursor = float(eval_start_ts)
    outage_intervals: List[Tuple[float, float]] = []
    non_slo_intervals: List[Tuple[float, float]] = []
    recovery_episodes: List[Dict[str, Any]] = []
    recovery_start: Optional[float] = None
    if not contract.state_normal_slo_eligible(state):
        recovery_start = cursor

    for ts, to_state, row in transitions:
        if ts <= eval_start_ts:
            continue
        if ts >= eval_end_ts:
            break
        if not contract.state_available(state):
            outage_intervals.append((cursor, ts))
        if not contract.state_normal_slo_eligible(state):
            non_slo_intervals.append((cursor, ts))
        previous_normal = contract.state_normal_slo_eligible(state)
        next_normal = contract.state_normal_slo_eligible(to_state)
        if previous_normal and not next_normal:
            recovery_start = ts
        elif not previous_normal and next_normal and recovery_start is not None:
            duration_s = max(0.0, ts - recovery_start)
            recovery_episodes.append(
                {
                    "start_ts": recovery_start,
                    "end_ts": ts,
                    "duration_s": duration_s,
                    "open": False,
                    "deadline_s": contract.recovery_deadline_s,
                    "deadline_violation": duration_s
                    > contract.recovery_deadline_s + 1e-9,
                    "recovery_debt_s": max(
                        0.0, duration_s - contract.recovery_deadline_s
                    ),
                    "restored_by_reason": str(row.get("reason", "") or ""),
                }
            )
            recovery_start = None
        state = to_state
        cursor = ts

    if cursor < eval_end_ts:
        if not contract.state_available(state):
            outage_intervals.append((cursor, eval_end_ts))
        if not contract.state_normal_slo_eligible(state):
            non_slo_intervals.append((cursor, eval_end_ts))
    if recovery_start is not None:
        duration_s = max(0.0, eval_end_ts - recovery_start)
        recovery_episodes.append(
            {
                "start_ts": recovery_start,
                "end_ts": eval_end_ts,
                "duration_s": duration_s,
                "open": True,
                "deadline_s": contract.recovery_deadline_s,
                "deadline_violation": duration_s > contract.recovery_deadline_s + 1e-9,
                "recovery_debt_s": max(0.0, duration_s - contract.recovery_deadline_s),
                "restored_by_reason": "",
            }
        )
    return (
        _merge_intervals(outage_intervals),
        _merge_intervals(non_slo_intervals),
        recovery_episodes,
        state,
    )


def _budget_windows(
    intervals: Sequence[Tuple[float, float]],
    *,
    eval_start_ts: float,
    eval_end_ts: float,
    window_s: float,
    budget_s: float,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    cursor = float(eval_start_ts)
    index = 0
    while cursor < eval_end_ts - 1e-9:
        end = min(eval_end_ts, cursor + window_s)
        outage_s = _interval_overlap_s(intervals, cursor, end)
        windows.append(
            {
                "window_index": index,
                "start_ts": cursor,
                "end_ts": end,
                "outage_s": outage_s,
                "budget_s": budget_s,
                "pass": outage_s <= budget_s + 1e-9,
            }
        )
        index += 1
        cursor = end
    return windows


def _request_terminal_ts(row: Mapping[str, Any]) -> float:
    return _row_ts(row, "finish_ts_wall", "fail_ts_wall", "submit_ts_wall")


def _is_hard_failure_text(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return bool(normalized) and any(
        marker in normalized for marker in HARD_FAILURE_MARKERS
    )


def _summarize_llm(
    rows: Sequence[Mapping[str, Any]],
    *,
    eval_start_ts: float,
    eval_end_ts: float,
    llm_contract: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    default_ttft_mode = (
        TTFT_SLO_MODE_ABSOLUTE
        if "ttft_p95_s" in llm_contract
        else TTFT_SLO_MODE_REPORT_ONLY
    )
    ttft_slo_mode = (
        str(llm_contract.get("ttft_slo_mode", default_ttft_mode) or default_ttft_mode)
        .strip()
        .lower()
    )
    if ttft_slo_mode not in _VALID_TTFT_SLO_MODES:
        raise ValueError(f"invalid ttft_slo_mode: {ttft_slo_mode}")
    ttft_limit: Optional[float] = None
    if ttft_slo_mode == TTFT_SLO_MODE_ABSOLUTE:
        ttft_limit = _to_float(llm_contract.get("ttft_p95_s"), 0.0)
        if ttft_limit <= 0.0:
            raise ValueError("absolute mode requires a positive ttft_p95_s")
    tpot_limit = max(0.0, _to_float(llm_contract.get("tpot_p95_s"), 0.05))
    target_attainment = min(
        1.0,
        max(0.0, _to_float(llm_contract.get("request_slo_attainment"), 0.95)),
    )
    terminals: List[Dict[str, Any]] = []
    hard_failures: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        event = str(row.get("event", "") or "").strip().lower()
        if event not in {"finish", "fail", "cancel", "forced_abort"}:
            continue
        ts = _row_ts(row)
        if not (eval_start_ts <= ts <= eval_end_ts):
            continue
        output_tokens = int(_to_float(row.get("output_tokens"), 0.0))
        ttft_s = max(0.0, _to_float(row.get("ttft_s"), 0.0))
        tpot_s = compute_tpot_s(
            output_tokens=output_tokens,
            decode_duration_s=_to_float(row.get("decode_duration_s"), 0.0),
            first_token_ts=_to_float(row.get("first_token_ts"), 0.0),
            finish_ts=_to_float(row.get("finish_ts"), 0.0),
        )
        success = event == "finish" and _to_bool(row.get("success", True))
        ttft_met = ttft_limit is None or ttft_s <= ttft_limit + 1e-9
        slo_met = success and ttft_met and tpot_s <= tpot_limit + 1e-9
        terminal = {
            "ts": ts,
            "req_id": str(row.get("req_id", "") or ""),
            "event": event,
            "success": success,
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "slo_met": slo_met,
            "failure_type": str(row.get("failure_type", "") or ""),
        }
        terminals.append(terminal)
        if _is_hard_failure_text(terminal["failure_type"]):
            hard_failures.append(
                {
                    "ts": ts,
                    "source": "llm_request",
                    "service_name": "llm",
                    "failure_type": terminal["failure_type"],
                    "req_id": terminal["req_id"],
                }
            )
    finished = [item for item in terminals if item["success"]]
    ttft_values = [float(item["ttft_s"]) for item in finished]
    tpot_values = [float(item["tpot_s"]) for item in finished]
    slo_count = sum(1 for item in terminals if item["slo_met"])
    attainment = slo_count / max(len(terminals), 1)
    ttft_p95_s = _percentile(ttft_values, 0.95)
    ttft_contract_pass: Optional[bool] = None
    if ttft_limit is not None:
        ttft_contract_pass = bool(finished) and ttft_p95_s <= ttft_limit + 1e-9
    tpot_p95_s = _percentile(tpot_values, 0.95)
    summary = {
        "terminal_requests": len(terminals),
        "successful_requests": len(finished),
        "request_slo_met": slo_count,
        "request_slo_attainment": attainment,
        "request_slo_attainment_target": target_attainment,
        "ttft_slo_mode": ttft_slo_mode,
        "ttft_contract_required": ttft_limit is not None,
        "ttft_contract_pass": ttft_contract_pass,
        "ttft_p95_s": ttft_p95_s,
        "ttft_p95_target_s": ttft_limit,
        "tpot_p95_s": tpot_p95_s,
        "tpot_p95_target_s": tpot_limit,
        "contract_pass": (
            bool(terminals)
            and attainment + 1e-9 >= target_attainment
            and (ttft_contract_pass is not False)
            and tpot_p95_s <= tpot_limit + 1e-9
        ),
    }
    return summary, terminals, hard_failures


def summarize_service_records(
    *,
    dnn_request_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    llm_request_rows: Sequence[Mapping[str, Any]],
    contracts: Mapping[str, ServiceContract],
    llm_contract: Mapping[str, Any],
    eval_start_ts: float,
    eval_end_ts: float,
    joint_window_s: float = 10.0,
    external_hard_failures: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate normalized trace records against service contracts."""
    if eval_end_ts <= eval_start_ts:
        raise ValueError("eval_end_ts must be greater than eval_start_ts")
    duration_s = eval_end_ts - eval_start_ts
    service_summaries: Dict[str, Dict[str, Any]] = {}
    service_outages: Dict[str, List[Tuple[float, float]]] = {}
    request_rows_by_service: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in contracts
    }
    hard_failures: List[Dict[str, Any]] = [
        dict(item) for item in (external_hard_failures or [])
    ]

    for raw in dnn_request_rows:
        row = dict(raw)
        service_name = str(row.get("service_name", row.get("dnn_name", "")) or "")
        if service_name not in contracts:
            continue
        ts = _request_terminal_ts(row)
        if eval_start_ts <= ts <= eval_end_ts:
            row["_terminal_ts"] = ts
            request_rows_by_service[service_name].append(row)

    for service_name, contract in contracts.items():
        outage_intervals, non_slo_intervals, recovery_episodes, final_state = (
            _service_state_intervals(
                contract,
                lifecycle_rows,
                eval_start_ts=eval_start_ts,
                eval_end_ts=eval_end_ts,
            )
        )
        service_outages[service_name] = outage_intervals
        requests = request_rows_by_service[service_name]
        total_requests = len(requests)
        available_requests = sum(
            1 for row in requests if _to_bool(row.get("service_available"))
        )
        successful_requests = sum(1 for row in requests if _to_bool(row.get("success")))
        slo_requests = sum(1 for row in requests if _to_bool(row.get("slo_met")))
        fallback_requests = sum(
            1 for row in requests if _to_bool(row.get("fallback_used"))
        )
        illegal_fallback_requests = sum(
            1
            for row in requests
            if _to_bool(row.get("fallback_used"))
            and not contract.fallback_legal(str(row.get("backend", "") or ""))
        )
        survival_only_requests = sum(
            1
            for row in requests
            if contract.fallback_mode == FALLBACK_SURVIVAL_ONLY
            and _to_bool(row.get("fallback_used"))
        )
        latencies = [
            max(0.0, _to_float(row.get("latency_s"), 0.0))
            for row in requests
            if _to_bool(row.get("success"))
        ]
        latency_p95_s = _percentile(latencies, 0.95)
        latency_contract_required = contract.latency_sla_s > 0.0
        latency_contract_pass = not latency_contract_required or (
            bool(latencies) and latency_p95_s <= contract.latency_sla_s + 1e-9
        )
        outage_total_s = sum(end - start for start, end in outage_intervals)
        non_slo_total_s = sum(end - start for start, end in non_slo_intervals)
        availability = max(0.0, 1.0 - outage_total_s / duration_s)
        normal_slo_availability = max(0.0, 1.0 - non_slo_total_s / duration_s)
        budget_windows = _budget_windows(
            outage_intervals,
            eval_start_ts=eval_start_ts,
            eval_end_ts=eval_end_ts,
            window_s=contract.outage_budget_window_s,
            budget_s=contract.outage_budget_s_per_window,
        )
        max_continuous_outage_s = max(
            (end - start for start, end in outage_intervals),
            default=0.0,
        )
        recovery_deadline_violations = sum(
            1 for episode in recovery_episodes if bool(episode["deadline_violation"])
        )
        recovery_debt_s = sum(
            float(episode["recovery_debt_s"]) for episode in recovery_episodes
        )
        contract_pass = (
            availability + 1e-9 >= contract.minimum_availability
            and all(bool(window["pass"]) for window in budget_windows)
            and max_continuous_outage_s <= contract.max_continuous_outage_s + 1e-9
            and recovery_deadline_violations == 0
            and illegal_fallback_requests == 0
            and latency_contract_pass
        )
        if illegal_fallback_requests:
            hard_failures.append(
                {
                    "ts": min(
                        float(row.get("_terminal_ts", eval_end_ts))
                        for row in requests
                        if _to_bool(row.get("fallback_used"))
                        and not contract.fallback_legal(
                            str(row.get("backend", "") or "")
                        )
                    ),
                    "source": "dnn_request",
                    "service_name": service_name,
                    "failure_type": "illegal_fallback",
                }
            )
        for row in requests:
            if _is_hard_failure_text(row.get("failure_type")):
                hard_failures.append(
                    {
                        "ts": float(row.get("_terminal_ts", eval_end_ts)),
                        "source": "dnn_request",
                        "service_name": service_name,
                        "failure_type": str(row.get("failure_type", "") or ""),
                        "req_id": str(row.get("request_id", "") or ""),
                    }
                )
        if not contract.state_available(final_state):
            hard_failures.append(
                {
                    "ts": eval_end_ts,
                    "source": "lifecycle",
                    "service_name": service_name,
                    "failure_type": "no_legal_service_path_at_end",
                }
            )
        service_summaries[service_name] = {
            "contract": contract_to_dict(contract),
            "request_total": total_requests,
            "request_success": successful_requests,
            "request_available": available_requests,
            "request_availability": available_requests / max(total_requests, 1),
            "request_slo_met": slo_requests,
            "request_slo_attainment": slo_requests / max(total_requests, 1),
            "latency_p95_s": latency_p95_s,
            "latency_contract_required": latency_contract_required,
            "latency_contract_pass": latency_contract_pass,
            "fallback_requests": fallback_requests,
            "survival_only_requests": survival_only_requests,
            "illegal_fallback_requests": illegal_fallback_requests,
            "time_availability": availability,
            "normal_slo_time_availability": normal_slo_availability,
            "outage_total_s": outage_total_s,
            "outage_intervals": [
                {"start_ts": start, "end_ts": end, "duration_s": end - start}
                for start, end in outage_intervals
            ],
            "max_continuous_outage_s": max_continuous_outage_s,
            "outage_budget_windows": budget_windows,
            "outage_budget_violation_count": sum(
                1 for window in budget_windows if not bool(window["pass"])
            ),
            "recovery_episodes": recovery_episodes,
            "recovery_deadline_violation_count": recovery_deadline_violations,
            "recovery_debt_s": recovery_debt_s,
            "final_state": final_state,
            "contract_pass": contract_pass,
        }

    llm_summary, llm_terminals, llm_hard_failures = _summarize_llm(
        llm_request_rows,
        eval_start_ts=eval_start_ts,
        eval_end_ts=eval_end_ts,
        llm_contract=llm_contract,
    )
    hard_failures.extend(llm_hard_failures)
    hard_failures.sort(key=lambda item: _to_float(item.get("ts"), eval_end_ts))

    window_s = max(1e-6, float(joint_window_s))
    joint_windows: List[Dict[str, Any]] = []
    cursor = float(eval_start_ts)
    index = 0
    critical_names = [name for name, contract in contracts.items() if contract.critical]
    while cursor < eval_end_ts - 1e-9:
        end = min(eval_end_ts, cursor + window_s)
        llm_in_window = [
            item for item in llm_terminals if cursor <= float(item["ts"]) < end
        ]
        hard_in_window = [
            item
            for item in hard_failures
            if cursor <= _to_float(item.get("ts"), end) < end
        ]
        critical_details: Dict[str, Any] = {}
        critical_pass = True
        for service_name in critical_names:
            request_in_window = [
                row
                for row in request_rows_by_service[service_name]
                if cursor <= float(row.get("_terminal_ts", 0.0)) < end
            ]
            outage_overlap_s = _interval_overlap_s(
                service_outages[service_name],
                cursor,
                end,
            )
            service_pass = outage_overlap_s <= 1e-9 and all(
                _to_bool(row.get("slo_met")) for row in request_in_window
            )
            critical_pass = critical_pass and service_pass
            critical_details[service_name] = {
                "request_count": len(request_in_window),
                "outage_overlap_s": outage_overlap_s,
                "pass": service_pass,
            }
        llm_pass = all(bool(item["slo_met"]) for item in llm_in_window)
        window_pass = llm_pass and critical_pass and not hard_in_window
        joint_windows.append(
            {
                "window_index": index,
                "start_ts": cursor,
                "end_ts": end,
                "llm_terminal_count": len(llm_in_window),
                "llm_pass": llm_pass,
                "critical_services": critical_details,
                "hard_failure_count": len(hard_in_window),
                "pass": window_pass,
            }
        )
        index += 1
        cursor = end

    joint_pass_count = sum(1 for window in joint_windows if bool(window["pass"]))
    return {
        "spec": "resimem.service_summary.v1",
        "evaluation": {
            "start_ts": eval_start_ts,
            "end_ts": eval_end_ts,
            "duration_s": duration_s,
            "joint_slo_window_s": window_s,
        },
        "llm": llm_summary,
        "services": service_summaries,
        "joint_slo": {
            "window_count": len(joint_windows),
            "pass_count": joint_pass_count,
            "attainment": joint_pass_count / max(len(joint_windows), 1),
            "windows": joint_windows,
        },
        "hard_failures": hard_failures,
        "hard_failure_count": len(hard_failures),
        "all_service_contracts_pass": all(
            bool(item["contract_pass"]) for item in service_summaries.values()
        ),
        "overall_contract_pass": (
            bool(llm_summary["contract_pass"])
            and all(bool(item["contract_pass"]) for item in service_summaries.values())
            and not hard_failures
        ),
    }


def summarize_service_contracts(
    *,
    dnn_request_trace_path: str | Path,
    lifecycle_trace_path: str | Path,
    llm_request_trace_path: str | Path,
    contract_path: str | Path,
    eval_start_ts: float,
    eval_end_ts: float,
    output_path: str | Path | None = None,
    external_hard_failures: Optional[Sequence[Mapping[str, Any]]] = None,
    service_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Aggregate persisted traces and optionally write the service summary as JSON."""
    payload = load_service_contract_payload(contract_path)
    contracts = load_service_contracts(contract_path, service_names=service_names)
    summary = summarize_service_records(
        dnn_request_rows=_read_trace(dnn_request_trace_path),
        lifecycle_rows=_read_trace(lifecycle_trace_path),
        llm_request_rows=_read_trace(llm_request_trace_path),
        contracts=contracts,
        llm_contract=dict(payload.get("llm", {}) or {}),
        eval_start_ts=float(eval_start_ts),
        eval_end_ts=float(eval_end_ts),
        joint_window_s=_to_float(
            payload.get("windowing", {}).get("joint_slo_window_s", 10.0),
            10.0,
        ),
        external_hard_failures=external_hard_failures,
    )
    summary["contract_id"] = str(payload.get("contract_id", "") or "")
    summary["contract_path"] = str(_resolve_path(contract_path))
    if output_path is not None and str(output_path).strip():
        resolved_output = Path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


__all__ = [
    "DEFAULT_SERVICE_CONTRACT_PATH",
    "FALLBACK_NONE",
    "FALLBACK_SLO_ELIGIBLE",
    "FALLBACK_SURVIVAL_ONLY",
    "TTFT_SLO_MODE_ABSOLUTE",
    "TTFT_SLO_MODE_REPORT_ONLY",
    "ServiceContract",
    "compute_tpot_s",
    "contract_to_dict",
    "load_service_contract_payload",
    "load_service_contracts",
    "resolve_service_contract",
    "summarize_service_records",
    "summarize_service_contracts",
]
