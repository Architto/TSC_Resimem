from __future__ import annotations

import pytest

from core.monitoring.service_contract_metrics import (
    DEFAULT_SERVICE_CONTRACT_PATH,
    compute_tpot_s,
    load_service_contract_payload,
    load_service_contracts,
    summarize_service_records,
)


CONTRACT_PATH = DEFAULT_SERVICE_CONTRACT_PATH


def _lifecycle(service: str, ts: float, from_state: str, to_state: str) -> dict:
    return {
        "service_name": service,
        "transition_ts_wall": ts,
        "from_state": from_state,
        "to_state": to_state,
        "reason": "test",
    }


def _dnn_request(
    service: str,
    ts: float,
    *,
    success: bool = True,
    available: bool = True,
    slo_met: bool = True,
    backend: str = "gpu",
    fallback_mode: str = "none",
    fallback_legal: bool = True,
) -> dict:
    return {
        "request_id": f"{service}-{ts}",
        "service_name": service,
        "finish_ts_wall": ts,
        "latency_s": 0.1,
        "success": success,
        "service_available": available,
        "slo_met": slo_met,
        "backend": backend,
        "fallback_used": backend == "cpu",
        "fallback_mode": fallback_mode,
        "fallback_legal": fallback_legal,
        "failure_type": "",
    }


def _llm_finish(ts: float, *, ttft_s: float = 0.2, decode_s: float = 0.4) -> dict:
    return {
        "event": "finish",
        "ts": ts,
        "req_id": f"llm-{ts}",
        "success": True,
        "ttft_s": ttft_s,
        "decode_duration_s": decode_s,
        "output_tokens": 11,
        "failure_type": "",
    }


def test_contract_file_and_tpot_are_machine_readable() -> None:
    payload = load_service_contract_payload(CONTRACT_PATH)
    contracts = load_service_contracts(CONTRACT_PATH)

    assert payload["schema_version"] == 2
    assert payload["contract_id"] == "resimem_service_contract_v2"
    assert payload["llm"]["ttft_slo_mode"] == "report_only"
    assert "ttft_p95_s" not in payload["llm"]
    assert contracts["vision"].minimum_availability == pytest.approx(0.99)
    assert not contracts["speech"].fallback_legal("cpu")
    assert not contracts["retrieval"].fallback_legal("cpu")
    assert compute_tpot_s(output_tokens=11, decode_duration_s=0.5) == pytest.approx(0.05)


def test_ttft_is_report_only_in_v2_contract() -> None:
    payload = load_service_contract_payload(CONTRACT_PATH)
    summary = summarize_service_records(
        dnn_request_rows=[],
        lifecycle_rows=[],
        llm_request_rows=[_llm_finish(110.0, ttft_s=30.0)],
        contracts={},
        llm_contract=payload["llm"],
        eval_start_ts=100.0,
        eval_end_ts=120.0,
    )

    llm = summary["llm"]
    assert llm["ttft_slo_mode"] == "report_only"
    assert llm["ttft_contract_required"] is False
    assert llm["ttft_contract_pass"] is None
    assert llm["ttft_p95_s"] == pytest.approx(30.0)
    assert llm["ttft_p95_target_s"] is None
    assert llm["request_slo_attainment"] == pytest.approx(1.0)
    assert llm["contract_pass"] is True


def test_legacy_explicit_ttft_threshold_remains_replayable() -> None:
    summary = summarize_service_records(
        dnn_request_rows=[],
        lifecycle_rows=[],
        llm_request_rows=[_llm_finish(110.0, ttft_s=2.0)],
        contracts={},
        llm_contract={
            "ttft_p95_s": 1.0,
            "tpot_p95_s": 0.05,
            "request_slo_attainment": 0.95,
        },
        eval_start_ts=100.0,
        eval_end_ts=120.0,
    )

    llm = summary["llm"]
    assert llm["ttft_slo_mode"] == "absolute"
    assert llm["ttft_contract_required"] is True
    assert llm["ttft_contract_pass"] is False
    assert llm["request_slo_attainment"] == pytest.approx(0.0)
    assert llm["contract_pass"] is False


def test_outage_merge_budget_recovery_and_joint_window() -> None:
    payload = load_service_contract_payload(CONTRACT_PATH)
    contracts = load_service_contracts(CONTRACT_PATH, service_names=["vision"])
    lifecycle = [
        _lifecycle("vision", 90.0, "restore_pending", "gpu_resident"),
        _lifecycle("vision", 120.0, "gpu_resident", "restore_pending"),
        _lifecycle("vision", 120.4, "restore_pending", "gpu_resident"),
    ]
    summary = summarize_service_records(
        dnn_request_rows=[
            _dnn_request("vision", 110.0),
            _dnn_request("vision", 120.2, success=False, available=False, slo_met=False),
            _dnn_request("vision", 126.0),
        ],
        lifecycle_rows=lifecycle,
        llm_request_rows=[_llm_finish(111.0), _llm_finish(126.0)],
        contracts=contracts,
        llm_contract=payload["llm"],
        eval_start_ts=100.0,
        eval_end_ts=140.0,
        joint_window_s=10.0,
    )

    vision = summary["services"]["vision"]
    assert vision["outage_total_s"] == pytest.approx(0.4)
    assert vision["max_continuous_outage_s"] == pytest.approx(0.4)
    assert vision["recovery_deadline_violation_count"] == 0
    assert vision["contract_pass"]
    assert summary["joint_slo"]["window_count"] == 4
    assert summary["joint_slo"]["pass_count"] == 3


def test_default_contract_rejects_cpu_fallback() -> None:
    payload = load_service_contract_payload(CONTRACT_PATH)
    contracts = load_service_contracts(CONTRACT_PATH, service_names=["speech"])
    lifecycle = [
        _lifecycle("speech", 90.0, "restore_pending", "gpu_resident"),
        _lifecycle("speech", 110.0, "gpu_resident", "cpu_degraded"),
        _lifecycle("speech", 125.0, "cpu_degraded", "gpu_resident"),
    ]
    summary = summarize_service_records(
        dnn_request_rows=[
            _dnn_request(
                "speech",
                115.0,
                backend="cpu",
                fallback_mode="survival_only",
                fallback_legal=True,
                slo_met=False,
            )
        ],
        lifecycle_rows=lifecycle,
        llm_request_rows=[_llm_finish(115.0)],
        contracts=contracts,
        llm_contract=payload["llm"],
        eval_start_ts=100.0,
        eval_end_ts=130.0,
    )

    speech = summary["services"]["speech"]
    assert speech["outage_total_s"] == pytest.approx(15.0)
    assert speech["normal_slo_time_availability"] == pytest.approx(0.5)
    assert speech["survival_only_requests"] == 0
    assert speech["illegal_fallback_requests"] == 1
    assert speech["recovery_deadline_violation_count"] == 0
    assert not speech["contract_pass"]
    assert "illegal_fallback" in {
        item["failure_type"] for item in summary["hard_failures"]
    }


def test_illegal_fallback_and_open_recovery_are_hard_failures() -> None:
    payload = load_service_contract_payload(CONTRACT_PATH)
    contracts = load_service_contracts(CONTRACT_PATH, service_names=["retrieval"])
    lifecycle = [
        _lifecycle("retrieval", 90.0, "restore_pending", "gpu_resident"),
        _lifecycle("retrieval", 105.0, "gpu_resident", "cpu_degraded"),
    ]
    summary = summarize_service_records(
        dnn_request_rows=[
            _dnn_request(
                "retrieval",
                110.0,
                backend="cpu",
                fallback_mode="none",
                fallback_legal=False,
                slo_met=False,
            )
        ],
        lifecycle_rows=lifecycle,
        llm_request_rows=[_llm_finish(110.0)],
        contracts=contracts,
        llm_contract=payload["llm"],
        eval_start_ts=100.0,
        eval_end_ts=140.0,
    )

    retrieval = summary["services"]["retrieval"]
    assert retrieval["illegal_fallback_requests"] == 1
    assert retrieval["recovery_deadline_violation_count"] == 1
    assert retrieval["recovery_debt_s"] == pytest.approx(5.0)
    failure_types = {item["failure_type"] for item in summary["hard_failures"]}
    assert "illegal_fallback" in failure_types
    assert "no_legal_service_path_at_end" in failure_types


def test_critical_latency_p95_is_part_of_contract_verdict() -> None:
    payload = load_service_contract_payload(CONTRACT_PATH)
    contracts = load_service_contracts(CONTRACT_PATH, service_names=["vision"])
    lifecycle = [
        _lifecycle("vision", 90.0, "restore_pending", "gpu_resident"),
    ]
    slow_request = _dnn_request("vision", 110.0, slo_met=False)
    slow_request["latency_s"] = 1.2

    summary = summarize_service_records(
        dnn_request_rows=[slow_request],
        lifecycle_rows=lifecycle,
        llm_request_rows=[_llm_finish(110.0)],
        contracts=contracts,
        llm_contract=payload["llm"],
        eval_start_ts=100.0,
        eval_end_ts=120.0,
    )

    vision = summary["services"]["vision"]
    assert vision["time_availability"] == pytest.approx(1.0)
    assert vision["latency_p95_s"] == pytest.approx(1.2)
    assert vision["latency_contract_required"] is True
    assert vision["latency_contract_pass"] is False
    assert vision["contract_pass"] is False
