"""ResiMem local quantum module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


LOCAL_QUANTUM_GB = 0.05


@dataclass(frozen=True)
class KnownManifestEntry:
    family_key: str
    surface_key: str
    family_signature: Dict[str, Any]
    surface_signature: Dict[str, Any]
    semantic_mapping_key: str
    capacity_scaling_key: str
    manifest_status: str
    contract_status: str
    runtime_policy: str
    evidence_basis: str
    latent_local_quantum_blocks: Optional[int] = None
    observable_local_quantum_blocks: Optional[int] = None
    latent_local_quantum_block_candidates: list[int] = field(default_factory=list)
    observable_local_quantum_block_candidates: list[int] = field(default_factory=list)
    dual_view_resolution: str = "not_applicable"
    floor_bypass_policy: str = "none"
    soft_snap_allowed: bool = False


@dataclass
class ResolvedLocalQuantumContract:
    family_key: str
    surface_key: str
    family_signature: Dict[str, Any]
    surface_signature: Dict[str, Any]
    semantic_mapping_key: str
    capacity_scaling_key: str
    manifest_status: str
    contract_status: str
    runtime_policy: str
    evidence_basis: str
    latent_local_quantum_blocks: Optional[int] = None
    observable_local_quantum_blocks: Optional[int] = None
    latent_local_quantum_block_candidates: list[int] = field(default_factory=list)
    observable_local_quantum_block_candidates: list[int] = field(default_factory=list)
    dual_view_resolution: str = "not_applicable"
    floor_bypass_policy: str = "none"
    local_quantum_gb: float = LOCAL_QUANTUM_GB
    resolution_source: str = ""
    probe_required: bool = False
    probe_reason: str = ""
    soft_snap_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_read_config_json(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return {}
    try:
        path = Path(model_path)
        if path.is_file():
            candidate = path
        else:
            candidate = path / "config.json"
        if candidate.exists() and candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _dtype_bytes_from_config(config_dict: Dict[str, Any]) -> int:
    dtype = str(config_dict.get("torch_dtype", "") or "").strip().lower()
    if dtype in {"bfloat16", "bf16", "float16", "fp16", "half"}:
        return 2
    if dtype in {"float32", "fp32"}:
        return 4
    if dtype in {"float64", "fp64", "double"}:
        return 8

    return 2


def estimate_bootstrap_kv_floor_from_model_config(
    model_path: str,
    *,
    max_model_len: int,
) -> Dict[str, Any]:
    """Estimate bootstrap kv floor from model config."""
    config_dict = _safe_read_config_json(model_path)
    num_hidden_layers = int(
        config_dict.get("num_hidden_layers") or config_dict.get("num_layers") or 0
    )
    num_attention_heads = int(config_dict.get("num_attention_heads") or 0)
    num_key_value_heads = int(
        config_dict.get("num_key_value_heads")
        or config_dict.get("num_kv_heads")
        or config_dict.get("multi_query_group_num")
        or num_attention_heads
        or 0
    )
    hidden_size = int(config_dict.get("hidden_size") or 0)
    head_dim = int(config_dict.get("head_dim") or 0)
    if head_dim <= 0 and hidden_size > 0 and num_attention_heads > 0:
        head_dim = hidden_size // num_attention_heads

    kv_layers = num_hidden_layers
    full_attention_layer_count = int(config_dict.get("full_attention_layer_count") or 0)
    if full_attention_layer_count > 0:
        kv_layers = full_attention_layer_count
        kv_layer_source = "config.full_attention_layer_count"
    else:
        kv_layer_source = "config.num_hidden_layers"

    dtype_bytes = _dtype_bytes_from_config(config_dict)
    valid = bool(
        max_model_len > 0 and kv_layers > 0 and num_key_value_heads > 0 and head_dim > 0
    )
    bytes_per_token = (
        int(kv_layers * 2 * num_key_value_heads * head_dim * dtype_bytes)
        if valid
        else 0
    )
    bootstrap_floor_bytes = int(bytes_per_token * max_model_len) if valid else 0
    bootstrap_floor_gb = float(bootstrap_floor_bytes / 1024**3) if valid else 0.0

    return {
        "estimate_valid": valid,
        "estimate_source": "model_config_formula",
        "model_type": str(config_dict.get("model_type", "") or "unknown"),
        "torch_dtype": str(config_dict.get("torch_dtype", "") or "unknown"),
        "max_model_len": int(max_model_len),
        "kv_layers": int(kv_layers),
        "kv_layer_source": kv_layer_source,
        "num_hidden_layers": int(num_hidden_layers),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "hidden_size": int(hidden_size),
        "dtype_bytes": int(dtype_bytes),
        "bytes_per_token": int(bytes_per_token),
        "bootstrap_floor_bytes": int(bootstrap_floor_bytes),
        "bootstrap_floor_gb": float(bootstrap_floor_gb),
    }


def _normalize_attention_backend(attention_backend: str) -> str:
    backend = str(attention_backend or "").strip().upper()
    if not backend:
        return "DEFAULT"
    return backend


def _infer_family_key(model_path: str, config_dict: Dict[str, Any]) -> str:
    lowered = str(model_path or "").lower()
    model_type = str(config_dict.get("model_type", "")).lower()
    architectures = " ".join(config_dict.get("architectures", []) or []).lower()
    quantization_method = str(
        (config_dict.get("quantization_config") or {}).get("quant_method", "")
    ).lower()

    if "opt-125m" in lowered or (
        model_type == "opt" and int(config_dict.get("num_hidden_layers", 0) or 0) == 12
    ):
        return "opt_125m_pure"
    if "granite" in lowered and "350m" in lowered:
        return "granite_350m_hybrid"
    if (
        ("qwen2.5" in lowered or "qwen2" in lowered)
        and "14b" in lowered
        and quantization_method == "gptq"
        and model_type == "qwen2"
    ):
        return "qwen25_14b_gptq_pure"
    if "qwen3.5-0.8b" in lowered or "qwen3-0.6b" in lowered:
        return "qwen35_08b_hybrid"
    if "qwen3.5-4b" in lowered:
        return "qwen35_4b_hybrid"
    if "qwen" in lowered and "0.8b" in lowered:
        return "qwen35_08b_hybrid"
    if "qwen" in lowered and "4b" in lowered:
        return "qwen35_4b_hybrid"
    if "granite" in architectures and "350m" in lowered:
        return "granite_350m_hybrid"
    return "unknown_family"


def _infer_text_surface(language_model_only: bool, config_dict: Dict[str, Any]) -> str:
    if bool(language_model_only):
        return "text_only"
    if bool(config_dict.get("vision_config")) or bool(config_dict.get("audio_config")):
        return "multimodal_or_mixed"
    return "text_only"


def _infer_attention_regime(family_key: str) -> str:
    if family_key == "opt_125m_pure":
        return "pure_full_attention"
    if family_key == "qwen25_14b_gptq_pure":
        return "pure_full_attention_quantized"
    if family_key in {"granite_350m_hybrid", "qwen35_08b_hybrid", "qwen35_4b_hybrid"}:
        return "hybrid_mixed_attention"
    return "unknown_attention_regime"


def _infer_family_signature(
    family_key: str,
    model_path: str,
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    hidden_size = int(config_dict.get("hidden_size", 0) or 0)
    num_attention_heads = int(config_dict.get("num_attention_heads", 0) or 0)
    num_key_value_heads = int(
        config_dict.get("num_key_value_heads")
        or config_dict.get("num_kv_heads")
        or config_dict.get("multi_query_group_num")
        or 0
    )
    head_dim = int(config_dict.get("head_dim", 0) or 0)
    if head_dim <= 0 and hidden_size > 0 and num_attention_heads > 0:
        head_dim = hidden_size // num_attention_heads

    signature = {
        "family_key": family_key,
        "attention_regime": _infer_attention_regime(family_key),
        "model_type": str(config_dict.get("model_type", "") or "unknown"),
        "num_hidden_layers": int(config_dict.get("num_hidden_layers", 0) or 0),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "source_model_path_tail": str(Path(model_path).name if model_path else ""),
    }

    if family_key == "opt_125m_pure":
        signature.update(
            {
                "full_attention_layer_count": 12,
                "non_full_attention_layer_count": 0,
                "layer_layout_signature": "pure_full_attention_12_0",
                "kv_tensor_shape_signature": "canonical_pure_main_view",
                "kv_bytes_per_block": None,
            }
        )
    elif family_key == "qwen25_14b_gptq_pure":
        signature.update(
            {
                "full_attention_layer_count": int(
                    config_dict.get("num_hidden_layers", 0) or 48
                ),
                "non_full_attention_layer_count": 0,
                "layer_layout_signature": "qwen25_pure_full_attention_48_0_gptq_int4",
                "kv_tensor_shape_signature": "pure_main_view_quantized_candidate",
                "kv_bytes_per_block": None,
                "quantization_enabled": True,
                "quantization_method": "gptq_int4",
                "quantization_group_size": int(
                    (
                        (config_dict.get("quantization_config") or {}).get(
                            "group_size", 0
                        )
                        or 0
                    )
                ),
            }
        )
    elif family_key == "granite_350m_hybrid":
        signature.update(
            {
                "full_attention_layer_count": 4,
                "non_full_attention_layer_count": 28,
                "layer_layout_signature": "granite_hybrid_4_28",
                "kv_tensor_shape_signature": "canonical_hybrid_main_view",
                "kv_bytes_per_block": None,
            }
        )
    elif family_key == "qwen35_08b_hybrid":
        signature.update(
            {
                "full_attention_layer_count": 6,
                "non_full_attention_layer_count": 18,
                "layer_layout_signature": "qwen35_08b_6_full_18_linear",
                "kv_tensor_shape_signature": "hybrid_main_aux_dual_view",
                "kv_bytes_per_block": None,
            }
        )
    elif family_key == "qwen35_4b_hybrid":
        signature.update(
            {
                "full_attention_layer_count": 8,
                "non_full_attention_layer_count": 24,
                "layer_layout_signature": "qwen35_4b_8_full_24_linear",
                "kv_tensor_shape_signature": "hybrid_main_aux_dual_view",
                "kv_bytes_per_block": None,
            }
        )
    else:
        signature.update(
            {
                "full_attention_layer_count": None,
                "non_full_attention_layer_count": None,
                "layer_layout_signature": "unknown",
                "kv_tensor_shape_signature": "unknown",
                "kv_bytes_per_block": None,
            }
        )

    return signature


def _bucket_kv_budget(kv_budget_gb: Optional[float]) -> str:
    if kv_budget_gb is None or kv_budget_gb <= 0:
        return "unspecified_budget"
    if kv_budget_gb <= 0.5:
        return "small_budget"
    if kv_budget_gb <= 2.0:
        return "medium_budget"
    return "large_budget"


def _infer_surface_key(
    family_key: str,
    *,
    attention_backend: str,
    kv_budget_gb: Optional[float],
    language_model_only: bool,
) -> str:
    backend = _normalize_attention_backend(attention_backend)
    kv_bucket = _bucket_kv_budget(kv_budget_gb)
    text_surface = "text_only" if language_model_only else "default_text_surface"

    if family_key == "opt_125m_pure":
        return "canonical_text_only_default_surface"
    if family_key == "qwen25_14b_gptq_pure":
        if backend == "TRITON_ATTN":
            return "text_only_quantized_triton_surface_probe_required"
        return "text_only_quantized_default_surface_probe_required"
    if family_key == "granite_350m_hybrid":
        return "canonical_text_only_default_surface"
    if family_key == "qwen35_08b_hybrid":
        if backend == "TRITON_ATTN":
            return "text_triton_attn_matched_toggle_surface"
        if kv_bucket == "small_budget":
            return "small_budget_text_only_default_surface"
        return f"{text_surface}_default_surface_probe_required"
    if family_key == "qwen35_4b_hybrid":
        if backend == "TRITON_ATTN":
            return "triton_attn_toggle_surface"
        return "default_backend_surface"
    return "unknown_surface"


def _build_surface_signature(
    family_key: str,
    config_dict: Dict[str, Any],
    *,
    attention_backend: str,
    kv_budget_gb: Optional[float],
    initial_kv_cache_gb: Optional[float],
    max_kv_cache_gb: Optional[float],
    gpu_utilization: Optional[float],
    max_num_seqs: Optional[int],
    max_num_batched_tokens: Optional[int],
    max_model_len: Optional[int],
    language_model_only: bool,
) -> Dict[str, Any]:
    backend = _normalize_attention_backend(attention_backend)
    kv_surface = _bucket_kv_budget(
        kv_budget_gb if kv_budget_gb is not None else initial_kv_cache_gb
    )
    text_surface = _infer_text_surface(language_model_only, config_dict)
    blocks_dim_layout_surface = "unknown"
    layout_mode_signature = "default_runtime_layout"

    if family_key in {"qwen35_08b_hybrid", "qwen35_4b_hybrid"}:
        if backend == "TRITON_ATTN":
            blocks_dim_layout_surface = "blocks_dim_idx_0_aligned"
            layout_mode_signature = "triton_attn_aligned_layout"
        else:
            blocks_dim_layout_surface = "blocks_dim_idx_1_mismatch"
            layout_mode_signature = "default_backend_dual_view_layout"
    elif family_key == "qwen25_14b_gptq_pure":
        if backend == "TRITON_ATTN":
            blocks_dim_layout_surface = "quantized_blocks_dim_probe_required"
            layout_mode_signature = "quantized_triton_layout_probe_required"
        else:
            blocks_dim_layout_surface = "quantized_blocks_dim_probe_required"
            layout_mode_signature = "quantized_default_layout_probe_required"
    elif family_key in {"opt_125m_pure", "granite_350m_hybrid"}:
        blocks_dim_layout_surface = "canonical_main_view"
        layout_mode_signature = "canonical_default_layout"

    return {
        "text_only_or_multimodal_surface": text_surface,
        "attention_backend_surface": backend,
        "layout_mode_signature": layout_mode_signature,
        "blocks_dim_layout_surface": blocks_dim_layout_surface,
        "kv_budget_surface": kv_surface,
        "requested_kv_budget_gb": kv_budget_gb,
        "initial_kv_cache_gb": initial_kv_cache_gb,
        "max_kv_cache_gb": max_kv_cache_gb,
        "gpu_utilization_surface": gpu_utilization,
        "scheduler_surface": {
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
        },
        "max_model_len_surface": max_model_len,
        "language_model_only": bool(language_model_only),
    }


KNOWN_MANIFESTS: tuple[KnownManifestEntry, ...] = (
    KnownManifestEntry(
        family_key="opt_125m_pure",
        surface_key="canonical_text_only_default_surface",
        family_signature={
            "attention_regime": "pure_full_attention",
            "num_hidden_layers": 12,
            "full_attention_layer_count": 12,
            "non_full_attention_layer_count": 0,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "layer_layout_signature": "pure_full_attention_12_0",
            "kv_tensor_shape_signature": "canonical_pure_main_view",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "DEFAULT",
            "layout_mode_signature": "canonical_default_layout",
            "blocks_dim_layout_surface": "canonical_main_view",
            "kv_budget_surface": "canonical_budget",
        },
        semantic_mapping_key="canonical_aligned_main_view",
        capacity_scaling_key="canonical_pure_anchor_scaling_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis="calibrated mapping: 91 blocks / 0.05 GB",
        latent_local_quantum_blocks=91,
        observable_local_quantum_blocks=91,
        latent_local_quantum_block_candidates=[91],
        observable_local_quantum_block_candidates=[91],
        dual_view_resolution="not_applicable",
        floor_bypass_policy="none",
    ),
    KnownManifestEntry(
        family_key="granite_350m_hybrid",
        surface_key="canonical_text_only_default_surface",
        family_signature={
            "attention_regime": "hybrid_mixed_attention",
            "num_hidden_layers": 32,
            "full_attention_layer_count": 4,
            "non_full_attention_layer_count": 28,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "layer_layout_signature": "granite_hybrid_4_28",
            "kv_tensor_shape_signature": "canonical_hybrid_main_view",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "DEFAULT",
            "layout_mode_signature": "canonical_default_layout",
            "blocks_dim_layout_surface": "canonical_main_view",
            "kv_budget_surface": "canonical_budget",
        },
        semantic_mapping_key="canonical_hybrid_main_view",
        capacity_scaling_key="canonical_hybrid_anchor_scaling_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis="calibrated mapping: 32-33 blocks / 0.05 GB",
        latent_local_quantum_blocks=33,
        observable_local_quantum_blocks=33,
        latent_local_quantum_block_candidates=[32, 33],
        observable_local_quantum_block_candidates=[32, 33],
        dual_view_resolution="not_applicable",
        floor_bypass_policy="none",
    ),
    KnownManifestEntry(
        family_key="qwen25_14b_gptq_pure",
        surface_key="text_only_quantized_default_surface_probe_required",
        family_signature={
            "attention_regime": "pure_full_attention_quantized",
            "num_hidden_layers": 48,
            "full_attention_layer_count": 48,
            "non_full_attention_layer_count": 0,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_layout_signature": "qwen25_pure_full_attention_48_0_gptq_int4",
            "kv_tensor_shape_signature": "pure_main_view_quantized_candidate",
            "quantization_enabled": True,
            "quantization_method": "gptq_int4",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "DEFAULT",
            "layout_mode_signature": "quantized_default_aligned_layout",
            "blocks_dim_layout_surface": "blocks_dim_idx_1_aligned_main_view",
            "kv_budget_surface": "small_budget",
        },
        semantic_mapping_key="supplementary_pure_surface_semantic_factor_1",
        capacity_scaling_key="supplementary_quantized_default_capacity_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis="calibrated aligned mapping: 17 blocks / 0.05 GB",
        latent_local_quantum_blocks=17,
        observable_local_quantum_blocks=17,
        latent_local_quantum_block_candidates=[17],
        observable_local_quantum_block_candidates=[17],
        dual_view_resolution="aligned_main_view_direct_load_ready",
        floor_bypass_policy="none",
    ),
    KnownManifestEntry(
        family_key="qwen35_08b_hybrid",
        surface_key="small_budget_text_only_default_surface",
        family_signature={
            "attention_regime": "hybrid_mixed_attention",
            "num_hidden_layers": 24,
            "full_attention_layer_count": 6,
            "non_full_attention_layer_count": 18,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_layout_signature": "qwen35_08b_6_full_18_linear",
            "kv_tensor_shape_signature": "hybrid_main_aux_dual_view",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "DEFAULT",
            "layout_mode_signature": "default_backend_dual_view_layout",
            "blocks_dim_layout_surface": "blocks_dim_idx_1_mismatch",
            "kv_budget_surface": "small_budget",
        },
        semantic_mapping_key="default_surface_semantic_factor_17",
        capacity_scaling_key="small_budget_text_only_capacity_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis=(
            "calibrated dual-view mapping: latent=8 blocks, observable=136 blocks "
            "/ 0.05 GB"
        ),
        latent_local_quantum_blocks=8,
        observable_local_quantum_blocks=136,
        latent_local_quantum_block_candidates=[8],
        observable_local_quantum_block_candidates=[136],
        dual_view_resolution="observable_direct_load_with_floor_bypass_ready",
        floor_bypass_policy="preserve_min_floor_when_floor_dominated_and_delta_below_observable_quantum",
    ),
    KnownManifestEntry(
        family_key="qwen35_08b_hybrid",
        surface_key="text_triton_attn_matched_toggle_surface",
        family_signature={
            "attention_regime": "hybrid_mixed_attention",
            "num_hidden_layers": 24,
            "full_attention_layer_count": 6,
            "non_full_attention_layer_count": 18,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_layout_signature": "qwen35_08b_6_full_18_linear",
            "kv_tensor_shape_signature": "hybrid_main_aux_dual_view",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "TRITON_ATTN",
            "layout_mode_signature": "triton_attn_aligned_layout",
            "blocks_dim_layout_surface": "blocks_dim_idx_0_aligned",
            "kv_budget_surface": "matched_toggle_probe",
        },
        semantic_mapping_key="aligned_surface_semantic_factor_1",
        capacity_scaling_key="matched_toggle_text_only_capacity_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis="calibrated aligned mapping: 8 blocks / 0.05 GB",
        latent_local_quantum_blocks=8,
        observable_local_quantum_blocks=8,
        latent_local_quantum_block_candidates=[8],
        observable_local_quantum_block_candidates=[8],
        dual_view_resolution="aligned_main_view_direct_load_ready",
        floor_bypass_policy="none",
    ),
    KnownManifestEntry(
        family_key="qwen35_4b_hybrid",
        surface_key="default_backend_surface",
        family_signature={
            "attention_regime": "hybrid_mixed_attention",
            "num_hidden_layers": 32,
            "full_attention_layer_count": 8,
            "non_full_attention_layer_count": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "layer_layout_signature": "qwen35_4b_8_full_24_linear",
            "kv_tensor_shape_signature": "hybrid_main_aux_dual_view",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "DEFAULT",
            "layout_mode_signature": "default_backend_dual_view_layout",
            "blocks_dim_layout_surface": "blocks_dim_idx_1_mismatch",
            "kv_budget_surface": "default_budget",
        },
        semantic_mapping_key="default_surface_semantic_factor_33",
        capacity_scaling_key="default_backend_text_only_capacity_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis=(
            "calibrated dual-view mapping: latent=3 blocks, observable=99 blocks "
            "/ 0.05 GB"
        ),
        latent_local_quantum_blocks=3,
        observable_local_quantum_blocks=99,
        latent_local_quantum_block_candidates=[3],
        observable_local_quantum_block_candidates=[99],
        dual_view_resolution="observable_direct_load_ready",
        floor_bypass_policy="none",
    ),
    KnownManifestEntry(
        family_key="qwen35_4b_hybrid",
        surface_key="triton_attn_toggle_surface",
        family_signature={
            "attention_regime": "hybrid_mixed_attention",
            "num_hidden_layers": 32,
            "full_attention_layer_count": 8,
            "non_full_attention_layer_count": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "layer_layout_signature": "qwen35_4b_8_full_24_linear",
            "kv_tensor_shape_signature": "hybrid_main_aux_dual_view",
        },
        surface_signature={
            "text_only_or_multimodal_surface": "text_only",
            "attention_backend_surface": "TRITON_ATTN",
            "layout_mode_signature": "triton_attn_aligned_layout",
            "blocks_dim_layout_surface": "blocks_dim_idx_0_aligned",
            "kv_budget_surface": "default_budget",
        },
        semantic_mapping_key="aligned_surface_semantic_factor_1",
        capacity_scaling_key="triton_attn_text_only_capacity_surface",
        manifest_status="verified_manifest_ready",
        contract_status="canonical_same_device_contract_ready",
        runtime_policy="direct_load_manifest",
        evidence_basis="calibrated aligned mapping: 3 blocks / 0.05 GB",
        latent_local_quantum_blocks=3,
        observable_local_quantum_blocks=3,
        latent_local_quantum_block_candidates=[3],
        observable_local_quantum_block_candidates=[3],
        dual_view_resolution="aligned_main_view_direct_load_ready",
        floor_bypass_policy="none",
    ),
)


def _resolve_known_manifest(
    family_key: str,
    surface_key: str,
    family_signature: Dict[str, Any],
    surface_signature: Dict[str, Any],
) -> Optional[KnownManifestEntry]:
    for entry in KNOWN_MANIFESTS:
        if entry.family_key == family_key and entry.surface_key == surface_key:
            return entry
    return None


def resolve_local_quantum_contract(
    *,
    model_path: str,
    attention_backend: str = "",
    kv_budget_gb: Optional[float] = None,
    initial_kv_cache_gb: Optional[float] = None,
    max_kv_cache_gb: Optional[float] = None,
    gpu_utilization: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    max_model_len: Optional[int] = None,
    language_model_only: bool = False,
) -> ResolvedLocalQuantumContract:
    config_dict = _safe_read_config_json(model_path)
    family_key = _infer_family_key(model_path, config_dict)
    family_signature = _infer_family_signature(family_key, model_path, config_dict)
    surface_key = _infer_surface_key(
        family_key,
        attention_backend=attention_backend,
        kv_budget_gb=kv_budget_gb if kv_budget_gb is not None else initial_kv_cache_gb,
        language_model_only=language_model_only,
    )
    surface_signature = _build_surface_signature(
        family_key,
        config_dict,
        attention_backend=attention_backend,
        kv_budget_gb=kv_budget_gb,
        initial_kv_cache_gb=initial_kv_cache_gb,
        max_kv_cache_gb=max_kv_cache_gb,
        gpu_utilization=gpu_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        language_model_only=language_model_only,
    )

    manifest_entry = _resolve_known_manifest(
        family_key, surface_key, family_signature, surface_signature
    )
    if manifest_entry is not None:
        probe_required = (
            manifest_entry.manifest_status == "surface_semantic_key_verified"
        )
        probe_reason = ""
        if probe_required:
            probe_reason = "semantic_mapping_key resolve local quantum contract, observable/latent local quantum  staircase probe."
        return ResolvedLocalQuantumContract(
            family_key=family_key,
            surface_key=surface_key,
            family_signature={**family_signature, **manifest_entry.family_signature},
            surface_signature={**surface_signature, **manifest_entry.surface_signature},
            semantic_mapping_key=manifest_entry.semantic_mapping_key,
            capacity_scaling_key=manifest_entry.capacity_scaling_key,
            manifest_status=manifest_entry.manifest_status,
            contract_status=manifest_entry.contract_status,
            runtime_policy=manifest_entry.runtime_policy,
            evidence_basis=manifest_entry.evidence_basis,
            latent_local_quantum_blocks=manifest_entry.latent_local_quantum_blocks,
            observable_local_quantum_blocks=manifest_entry.observable_local_quantum_blocks,
            latent_local_quantum_block_candidates=list(
                manifest_entry.latent_local_quantum_block_candidates
            ),
            observable_local_quantum_block_candidates=list(
                manifest_entry.observable_local_quantum_block_candidates
            ),
            dual_view_resolution=manifest_entry.dual_view_resolution,
            floor_bypass_policy=manifest_entry.floor_bypass_policy,
            resolution_source=f"known_manifest:{family_key}/{surface_key}",
            probe_required=probe_required,
            probe_reason=probe_reason,
            soft_snap_allowed=manifest_entry.soft_snap_allowed,
        )

    backend = _normalize_attention_backend(attention_backend)
    aligned_semantic_probe_families = {"qwen35_08b_hybrid", "qwen35_4b_hybrid"}
    if family_key == "unknown_family":
        semantic_mapping_key = "unknown_surface_semantic_mapping"
        capacity_scaling_key = "unknown_capacity_scaling"
        probe_reason = "unknown family+surface,resolve local quantum contract manifest,invalid configuration small-probe fallback."
    else:
        semantic_mapping_key = (
            "aligned_surface_semantic_factor_1"
            if backend == "TRITON_ATTN"
            and family_key in aligned_semantic_probe_families
            else "unknown_family_known_surface_semantic_pending"
        )
        capacity_scaling_key = "probe_required_capacity_scaling"
        probe_reason = "family resolve local quantum contract, surface  manifest registry, small-probe fallback."

    return ResolvedLocalQuantumContract(
        family_key=family_key,
        surface_key=surface_key,
        family_signature=family_signature,
        surface_signature=surface_signature,
        semantic_mapping_key=semantic_mapping_key,
        capacity_scaling_key=capacity_scaling_key,
        manifest_status="candidate_probe_only",
        contract_status="probe_required_before_runtime_contract",
        runtime_policy="small_probe_required",
        evidence_basis="runtime fallback route",
        resolution_source=f"probe_required:{family_key}/{surface_key}",
        probe_required=True,
        probe_reason=probe_reason,
    )


def serialize_contract_for_env(contract: Dict[str, Any]) -> str:
    return json.dumps(contract, ensure_ascii=True, sort_keys=True)


__all__ = [
    "KnownManifestEntry",
    "ResolvedLocalQuantumContract",
    "estimate_bootstrap_kv_floor_from_model_config",
    "resolve_local_quantum_contract",
    "serialize_contract_for_env",
]
