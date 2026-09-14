# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Support matrix checks for the Prims-TS BF16 MoE MVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from tensorrt_llm._torch.moe.flashinfer.prims_ts.utils import is_prims_ts_available
from tensorrt_llm._torch.moe.flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)
from flashinfer.utils import get_compute_capability

_SUPPORTED_ACTIVATIONS = (
    ActivationType.Identity,
    ActivationType.Swiglu,
    ActivationType.Geglu,
    ActivationType.Silu,
    ActivationType.Relu2,
    ActivationType.Situ,
)


@dataclass(frozen=True)
class SupportResult:
    ok: bool
    reason: str = ""

    def as_tuple(self) -> tuple[bool, str]:
        return self.ok, self.reason


def _enum_eq(value: Any, expected: Any) -> bool:
    try:
        return int(value) == int(expected)
    except Exception:
        return value == expected


def _weight_layout_arg(runner: Any, kwargs: dict[str, Any]) -> int:
    return int(kwargs.get("weight_layout", runner.weight_layout))


def _is_supported_weight_layout(value: Any) -> bool:
    return _enum_eq(value, WeightLayout.MajorK) or _enum_eq(
        value, WeightLayout.BlockMajorK
    )


def _uses_block_major_k(runner: Any, kwargs: dict[str, Any]) -> bool:
    return _enum_eq(_weight_layout_arg(runner, kwargs), WeightLayout.BlockMajorK)


def _fp4_block_major_k_candidate_bytes(runner: Any) -> tuple[int, ...]:
    if _enum_eq(
        getattr(runner, "dtype_weights", None), DtypeTrtllmGen.MxE2m1
    ) or _enum_eq(getattr(runner, "dtype_weights", None), DtypeTrtllmGen.E2m1):
        return (64, 128)
    return (128,)


def _block_major_k_storage_bytes(tensor: Any) -> int | None:
    if tensor is None or not hasattr(tensor, "dim") or tensor.dim() != 4:
        return None
    return int(tensor.shape[-1]) * int(tensor.element_size())


def _validate_fp4_block_major_k_alignment(runner: Any, kwargs: dict[str, Any]):
    if not _uses_block_major_k(runner, kwargs):
        return True, ""
    candidate_bytes = _fp4_block_major_k_candidate_bytes(runner)
    observed = {
        _block_major_k_storage_bytes(kwargs.get("gemm1_weights")),
        _block_major_k_storage_bytes(kwargs.get("gemm2_weights")),
    }
    observed.discard(None)
    if len(observed) > 1:
        return (
            False,
            "BlockMajorK FP4 gemm1/gemm2 weights use different K-block byte sizes",
        )
    block_bytes = next(iter(observed), None)
    if block_bytes is not None and block_bytes not in candidate_bytes:
        return (
            False,
            "BlockMajorK FP4 weights have unsupported K-block byte size "
            f"{block_bytes}; expected one of {candidate_bytes}",
        )
    if block_bytes is None:
        block_bytes = next(
            (
                candidate
                for candidate in candidate_bytes
                if (runner.hidden_size // 2) % candidate == 0
                and (runner.intermediate_size // 2) % candidate == 0
            ),
            candidate_bytes[-1],
        )
    if (runner.hidden_size // 2) % block_bytes != 0 or (
        runner.intermediate_size // 2
    ) % block_bytes != 0:
        return (
            False,
            "BlockMajorK FP4 weights require packed hidden_size / 2 and "
            f"intermediate_size / 2 to be multiples of {block_bytes} bytes",
        )
    return True, ""


def _validate_block_major_k_storage_matches_config(
    runner: Any, kwargs: dict[str, Any], fc1_cfg: Any, fc2_cfg: Any
) -> tuple[bool, str]:
    if not _uses_block_major_k(runner, kwargs):
        return True, ""
    expected = (
        ("gemm1_weights", int(fc1_cfg.block_major_k_bytes)),
        ("gemm2_weights", int(fc2_cfg.block_major_k_bytes)),
    )
    for name, expected_bytes in expected:
        actual_bytes = _block_major_k_storage_bytes(kwargs.get(name))
        if actual_bytes is None:
            continue
        if actual_bytes != expected_bytes:
            return (
                False,
                f"{name} BlockMajorK K-block is {actual_bytes} bytes but the "
                f"resolved Prims-TS config expects {expected_bytes} bytes",
            )
    return True, ""


def _device_supports_prims_ts(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, minor = get_compute_capability(device)
    return (major, minor) in ((10, 0), (10, 3))


def _has_gemm1_oa_params(kwargs: dict[str, Any]) -> bool:
    return (
        kwargs.get("gemm1_alpha") is not None
        or kwargs.get("gemm1_beta") is not None
        or kwargs.get("gemm1_clamp_limit") is not None
    )


def _gemm1_oa_flags(kwargs: dict[str, Any]) -> dict[str, bool]:
    return {
        "has_gemm1_alpha": kwargs.get("gemm1_alpha") is not None,
        "has_gemm1_beta": kwargs.get("gemm1_beta") is not None,
        "has_gemm1_clamp_limit": kwargs.get("gemm1_clamp_limit") is not None,
    }


def _per_token_sf_dtype_value(tensor: torch.Tensor) -> int:
    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    if tensor.dtype == torch.bfloat16:
        return int(DType.BF16)
    if tensor.dtype == torch.float16:
        return int(DType.FP16)
    if tensor.dtype == torch.float32:
        return int(DType.FP32)
    raise ValueError(f"Unsupported per-token scale dtype {tensor.dtype}")


def _merge_per_token_sf_dtype(
    current: int | None, candidate: int, *, current_name: str, candidate_name: str
) -> int:
    if current is None:
        return int(candidate)
    if int(current) != int(candidate):
        raise ValueError(f"{current_name} and {candidate_name} must use the same dtype")
    return int(current)


def _split_per_channel_weight_scale_from_kwargs(
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return (
        kwargs.get("fc1_per_channel_weight_scale"),
        kwargs.get("fc2_per_channel_weight_scale"),
    )


def _is_gated_activation(activation_type: Any) -> bool:
    return any(
        _enum_eq(activation_type, act)
        for act in (ActivationType.Swiglu, ActivationType.Geglu, ActivationType.Situ)
    )


def _validate_gemm1_oa_params(
    runner: Any,
    moe_inputs: Any,
    activation_type: Any,
    kwargs: dict[str, Any],
) -> tuple[bool, str]:
    if not _has_gemm1_oa_params(kwargs):
        return True, ""
    if not any(
        _enum_eq(activation_type, act)
        for act in (ActivationType.Swiglu, ActivationType.Situ)
    ):
        return False, "gemm1_alpha/beta/clamp_limit require Swiglu or Situ activation"
    device = moe_inputs.hidden_states.device
    num_experts = int(
        kwargs.get("local_num_experts", getattr(runner, "num_local_experts", 0))
    )
    if num_experts <= 0:
        return False, "num_experts must be available for OA parameter validation"
    for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
        tensor = kwargs.get(name)
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            return False, f"{name} must be a torch.Tensor"
        if tensor.dtype != torch.float32:
            return False, f"{name} must be float32"
        if tensor.dim() != 1:
            return False, f"{name} must be 1D"
        if not tensor.is_contiguous():
            return False, f"{name} must be contiguous"
        if tensor.device != device:
            return False, f"{name} must be on {device}, got {tensor.device}"
        if tensor.numel() < num_experts:
            return False, f"{name} must have at least {num_experts} elements"
    return True, ""


def is_prims_ts_bf16_supported(
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return whether this BF16 MoE call is eligible for the Prims-TS path."""

    hidden_states = moe_inputs.hidden_states
    if not is_prims_ts_available():
        return False, "nvidia-cutlass-dsl Prims-TS dependencies are not importable"
    if not _device_supports_prims_ts(hidden_states.device):
        return False, "requires an SM100 or SM103 CUDA device"
    if not _enum_eq(runner.dtype_act, DtypeTrtllmGen.Bfloat16) or not _enum_eq(
        runner.dtype_weights, DtypeTrtllmGen.Bfloat16
    ):
        return False, "only BF16 activations and BF16 weights are supported"
    activation_type = kwargs.get("activation_type", runner.activation_type)
    if not any(_enum_eq(activation_type, act) for act in _SUPPORTED_ACTIVATIONS):
        return False, "activation must be Identity, Swiglu, Geglu, Silu, Relu2, or Situ"
    routing_method_type = kwargs.get(
        "routing_method_type", getattr(runner, "routing_method_type", None)
    )
    if any(
        _enum_eq(routing_method_type, routing)
        for routing in (RoutingMethodType.Sigmoid, RoutingMethodType.DeepSeekV3)
    ):
        return False, "Sigmoid and DeepSeekV3 routing are not supported"
    if not _is_supported_weight_layout(
        kwargs.get("weight_layout", runner.weight_layout)
    ):
        return False, "weight_layout must be MajorK or BlockMajorK"
    if not kwargs.get("use_shuffled_weight", runner.use_shuffled_weight):
        return False, "Prims-TS BF16 path expects shuffled weights"
    if moe_inputs.gemm1_lora_delta is not None:
        return False, "gemm1_lora_delta is not supported"
    ok, reason = _validate_gemm1_oa_params(runner, moe_inputs, activation_type, kwargs)
    if not ok:
        return False, reason
    if runner.intermediate_size % 128 != 0:
        return False, "intermediate_size must be a multiple of 128"
    if getattr(runner, "use_per_token_scaling", False):
        return False, "per-token scaling is not supported"
    if moe_inputs.hidden_states_scale is not None:
        return False, "scaled activation quantization is outside the BF16 MVP"
    if moe_inputs.per_token_scale is not None:
        return False, "per-token output scaling is outside the BF16 MVP"
    if kwargs.get("num_fused_shared_experts", 0):
        return False, "fused shared experts are not supported"

    try:
        from .config_mapper import map_trtllm_bf16_moe_tactic

        pair = map_trtllm_bf16_moe_tactic(
            tactic,
            activation_type=int(activation_type),
            num_tokens=int(hidden_states.shape[0]),
            top_k=getattr(runner, "top_k", None),
            num_local_experts=getattr(runner, "num_local_experts", None),
            weight_layout=_weight_layout_arg(runner, kwargs),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags(kwargs),
        )
        ok, reason = _validate_block_major_k_storage_matches_config(
            runner, kwargs, pair.fc1.cfg.build(), pair.fc2.cfg.build()
        )
        if not ok:
            return False, reason
    except Exception as exc:
        return False, str(exc)

    return True, ""


def is_prims_ts_nvfp4_supported(
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return whether this NVFP4xNVFP4 MoE call is eligible for Prims-TS."""

    hidden_states = moe_inputs.hidden_states
    if not is_prims_ts_available():
        return False, "nvidia-cutlass-dsl Prims-TS dependencies are not importable"
    if not _device_supports_prims_ts(hidden_states.device):
        return False, "requires an SM100 or SM103 CUDA device"
    if not _enum_eq(runner.dtype_act, DtypeTrtllmGen.E2m1) or not _enum_eq(
        runner.dtype_weights, DtypeTrtllmGen.E2m1
    ):
        return False, "only NVFP4 activations and NVFP4 weights are supported"
    activation_type = kwargs.get("activation_type", runner.activation_type)
    if not any(_enum_eq(activation_type, act) for act in _SUPPORTED_ACTIVATIONS):
        return False, "activation must be Identity, Swiglu, Geglu, Silu, Relu2, or Situ"
    if not _is_supported_weight_layout(
        kwargs.get("weight_layout", runner.weight_layout)
    ):
        return False, "weight_layout must be MajorK or BlockMajorK"
    ok, reason = _validate_fp4_block_major_k_alignment(runner, kwargs)
    if not ok:
        return False, reason
    if not kwargs.get("use_shuffled_weight", runner.use_shuffled_weight):
        return False, "Prims-TS NVFP4 path expects shuffled weights"
    if moe_inputs.hidden_states_scale is None:
        return False, "hidden_states_scale is required for NVFP4 activations"
    if moe_inputs.gemm1_lora_delta is not None:
        return False, "gemm1_lora_delta is not supported"
    ok, reason = _validate_gemm1_oa_params(runner, moe_inputs, activation_type, kwargs)
    if not ok:
        return False, reason
    if moe_inputs.per_token_scale is not None:
        if moe_inputs.per_token_scale.dtype != torch.float32:
            return False, "NVFP4 per-token scale must be float32"
        if moe_inputs.per_token_scale.dim() != 1:
            return False, "NVFP4 per-token scale must be 1D"
        if moe_inputs.per_token_scale.shape[0] < hidden_states.shape[0]:
            return False, "NVFP4 per-token scale must cover all input tokens"
    if kwargs.get("num_fused_shared_experts", 0):
        return False, "fused shared experts are not supported"
    if runner.intermediate_size % 128 != 0:
        return False, "intermediate_size must be a multiple of 128"

    required = (
        "gemm1_weights_scale",
        "gemm2_weights_scale",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    )
    for name in required:
        if kwargs.get(name) is None:
            return False, f"{name} is required"

    try:
        from .config_mapper import map_trtllm_nvfp4_moe_tactic

        pair = map_trtllm_nvfp4_moe_tactic(
            tactic,
            activation_type=int(activation_type),
            num_tokens=int(hidden_states.shape[0]),
            top_k=getattr(runner, "top_k", None),
            num_local_experts=getattr(runner, "num_local_experts", None),
            weight_layout=_weight_layout_arg(runner, kwargs),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            use_per_token_sf_b=moe_inputs.per_token_scale is not None,
            per_token_sf_dtype=1,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags(kwargs),
        )
        ok, reason = _validate_block_major_k_storage_matches_config(
            runner, kwargs, pair.fc1.cfg.build(), pair.fc2.cfg.build()
        )
        if not ok:
            return False, reason
    except Exception as exc:
        return False, str(exc)

    return True, ""


def is_prims_ts_mxfp4_mxfp8_supported(
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return whether this MXFP4xMXFP8 MoE call is eligible for Prims-TS."""

    hidden_states = moe_inputs.hidden_states
    if not is_prims_ts_available():
        return False, "nvidia-cutlass-dsl Prims-TS dependencies are not importable"
    if not _device_supports_prims_ts(hidden_states.device):
        return False, "requires an SM100 or SM103 CUDA device"
    if not _enum_eq(runner.dtype_act, DtypeTrtllmGen.MxE4m3) or not _enum_eq(
        runner.dtype_weights, DtypeTrtllmGen.MxE2m1
    ):
        return False, "only MXFP8 activations and MXFP4 weights are supported"
    if hidden_states.dtype != torch.float8_e4m3fn:
        return False, "hidden_states must be float8_e4m3fn MXFP8 storage"
    if moe_inputs.hidden_states_scale is None:
        return False, "hidden_states_scale is required for MXFP8 activations"
    activation_type = kwargs.get("activation_type", runner.activation_type)
    if not any(_enum_eq(activation_type, act) for act in _SUPPORTED_ACTIVATIONS):
        return False, "activation must be Identity, Swiglu, Geglu, Silu, Relu2, or Situ"
    if not _is_supported_weight_layout(
        kwargs.get("weight_layout", runner.weight_layout)
    ):
        return False, "weight_layout must be MajorK or BlockMajorK"
    ok, reason = _validate_fp4_block_major_k_alignment(runner, kwargs)
    if not ok:
        return False, reason
    if not kwargs.get("use_shuffled_weight", runner.use_shuffled_weight):
        return False, "Prims-TS MXFP4xMXFP8 path expects shuffled weights"
    if moe_inputs.gemm1_lora_delta is not None:
        return False, "gemm1_lora_delta is not supported"
    ok, reason = _validate_gemm1_oa_params(runner, moe_inputs, activation_type, kwargs)
    if not ok:
        return False, reason
    if moe_inputs.per_token_scale is not None or getattr(
        runner, "use_per_token_scaling", False
    ):
        return False, "per-token scaling is not supported"
    if kwargs.get("num_fused_shared_experts", 0):
        return False, "fused shared experts are not supported"
    if runner.intermediate_size % 128 != 0 or runner.hidden_size % 128 != 0:
        return False, "hidden_size and intermediate_size must be multiples of 128"
    required = (
        "gemm1_weights_scale",
        "gemm2_weights_scale",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    )
    for name in required:
        if kwargs.get(name) is None:
            return False, f"{name} is required"

    try:
        from .config_mapper import map_trtllm_mxfp4_mxfp8_moe_tactic

        pair = map_trtllm_mxfp4_mxfp8_moe_tactic(
            tactic,
            activation_type=int(activation_type),
            num_tokens=int(hidden_states.shape[0]),
            top_k=getattr(runner, "top_k", None),
            num_local_experts=getattr(runner, "num_local_experts", None),
            weight_layout=_weight_layout_arg(runner, kwargs),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags(kwargs),
        )
        ok, reason = _validate_block_major_k_storage_matches_config(
            runner, kwargs, pair.fc1.cfg.build(), pair.fc2.cfg.build()
        )
        if not ok:
            return False, reason
    except Exception as exc:
        return False, str(exc)

    return True, ""


def is_prims_ts_mxfp4_bf16_supported(
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return whether this MXFP4xBF16 MoE call is eligible for Prims-TS."""

    hidden_states = moe_inputs.hidden_states
    if not is_prims_ts_available():
        return False, "nvidia-cutlass-dsl Prims-TS dependencies are not importable"
    if not _device_supports_prims_ts(hidden_states.device):
        return False, "requires an SM100 or SM103 CUDA device"
    if not _enum_eq(runner.dtype_act, DtypeTrtllmGen.Bfloat16) or not _enum_eq(
        runner.dtype_weights, DtypeTrtllmGen.MxE2m1
    ):
        return False, "only BF16 activations and MXFP4 weights are supported"
    if hidden_states.dtype != torch.bfloat16:
        return False, "hidden_states must be bfloat16"
    if moe_inputs.hidden_states_scale is not None:
        return False, "hidden_states_scale must be None for BF16 activations"
    activation_type = kwargs.get("activation_type", runner.activation_type)
    if not any(_enum_eq(activation_type, act) for act in _SUPPORTED_ACTIVATIONS):
        return False, "activation must be Identity, Swiglu, Geglu, Silu, Relu2, or Situ"
    if not _is_supported_weight_layout(
        kwargs.get("weight_layout", runner.weight_layout)
    ):
        return False, "weight_layout must be MajorK or BlockMajorK"
    ok, reason = _validate_fp4_block_major_k_alignment(runner, kwargs)
    if not ok:
        return False, reason
    if not kwargs.get("use_shuffled_weight", runner.use_shuffled_weight):
        return False, "Prims-TS MXFP4xBF16 path expects shuffled weights"
    if moe_inputs.gemm1_lora_delta is not None:
        return False, "gemm1_lora_delta is not supported"
    ok, reason = _validate_gemm1_oa_params(runner, moe_inputs, activation_type, kwargs)
    if not ok:
        return False, reason
    if moe_inputs.per_token_scale is not None or getattr(
        runner, "use_per_token_scaling", False
    ):
        return False, "per-token scaling is not supported"
    if kwargs.get("num_fused_shared_experts", 0):
        return False, "fused shared experts are not supported"
    if runner.intermediate_size % 128 != 0 or runner.hidden_size % 128 != 0:
        return False, "hidden_size and intermediate_size must be multiples of 128"

    required = (
        "gemm1_weights_scale",
        "gemm2_weights_scale",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    )
    for name in required:
        if kwargs.get(name) is None:
            return False, f"{name} is required"

    try:
        from .config_mapper import map_trtllm_mxfp4_bf16_moe_tactic

        pair = map_trtllm_mxfp4_bf16_moe_tactic(
            tactic,
            activation_type=int(activation_type),
            num_tokens=int(hidden_states.shape[0]),
            top_k=getattr(runner, "top_k", None),
            num_local_experts=getattr(runner, "num_local_experts", None),
            weight_layout=_weight_layout_arg(runner, kwargs),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags(kwargs),
        )
        ok, reason = _validate_block_major_k_storage_matches_config(
            runner, kwargs, pair.fc1.cfg.build(), pair.fc2.cfg.build()
        )
        if not ok:
            return False, reason
    except Exception as exc:
        return False, str(exc)

    return True, ""


def is_prims_ts_fp8_per_tensor_supported(
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return whether this FP8 per-tensor MoE call is eligible for Prims-TS."""

    hidden_states = moe_inputs.hidden_states
    if not is_prims_ts_available():
        return False, "nvidia-cutlass-dsl Prims-TS dependencies are not importable"
    if not _device_supports_prims_ts(hidden_states.device):
        return False, "requires an SM100 or SM103 CUDA device"
    if not _enum_eq(runner.dtype_act, DtypeTrtllmGen.E4m3) or not _enum_eq(
        runner.dtype_weights, DtypeTrtllmGen.E4m3
    ):
        return False, "only FP8 activations and FP8 weights are supported"
    activation_type = kwargs.get("activation_type", runner.activation_type)
    if not any(_enum_eq(activation_type, act) for act in _SUPPORTED_ACTIVATIONS):
        return False, "activation must be Identity, Swiglu, Geglu, Silu, Relu2, or Situ"
    routing_method_type = kwargs.get(
        "routing_method_type", getattr(runner, "routing_method_type", None)
    )
    if _enum_eq(routing_method_type, RoutingMethodType.Sigmoid):
        return False, "Sigmoid routing is not supported"
    if _enum_eq(routing_method_type, RoutingMethodType.DeepSeekV3) and not (
        _is_gated_activation(activation_type)
    ):
        return False, "DeepSeekV3 routing requires a gated activation"
    if not _is_supported_weight_layout(
        kwargs.get("weight_layout", runner.weight_layout)
    ):
        return False, "weight_layout must be MajorK or BlockMajorK"
    if not kwargs.get("use_shuffled_weight", runner.use_shuffled_weight):
        return False, "Prims-TS FP8 per-tensor path expects shuffled weights"
    if moe_inputs.gemm1_lora_delta is not None:
        return False, "gemm1_lora_delta is not supported"
    ok, reason = _validate_gemm1_oa_params(runner, moe_inputs, activation_type, kwargs)
    if not ok:
        return False, reason
    if moe_inputs.per_token_scale is not None or getattr(
        runner, "use_per_token_scaling", False
    ):
        return False, "per-token scaling is not supported"
    fc1_per_channel_weight_scale, fc2_per_channel_weight_scale = (
        _split_per_channel_weight_scale_from_kwargs(kwargs)
    )
    fc1_use_per_channel_weight_scale = fc1_per_channel_weight_scale is not None
    fc2_use_per_channel_weight_scale = fc2_per_channel_weight_scale is not None
    per_token_sf_dtype: int | None = None
    for name, tensor in (
        ("fc1_per_channel_weight_scale", fc1_per_channel_weight_scale),
        ("fc2_per_channel_weight_scale", fc2_per_channel_weight_scale),
    ):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            return False, f"{name} must be a torch.Tensor"
        if not tensor.is_contiguous():
            return False, f"{name} must be contiguous"
        try:
            per_token_sf_dtype = _merge_per_token_sf_dtype(
                per_token_sf_dtype,
                _per_token_sf_dtype_value(tensor),
                current_name="fc1_per_channel_weight_scale",
                candidate_name=name,
            )
        except ValueError as exc:
            return False, str(exc)

    use_routing_scales_on_input = bool(kwargs.get("use_routing_scales_on_input", False))
    if use_routing_scales_on_input:
        if not _enum_eq(kwargs.get("routing_method_type"), RoutingMethodType.Llama4):
            return False, "routing scales on input are only supported for Llama4"
        routing_logits = getattr(moe_inputs, "routing_logits", None)
        if routing_logits is None:
            return False, "routing logits are required for routing scales on input"
        try:
            per_token_sf_dtype = _merge_per_token_sf_dtype(
                per_token_sf_dtype,
                _per_token_sf_dtype_value(routing_logits),
                current_name="fc1_per_channel_weight_scale",
                candidate_name="routing_logits",
            )
        except ValueError as exc:
            return False, str(exc)
    elif kwargs.get("gemm1_bias") is not None:
        return False, "gemm1_bias is not supported"
    if kwargs.get("num_fused_shared_experts", 0):
        return False, "fused shared experts are not supported"
    if runner.intermediate_size % 128 != 0 or runner.hidden_size % 128 != 0:
        return False, "hidden_size and intermediate_size must be multiples of 128"

    required = (
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    )
    for name in required:
        if kwargs.get(name) is None:
            return False, f"{name} is required"

    try:
        from .config_mapper import map_trtllm_fp8_per_tensor_moe_tactic

        pair = map_trtllm_fp8_per_tensor_moe_tactic(
            tactic,
            activation_type=int(activation_type),
            num_tokens=int(hidden_states.shape[0]),
            top_k=getattr(runner, "top_k", None),
            num_local_experts=getattr(runner, "num_local_experts", None),
            weight_layout=_weight_layout_arg(runner, kwargs),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            fc1_use_per_token_sf_a=fc1_use_per_channel_weight_scale,
            fc2_use_per_token_sf_a=fc2_use_per_channel_weight_scale,
            use_per_token_sf_b=use_routing_scales_on_input,
            per_token_sf_dtype=per_token_sf_dtype or 1,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags(kwargs),
        )
        ok, reason = _validate_block_major_k_storage_matches_config(
            runner, kwargs, pair.fc1.cfg.build(), pair.fc2.cfg.build()
        )
        if not ok:
            return False, reason
    except Exception as exc:
        return False, str(exc)

    return True, ""


def is_prims_ts_fp8_block_scale_supported(
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return whether this FP8 block-scale MoE call is eligible for Prims-TS."""

    hidden_states = moe_inputs.hidden_states
    if not is_prims_ts_available():
        return False, "nvidia-cutlass-dsl Prims-TS dependencies are not importable"
    if not _device_supports_prims_ts(hidden_states.device):
        return False, "requires an SM100 or SM103 CUDA device"
    if hidden_states.dtype != torch.float8_e4m3fn:
        return False, "hidden_states must be float8_e4m3fn"
    if moe_inputs.hidden_states_scale is None:
        return False, "hidden_states_scale is required for FP8 block-scale"

    quantization_type = Fp8QuantizationType(
        int(getattr(runner, "fp8_quantization_type", Fp8QuantizationType.NoneFp8))
    )
    is_deepseek = quantization_type == Fp8QuantizationType.DeepSeekFp8
    is_mxfp8 = quantization_type == Fp8QuantizationType.MxFp8
    if not (is_deepseek or is_mxfp8):
        return False, "only DeepSeekFp8 and MxFp8 block-scale modes are supported"

    expected_dtype = DtypeTrtllmGen.E4m3 if is_deepseek else DtypeTrtllmGen.MxE4m3
    if not _enum_eq(runner.dtype_act, expected_dtype) or not _enum_eq(
        runner.dtype_weights, expected_dtype
    ):
        return False, "runner dtype does not match fp8_quantization_type"

    activation_type = kwargs.get("activation_type", runner.activation_type)
    if not any(_enum_eq(activation_type, act) for act in _SUPPORTED_ACTIVATIONS):
        return False, "activation must be Identity, Swiglu, Geglu, Silu, Relu2, or Situ"
    if is_deepseek and not _enum_eq(activation_type, ActivationType.Swiglu):
        return False, "DeepSeek FP8 Prims-TS integration currently exposes Swiglu only"
    if not _is_supported_weight_layout(
        kwargs.get("weight_layout", runner.weight_layout)
    ):
        return False, "weight_layout must be MajorK or BlockMajorK"
    if not kwargs.get("use_shuffled_weight", runner.use_shuffled_weight):
        return False, "Prims-TS FP8 block-scale path expects shuffled weights"
    if moe_inputs.gemm1_lora_delta is not None:
        return False, "gemm1_lora_delta is not supported"
    if moe_inputs.per_token_scale is not None or getattr(
        runner, "use_per_token_scaling", False
    ):
        return False, "per-token scaling is not supported"
    if is_deepseek and _has_gemm1_oa_params(kwargs):
        return False, "DeepSeek FP8 Prims-TS OA params are not supported"
    ok, reason = _validate_gemm1_oa_params(runner, moe_inputs, activation_type, kwargs)
    if not ok:
        return False, reason
    if kwargs.get("num_fused_shared_experts", 0):
        return False, "fused shared experts are not supported"
    if is_deepseek and (
        kwargs.get("gemm1_bias") is not None or kwargs.get("gemm2_bias") is not None
    ):
        return False, "DeepSeek FP8 Prims-TS bias is not supported"
    if runner.intermediate_size % 128 != 0 or runner.hidden_size % 128 != 0:
        return False, "hidden_size and intermediate_size must be multiples of 128"

    if is_deepseek:
        if moe_inputs.hidden_states_scale.dtype != torch.float32:
            return False, "DeepSeek FP8 hidden_states_scale must be float32"
        if (
            kwargs.get("gemm1_weights_scale") is None
            or kwargs.get("gemm2_weights_scale") is None
        ):
            return False, "DeepSeek FP8 weight scales are required"
        if kwargs["gemm1_weights_scale"].dtype != torch.float32:
            return False, "DeepSeek FP8 gemm1_weights_scale must be float32"
        if kwargs["gemm2_weights_scale"].dtype != torch.float32:
            return False, "DeepSeek FP8 gemm2_weights_scale must be float32"
        mapper_name = "map_trtllm_deepseek_fp8_moe_tactic"
    else:
        if (
            kwargs.get("gemm1_weights_scale") is None
            or kwargs.get("gemm2_weights_scale") is None
        ):
            return False, "MXFP8 weight scales are required"
        if moe_inputs.hidden_states_scale.dtype != torch.uint8:
            return False, "MXFP8 hidden_states_scale must be uint8"
        if kwargs["gemm1_weights_scale"].dtype != torch.uint8:
            return False, "MXFP8 gemm1_weights_scale must be uint8"
        if kwargs["gemm2_weights_scale"].dtype != torch.uint8:
            return False, "MXFP8 gemm2_weights_scale must be uint8"
        mapper_name = "map_trtllm_mxfp8_mxfp8_moe_tactic"

    try:
        from . import config_mapper

        mapper = getattr(config_mapper, mapper_name)
        if is_deepseek:
            pair = mapper(
                tactic,
                num_tokens=int(hidden_states.shape[0]),
                top_k=getattr(runner, "top_k", None),
                num_local_experts=getattr(runner, "num_local_experts", None),
                weight_layout=_weight_layout_arg(runner, kwargs),
                enable_pdl=bool(kwargs.get("enable_pdl", False)),
            )
        else:
            pair = mapper(
                tactic,
                activation_type=int(activation_type),
                num_tokens=int(hidden_states.shape[0]),
                top_k=getattr(runner, "top_k", None),
                num_local_experts=getattr(runner, "num_local_experts", None),
                weight_layout=_weight_layout_arg(runner, kwargs),
                fc1_has_bias=kwargs.get("gemm1_bias") is not None,
                fc2_has_bias=kwargs.get("gemm2_bias") is not None,
                enable_pdl=bool(kwargs.get("enable_pdl", False)),
                **_gemm1_oa_flags(kwargs),
            )
        ok, reason = _validate_block_major_k_storage_matches_config(
            runner, kwargs, pair.fc1.cfg.build(), pair.fc2.cfg.build()
        )
        if not ok:
            return False, reason
    except Exception as exc:
        return False, str(exc)

    return True, ""


_SUPPORT_CHECKS = {
    "bf16": is_prims_ts_bf16_supported,
    "nvfp4": is_prims_ts_nvfp4_supported,
    "mxfp4_mxfp8": is_prims_ts_mxfp4_mxfp8_supported,
    "mxfp4_bf16": is_prims_ts_mxfp4_bf16_supported,
    "fp8_per_tensor": is_prims_ts_fp8_per_tensor_supported,
    "fp8_block_scale": is_prims_ts_fp8_block_scale_supported,
}


def check_prims_ts_moe_supported(
    kind: str,
    runner: Any,
    moe_inputs: Any,
    tactic: int | Sequence[int],
    **kwargs: Any,
) -> SupportResult:
    try:
        check = _SUPPORT_CHECKS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown Prims-TS MoE support kind: {kind!r}") from exc
    ok, reason = check(runner, moe_inputs, tactic, **kwargs)
    return SupportResult(ok=ok, reason=reason)
