# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch custom ops wrapping the vendored PrimsTS GEMM kernels."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .flux3_gemm_config import autotune_enabled, gemm_config
from .nvfp4_metadata import find_nvfp4_owner, nvfp4_owner

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm import PrimsTsGemmConfig
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm.api import PreparedFp4Linear


def is_sm120(device: torch.device) -> bool:
    """SM120/SM121 run FLUX.3 GEMMs through the CuTe DSL kernels instead of PrimsTS."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))


@lru_cache(maxsize=32)
def _token_positions(rows: int, device: torch.device) -> torch.Tensor:
    """Cache identity RoPE indices across blocks and denoising steps."""
    return torch.arange(rows, device=device, dtype=torch.int64)


def _version(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    # Offload can replace storage without changing tensor identity/version.
    return (
        tensor.device,
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        None if tensor.is_inference() else tensor._version,
    )


def prepare_nvfp4_gemm(
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    epilogue: str = "linear",
    output_inverse_scale: torch.Tensor | None = None,
    q_norm: torch.Tensor | None = None,
    k_norm: torch.Tensor | None = None,
    qkv_scale: torch.Tensor | None = None,
    config: PrimsTsGemmConfig | None = None,
) -> PreparedFp4Linear:
    """Cache launch descriptors on the adapter, or the weight for standalone calls."""
    versions = tuple(
        _version(t) for t in (weight, scale, output_inverse_scale, q_norm, k_norm, qkv_scale)
    )
    owner = find_nvfp4_owner(weight)
    cache_owner = weight if owner is None else owner
    cache = getattr(cache_owner, "_prepared_gemm_cache", None)
    if cache is None:
        cache = OrderedDict()
        cache_owner._prepared_gemm_cache = cache
    key = (epilogue, input_global_scale, weight_global_scale, versions, config)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    # Drop descriptors for replaced storage or changed auxiliary tensors.
    for old_key in list(cache):
        same_output = (old_key[3][2] is None) == (output_inverse_scale is None)
        if old_key[3][:2] != versions[:2] or (
            old_key[0] == epilogue and same_output and old_key[3] != versions
        ):
            del cache[old_key]
    prepared = _prepared_nvfp4_gemm(
        weight,
        scale,
        input_global_scale,
        weight_global_scale,
        epilogue,
        output_inverse_scale,
        q_norm,
        k_norm,
        qkv_scale,
        config,
    )
    cache[key] = prepared
    if len(cache) > 8:
        cache.popitem(last=False)
    return prepared


def _prepared_nvfp4_gemm(
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    epilogue: str,
    output_inverse_scale: torch.Tensor | None,
    q_norm: torch.Tensor | None,
    k_norm: torch.Tensor | None,
    qkv_scale: torch.Tensor | None,
    config: PrimsTsGemmConfig | None,
) -> PreparedFp4Linear:
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    args = (weight.detach(), scale.detach(), input_global_scale, weight_global_scale)
    if epilogue == "linear":
        return gemm.prepare_fp4_linear(*args, config=config)
    if epilogue == "swiglu":
        return gemm.prepare_fp4_linear_swiglu(
            *args,
            out_dtype=torch.bfloat16 if output_inverse_scale is None else torch.uint8,
            output_quant_scale=output_inverse_scale,
            config=config,
        )
    if epilogue == "qkv_qknorm_rope":
        if q_norm is None or k_norm is None:
            raise ValueError("QKV epilogue requires Q/K normalization weights")
        heads = weight.shape[0] // (3 * q_norm.numel())
        return gemm.prepare_fp4_qkv_qknorm_rope(
            *args,
            q_norm.detach(),
            k_norm.detach(),
            num_q_heads=heads,
            num_kv_heads=heads,
            head_dim=q_norm.numel(),
            is_neox=False,
            qkv_scale=qkv_scale,
            config=config,
        )
    raise ValueError(f"Unsupported PrimsTS epilogue: {epilogue}")


def pack_swiglu_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [gate | value] rows and their scales into [value, gate] pairs."""
    if weight.ndim != 2 or weight.shape[0] % 2 or scale.shape != (weight.shape[0],):
        raise ValueError("SwiGLU requires [2H,K] weights and [2H] row scales")
    half = weight.shape[0] // 2
    packed = torch.stack((weight[half:], weight[:half]), dim=1).reshape_as(weight)
    scales = torch.stack((scale[half:], scale[:half]), dim=1).reshape_as(scale)
    return packed.contiguous(), scales.float().contiguous()


def _gemm_config(
    x: torch.Tensor,
    weight: torch.Tensor,
    epilogue: str,
    head_dim: int | None = None,
    *,
    qkv_scale: bool = False,
    packed_output: bool = False,
) -> PrimsTsGemmConfig | None:
    major, minor = torch.cuda.get_device_capability(x.device)
    fp4 = weight.dtype == torch.uint8
    return gemm_config(
        major * 10 + minor,
        "nvfp4_e2m1" if fp4 else "fp8_e4m3",
        epilogue,
        x.shape[0],
        weight.shape[0],
        weight.shape[1] * (2 if fp4 else 1),
        "nvfp4_e2m1" if packed_output else "bf16",
        head_dim,
        qkv_scale,
        autotune_enabled(),
    )


def qkv_gemm_config(
    activation: torch.Tensor,
    weight: torch.Tensor,
    head_dim: int,
    *,
    qkv_scale: bool = False,
) -> PrimsTsGemmConfig | None:
    """Config the fused QKV GEMM runs with, None on SM120 and under autotuning.

    The config decides the output layout. With separate_qkv_output the kernel writes
    Q, K and V as three [M, N/3] planes, otherwise the output is packed [M, N].
    """
    if is_sm120(activation.device):
        return None
    return _gemm_config(activation, weight, "qkv_qknorm_rope", head_dim, qkv_scale=qkv_scale)


@torch.library.custom_op("trtllm::flux3_primsts_fp8_linear", mutates_args=(), device_types="cuda")
def fp8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 [M,K] times [N,K], with FP32 [M]/[N] scales, returning BF16 [M,N]."""
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    return gemm.fp8_linear(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        config=_gemm_config(activation, weight, "linear"),
    )


@fp8_linear.register_fake
def _fp8_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_primsts_fp8_swiglu", mutates_args=(), device_types="cuda")
def fp8_linear_swiglu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 [M,K] times paired [2H,K] weights, returning BF16 SwiGLU [M,H]."""
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    return gemm.fp8_linear_swiglu(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        config=_gemm_config(activation, weight, "swiglu"),
    )


@fp8_linear_swiglu.register_fake
def _fp8_linear_swiglu_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0] // 2), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_primsts_fp8_qkv_rope", mutates_args=(), device_types="cuda")
def fp8_qkv_qknorm_rope(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    """FP8 QKV GEMM with BF16 per-head RMSNorm and interleaved RoPE.

    Norm weights are [128], cos_sin is FP32 [M,128] (half cos, half sin),
    positions is int64 [M] or None for identity indices, and output is BF16 [M,3*heads*128].
    The kernel has fixed RMSNorm epsilon 1e-6 and equal Q/KV head counts.
    """
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    if positions is None:
        positions = _token_positions(activation.shape[0], activation.device)
    head_dim = q_norm.numel()
    heads = weight.shape[0] // (3 * head_dim)
    return gemm.fp8_qkv_qknorm_rope(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions.detach(),
        num_q_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        is_neox=False,
        config=qkv_gemm_config(activation, weight, head_dim),
    )


@fp8_qkv_qknorm_rope.register_fake
def _fp8_qkv_qknorm_rope_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def nvfp4_scale_size(rows: int, k: int) -> int:
    """Bytes in FlashInfer's padded 128x4 E4M3 block-scale layout."""
    return ((rows + 127) // 128) * 128 * ((k // 16 + 3) // 4) * 4


def pack_nvfp4_swiglu_weight(
    weight: torch.Tensor,
    linear_scales: torch.Tensor,
    *,
    interleaved_64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert native gate/up rows to PrimsTS adjacent [value, gate] pairs.

    TRT-LLM's fused CuteDSL path already uses [value64, gate64] groups;
    ordinary linears use [gate | value]. Apply the same permutation to scales.
    """
    n = weight.shape[0]
    if n % 128 or linear_scales.shape[0] != n:
        raise ValueError("NVFP4 SwiGLU requires aligned weight and block-scale rows")

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        if interleaved_64:
            return (
                tensor.reshape(n // 128, 2, 64, -1).transpose(1, 2).reshape_as(tensor).contiguous()
            )
        half = n // 2
        return torch.stack((tensor[half:], tensor[:half]), dim=1).reshape_as(tensor).contiguous()

    return pack(weight), pack(linear_scales)


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_quantize", mutates_args=(), device_types="cuda"
)
def nvfp4_quantize(
    activation: torch.Tensor, inverse_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with the checkpoint's inverse global scale, without row shuffling."""
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    data, scales = quantize(
        activation.detach(),
        inverse_scale.detach(),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    return data.view(torch.uint8).contiguous(), scales.view(torch.uint8).reshape(-1).contiguous()


@nvfp4_quantize.register_fake
def _nvfp4_quantize_fake(
    activation: torch.Tensor,
    inverse_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = activation.shape
    return (
        activation.new_empty((m, k // 2), dtype=torch.uint8),
        activation.new_empty((nvfp4_scale_size(m, k),), dtype=torch.uint8),
    )


@torch.library.custom_op("trtllm::flux3_primsts_nvfp4_linear", mutates_args=(), device_types="cuda")
def nvfp4_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    if is_sm120(activation.device):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

        return flux3_sm120_nvfp4.run(
            "linear",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        config=_gemm_config(activation, weight, "linear"),
    )
    return prepared(activation.detach(), activation_scale.detach())


@nvfp4_linear.register_fake
def _nvfp4_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_primsts_nvfp4_swiglu", mutates_args=(), device_types="cuda")
def nvfp4_linear_swiglu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    if is_sm120(activation.device):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

        return flux3_sm120_nvfp4.run(
            "swiglu",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        "swiglu",
        config=_gemm_config(activation, weight, "swiglu"),
    )
    return prepared(activation.detach(), activation_scale.detach())


@nvfp4_linear_swiglu.register_fake
def _nvfp4_swiglu_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0] // 2), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_swiglu_packed", mutates_args=(), device_types="cuda"
)
def nvfp4_swiglu_packed(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    output_inverse_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU with fused NVFP4 output using the next projection's static scale."""
    if is_sm120(activation.device):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

        return flux3_sm120_nvfp4.run(
            "packed",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            output_inverse_scale=output_inverse_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        "swiglu",
        output_inverse_scale,
        config=_gemm_config(activation, weight, "swiglu", packed_output=True),
    )
    return prepared(activation.detach(), activation_scale.detach())


@nvfp4_swiglu_packed.register_fake
def _nvfp4_swiglu_packed_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    output_inverse_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = activation.shape[0], weight.shape[0] // 2
    return activation.new_empty((m, n // 2)), activation.new_empty((nvfp4_scale_size(m, n),))


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_qkv_rope", mutates_args=(), device_types="cuda"
)
def nvfp4_qkv_qknorm_rope(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    qkv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if positions is None:
        positions = _token_positions(activation.shape[0], activation.device)

    if is_sm120(activation.device):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

        return flux3_sm120_nvfp4.run(
            "qkv",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_sin=cos_sin,
            positions=positions,
            qkv_scale=qkv_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        "qkv_qknorm_rope",
        q_norm=q_norm,
        k_norm=k_norm,
        qkv_scale=qkv_scale,
        config=qkv_gemm_config(activation, weight, q_norm.numel(), qkv_scale=qkv_scale is not None),
    )
    return prepared(
        activation.detach(),
        activation_scale.detach(),
        cos_sin=cos_sin.detach(),
        positions=positions.detach(),
    )


@nvfp4_qkv_qknorm_rope.register_fake
def _nvfp4_qkv_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    qkv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_nvfp4_quantize_prepared", mutates_args=())
def nvfp4_quantize_prepared(
    activation: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a host scale without exposing a per-layer float to Dynamo."""
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    data, scales = quantize(
        activation.detach(),
        1.0 / nvfp4_owner(weight).input_global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    return data.view(torch.uint8).contiguous(), scales.view(torch.uint8).reshape(-1).contiguous()


@nvfp4_quantize_prepared.register_fake
def _nvfp4_quantize_prepared_fake(activation, weight):
    return _nvfp4_quantize_fake(activation, weight)


def _prepared_activation(activation, activation_scale, original, source_weight, weight):
    if (
        source_weight is not None
        and nvfp4_owner(source_weight).input_global_scale != nvfp4_owner(weight).input_global_scale
    ):
        return nvfp4_quantize_prepared(
            original.reshape(-1, original.shape[-1]).contiguous(), weight
        )
    return activation, activation_scale


@torch.library.custom_op("trtllm::flux3_nvfp4_linear_prepared", mutates_args=())
def nvfp4_linear_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    original: torch.Tensor | None = None,
    source_weight: torch.Tensor | None = None,
    swiglu: bool = False,
) -> torch.Tensor:
    owner = nvfp4_owner(weight)
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    op = nvfp4_linear_swiglu if swiglu else nvfp4_linear
    return op(
        activation,
        weight,
        activation_scale,
        weight_scale,
        owner.input_global_scale,
        owner.weight_global_scale,
    )


@nvfp4_linear_prepared.register_fake
def _nvfp4_linear_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    original=None,
    source_weight=None,
    swiglu=False,
):
    return activation.new_empty(
        (activation.shape[0], weight.shape[0] // (2 if swiglu else 1)), dtype=torch.bfloat16
    )


@torch.library.custom_op("trtllm::flux3_nvfp4_swiglu_prepared", mutates_args=())
def nvfp4_swiglu_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_inverse_scale: torch.Tensor,
    original: torch.Tensor,
    source_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner = nvfp4_owner(weight)
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    return nvfp4_swiglu_packed(
        activation,
        weight,
        activation_scale,
        weight_scale,
        owner.input_global_scale,
        owner.weight_global_scale,
        output_inverse_scale,
    )


@nvfp4_swiglu_prepared.register_fake
def _nvfp4_swiglu_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    output_inverse_scale,
    original,
    source_weight=None,
):
    return _nvfp4_swiglu_packed_fake(
        activation, weight, activation_scale, weight_scale, 1.0, 1.0, output_inverse_scale
    )


@torch.library.custom_op("trtllm::flux3_nvfp4_qkv_prepared", mutates_args=())
def nvfp4_qkv_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    qkv_scale: torch.Tensor | None,
    original: torch.Tensor,
    source_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    owner = nvfp4_owner(weight)
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    return nvfp4_qkv_qknorm_rope(
        activation,
        weight,
        activation_scale,
        weight_scale,
        owner.input_global_scale,
        owner.weight_global_scale,
        q_norm,
        k_norm,
        cos_sin,
        None,
        qkv_scale,
    )


@nvfp4_qkv_prepared.register_fake
def _nvfp4_qkv_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    q_norm,
    k_norm,
    cos_sin,
    qkv_scale,
    original,
    source_weight=None,
):
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)
