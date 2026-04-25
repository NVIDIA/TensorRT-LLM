# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton-kernels-based MXFP4 MoE ops (GPT-OSS style) with routing, swizzling, and fused activation

import weakref
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton_kernels.matmul_ogs import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    GatherIndx,
    PrecisionConfig,
    RoutingData,
    ScatterIndx,
    matmul_ogs,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import StridedLayout

from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonEPRouter


# copied from transformers.integrations.mxfp4::swizzle_mxfp4 with minor modification
def _swizzle_mxfp4(w, w_scale):
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


@triton.jit
def _deepseek_v4_swiglu_fn(input, alpha, limit):
    gate, up = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    if limit is not None:
        gate = tl.minimum(gate, limit)
        up = tl.clamp(up, -limit, limit)
    return gate / (1 + tl.exp(-alpha * gate)) * up


def _deepseek_v4_swiglu_torch(input: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = input[..., 0::2].float()
    up = input[..., 1::2].float()
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * up).to(input.dtype)


def _interleave_deepseek_v4_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    """Convert DeepSeek checkpoint order [w3, w1] to activation lane order [w1, w3]."""
    if tensor.dim() < 2:
        raise ValueError(f"gate/up tensor must have rank at least 2, got rank {tensor.dim()}")
    if tensor.shape[1] % 2 != 0:
        raise ValueError(f"gate/up dimension must be even, got {tensor.shape[1]}")

    intermediate_size = tensor.shape[1] // 2
    up = tensor[:, :intermediate_size]
    gate = tensor[:, intermediate_size:]
    return torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()


RouteFn = Callable[[torch.Tensor], Tuple[RoutingData, GatherIndx, ScatterIndx]]

_TensorCacheKey = tuple[
    int,
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype,
    str,
    int | None,
    torch.layout,
    int | None,
]
_PrepareCacheKey = tuple[
    int, bool, _TensorCacheKey, _TensorCacheKey, _TensorCacheKey, _TensorCacheKey
]
_PreparedWeightsScales = tuple[object, object, object, object]
_TensorRef = weakref.ReferenceType[torch.Tensor] | None
_PrepareCacheEntry = tuple[
    tuple[_TensorRef, _TensorRef, _TensorRef, _TensorRef], _PreparedWeightsScales
]
_PREPARED_WEIGHTS_SCALES_CACHE: dict[_PrepareCacheKey, _PrepareCacheEntry] = {}
_INFERENCE_TENSOR_VERSION_ERROR = "Inference tensors do not track version counter."


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError as error:
        if str(error) == _INFERENCE_TENSOR_VERSION_ERROR:
            return None
        raise


def _tensor_cache_key(tensor: torch.Tensor) -> _TensorCacheKey:
    storage = tensor.untyped_storage()
    device = tensor.device
    return (
        id(tensor),
        storage.data_ptr(),
        storage.nbytes(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        device.type,
        device.index,
        tensor.layout,
        _tensor_version(tensor),
    )


def _prepare_cache_key(
    hidden_size: int,
    interleave_gate_up: bool,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
) -> _PrepareCacheKey:
    return (
        hidden_size,
        interleave_gate_up,
        _tensor_cache_key(gate_up_blocks),
        _tensor_cache_key(gate_up_scales),
        _tensor_cache_key(down_blocks),
        _tensor_cache_key(down_scales),
    )


def _make_tensor_ref(tensor: torch.Tensor) -> _TensorRef:
    try:
        return weakref.ref(tensor)
    except TypeError:
        return None


def _cache_entry_matches(
    tensor_refs: tuple[_TensorRef, _TensorRef, _TensorRef, _TensorRef],
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    return all(
        tensor_ref is None or tensor_ref() is tensor
        for tensor_ref, tensor in zip(tensor_refs, tensors)
    )


def _remove_prepare_cache_entry(cache_key: _PrepareCacheKey) -> None:
    _PREPARED_WEIGHTS_SCALES_CACHE.pop(cache_key, None)


def _register_cache_finalizers(
    cache_key: _PrepareCacheKey,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for tensor in tensors:
        try:
            weakref.finalize(tensor, _remove_prepare_cache_entry, cache_key)
        except TypeError:
            pass


def _clear_mxfp4_weights_scales_cache() -> None:
    _PREPARED_WEIGHTS_SCALES_CACHE.clear()


def _routing_from_precomputed(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> Tuple[RoutingData, GatherIndx, ScatterIndx]:
    from triton_kernels.tensor import make_ragged_tensor_metadata

    flat_experts = selected_experts.reshape(-1).to(torch.int64)
    col_sorted_indx = torch.argsort(flat_experts, stable=True).to(torch.int32)
    row_sorted_indx = torch.argsort(col_sorted_indx, stable=True).to(torch.int32)
    col_sum = torch.zeros(num_experts, dtype=torch.int32, device=flat_experts.device)
    col_sum.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=torch.int32))
    expt_data = make_ragged_tensor_metadata(
        col_sum, routing_weights.shape[0] * routing_weights.shape[1]
    )
    gate_scal_sorted = routing_weights.reshape(-1)[col_sorted_indx.to(torch.int64)]
    routing_data = RoutingData(
        gate_scal=gate_scal_sorted,
        expt_hist=col_sum,
        n_expts_tot=num_experts,
        n_expts_act=routing_weights.shape[1],
        expt_data=expt_data,
    )
    gather_idx = GatherIndx(
        src_indx=col_sorted_indx,
        dst_indx=row_sorted_indx,
    )
    scatter_idx = ScatterIndx(
        src_indx=row_sorted_indx,
        dst_indx=col_sorted_indx,
    )
    return routing_data, gather_idx, scatter_idx


def _prepare_weights_scales(
    hidden_size: int,
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in unit8
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in unit8
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
    *,
    interleave_gate_up: bool = False,
) -> _PreparedWeightsScales:
    cache_tensors = (gate_up_blocks, gate_up_scales, down_blocks, down_scales)
    cache_key = _prepare_cache_key(
        hidden_size,
        interleave_gate_up,
        gate_up_blocks,
        gate_up_scales,
        down_blocks,
        down_scales,
    )
    cache_entry = _PREPARED_WEIGHTS_SCALES_CACHE.get(cache_key)
    if cache_entry is not None:
        tensor_refs, cached_weights_scales = cache_entry
        if _cache_entry_matches(tensor_refs, cache_tensors):
            return cached_weights_scales
        del _PREPARED_WEIGHTS_SCALES_CACHE[cache_key]

    local_experts = gate_up_blocks.size(0)
    intermediate_size = gate_up_blocks.shape[1] // 2
    if interleave_gate_up:
        gate_up_blocks = _interleave_deepseek_v4_gate_up(gate_up_blocks)
        gate_up_scales = _interleave_deepseek_v4_gate_up(gate_up_scales)

    # canon shapes for swizzling (use last two dims as [K, N] style)
    gate_up_blocks = gate_up_blocks.view(local_experts, intermediate_size * 2, -1)
    triton_gate_up_w, gate_up_w_scale_raw = _swizzle_mxfp4(
        gate_up_blocks.transpose(-2, -1), gate_up_scales.transpose(-2, -1)
    )
    triton_gate_up_w.shape = torch.Size([local_experts, hidden_size, intermediate_size * 2])

    down_blocks = down_blocks.view(local_experts, -1, intermediate_size // 2)
    triton_down_w, down_w_scale_raw = _swizzle_mxfp4(
        down_blocks.transpose(-2, -1), down_scales.transpose(-2, -1)
    )
    triton_down_w.shape = torch.Size([local_experts, intermediate_size, hidden_size])

    prepared_weights_scales = (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    )
    _PREPARED_WEIGHTS_SCALES_CACHE[cache_key] = (
        tuple(_make_tensor_ref(tensor) for tensor in cache_tensors),
        prepared_weights_scales,
    )
    _register_cache_finalizers(cache_key, cache_tensors)
    return prepared_weights_scales


def _run_mxfp4_mlp_core(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    route_fn: RouteFn,  # injects routing variant
) -> torch.Tensor:
    """
    Shared core for both triton_mxfp4_moe and triton_mxfp4_moe_ep.
    - route_fn encapsulates the only difference: how we produce (routing_data, gather_idx, scatter_idx).
    """
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)
    router_logits = F.linear(x, router_weight, router_bias)
    # route (global vs EP-aware)
    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = route_fn(router_logits)

    return _run_mxfp4_mlp_core_from_routing(
        hidden_states,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        routing_data,
        gather_idx,
        scatter_idx,
    )


def _run_mxfp4_mlp_core_from_routing(
    hidden_states: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    routing_data: RoutingData,
    gather_idx: GatherIndx,
    scatter_idx: ScatterIndx,
    *,
    activation_fn=swiglu_fn,
    activation_name: str = "swiglu",
    interleave_gate_up: bool = False,
) -> torch.Tensor:
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)

    (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    ) = _prepare_weights_scales(
        hidden_size,
        gate_up_blocks,
        gate_up_scales,
        down_blocks,
        down_scales,
        interleave_gate_up=interleave_gate_up,
    )
    if interleave_gate_up:
        gate_up_bias = _interleave_deepseek_v4_gate_up(gate_up_bias)

    gate_pc = PrecisionConfig(
        weight_scale=gate_up_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )
    down_pc = PrecisionConfig(
        weight_scale=down_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )

    act = FusedActivation(
        FnSpecs(activation_name, activation_fn, ("alpha", "limit"), reduction_n=2),
        (float(alpha), float(limit)),
    )

    # gate_up (with SWiGLU fused)
    inter = matmul_ogs(
        x,
        triton_gate_up_w,
        gate_up_bias.to(torch.float32),
        routing_data,
        gather_indx=gather_idx,
        precision_config=gate_pc,
        gammas=None,
        fused_activation=act,
    )

    # down
    y = matmul_ogs(
        inter,
        triton_down_w,
        down_bias.to(torch.float32),
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=down_pc,
        gammas=routing_data.gate_scal,
    )

    y = y.reshape(*leading_shape, hidden_size)
    return y


@torch.library.custom_op("auto_deploy::triton_mxfp4_moe", mutates_args=())
def triton_mxfp4_moe(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router
    router_weight: torch.Tensor,  # [E, H]
    router_bias: torch.Tensor,  # [E]
    top_k: int,
    # gate_up path
    gate_up_blocks: torch.Tensor,  # [E, 2I, H//32, 16] in unit8
    gate_up_bias: torch.Tensor,  # [E, 2I]
    gate_up_scales: torch.Tensor,  # [E, 2I, H//32] in unit8
    alpha: float,
    limit: float,
    # down path
    down_blocks: torch.Tensor,  # [E, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E, H]
    down_scales: torch.Tensor,  # [E, H, I//32] in uint8
    layer_type: str = "moe",
) -> torch.Tensor:
    def _global_route_fn(logits: torch.Tensor):
        # routing() removed in triton_kernels 3.6.0
        # TritonEPRouter(ep=1) is equivalent
        return TritonEPRouter()(logits, top_k)

    return _run_mxfp4_mlp_core(
        hidden_states,
        router_weight,
        router_bias,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        route_fn=_global_route_fn,
    )


@triton_mxfp4_moe.register_fake
def _mxfp4_mlp_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_mxfp4_moe_from_routing", mutates_args=())
def triton_deepseek_v4_mxfp4_moe_from_routing(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    layer_type: str = "moe",
) -> torch.Tensor:
    del layer_type

    with torch.cuda.device(hidden_states.device):
        routing_data, gather_idx, scatter_idx = _routing_from_precomputed(
            selected_experts, routing_weights, gate_up_blocks.shape[0]
        )

    return _run_mxfp4_mlp_core_from_routing(
        hidden_states,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        routing_data,
        gather_idx,
        scatter_idx,
        activation_fn=_deepseek_v4_swiglu_fn,
        activation_name="deepseek_v4_swiglu",
        interleave_gate_up=True,
    )


@triton_deepseek_v4_mxfp4_moe_from_routing.register_fake
def _deepseek_v4_mxfp4_moe_from_routing_fake(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    layer_type: str = "moe",
) -> torch.Tensor:
    del (
        selected_experts,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        layer_type,
    )
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::triton_mxfp4_moe_ep", mutates_args=())
def triton_mxfp4_moe_ep(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router (replicated across EP)
    router_weight: torch.Tensor,  # [E_total, H]
    router_bias: torch.Tensor,  # [E_total]
    top_k: int,
    # expert params (already sharded along dim 0)
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in unit8
    gate_up_bias: torch.Tensor,  # [E_local, 2I]
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in unit8
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E_local, H]
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
    # EP topology
    ep_size: int,
    ep_rank: int,
    layer_type: str = "moe",
) -> torch.Tensor:
    triton_ep_router = TritonEPRouter()

    def _ep_route_fn(logits: torch.Tensor):
        return triton_ep_router(logits, top_k, ep=ep_size, node_idx=ep_rank)

    return _run_mxfp4_mlp_core(
        hidden_states,
        router_weight,
        router_bias,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        route_fn=_ep_route_fn,
    )


@triton_mxfp4_moe_ep.register_fake
def _mxfp4_mlp_ep_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    ep_size: int,
    ep_rank: int,
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)
