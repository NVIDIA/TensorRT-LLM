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

import contextlib
import weakref
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

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
from triton_kernels.tensor import FP4, RaggedTensorMetadata, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import StridedLayout

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_router import (
    DeepSeekV4ExpertParallelRoutingMetadata,
    deepseek_v4_localize_expert_ids,
)
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


@dataclass(frozen=True)
class DeepSeekV4RouteMetadata:
    """Graph-safe route metadata for ``triton_deepseek_v4_mxfp4_moe_from_routing``.

    ``selected_experts`` must already be in the coordinate system of the MXFP4
    expert weight slice: global ids when ``moe_ep == 1`` and EP-local ids when
    ``moe_ep > 1``. Use ``deepseek_v4_localize_expert_ids`` to convert router
    global ids before calling this path on EP-sharded weights. Nonlocal EP
    routes keep their ``[T, top_k]`` slots but carry routing weight 0.

    Attributes:
        selected_experts: EP-local expert ids with shape ``[T, top_k]``.
        routing_weights: Routing weights with shape ``[T, top_k]``.
        flat_expert_ids: Flattened local expert ids with shape ``[T * top_k]``.
        sorted_route_indices: Stable permutation of flattened routes by expert.
        inverse_route_indices: Inverse permutation back to original route order.
        expert_histogram: Number of route slots assigned to each local expert.
        sorted_routing_weights: Routing weights ordered by ``sorted_route_indices``.
        num_experts: Number of experts in the local MXFP4 weight slice.
        top_k: Number of route slots per token.
        num_routes: Total route slots, ``T * top_k``.
    """

    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    flat_expert_ids: torch.Tensor
    sorted_route_indices: torch.Tensor
    inverse_route_indices: torch.Tensor
    expert_histogram: torch.Tensor
    sorted_routing_weights: torch.Tensor
    num_experts: int
    top_k: int
    num_routes: int


@dataclass(frozen=True)
class DeepSeekV4RouteMetadataTensors:
    """Prebuilt matmul_ogs route metadata tensors.

    Attributes:
        sorted_routing_weights: Routing weights sorted by expert, shape ``[T * top_k]``.
        expert_histogram: Per-local-expert route counts, shape ``[E_local]``.
        sorted_route_indices: Stable gather permutation into expert-major route order.
        inverse_route_indices: Inverse permutation back to original route order.
        expert_offsets: Ragged slice offsets, shape ``[E_local + 1]``.
        expert_block_offsets: Ragged block offsets for supported matmul_ogs block sizes.
        expert_block_schedule: Ragged block schedules for supported matmul_ogs block sizes.
    """

    sorted_routing_weights: torch.Tensor
    expert_histogram: torch.Tensor
    sorted_route_indices: torch.Tensor
    inverse_route_indices: torch.Tensor
    expert_offsets: torch.Tensor
    expert_block_offsets: torch.Tensor
    expert_block_schedule: torch.Tensor

    def as_tuple(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.sorted_routing_weights,
            self.expert_histogram,
            self.sorted_route_indices,
            self.inverse_route_indices,
            self.expert_offsets,
            self.expert_block_offsets,
            self.expert_block_schedule,
        )


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


def _device_guard(tensor: torch.Tensor) -> contextlib.AbstractContextManager[None]:
    if tensor.device.type == "cuda":
        return torch.cuda.device(tensor.device)
    return contextlib.nullcontext()


def prepare_deepseek_v4_route_metadata(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> DeepSeekV4RouteMetadata:
    """Prepare stable sort, histogram, gather, and scatter metadata.

    Args:
        selected_experts: Local expert ids with shape ``[T, top_k]``. These are
            global ids only in the non-EP case where the weight slice contains
            all routed experts.
        routing_weights: Per-route weights with shape ``[T, top_k]``. For EP,
            nonlocal routes should already be zeroed by rank mask.
        num_experts: Number of experts in the current MXFP4 weight slice.

    Returns:
        ``DeepSeekV4RouteMetadata`` consumed by the matmul_ogs metadata bridge.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if selected_experts.dim() != 2:
        raise ValueError(f"selected_experts must have rank 2, got rank {selected_experts.dim()}")
    if selected_experts.shape[1] <= 0:
        raise ValueError("selected_experts must include at least one route per token")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            "routing_weights must match selected_experts shape "
            f"{tuple(selected_experts.shape)}, got {tuple(routing_weights.shape)}"
        )
    if routing_weights.device != selected_experts.device:
        raise ValueError(
            "routing_weights must be on the same device as selected_experts, got "
            f"{routing_weights.device} and {selected_experts.device}"
        )
    if selected_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_experts must be int32 or int64, got {selected_experts.dtype}")
    if not routing_weights.is_floating_point():
        raise TypeError(f"routing_weights must be floating point, got {routing_weights.dtype}")

    flat_expert_ids = selected_experts.reshape(-1).to(torch.int64)
    sorted_route_indices = torch.argsort(flat_expert_ids, stable=True).to(torch.int32)
    inverse_route_indices = torch.argsort(sorted_route_indices, stable=True).to(torch.int32)
    expert_histogram = torch.zeros(num_experts, dtype=torch.int32, device=flat_expert_ids.device)
    expert_histogram.scatter_add_(
        0,
        flat_expert_ids,
        torch.ones_like(flat_expert_ids, dtype=torch.int32),
    )
    sorted_routing_weights = routing_weights.reshape(-1)[sorted_route_indices.to(torch.int64)]

    return DeepSeekV4RouteMetadata(
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        flat_expert_ids=flat_expert_ids,
        sorted_route_indices=sorted_route_indices,
        inverse_route_indices=inverse_route_indices,
        expert_histogram=expert_histogram,
        sorted_routing_weights=sorted_routing_weights,
        num_experts=num_experts,
        top_k=selected_experts.shape[1],
        num_routes=selected_experts.numel(),
    )


def _make_deepseek_v4_ragged_metadata(
    expert_histogram: torch.Tensor,
    num_routes: int,
) -> RaggedTensorMetadata:
    if expert_histogram.device.type == "cpu":
        from triton_kernels.tensor import make_ragged_tensor_metadata_torch

        return make_ragged_tensor_metadata_torch(expert_histogram, num_routes)

    from triton_kernels.tensor import make_ragged_tensor_metadata

    return make_ragged_tensor_metadata(expert_histogram, num_routes)


def _route_metadata_to_tensors(
    route_metadata: DeepSeekV4RouteMetadata,
) -> DeepSeekV4RouteMetadataTensors:
    expert_data = _make_deepseek_v4_ragged_metadata(
        route_metadata.expert_histogram,
        route_metadata.num_routes,
    )
    return DeepSeekV4RouteMetadataTensors(
        sorted_routing_weights=route_metadata.sorted_routing_weights,
        expert_histogram=route_metadata.expert_histogram,
        sorted_route_indices=route_metadata.sorted_route_indices,
        inverse_route_indices=route_metadata.inverse_route_indices,
        expert_offsets=expert_data.slice_offs.clone(),
        expert_block_offsets=expert_data.block_offs_data.clone(),
        expert_block_schedule=expert_data.block_schedule_data.clone(),
    )


def prepare_deepseek_v4_route_metadata_tensors(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> DeepSeekV4RouteMetadataTensors:
    """Prepare tensor metadata that can be passed to the MXFP4 fast path."""
    return _route_metadata_to_tensors(
        prepare_deepseek_v4_route_metadata(
            selected_experts,
            routing_weights,
            num_experts,
        )
    )


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_mxfp4_route_metadata", mutates_args=())
def torch_deepseek_v4_mxfp4_route_metadata(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts,
    ).as_tuple()


@torch_deepseek_v4_mxfp4_route_metadata.register_fake
def _torch_deepseek_v4_mxfp4_route_metadata_fake(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_routes = selected_experts.numel()
    device = selected_experts.device
    block_size_count = len(RaggedTensorMetadata.block_sizes_log2())
    max_num_blocks = RaggedTensorMetadata.max_n_tiles(num_experts, num_routes)
    return (
        torch.empty((num_routes,), dtype=routing_weights.dtype, device=device),
        torch.empty((num_experts,), dtype=torch.int32, device=device),
        torch.empty((num_routes,), dtype=torch.int32, device=device),
        torch.empty((num_routes,), dtype=torch.int32, device=device),
        torch.empty((num_experts + 1,), dtype=torch.int32, device=device),
        torch.empty((block_size_count, num_experts + 1), dtype=torch.int32, device=device),
        torch.empty((block_size_count, max_num_blocks), dtype=torch.int32, device=device),
    )


def _routing_from_route_metadata(
    route_metadata: DeepSeekV4RouteMetadata,
) -> Tuple[RoutingData, GatherIndx, ScatterIndx]:
    return _routing_from_prebuilt_metadata(
        *_route_metadata_to_tensors(route_metadata).as_tuple(),
        num_experts=route_metadata.num_experts,
        top_k=route_metadata.top_k,
    )


def _validate_route_metadata_tensor(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _validate_route_metadata_devices(
    reference: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> None:
    mismatched = [
        f"{name}={tensor.device}"
        for name, tensor in tensors.items()
        if tensor.device != reference.device
    ]
    if mismatched:
        raise ValueError(
            "prebuilt route metadata tensors must be on the same device as "
            f"sorted_routing_weights={reference.device}, got {', '.join(mismatched)}"
        )


def _routing_from_prebuilt_metadata(
    sorted_routing_weights: torch.Tensor,
    expert_histogram: torch.Tensor,
    sorted_route_indices: torch.Tensor,
    inverse_route_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_block_offsets: torch.Tensor,
    expert_block_schedule: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
) -> Tuple[RoutingData, GatherIndx, ScatterIndx]:
    """Build matmul_ogs routing objects from prebuilt tensor metadata."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not sorted_routing_weights.is_floating_point():
        raise TypeError(
            f"sorted_routing_weights must be floating point, got {sorted_routing_weights.dtype}"
        )
    _validate_route_metadata_tensor("expert_histogram", expert_histogram, torch.int32)
    _validate_route_metadata_tensor("sorted_route_indices", sorted_route_indices, torch.int32)
    _validate_route_metadata_tensor("inverse_route_indices", inverse_route_indices, torch.int32)
    _validate_route_metadata_tensor("expert_offsets", expert_offsets, torch.int32)
    _validate_route_metadata_tensor("expert_block_offsets", expert_block_offsets, torch.int32)
    _validate_route_metadata_tensor("expert_block_schedule", expert_block_schedule, torch.int32)
    _validate_route_metadata_devices(
        sorted_routing_weights,
        {
            "expert_histogram": expert_histogram,
            "sorted_route_indices": sorted_route_indices,
            "inverse_route_indices": inverse_route_indices,
            "expert_offsets": expert_offsets,
            "expert_block_offsets": expert_block_offsets,
            "expert_block_schedule": expert_block_schedule,
        },
    )

    if sorted_routing_weights.dim() != 1:
        raise ValueError("sorted_routing_weights must have rank 1")
    if sorted_routing_weights.shape[0] % top_k != 0:
        raise ValueError(
            "sorted_routing_weights route count must be divisible by top_k, got "
            f"{sorted_routing_weights.shape[0]} routes and top_k={top_k}"
        )
    if expert_histogram.shape != (num_experts,):
        raise ValueError(
            f"expert_histogram must have shape ({num_experts},), got "
            f"{tuple(expert_histogram.shape)}"
        )
    if sorted_route_indices.shape != sorted_routing_weights.shape:
        raise ValueError(
            "sorted_route_indices must match sorted_routing_weights shape "
            f"{tuple(sorted_routing_weights.shape)}, got {tuple(sorted_route_indices.shape)}"
        )
    if inverse_route_indices.shape != sorted_route_indices.shape:
        raise ValueError(
            "inverse_route_indices must match sorted_route_indices shape "
            f"{tuple(sorted_route_indices.shape)}, got {tuple(inverse_route_indices.shape)}"
        )
    if expert_offsets.shape != (num_experts + 1,):
        raise ValueError(
            f"expert_offsets must have shape ({num_experts + 1},), got "
            f"{tuple(expert_offsets.shape)}"
        )
    if expert_block_offsets.dim() != 2 or expert_block_offsets.shape[1] != num_experts + 1:
        raise ValueError(
            "expert_block_offsets must have shape "
            f"[block_size_count, {num_experts + 1}], got {tuple(expert_block_offsets.shape)}"
        )
    block_size_count = len(RaggedTensorMetadata.block_sizes_log2())
    if expert_block_offsets.shape[0] != block_size_count:
        raise ValueError(
            "expert_block_offsets first dimension must match supported ragged block size count "
            f"{block_size_count}, got {expert_block_offsets.shape[0]}"
        )
    if expert_block_schedule.dim() != 2 or expert_block_schedule.shape[0] != block_size_count:
        raise ValueError(
            "expert_block_schedule must have shape "
            f"[{block_size_count}, max_num_blocks], got {tuple(expert_block_schedule.shape)}"
        )

    expert_data = RaggedTensorMetadata(
        expert_histogram,
        expert_offsets,
        expert_block_offsets,
        expert_block_schedule,
    )
    routing_data = RoutingData(
        gate_scal=sorted_routing_weights,
        expt_hist=expert_histogram,
        n_expts_tot=num_experts,
        n_expts_act=top_k,
        expt_data=expert_data,
    )
    gather_idx = GatherIndx(
        src_indx=sorted_route_indices,
        dst_indx=inverse_route_indices,
    )
    scatter_idx = ScatterIndx(
        src_indx=inverse_route_indices,
        dst_indx=sorted_route_indices,
    )
    return routing_data, gather_idx, scatter_idx


def _routing_from_precomputed(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> Tuple[RoutingData, GatherIndx, ScatterIndx]:
    route_metadata = prepare_deepseek_v4_route_metadata(
        selected_experts,
        routing_weights,
        num_experts,
    )
    return _routing_from_route_metadata(route_metadata)


def localize_deepseek_v4_routes_for_mxfp4(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    moe_ep_size: int,
    moe_ep_rank: int,
    num_global_experts: int,
) -> DeepSeekV4ExpertParallelRoutingMetadata:
    """Return EP-local route tensors for a local MXFP4 expert weight slice."""
    return deepseek_v4_localize_expert_ids(
        selected_experts,
        routing_weights,
        moe_ep_size=moe_ep_size,
        moe_ep_rank=moe_ep_rank,
        num_global_experts=num_global_experts,
    )


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
    with _device_guard(router_logits):
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
    sorted_routing_weights: Optional[torch.Tensor] = None,
    expert_histogram: Optional[torch.Tensor] = None,
    sorted_route_indices: Optional[torch.Tensor] = None,
    inverse_route_indices: Optional[torch.Tensor] = None,
    expert_offsets: Optional[torch.Tensor] = None,
    expert_block_offsets: Optional[torch.Tensor] = None,
    expert_block_schedule: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del layer_type

    prebuilt_metadata = (
        sorted_routing_weights,
        expert_histogram,
        sorted_route_indices,
        inverse_route_indices,
        expert_offsets,
        expert_block_offsets,
        expert_block_schedule,
    )
    with _device_guard(hidden_states):
        if any(tensor is not None for tensor in prebuilt_metadata):
            if not all(tensor is not None for tensor in prebuilt_metadata):
                raise ValueError("prebuilt route metadata must be fully specified")
            routing_data, gather_idx, scatter_idx = _routing_from_prebuilt_metadata(
                *prebuilt_metadata,
                num_experts=gate_up_blocks.shape[0],
                top_k=selected_experts.shape[1],
            )
        else:
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
    sorted_routing_weights: Optional[torch.Tensor] = None,
    expert_histogram: Optional[torch.Tensor] = None,
    sorted_route_indices: Optional[torch.Tensor] = None,
    inverse_route_indices: Optional[torch.Tensor] = None,
    expert_offsets: Optional[torch.Tensor] = None,
    expert_block_offsets: Optional[torch.Tensor] = None,
    expert_block_schedule: Optional[torch.Tensor] = None,
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
        sorted_routing_weights,
        expert_histogram,
        sorted_route_indices,
        inverse_route_indices,
        expert_offsets,
        expert_block_offsets,
        expert_block_schedule,
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
