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

"""Packed DeepSeek V4 MXFP4 routed expert loading helpers.

DeepSeek V4 stores routed expert checkpoint weights as signed I8 tensors that
contain two packed FP4 values per byte. The tensors are already quantized; this
module only validates, reinterprets, blocks, and stacks their raw bytes for the
AutoDeploy MXFP4 MoE backend.

The stable output layout is:

* ``gate_up_blocks``: ``[E_local, 2 * I, H / 32, 16]`` ``torch.uint8``
* ``gate_up_scales``: ``[E_local, 2 * I, H / 32]`` ``torch.uint8``
* ``down_blocks``: ``[E_local, H, I / 32, 16]`` ``torch.uint8``
* ``down_scales``: ``[E_local, H, I / 32]`` ``torch.uint8``

``H`` is the model hidden size and ``I`` is the MoE intermediate size. The
gate/up tensors are stacked in runtime order ``[w3, w1]`` along the ``2 * I``
dimension, matching the existing fused MoE and ``triton_mxfp4_moe`` contract.
E8M0 scales are preserved as exponent bytes with ``view(torch.uint8)`` via the
shared E8M0 helper; they are never numerically converted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from ...utils.e8m0 import e8m0_to_uint8

DEEPSEEK_V4_MXFP4_BLOCK_SIZE = 32
DEEPSEEK_V4_MXFP4_PACKED_VALUES_PER_BYTE = 2
DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK = (
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE // DEEPSEEK_V4_MXFP4_PACKED_VALUES_PER_BYTE
)

DeepSeekV4ExpertProj = Literal["w1", "w2", "w3"]
DeepSeekV4ExpertSuffix = Literal["weight", "scale"]

_EXPERT_KEY_RE = re.compile(
    r"^layers\.(?P<layer_idx>\d+)\.ffn\.experts\.(?P<expert_idx>\d+)\."
    r"(?P<proj>w[123])\.(?P<suffix>weight|scale)$"
)

__all__ = [
    "DEEPSEEK_V4_MXFP4_BLOCK_SIZE",
    "DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK",
    "DeepSeekV4ExpertKey",
    "DeepSeekV4MXFP4Layout",
    "DeepSeekV4MXFP4LoaderError",
    "expert_parallel_slice",
    "load_deepseek_v4_mxfp4_experts",
    "parse_deepseek_v4_expert_key",
    "slice_deepseek_v4_mxfp4_experts",
]


class DeepSeekV4MXFP4LoaderError(ValueError):
    """Raised when DeepSeek V4 packed MXFP4 expert tensors are inconsistent."""


@dataclass(frozen=True)
class DeepSeekV4ExpertKey:
    """Parsed DeepSeek V4 per-expert checkpoint key components."""

    layer_idx: int
    expert_idx: int
    proj: DeepSeekV4ExpertProj
    suffix: DeepSeekV4ExpertSuffix


@dataclass(frozen=True)
class DeepSeekV4MXFP4Layout:
    """Backend-ready packed routed expert tensors.

    Attributes:
        gate_up_blocks: Packed FP4 bytes in ``[E_local, 2 * I, H / 32, 16]`` layout.
            The ``2 * I`` dimension is ordered as ``[w3, w1]``.
        gate_up_scales: Raw E8M0 exponent bytes in ``[E_local, 2 * I, H / 32]`` layout,
            ordered as ``[w3, w1]``.
        down_blocks: Packed FP4 bytes in ``[E_local, H, I / 32, 16]`` layout.
        down_scales: Raw E8M0 exponent bytes in ``[E_local, H, I / 32]`` layout.
        expert_indices: Global expert ids represented by the local expert dimension.
        hidden_size: Model hidden size ``H``.
        intermediate_size: MoE intermediate size ``I``.
    """

    gate_up_blocks: torch.Tensor
    gate_up_scales: torch.Tensor
    down_blocks: torch.Tensor
    down_scales: torch.Tensor
    expert_indices: tuple[int, ...]
    hidden_size: int
    intermediate_size: int


def parse_deepseek_v4_expert_key(name: str) -> DeepSeekV4ExpertKey | None:
    """Parse a DeepSeek V4 routed expert checkpoint key.

    Args:
        name: Candidate checkpoint key, for example
            ``layers.3.ffn.experts.42.w1.weight``.

    Returns:
        Parsed key components, or ``None`` when ``name`` is not a routed expert
        ``w1``/``w2``/``w3`` weight or scale key.
    """
    match = _EXPERT_KEY_RE.match(name)
    if match is None:
        return None
    return DeepSeekV4ExpertKey(
        layer_idx=int(match.group("layer_idx")),
        expert_idx=int(match.group("expert_idx")),
        proj=cast(DeepSeekV4ExpertProj, match.group("proj")),
        suffix=cast(DeepSeekV4ExpertSuffix, match.group("suffix")),
    )


def load_deepseek_v4_mxfp4_experts(
    state_dict: Mapping[str, torch.Tensor],
    *,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int | None = None,
    expert_indices: Sequence[int] | None = None,
) -> DeepSeekV4MXFP4Layout:
    """Load and stack packed MXFP4 routed expert tensors for one DeepSeek V4 layer.

    Args:
        state_dict: Checkpoint tensors keyed by the DeepSeek V4 HF names
            ``layers.{layer_idx}.ffn.experts.{expert_idx}.w{1,2,3}.{weight,scale}``.
        layer_idx: Decoder layer index to load.
        hidden_size: Model hidden size ``H``. Must be divisible by 32.
        intermediate_size: MoE intermediate size ``I``. Must be divisible by 32.
        num_experts: Optional total routed expert count. When provided without
            ``expert_indices``, the loader requires ``range(num_experts)``.
        expert_indices: Optional explicit global expert ids to load and stack,
            in output order.

    Returns:
        ``DeepSeekV4MXFP4Layout`` containing uint8 packed blocks and raw scale bytes.
    """
    _validate_model_dims(hidden_size, intermediate_size)
    experts = _resolve_expert_indices(state_dict, layer_idx, num_experts, expert_indices)

    gate_up_block_list = []
    gate_up_scale_list = []
    down_block_list = []
    down_scale_list = []
    for expert_idx in experts:
        w3_blocks = _load_weight_blocks(
            state_dict, layer_idx, expert_idx, "w3", intermediate_size, hidden_size
        )
        w1_blocks = _load_weight_blocks(
            state_dict, layer_idx, expert_idx, "w1", intermediate_size, hidden_size
        )
        w3_scales = _load_scale_bytes(
            state_dict, layer_idx, expert_idx, "w3", intermediate_size, hidden_size
        )
        w1_scales = _load_scale_bytes(
            state_dict, layer_idx, expert_idx, "w1", intermediate_size, hidden_size
        )

        down_blocks = _load_weight_blocks(
            state_dict, layer_idx, expert_idx, "w2", hidden_size, intermediate_size
        )
        down_scales = _load_scale_bytes(
            state_dict, layer_idx, expert_idx, "w2", hidden_size, intermediate_size
        )

        gate_up_block_list.append(torch.cat((w3_blocks, w1_blocks), dim=0).contiguous())
        gate_up_scale_list.append(torch.cat((w3_scales, w1_scales), dim=0).contiguous())
        down_block_list.append(down_blocks.contiguous())
        down_scale_list.append(down_scales.contiguous())

    return DeepSeekV4MXFP4Layout(
        gate_up_blocks=torch.stack(gate_up_block_list, dim=0).contiguous(),
        gate_up_scales=torch.stack(gate_up_scale_list, dim=0).contiguous(),
        down_blocks=torch.stack(down_block_list, dim=0).contiguous(),
        down_scales=torch.stack(down_scale_list, dim=0).contiguous(),
        expert_indices=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def expert_parallel_slice(num_experts: int, ep_size: int, ep_rank: int) -> slice:
    """Return the contiguous expert slice owned by an expert-parallel rank.

    The partitioning mirrors the existing ``triton_mxfp4_moe`` sharding path:
    every rank receives ``num_experts // ep_size`` experts, and the last rank
    receives any remainder.
    """
    if num_experts < 0:
        raise DeepSeekV4MXFP4LoaderError(f"num_experts must be non-negative, got {num_experts}.")
    if ep_size <= 0:
        raise DeepSeekV4MXFP4LoaderError(f"ep_size must be positive, got {ep_size}.")
    if ep_rank < 0 or ep_rank >= ep_size:
        raise DeepSeekV4MXFP4LoaderError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}.")

    base = num_experts // ep_size
    start = base * ep_rank
    stop = num_experts if ep_rank == ep_size - 1 else base * (ep_rank + 1)
    return slice(start, stop)


def slice_deepseek_v4_mxfp4_experts(
    layout: DeepSeekV4MXFP4Layout,
    *,
    ep_size: int,
    ep_rank: int,
) -> DeepSeekV4MXFP4Layout:
    """Slice a stacked MXFP4 expert layout along the expert dimension only."""
    expert_slice = expert_parallel_slice(len(layout.expert_indices), ep_size, ep_rank)
    return DeepSeekV4MXFP4Layout(
        gate_up_blocks=layout.gate_up_blocks[expert_slice].contiguous(),
        gate_up_scales=layout.gate_up_scales[expert_slice].contiguous(),
        down_blocks=layout.down_blocks[expert_slice].contiguous(),
        down_scales=layout.down_scales[expert_slice].contiguous(),
        expert_indices=layout.expert_indices[expert_slice],
        hidden_size=layout.hidden_size,
        intermediate_size=layout.intermediate_size,
    )


def _validate_model_dims(hidden_size: int, intermediate_size: int) -> None:
    for name, dim in (("hidden_size", hidden_size), ("intermediate_size", intermediate_size)):
        if dim <= 0:
            raise DeepSeekV4MXFP4LoaderError(f"{name} must be positive, got {dim}.")
        if dim % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
            raise DeepSeekV4MXFP4LoaderError(
                f"{name} must be divisible by {DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, got {dim}."
            )


def _resolve_expert_indices(
    state_dict: Mapping[str, torch.Tensor],
    layer_idx: int,
    num_experts: int | None,
    expert_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if expert_indices is not None:
        experts = tuple(int(expert_idx) for expert_idx in expert_indices)
    elif num_experts is not None:
        if num_experts <= 0:
            raise DeepSeekV4MXFP4LoaderError(f"num_experts must be positive, got {num_experts}.")
        experts = tuple(range(num_experts))
    else:
        experts = tuple(
            sorted(
                {
                    parsed.expert_idx
                    for name in state_dict
                    if (parsed := parse_deepseek_v4_expert_key(name)) is not None
                    and parsed.layer_idx == layer_idx
                }
            )
        )

    if not experts:
        raise DeepSeekV4MXFP4LoaderError(
            f"No DeepSeek V4 MXFP4 routed experts found for layer {layer_idx}."
        )
    if len(set(experts)) != len(experts):
        raise DeepSeekV4MXFP4LoaderError(f"Duplicate expert indices requested: {experts}.")
    if any(expert_idx < 0 for expert_idx in experts):
        raise DeepSeekV4MXFP4LoaderError(f"Expert indices must be non-negative: {experts}.")
    if num_experts is not None and any(expert_idx >= num_experts for expert_idx in experts):
        raise DeepSeekV4MXFP4LoaderError(
            f"Expert indices {experts} are out of range for num_experts={num_experts}."
        )
    return experts


def _expert_tensor_key(
    layer_idx: int, expert_idx: int, proj: DeepSeekV4ExpertProj, suffix: DeepSeekV4ExpertSuffix
) -> str:
    return f"layers.{layer_idx}.ffn.experts.{expert_idx}.{proj}.{suffix}"


def _load_weight_blocks(
    state_dict: Mapping[str, torch.Tensor],
    layer_idx: int,
    expert_idx: int,
    proj: DeepSeekV4ExpertProj,
    rows: int,
    logical_cols: int,
) -> torch.Tensor:
    key = _expert_tensor_key(layer_idx, expert_idx, proj, "weight")
    tensor = _require_tensor(state_dict, key)
    expected_shape = (rows, logical_cols // DEEPSEEK_V4_MXFP4_PACKED_VALUES_PER_BYTE)
    _check_shape(key, tensor, expected_shape)

    if tensor.dtype == torch.int8:
        raw = tensor.view(torch.uint8)
    elif tensor.dtype == torch.uint8:
        raw = tensor
    else:
        raise DeepSeekV4MXFP4LoaderError(
            f"{key} must be packed torch.int8 or torch.uint8 bytes, got {tensor.dtype}."
        )

    return raw.contiguous().view(
        rows,
        logical_cols // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK,
    )


def _load_scale_bytes(
    state_dict: Mapping[str, torch.Tensor],
    layer_idx: int,
    expert_idx: int,
    proj: DeepSeekV4ExpertProj,
    rows: int,
    logical_cols: int,
) -> torch.Tensor:
    key = _expert_tensor_key(layer_idx, expert_idx, proj, "scale")
    tensor = _require_tensor(state_dict, key)
    expected_shape = (rows, logical_cols // DEEPSEEK_V4_MXFP4_BLOCK_SIZE)
    _check_shape(key, tensor, expected_shape)

    if tensor.dtype == torch.uint8:
        raw = tensor
    else:
        try:
            raw = e8m0_to_uint8(tensor)
        except (RuntimeError, TypeError) as err:
            raise DeepSeekV4MXFP4LoaderError(
                f"{key} must be torch.float8_e8m0fnu or raw torch.uint8 scale bytes, "
                f"got {tensor.dtype}."
            ) from err
    return raw.contiguous()


def _require_tensor(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    tensor = state_dict.get(key)
    if tensor is None:
        raise DeepSeekV4MXFP4LoaderError(f"Missing DeepSeek V4 MXFP4 tensor: {key}.")
    return tensor


def _check_shape(key: str, tensor: torch.Tensor, expected_shape: tuple[int, int]) -> None:
    actual_shape = tuple(int(dim) for dim in tensor.shape)
    if actual_shape != expected_shape:
        raise DeepSeekV4MXFP4LoaderError(
            f"{key} has shape {list(actual_shape)}, expected {list(expected_shape)}."
        )
