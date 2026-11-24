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
"""What the kernels of more than one CuteDSL implementation take.

Free functions taking the implementation, so a leaf binds or calls them
without inheriting from another kernel's root.
"""

from dataclasses import dataclass

import torch

from tensorrt_llm.math_utils import ceil_div

from ....utils import Fp4QuantizedTensor
from ..impl_base import MoEImplBase
from ..impl_contract import MoERunContext


@dataclass
class NvFp4WeightView:
    """One layer's NVFP4 expert weights and scales, passed to the runner as one object."""

    w3_w1_weight: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc1_global_scale: torch.Tensor
    w2_weight: torch.Tensor
    fc2_weight_scale: torch.Tensor
    fc2_global_scale: torch.Tensor
    expert_size_per_partition: int
    slot_start: int


def presorted_routing(
    token_selected_experts: torch.Tensor, local_expert_offset: int, local_num_experts: int
) -> tuple[int, tuple[torch.Tensor, ...]] | None:
    """``(tile_size, moe_sort outputs)`` a router attached for this rank, or ``None``.

    Routers that also sort (the Kimi-K3 routing tail) attach
    ``((local_expert_offset, local_num_experts, tile_size), outputs)`` to the
    top-k ids as ``_k3_presorted``, where ``outputs`` are the six tensors
    ``moe_sort`` returns for that expert range and tile size.

    Args:
        token_selected_experts: Top-k expert ids ``[num_tokens, top_k]``.
        local_expert_offset: First expert of this rank.
        local_num_experts: Experts of this rank.

    Returns:
        ``(tile_size, outputs)`` when the attached tables cover this rank's
        experts, else ``None``.
    """
    presorted = getattr(token_selected_experts, "_k3_presorted", None)
    if presorted is None:
        return None
    (offset, num_local, tile_size), outputs = presorted
    if (offset, num_local) != (local_expert_offset, local_num_experts):
        return None
    return tile_size, outputs


def sort_routing(
    backend: MoEImplBase,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor | None,
    weight_view: NvFp4WeightView,
    tile_size: int,
) -> tuple[torch.Tensor, ...]:
    """``moe_sort`` outputs of this rank's experts for ``tile_size``.

    Uses the outputs a router attached to ``token_selected_experts`` (see
    :func:`presorted_routing`) when they match this rank and tile size, else
    runs ``moe_sort``.

    Args:
        backend: The leaf; read for ``num_slots``.
        token_selected_experts: Top-k expert ids ``[num_tokens, top_k]``.
        token_final_scales: ``[num_tokens, top_k]`` float32 routing weights.
        weight_view: This rank's experts (``slot_start``, ``expert_size_per_partition``).
        tile_size: Rows per routing tile.

    Returns:
        The six ``moe_sort`` outputs: ``(tile_idx_to_expert_idx,
        tile_idx_to_mn_limit, expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx, total_num_padded_tokens,
        num_non_exiting_tiles)``.
    """
    esp = weight_view.expert_size_per_partition
    slot_start = weight_view.slot_start
    presorted = presorted_routing(token_selected_experts, slot_start, esp)
    if presorted is not None and presorted[0] == tile_size:
        return presorted[1]
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=backend.num_slots,
        top_k=token_selected_experts.size(1),
        local_expert_offset=slot_start,
        local_num_experts=esp,
        tile_tokens_dim=tile_size,
    )


def finalize_routing_scales(backend: MoEImplBase, ctx: MoERunContext) -> torch.Tensor:
    """Per-token expert weights for a CuteDSL finalize epilogue.

    With ``apply_router_weight_on_input`` the scheduler has already folded the
    top-1 weight into ``x`` and may pass no scales; ones keep the finalize from
    weighting twice.

    Args:
        backend: The leaf running ``ctx``; read for ``apply_router_weight_on_input``.
        ctx: The run context; ``token_final_scales`` is ``[num_tokens, top_k]``
            float32 or ``None``.

    Returns:
        ``[num_tokens, top_k]`` float32 scales.
    """
    if ctx.token_final_scales is not None:
        return ctx.token_final_scales
    assert backend.apply_router_weight_on_input, (
        "token_final_scales is None without apply_router_weight_on_input"
    )
    return torch.ones_like(ctx.token_selected_experts, dtype=torch.float32)


def quantize_input_nvfp4(
    self: MoEImplBase,
    x: torch.Tensor | Fp4QuantizedTensor,
    post_quant_comm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to NVFP4, or unpack an already-quantized input.

    The ``quantize_input`` the CuteDSL NVFP4 leaves bind. On an NVFP4_AWQ /
    NVFP4_ARC layer, ``x`` is first scaled by ``fc31_act_scale``, the
    checkpoint's per-channel pre-quant scale.

    Args:
        x: ``[num_tokens, hidden_size]`` bfloat16 activations, or an
            unswizzled ``Fp4QuantizedTensor`` quantized upstream.
        post_quant_comm: Ignored; both paths produce the same layout.

    Returns:
        ``(x, x_sf)``: packed FP4 ``[num_tokens, hidden_size // 2]`` and linear
        (unswizzled) scale factors
        ``[num_tokens, ceil_div(hidden_size, scaling_vector_size)]``, the layout
        alltoall/allgather move.
    """
    del post_quant_comm
    if getattr(self, "fc31_act_scale", None) is not None:
        assert not isinstance(x, Fp4QuantizedTensor), (
            "Fp4QuantizedTensor is not expected for AWQ quantization."
        )
        x = x * self.fc31_act_scale
    x_row = x.shape[0]
    if isinstance(x, Fp4QuantizedTensor):
        assert not x.is_sf_swizzled, (
            "Fp4QuantizedTensor should not be swizzled before communication"
        )
        x, x_sf = x.fp4_tensor, x.scaling_factor
    else:
        x, x_sf = torch.ops.trtllm.fp4_quantize(
            x, self.fc31_input_scale, self.scaling_vector_size, False, False
        )
    # ``view(0, -1)`` is ambiguous for an empty micro-batch, so spell out the
    # scale width.
    return x, x_sf.view(x_row, ceil_div(self.hidden_size, self.scaling_vector_size))
