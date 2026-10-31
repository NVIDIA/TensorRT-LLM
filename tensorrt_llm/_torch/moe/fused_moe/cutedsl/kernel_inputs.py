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

from ....kernels.nvfp4_quant import can_use_nvfp4_quant, nvfp4_quant
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
    *,
    enable_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to NVFP4, or unpack an already-quantized input.

    The ``quantize_input`` the CuteDSL NVFP4 leaves bind. On an NVFP4_AWQ /
    NVFP4_ARC layer, ``x`` is first scaled by ``fc31_act_scale``, the
    checkpoint's per-channel pre-quant scale.

    Args:
        x: ``[num_tokens, hidden_size]`` bfloat16 activations, or an
            unswizzled ``Fp4QuantizedTensor`` quantized upstream.
        post_quant_comm: Ignored; both paths produce the same layout.
        enable_triton: Allow tiled quantization for eligible inputs on this
            NVFP4-capable backend. False uses the CUDA quantizer.

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
    elif (
        enable_triton
        and self.scaling_vector_size == 16
        and can_use_nvfp4_quant(x, self.fc31_input_scale)
    ):
        x, x_sf = nvfp4_quant(x, self.fc31_input_scale)
    else:
        x, x_sf = torch.ops.trtllm.fp4_quantize(
            x, self.fc31_input_scale, self.scaling_vector_size, False, False
        )
    # ``view(0, -1)`` is ambiguous for an empty micro-batch, so spell out the
    # scale width.
    return x, x_sf.view(x_row, ceil_div(self.hidden_size, self.scaling_vector_size))
