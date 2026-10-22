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

from ....utils import Fp4QuantizedTensor
from ..impl_base import MoEImplBase


@dataclass
class NvFp4WeightView:
    """Bundles all NVFP4 weight tensors for MoE computation.

    Under the VA-based DWDP pipeline ``param.data`` is swapped to a
    composite [num_experts, ...] tensor before the kernel call, so every
    field is a single tensor — the bundle is just a convenient grouping
    that lets the runner forward a single object instead of six.
    """

    w3_w1_weight: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc1_global_scale: torch.Tensor
    w2_weight: torch.Tensor
    fc2_weight_scale: torch.Tensor
    fc2_global_scale: torch.Tensor
    expert_size_per_partition: int
    slot_start: int


def quantize_input_nvfp4(
    self: MoEImplBase,
    x: torch.Tensor | Fp4QuantizedTensor,
    post_quant_comm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to NVFP4, or unpack an already-quantized input.

    The ``quantize_input`` the CuteDSL NVFP4 leaves bind. On an NVFP4_AWQ /
    NVFP4_ARC layer, ``x`` is first scaled by ``fc31_act_scale``, the
    checkpoint's per-channel pre-quant scale. Returns ``(x, x_sf)`` with
    ``x_sf`` linear (unswizzled) and shaped
    ``[num_tokens, ceil_div(hidden_size, scaling_vector_size)]``, the layout
    alltoall/allgather move.
    """
    del post_quant_comm  # Both paths produce the same layout.
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
    scale_cols = (self.hidden_size + self.scaling_vector_size - 1) // self.scaling_vector_size
    return x, x_sf.view(x_row, scale_cols)
