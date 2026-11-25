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
"""``FlashinferPrimsTsW4a8Mxfp4Mxfp8Impl``: ``flashinfer.prims_ts.batched_gemm.w4a8_mxfp4_mxfp8``."""

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod
from .base import PrimsTsFusedMoEBase
from .eligibility import check_prims_ts_leaf
from .identity import prims_ts_descriptor


@register_moe_impl
class FlashinferPrimsTsW4a8Mxfp4Mxfp8Impl(PrimsTsFusedMoEBase):
    """``flashinfer.prims_ts.batched_gemm.w4a8_mxfp4_mxfp8``: MXFP4 weights, MXFP8 activations.

    Weights are laid out by ``W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod``. It pads
    FC1's input to a multiple of 512 and FC2's output to a multiple of 128, so
    a hidden size that is a multiple of 512 keeps both at the layer's width;
    the MXFP8 input is quantized to the same 512 alignment with linear scale
    factors.
    """

    descriptor = prims_ts_descriptor(
        "w4a8_mxfp4_mxfp8",
        "FlashInfer PrimsTS FC1/FC2 batched GEMMs, MXFP4 weights and MXFP8 activations, "
        "SM100/SM103/SM107.",
    )
    hidden_size_alignment = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.input_hidden_alignment
    intermediate_shard_alignment = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.scaling_vector_size

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_prims_ts_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del post_quant_comm
        x, x_sf = torch.ops.trtllm.mxfp8_quantize(
            x, False, alignment=self.quant_method.input_hidden_alignment
        )
        return x, x_sf.view(x.shape[0], -1)
