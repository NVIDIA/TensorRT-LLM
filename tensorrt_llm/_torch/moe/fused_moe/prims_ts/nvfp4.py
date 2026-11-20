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
"""``FlashinferPrimsTsNvfp4Impl``: ``flashinfer.prims_ts.batched_gemm.nvfp4``."""

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERejectReason
from ..impl_identity import register_moe_impl
from ..interface import _reject
from ..quantization import NVFP4TRTLLMGenFusedMoEMethod
from .base import PrimsTsFusedMoEBase
from .eligibility import check_prims_ts_leaf
from .identity import prims_ts_descriptor


@register_moe_impl
class FlashinferPrimsTsNvfp4Impl(PrimsTsFusedMoEBase):
    """``flashinfer.prims_ts.batched_gemm.nvfp4``: NVFP4 weights and activations.

    Weights are laid out by ``NVFP4TRTLLMGenFusedMoEMethod``, which pads each
    rank's intermediate shard to the 128-multiple the kernels need; the input
    is quantized with linear (unswizzled) group-16 scale factors.
    """

    descriptor = prims_ts_descriptor(
        "nvfp4",
        "FlashInfer PrimsTS FC1/FC2 batched GEMMs, NVFP4 weights and activations, "
        "SM100/SM103/SM107.",
    )
    intermediate_shard_alignment = NVFP4TRTLLMGenFusedMoEMethod.scaling_vector_size

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_prims_ts_leaf(cls, p, d, cls._check_hidden_not_padded(p, d))

    @classmethod
    def _check_hidden_not_padded(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
        """Decline a hidden size the NVFP4 weight method would pad.

        The finalize writes FC2's rows into the layer output directly, so FC2's
        output width must be the layer's hidden size.
        """
        if p.hidden_size is None or p.intermediate_size is None:
            return None
        weight_alignment, input_alignment = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(
            p.hidden_size, p.intermediate_size // d.tp_size
        )
        for alignment in (weight_alignment, input_alignment):
            if p.hidden_size % alignment != 0:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"{cls.__name__} requires hidden_size ({p.hidden_size}) to stay unpadded, "
                    f"but the NVFP4 weight layout pads it to a multiple of {alignment}",
                )
        return None

    def _get_quant_method(self) -> object:
        return NVFP4TRTLLMGenFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor | Fp4QuantizedTensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del post_quant_comm
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, "PrimsTS reads linear NVFP4 scale factors"
            return x.fp4_tensor, x.scaling_factor.view(x.shape[0], -1)
        # The pre-quant scale of NVFP4_AWQ checkpoints, shape [1, hidden_size].
        if getattr(self, "fc31_act_scale", None) is not None:
            x = x * self.fc31_act_scale
        num_tokens = x.shape[0]
        x, x_sf = torch.ops.trtllm.fp4_quantize(
            x, self.fc31_input_scale, self.scaling_vector_size, False, False
        )
        return x, x_sf.view(num_tokens, -1)
