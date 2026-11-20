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
"""``FlashinferPrimsTsBf16Impl``: ``flashinfer.prims_ts.batched_gemm.none``."""

import torch

from ....utils import ActivationType
from ..activation import ActivationParamShape, MoEActivationSupport
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import BF16TRTLLMGenFusedMoEMethod
from .base import PrimsTsFusedMoEBase
from .eligibility import check_prims_ts_leaf
from .identity import prims_ts_descriptor


@register_moe_impl
class FlashinferPrimsTsBf16Impl(PrimsTsFusedMoEBase):
    """``flashinfer.prims_ts.batched_gemm.none``: BF16 activations and weights.

    Weights are stored in the shuffled BlockMajorK layout
    ``BF16TRTLLMGenFusedMoEMethod`` produces, which does not pad, so each
    rank's intermediate shard must already be a multiple of the kernels' 128.
    """

    descriptor = prims_ts_descriptor(
        "none",
        "FlashInfer PrimsTS FC1/FC2 batched GEMMs, BF16 activations and weights, "
        "SM100/SM103/SM107.",
    )
    intermediate_shard_alignment = 128

    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu}),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_prims_ts_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return BF16TRTLLMGenFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, None]:
        """BF16 activations go to the kernels as they are; no scale factors."""
        del post_quant_comm
        return x, None
