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
"""``trtllm.cutlass.grouped_gemm.none``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import UnquantizedFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor
from .lora import CutlassMoELoraMixin


@register_moe_impl
class TrtllmCutlassUnquantizedImpl(CutlassMoELoraMixin, CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.none``.

    No quantization, on every architecture the CUTLASS MoE is built for. One of
    the two formats the op fuses routed-expert LoRA for.
    """

    descriptor = cutlass_descriptor(
        "none",
        "CUTLASS grouped GEMM over unquantized fp16/bf16 weights, SM80+.",
        supports_moe_lora=True,
    )
    sm_support = SmSupport(minimum=80)
    supported_dtypes = HP_DTYPES
    supports_gptoss_style = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return UnquantizedFusedMoEMethod()
