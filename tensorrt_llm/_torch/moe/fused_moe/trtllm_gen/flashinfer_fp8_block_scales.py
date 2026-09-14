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
"""``flashinfer.trtllm_gen.fused_moe.fp8_block_scales``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .eligibility import check_flashinfer_provider, check_trtllm_gen_leaf
from .fp8_block_scale import TRTLLMGenFp8BlockScalesBase
from .identity import PROVIDER_FLASHINFER, FlashinferProviderTraits, trtllm_gen_descriptor


@register_moe_impl
class FlashinferTrtllmGenFp8BlockScalesImpl(FlashinferProviderTraits, TRTLLMGenFp8BlockScalesBase):
    """``flashinfer.trtllm_gen.fused_moe.fp8_block_scales``.

    No alignment gate: the padding rules belong to the FP4 weight layouts, and
    this format has none of them.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER,
        "fp8_block_scales",
        "FlashInfer's TRTLLM-Gen FP8 block-scale fused MoE, SM100 family.",
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d, check_flashinfer_provider(cls, p, d))
