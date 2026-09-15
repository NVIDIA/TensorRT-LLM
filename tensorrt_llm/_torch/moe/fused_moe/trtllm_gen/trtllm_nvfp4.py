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
"""``trtllm.trtllm_gen.fused_moe.nvfp4``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .eligibility import check_trtllm_gen_leaf
from .fp4_block_scale import TRTLLMGenNvfp4Base
from .identity import PROVIDER_TRTLLM, TrtllmProviderTraits, trtllm_gen_descriptor


@register_moe_impl
class TrtllmTrtllmGenNvfp4Impl(TrtllmProviderTraits, TRTLLMGenNvfp4Base):
    """``trtllm.trtllm_gen.fused_moe.nvfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "nvfp4", "TRTLLM-Gen batched-GEMM cubins over NVFP4, SM100 family."
    )

    #: The group-16 fused SiTu cubins exist for this format, but only the
    #: native op backend calls them, so eligibility is per leaf and not on
    #: ``TRTLLMGenNvfp4Base`` (which the FlashInfer leaf shares).
    supports_situ = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)
