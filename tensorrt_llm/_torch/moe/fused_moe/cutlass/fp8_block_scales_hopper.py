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
"""``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import DeepSeekFP8BlockScalesFusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags
from .eligibility import BF16_ONLY, SmSupport, check_cutlass_leaf
from .identity import (
    KERNEL_HOPPER_GROUPED_GEMM,
    PROVIDER_DEEPGEMM,
    TECHNIQUE_CUDA,
    cutlass_descriptor,
)


@register_moe_impl
class DeepgemmCudaHopperFp8BlockScalesImpl(CutlassFusedMoEBase):
    """``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``.

    FP8 block scales on SM90. ``use_deepseek_fp8_block_scale=True`` makes
    ``CutlassMoeFCRunner`` run FC1/FC2 through
    ``cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/``, which is
    built on TRT-LLM's DeepGEMM port (``cpp/include/tensorrt_llm/deep_gemm``),
    hence the identity; permute, activation and finalize stay in the CUTLASS
    runner, hence the family. The runner quantizes activations itself. Only
    ``kernel_name`` separates this identity from the SM100/103/107 leaf in
    ``..fused_moe_deepgemm``.
    """

    descriptor = cutlass_descriptor(
        "fp8_block_scales",
        "DeepGEMM grouped GEMM over FP8 block scales, driven by CutlassMoeFCRunner, SM90.",
        provider=PROVIDER_DEEPGEMM,
        technique=TECHNIQUE_CUDA,
        kernel_name=KERNEL_HOPPER_GROUPED_GEMM,
    )
    sm_support = SmSupport(allowed=frozenset({90}))
    #: The block-scale GEMM runner only has BF16 A / output instantiations.
    supported_dtypes = BF16_ONLY
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(use_deepseek_fp8_block_scale=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return DeepSeekFP8BlockScalesFusedMoEMethod()
